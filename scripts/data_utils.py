
#                                          LIBRARIES IMPORT
# ================================================================================================

import os
from pathlib import Path
import shutil
import zipfile
import requests
from tqdm import tqdm

# Torch
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Visualization
import matplotlib.pyplot as plt


#                                    DATASET UTILITIES
# ================================================================================================

def clean_datadir(data_dir: str) -> None:
    """Deletes the 'train' and 'test' folders (and temporary ones) from the dataset directory."""
    dir_to_delete = [
        f'{data_dir}/train', 
        f'{data_dir}/test', 
        f'{data_dir}/train__', 
        f'{data_dir}/test__'
    ]
    for d in dir_to_delete:
        try:
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"Deleted: {d}")
        except Exception as e:
            print(f'Error: Could not delete: {d} - {e}')


def reorganize_dataset_dir(dest_data_dir: str) -> list:
    """
    Reorganizes the dataset:
      - Normalizes and renames class names
      - Removes directory redundancy (e.g., data/train/train -> data/train)
    
    Args:
        dest_data_dir: Destination directory containing the zip extraction
    
    Returns:
        List of normalized dataset labels
    """
    labels = set()
    
    for split in ['train', 'test']:
        data_root_path = Path(f'{dest_data_dir}/{split}/{split}')
        
        if not (data_root_path.exists() and data_root_path.is_dir()):
            continue
            
        # Normalization of class names
        for dir_label in data_root_path.iterdir():
            print(f"Processing: {dir_label}")
            
            # Normalization of the name
            new_label = (dir_label.name
                        .replace('Grape ', '')
                        .replace(' leaf', '')
                        .replace('_leaf', '')
                        .replace(' disease', '')
                        .replace(' ', '_')
                        .lower())
            labels.add(new_label)
            
            # Renaming the directory
            new_dir_label = dir_label.parent / new_label
            if not new_dir_label.exists():
                try:
                    shutil.move(str(dir_label), str(new_dir_label))
                    print(f"Renamed: {dir_label} -> {new_dir_label}")
                except Exception as e:
                    print(f'Error renaming {dir_label} to {new_dir_label}: {e}')
            else:
                print(f'{new_dir_label} already exists, skipped')
        
        # Removal of split/split redundancy
        print(f'Removing directory redundancy for {split}...')
        try:
            tmp_dir = f'{dest_data_dir}/{split}__'
            shutil.move(str(data_root_path), tmp_dir)
            shutil.rmtree(str(data_root_path.parent))
            shutil.move(tmp_dir, f'{dest_data_dir}/{split}')
            print(f'Successfully reorganized {split} directory')
        except Exception as e:
            print(f'Error moving {data_root_path}: {e}')
    
    return sorted(list(labels))


def download_dataset(data_dir: str, url: str) -> str:
    """
    Downloads the dataset from Kaggle if necessary
    
    Args:
        data_dir: Destination directory
        url: Kaggle dataset URL
    
    Returns:
        Path to the downloaded ZIP file
    """
    data_zip_path = f'{data_dir}/grapes-leafs-disease-7-classes-plantcity-2025.zip'
    
    if os.path.exists(data_zip_path):
        print(f"File already exists: {data_zip_path}")
        print("Skipping download.")
        return data_zip_path
    
    print(f"Downloading dataset to {data_zip_path}...")
    
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(data_zip_path, 'wb') as file, tqdm(
        desc=data_zip_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)
    
    print(f"Download complete: {data_zip_path}")
    return data_zip_path


def extract_and_prepare_dataset(data_dir: str, data_zip_path: str) -> list:
    """
    Extracts and prepares the dataset
    
    Args:
        data_dir: Destination directory
        data_zip_path: Path to the ZIP file
    
    Returns:
        List of normalized labels
    """
    # Check if already extracted
    if os.path.exists(f'{data_dir}/train') and os.path.exists(f'{data_dir}/test'):
        print(f'{data_zip_path} already extracted, skipped')
        # Retrieve existing labels
        train_path = Path(f'{data_dir}/train')
        return sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    
    # Prior cleanup
    clean_datadir(data_dir)
    
    # Extraction
    print(f'Extracting {data_zip_path}...')
    with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Reorganization
    labels = reorganize_dataset_dir(data_dir)
    print(f"Labels found: {labels}")
    
    return labels


#                                 DATA LOADING AND PREPROCESSING
# ================================================================================================

def create_transforms():
    """Creates the image transformation pipeline"""
    # Image transformation pipeline 
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5), # Simple data augmentation
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet means
            std=[0.229, 0.224, 0.225]   # ImageNet standard deviations
        )
    ])


def prepare_datasets(data_dir: str, transform, train_split: float = 0.8):
    """
    Prepares the train, validation, and test datasets
    
    Args:
        data_dir: Data directory
        transform: Transformations to apply
        train_split: Proportion of the training set in the train/val split
    
    Returns:
        train_dataset, val_dataset, test_dataset, class_names
    """
    # Full train dataset (will be split into train/val)
    train_root = Path(f"{data_dir}/train")
    full_train_dataset = ImageFolder(root=train_root, transform=transform)
    
    # Train/validation split 
    train_size = int(train_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, 
        [train_size, val_size]
    )
    
    # Test dataset
    test_root = Path(f"{data_dir}/test")
    test_dataset = ImageFolder(root=test_root, transform=transform)
    
    class_names = full_train_dataset.classes
    
    print(f"Train dataset: {train_size} images ({train_split*100:.1f}%)")
    print(f"Validation dataset: {val_size} images ({(1-train_split)*100:.1f}%)")
    print(f"Test dataset: {len(test_dataset)} images")
    print(f"Classes: {class_names}")
    
    return train_dataset, val_dataset, test_dataset, class_names


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size: int = 32):
    """Creates the DataLoaders"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def visualize_batch(train_loader: DataLoader, class_names: list, n_images: int = 10):
    """Visualizes a batch of images"""
    imgs, labels = next(iter(train_loader))
    
    for i, (img, label) in enumerate(zip(imgs, labels)):
        if i >= n_images:
            break
        
        true_label_name = class_names[label]
        print(f"Label: {true_label_name}")
        
        # Denormalize and display image
        plt.imshow(img.permute(1, 2, 0).numpy())
        plt.title(true_label_name)
        plt.axis('off')
        plt.show()