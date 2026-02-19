#                                          LIBRARIES IMPORT
# ================================================================================================

import os
import random
import shutil
import zipfile
from pathlib import Path

import requests
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm


#                                    KAGGLE DATASET UTILITIES
# ================================================================================================

def download_dataset(data_dir: str, url: str) -> str:
    """
    Downloads the Kaggle dataset if not already present.

    Args:
        data_dir: Destination directory
        url: Kaggle dataset URL

    Returns:
        Path to the downloaded ZIP file
    """
    data_zip_path = f'{data_dir}/grapes-leafs-disease-7-classes-plantcity-2025.zip'

    if os.path.exists(data_zip_path):
        print(f"File already exists: {data_zip_path}, skipping download.")
        return data_zip_path

    print(f"Downloading dataset to {data_zip_path}...")
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    with open(data_zip_path, 'wb') as file, tqdm(
        desc=data_zip_path, total=total_size,
        unit='B', unit_scale=True, unit_divisor=1024
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)

    print(f"Download complete: {data_zip_path}")
    return data_zip_path


def extract_and_prepare_kaggle(data_dir: str, data_zip_path: str) -> None:
    """
    Extracts and reorganizes the Kaggle dataset.
    Fixes directory redundancy (data/train/train -> data/train)
    and normalizes class names.

    Args:
        data_dir: Destination directory
        data_zip_path: Path to the ZIP file
    """
    train_path = Path(data_dir) / "train"
    test_path  = Path(data_dir) / "test"

    if train_path.exists() and test_path.exists():
        print(f"Kaggle dataset already extracted at {data_dir}, skipping.")
        return

    # Cleanup and extraction
    _clean_datadir(data_dir)
    print(f"Extracting {data_zip_path}...")
    with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    # Fix structure and normalize names
    _reorganize_kaggle_dirs(data_dir)
    print("Kaggle dataset ready.")


def _clean_datadir(data_dir: str) -> None:
    """Removes train/test folders and temporary folders."""
    for d in [f'{data_dir}/train', f'{data_dir}/test',
              f'{data_dir}/train__', f'{data_dir}/test__']:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Deleted: {d}")


def _reorganize_kaggle_dirs(data_dir: str) -> None:
    """
    Fixes the Kaggle dataset structure:
    - Removes split/split redundancy (e.g. train/train -> train)
    - Normalizes class names (e.g. 'Grape Anthracnose leaf' -> 'anthracnose')
    """
    for split in ['train', 'test']:
        data_root_path = Path(f'{data_dir}/{split}/{split}')
        if not data_root_path.exists():
            continue

        # Normalize class names
        for dir_label in data_root_path.iterdir():
            if not dir_label.is_dir():
                continue
            new_label = (dir_label.name
                         .replace('Grape ', '')
                         .replace(' leaf', '')
                         .replace('_leaf', '')
                         .replace(' disease', '')
                         .replace(' ', '_')
                         .lower())
            new_dir = dir_label.parent / new_label
            if not new_dir.exists():
                shutil.move(str(dir_label), str(new_dir))

        # Remove split/split redundancy
        tmp = f'{data_dir}/{split}__'
        shutil.move(str(data_root_path), tmp)
        shutil.rmtree(str(data_root_path.parent))
        shutil.move(tmp, f'{data_dir}/{split}')
        print(f"Reorganized: {split}/")


#                                    COMBINED DATASET BUILDER
# ================================================================================================

def build_combined_dataset(inrae_dir: Path, kaggle_dir: Path,
                            output_dir: Path, target_nb: int = 350,
                            splits: dict = None, seed: int = 42) -> None:
    """
    Builds the final training dataset by combining:
    - 6 disease classes from INRAE (scientific names)
    - 1 healthy class ('healthy') from Kaggle (train + test merged)

    All classes are balanced to approximately target_nb images,
    then split into train/val/test.

    Args:
        inrae_dir:  Path to data-inrae/ (6 disease class folders at root)
        kaggle_dir: Path to data-kaggle/ (contains train/ and test/ subfolders)
        output_dir: Path where the final organized dataset will be created
        target_nb:  Target number of images per class for balancing
        splits:     Split ratios e.g. {'train': 0.7, 'val': 0.15, 'test': 0.15}
        seed:       Random seed for reproducibility
    """
    if splits is None:
        splits = {"train": 0.7, "val": 0.15, "test": 0.15}

    if output_dir.exists():
        print(f"Combined dataset already exists at {output_dir}, skipping.")
        return

    random.seed(seed)

    # --- Step 1: Collect images per class ---
    class_images = {}

    # Disease classes from INRAE
    for class_dir in sorted(inrae_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [img for img in class_dir.iterdir()
                  if img.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        class_images[class_dir.name] = images
        print(f"INRAE    - {class_dir.name:<40}: {len(images)} images")

    # Healthy class from Kaggle: merge train/healthy + test/healthy
    healthy_images = []
    for split in ["train", "test"]:
        
        for class_name in ["normal", "healthy", "sain"]:
            healthy_path = kaggle_dir / split / class_name
            if healthy_path.exists():
                imgs = [img for img in healthy_path.iterdir()
                        if img.suffix.lower() in {".jpg", ".jpeg", ".png"}]
                healthy_images.extend(imgs)
                print(f"Kaggle   - {split}/healthy: {len(imgs)} images collected")

    class_images["healthy"] = healthy_images
    print(f"Kaggle   - healthy (total):{len(healthy_images):>28} images")

    # --- Step 2: Balance all classes ---
    print(f"\nBalancing all classes to ~{target_nb} images per class:")
    balanced_images = {}
    for class_name, images in class_images.items():
        random.shuffle(images)
        kept = images[:target_nb]
        balanced_images[class_name] = kept
        print(f"  {class_name:<40}: {len(images)} -> {len(kept)} images kept")

    # --- Step 3: Create output folder structure ---
    for split in splits:
        for class_name in balanced_images:
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # --- Step 4: Split and copy images ---
    print("\nSplitting and copying images...")
    for class_name, images in balanced_images.items():
        n_total = len(images)
        n_train = int(n_total * splits["train"])
        n_val   = int(n_total * splits["val"])

        split_sets = {
            "train": images[:n_train],
            "val":   images[n_train:n_train + n_val],
            "test":  images[n_train + n_val:]
        }

        for split_name, split_imgs in split_sets.items():
            for img in split_imgs:
                shutil.copy(img, output_dir / split_name / class_name / img.name)

    # --- Step 5: Summary ---
    print(f"\nFinal dataset summary (output: {output_dir}):")
    for split in splits:
        total = 0
        print(f"\n  {split.upper()}")
        for class_dir in sorted((output_dir / split).iterdir()):
            if class_dir.is_dir():
                n = len(list(class_dir.glob("*")))
                total += n
                print(f"    {class_dir.name:<40}: {n}")
        print(f"    TOTAL: {total}")

    print(f"\nDataset successfully built at: {output_dir}")


#                                 DATA LOADING AND PREPROCESSING
# ================================================================================================

def create_transforms(input_size: int = 224, augment: bool = True):
    """
    Creates the image transformation pipeline.

    Args:
        input_size: Target image size (224 for ResNet, 240 for EfficientNet-B1, etc.)
        augment: If True, applies data augmentation (for training)

    Returns:
        transforms.Compose object
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet standard deviations
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def prepare_datasets(data_dir: str, input_size: int = 224):
    """
    Loads train, validation, and test datasets from the combined dataset.
    Expects data_dir to contain train/, val/, test/ subfolders.

    Args:
        data_dir: Path to the combined dataset (output of build_combined_dataset)
        input_size: Image size for transforms

    Returns:
        train_dataset, val_dataset, test_dataset, class_names
    """
    data_path = Path(data_dir)

    # Augmentation only for training set
    transform_train = create_transforms(input_size=input_size, augment=True)
    transform_eval  = create_transforms(input_size=input_size, augment=False)

    train_dataset = ImageFolder(root=data_path / "train", transform=transform_train)
    val_dataset   = ImageFolder(root=data_path / "val",   transform=transform_eval)
    test_dataset  = ImageFolder(root=data_path / "test",  transform=transform_eval)

    class_names = train_dataset.classes

    print(f"Train : {len(train_dataset)} images")
    print(f"Val   : {len(val_dataset)} images")
    print(f"Test  : {len(test_dataset)} images")
    print(f"Classes ({len(class_names)}): {class_names}")

    return train_dataset, val_dataset, test_dataset, class_names


def create_dataloaders(train_dataset, val_dataset, test_dataset,
                       batch_size: int = 32):
    """
    Creates DataLoaders for train, validation, and test sets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def visualize_batch(train_loader: DataLoader, class_names: list,
                    n_images: int = 9) -> None:
    """
    Visualizes a batch of training images in a grid.

    Args:
        train_loader: Training DataLoader
        class_names: List of class names
        n_images: Number of images to display
    """
    imgs, labels = next(iter(train_loader))

    cols = 3
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()

    for i in range(n_images):
        img = imgs[i].permute(1, 2, 0).numpy()
        # Denormalize
        img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        img = img.clip(0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[i]])
        axes[i].axis('off')

    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()