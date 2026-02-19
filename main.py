"""
====================================================================================================
                                            CNN MODEL - Vitiscan
                          Classification of grape diseases with a Resnet18 model
====================================================================================================

"""



#  I. LIBRARIES IMPORT
# ================================================================================================

import os
import mlflow
from dotenv import load_dotenv

# Torch
import torch.nn as nn
import torch.optim as optim

# Visualization
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Local functions (scripts/)
from scripts.data_utils import download_dataset, extract_and_prepare_dataset, create_transforms, prepare_datasets, create_dataloaders
from scripts.model import get_device, create_model
from scripts.training import train
from scripts.visualisation import plot_training_history

load_dotenv()


#  II. MAIN EXECUTION
# ================================================================================================

def main():
    """Main execution function"""
    
    # Retrieve device
    device = get_device() 
    
    # Configuration
    DATA_DIR = os.getenv('DATA_DIR', '../data')
    DATASET_URL = os.getenv('DATASET_URL','https://www.kaggle.com/api/v1/datasets/download/codewithsk/grapes-leafs-disease-7-classes-plantcity-2025')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE','32'))
    LEARNING_RATE =float(os.getenv('LEARNING_RATE','0.0005'))
    EPOCHS = int(os.getenv('EPOCHS','10'))
    TRAIN_SPLIT =float(os.getenv('TRAIN_SPLIT','0.8'))
    EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME',"Vitiscan_CNN_Resnet18")
    MLFLOW_URI=os.getenv('MLFLOW_URI',"https://gviel-mlflow37.hf.space/")
    MODEL_NAME=os.getenv('MODEL_NAME','resnet18')
    
    # MLFlow setup

    mlflow.set_tracking_uri(MLFLOW_URI)
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    
    print("="*80)
    print("VITISCAN - CNN MODEL TRAINING")
    print("="*80)
    
    # 1. Dataset preparation
    print("\n[1/6] Dataset preparation...")
    os.makedirs(DATA_DIR, exist_ok=True)
    data_zip_path = download_dataset(DATA_DIR, DATASET_URL)
    labels = extract_and_prepare_dataset(DATA_DIR, data_zip_path)
    
    # 2. Data loading
    print("\n[2/6] Data loading...")
    transform = create_transforms()
    train_dataset, val_dataset, test_dataset, class_names = prepare_datasets(
        DATA_DIR, transform, TRAIN_SPLIT
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE
    )
    
    # 3. Model creation
    print("\n[3/6] Model creation...")
    num_classes = len(class_names)
    # model, _ = create_model(MODEL_NAME,num_classes, device, freeze_base=True,unfreeze_layer="layer4")
    model= create_model(MODEL_NAME, num_classes, device,freeze_base=True,unfreeze_layer="layer4")

    # 4. Training setup
    print("\n[4/6] Training setup...")
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # 5. Training
    print("\n[5/6] Training...")
    history = train(
        model_name=MODEL_NAME,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=EPOCHS,
        patience=int(os.getenv('PATIENCE', '5')), 
        experiment_id=experiment.experiment_id,
        class_names=class_names
    )
    
    # 6. Visualization
    print("\n[6/6] Visualization...")
    plot_training_history(history, "ResNet18 Fine tuned")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()