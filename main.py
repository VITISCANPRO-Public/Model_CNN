"""
====================================================================================================
                                        CNN MODEL - Vitiscan
               Classification of grape leaf diseases with multiple CNN architectures
====================================================================================================
"""

#                                         LIBRARIES IMPORT
# ================================================================================================

import os
import yaml
import mlflow
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv

# Local functions (scripts/)
from scripts.data_utils import (download_dataset, extract_and_prepare_dataset,
                                 create_transforms, prepare_datasets, create_dataloaders)
from scripts.config_utils import load_config
from scripts.model import get_device, create_model
from scripts.training import train
from scripts.visualisation import plot_training_history

load_dotenv()


#                                           MAIN EXECUTION
# ================================================================================================

def main():
    """Main execution function — trains all models defined in config.yml"""

    # --- Load config ---
    config = load_config("config.yml")

    # --- Device ---
    device = get_device()

    # --- Global settings (from config, overridable by .env) ---
    DATA_DIR = os.getenv('DATA_DIR', config['data']['root_dir'])
    DATASET_URL = os.getenv('DATASET_URL')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', config['default_training']['batch_size']))
    TRAIN_SPLIT = float(os.getenv('TRAIN_SPLIT', config['data']['train_split']))
    MLFLOW_URI = os.getenv('MLFLOW_URI')
    EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', config['experiment_name'])

    # --- MLflow setup ---
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    print("=" * 50)
    print("VITISCAN - CNN MODEL TRAINING")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print("=" * 50)

    # --- 1. Dataset preparation (done once for all models) ---
    print("\n[1/3] Dataset preparation...")
    os.makedirs(DATA_DIR, exist_ok=True)
    data_zip_path = download_dataset(DATA_DIR, DATASET_URL)
    extract_and_prepare_dataset(DATA_DIR, data_zip_path)

    # --- 2. Data loading (done once for all models) ---
    print("\n[2/3] Data loading...")
    transform = create_transforms()
    train_dataset, val_dataset, test_dataset, class_names = prepare_datasets(
        DATA_DIR, transform, TRAIN_SPLIT
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE
    )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # --- 3. Training loop over all models in config ---
    print("\n[3/3] Training all models...")
    models_to_run = config['models_to_run']
    print(f"{len(models_to_run)} models to train: "
          f"{[m['name'] + ' (FT)' if not m['freeze_base'] else m['name'] + ' (TL)' for m in models_to_run]}")

    for i, model_cfg in enumerate(models_to_run):
        model_name = model_cfg['name']
        freeze_base = model_cfg['freeze_base']
        unfreeze_layer = model_cfg.get('unfreeze_layer', 'layer4')
        # Model-specific lr overrides default if specified in config
        lr = float(model_cfg.get('learning_rate',
                             config['default_training']['learning_rate']))
        epochs = int(os.getenv('EPOCHS',config['default_training']['epochs']))
        patience = int(os.getenv('PATIENCE', 5))

        mode = "Transfer Learning" if freeze_base else "Fine-tuning"
        run_label = f"{model_name} ({mode})"

        print("\n" + "=" * 50)
        print(f"Model {i+1}/{len(models_to_run)}: {run_label}")
        print(f"  Description : {model_cfg.get('description', '')}")
        print(f"  LR: {lr} | Epochs: {epochs} | Patience: {patience}")
        print("=" * 50)

        # Create model
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            device=device,
            freeze_base=freeze_base,
            unfreeze_layer=unfreeze_layer
        )

        # Optimizer — uses only trainable parameters (works for all architectures)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )

        # Train
        history = train(
            model_name=run_label,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            patience=patience,
            experiment_id=experiment.experiment_id,
            class_names=class_names
        )

        # Visualize
        plot_training_history(history, run_label)

    print("\n" + "=" * 50)
    print(f"ALL {len(models_to_run)} MODELS TRAINED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    main()
