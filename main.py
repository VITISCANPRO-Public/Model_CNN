"""
====================================================================================================
                                        CNN MODEL - Vitiscan
               Classification of grape leaf diseases with multiple CNN architectures
====================================================================================================
"""

#                                         LIBRARIES IMPORT
# ================================================================================================

import os
import mlflow
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from pathlib import Path

# Local functions (scripts/)
from scripts.config_utils import load_config
from scripts.data_utils import (download_dataset, extract_and_prepare_kaggle,
                                 build_combined_dataset, prepare_datasets,
                                 create_dataloaders)
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
    INRAE_DIR = Path(os.getenv('INRAE_DIR', config['data'].get('inrae_dir','./data-inrae')))
    KAGGLE_DIR = Path(os.getenv('KAGGLE_DIR',config['data'].get('kaggle_dir','./data-kaggle')))
    OUTPUT_DIR = Path(os.getenv('DATA_DIR',config['data'].get('output_dir','./data-combined')))
    KAGGLE_URL = os.getenv('DATASET_URL')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE',config['default_training']['batch_size']))
    MLFLOW_URI = os.getenv('MLFLOW_URI')
    EXPERIMENT_NAME = os.getenv('EXPERIMENT_NAME', config['experiment_name'])

    # --- MLflow setup ---
    mlflow.set_tracking_uri(MLFLOW_URI)

    print("=" * 60)
    print("VITISCAN - CNN MODEL TRAINING")
    print(f"Experiment : {EXPERIMENT_NAME}")
    print(f"INRAE data : {INRAE_DIR}")
    print(f"Kaggle data: {KAGGLE_DIR}")
    print(f"Output data: {OUTPUT_DIR}")
    print("=" * 60)

    # =========================================================================
    # STEP 1 — Dataset preparation (done once for all models)
    # =========================================================================
    print("\n[1/3] Dataset preparation...")

    # Download and extract Kaggle dataset (needed for the healthy class)
    os.makedirs(str(KAGGLE_DIR), exist_ok=True)
    data_zip_path = download_dataset(str(KAGGLE_DIR), KAGGLE_URL)
    extract_and_prepare_kaggle(str(KAGGLE_DIR), data_zip_path)

    # Build the combined dataset:
    # - 6 disease classes from INRAE (scientific names)
    # - 1 healthy class from Kaggle (train + test merged and balanced)
    build_combined_dataset(
        inrae_dir=INRAE_DIR,
        kaggle_dir=KAGGLE_DIR,
        output_dir=OUTPUT_DIR
    )

    # =========================================================================
    # STEP 2 — Data loading (done once for all models)
    # =========================================================================
    print("\n[2/3] Data loading...")

    # Input size 224 by default — EfficientNet variants may use different sizes
    # but we use 224 as a common baseline for fair model comparison
    train_dataset, val_dataset, test_dataset, class_names = prepare_datasets(
        data_dir=str(OUTPUT_DIR),
        input_size=224
    )
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, BATCH_SIZE
    )
    num_classes = len(class_names)

    # =========================================================================
    # STEP 3 — Training loop over all models defined in config.yml
    # =========================================================================
    print("\n[3/3] Training all models...")
    models_to_run = config['models_to_run']

    print(f"{len(models_to_run)} models to train: "
          f"{[m['name'] + ' (FT)' if not m['freeze_base'] else m['name'] + ' (TL)' for m in models_to_run]}\n")

    for i, model_cfg in enumerate(models_to_run):

        model_name = model_cfg['name']
        freeze_base = model_cfg['freeze_base']
        unfreeze_layer = model_cfg.get('unfreeze_layer', 'layer4')
        # Model-specific lr overrides global default if specified in config
        lr = float(model_cfg.get('learning_rate',
                               config['default_training']['learning_rate']))
        epochs = int(os.getenv('EPOCHS',  config['default_training']['epochs']))
        patience = int(os.getenv('PATIENCE', config['default_training'].get('patience', 5)))

        mode = "Transfer Learning" if freeze_base else "Fine-tuning"
        run_experiment_name = f"{EXPERIMENT_NAME}_{model_name}_{mode}"

        mlflow.set_experiment(run_experiment_name)
        experiment = mlflow.get_experiment_by_name(run_experiment_name)

        print("=" * 60)
        print(f"Model {i+1}/{len(models_to_run)}: {run_experiment_name}")
        print(f"  Description : {model_cfg.get('description', '')}")
        print(f"  LR: {lr} | Epochs: {epochs} | Patience: {patience}")
        print("=" * 60)

        # --- Create model ---
        model = create_model(
            model_name=model_name,
            num_classes=num_classes,
            device=device,
            freeze_base=freeze_base,
            unfreeze_layer=unfreeze_layer
        )

        # --- Optimizer and loss ---
        # filter(requires_grad) ensures compatibility with all architectures
        # (ResNet uses .fc, EfficientNet/MobileNet use .classifier)
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr
        )

        # --- Train ---
        history = train(
            model_name=run_experiment_name,
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

        # --- Visualize training curves ---
        plot_training_history(history, run_experiment_name)

    print("\n" + "=" * 60)
    print(f"ALL {len(models_to_run)} MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()