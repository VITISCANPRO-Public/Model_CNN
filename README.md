# Vitiscan — CNN Model Training

Grape leaf disease classification using fine-tuned CNN architectures (ResNet, EfficientNet).
Part of the **Vitiscan MLOps pipeline** for automated vineyard disease detection.

## Problem

Early detection of vine diseases is critical for yield protection. This module trains
a multi-class classifier to identify 6 diseases + healthy leaves from a single photo.

**Classes:** Colomerus vitis, Elsinoe ampelina, Erysiphe necator, Guignardia bidwellii,
Phaeomoniella chlamydospora, Plasmopara viticola, Healthy

## Dataset

Two sources combined:
- **INRAE** — 6 disease classes scraped from labelled scientific images
- **Kaggle** — Healthy class only (balanced to ~350 images/class)

> The Kaggle dataset was initially used for all classes but discarded due to
> labelling inconsistencies. INRAE data provided higher quality ground truth.

Final split: **1040 train / 222 val / 231 test** (70/15/15)

## Models & Results

| Model | Strategy | Test Accuracy | F1 Macro | Epochs |
|-------|----------|--------------|----------|--------|
| ResNet18 | Transfer Learning | 94.8% | 0.946 | 17 |
| **ResNet18** | **Fine-tuning** | **98.3%** | **0.982** | **13** |
| ResNet34 | Transfer Learning | 95.7% | 0.951 | 35 |
| EfficientNet B0 | Fine-tuning | 97.0% | 0.966 | 28 |

**Selected model: ResNet18 Fine-tuning** — best F1 macro with fastest convergence.
EfficientNet B0 is a strong alternative for edge/mobile deployment (4x lighter).

All experiments tracked with **MLflow** at https://mouniat-vitiscanpro-hf.hf.space

## Project Structure
```
Model-CNN/
├── main.py                 # Main training script
├── config.yml              # Models and hyperparameters configuration
├── scripts/
│   ├── config_utils.py     # YAML config loader
│   ├── data_utils.py       # Dataset download, preparation, dataloaders
│   ├── model.py            # Model creation (ResNet, EfficientNet, MobileNet)
│   ├── training.py         # Training loop with MLflow logging
│   └── visualisation.py    # Training curves
├── notebooks/
│   ├── CNN_model.ipynb     # Exploratory training (Kaggle dataset)
│   └── CNN_model_FT.ipynb  # Fine-tuning exploration (INRAE dataset)
└── tests/
    └── test_model.py
```

## Quickstart

**1. Setup environment**
```bash
conda env create -f environment.yml
conda activate vitiscan_cnn
```

**2. Configure your `.env`**
```bash
cp .env.example .env
# Fill in MLFLOW_URI, AWS credentials
```

**3. Run training**
```bash
python main.py
```

The script will automatically:
- Download the Kaggle dataset
- Build the combined INRAE + Kaggle dataset
- Train all models defined in `config.yml`
- Log metrics and models to MLflow

## Configuration

Edit `config.yml` to change models or hyperparameters:
```yaml
default_training:
  epochs: 45
  patience: 7
  learning_rate: 0.0001

models_to_run:
  - name: "resnet18"
    freeze_base: false
    unfreeze_layer: "layer4"
```

## Requirements

- Python 3.11
- PyTorch >= 2.5.0 with MPS/CUDA support
- MLflow 3.7.0
- See `environment.yml` for full list