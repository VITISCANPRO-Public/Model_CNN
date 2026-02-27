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
├── main.py                         # Training pipeline entry point
├── config.yml                      # Models and hyperparameters configuration
├── requirements.txt                # Python dependencies (training)
├── Dockerfile                      # Training container image
├── .env.template                   # Environment variables template
├── .dockerignore                   # Files excluded from Docker context
├── scripts/
│   ├── __init__.py
│   ├── config_utils.py             # YAML config loader
│   ├── data_utils.py               # Dataset download, preparation, dataloaders
│   ├── model.py                    # Model creation (ResNet, EfficientNet, MobileNet)
│   ├── training.py                 # Training loop with MLflow logging
│   └── visualisation.py            # Training curves (Plotly)
├── notebooks/
│   ├── CNN_model.ipynb             # Exploratory training (Kaggle dataset)
│   └── CNN_model_FT.ipynb          # Fine-tuning exploration (INRAE dataset)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared pytest fixtures
│   ├── test_model.py               # Unit tests (model, config, transforms)
│   ├── Dockerfile                  # Test runner container (used by CI)
│   └── requirements.txt            # Lightweight test dependencies
├── data-inrae/
│   └── README.md                   # INRAE dataset documentation
├── data-kaggle/
│   └── README.md                   # Kaggle dataset documentation
├── environment.yml                 # Conda env (local CPU development)
├── env_vitiscan_cnn_gpu.yml        # Conda env (EC2 GPU training)
└── .github/
    └── workflows/
        └── ci.yaml                 # CI pipeline (GitHub Actions)

## Quickstart

**1. Setup environment**
```bash
conda env create -f environment.yml
conda activate vitiscan_cnn
```

**2. Configure your `.env`**
```bash
cp .env.template .env
# Fill in MLFLOW_URI, AWS credentials
```

**3. Run training**
```bash
python main.py
```

The script will automatically:
- Download the Kaggle dataset (for the healthy class)
- Build the combined INRAE + Kaggle dataset (7 balanced classes)
- Train all models defined in `config.yml`
- Log metrics, confusion matrices and models to MLflow

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

## Testing

Tests cover model creation, config loading, data transforms, freeze/unfreeze logic, and edge cases.

**Run locally:**
```bash
PYTHONPATH=. pytest tests/ -v --tb=short
```

**Run with Docker (same as CI):**
```bash
docker build -t vitiscan-tests -f tests/Dockerfile .
docker run --rm vitiscan-tests
```

## CI/CD — GitHub Actions

Every push to `main` triggers the CI pipeline:

| Step | What it does |
|------|-------------|
| **Build** | Builds the test Docker image from `tests/Dockerfile` |
| **Test** | Runs all unit tests with `pytest` inside the container |

The pipeline can also be triggered manually (`workflow_dispatch`) or by Airflow via `repository_dispatch`.

## Docker

Two Dockerfiles serve different purposes:

| File | Purpose | Command |
|------|---------|---------|
| `Dockerfile` | Training image — runs `main.py` | `docker build -t vitiscan . && docker run vitiscan` |
| `tests/Dockerfile` | Test runner — runs `pytest` | `docker build -f tests/Dockerfile -t vitiscan-tests .` |

## Requirements

- Python 3.11
- PyTorch >= 2.5.0 with MPS/CUDA support
- MLflow 3.7.0
- See `environment.yml` for full list


## Author

**Mounia Tonazzini** — Agronomist Engineer & Data Scientist and Data Engineer

- HuggingFace: [huggingface.co/MouniaT](https://huggingface.co/MouniaT)
- LinkedIn: [www.linkedin.com/in/mounia-tonazzini](www.linkedin.com/in/mounia-tonazzini)
- GitHub: [github/Mounia-Agronomist-Datascientist](https://github.com/Mounia-Agronomist-Datascientist)
- Email : mounia.tonazzini@gmail.com