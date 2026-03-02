"""
scripts/ — Core modules for the Vitiscan CNN training pipeline.

Modules:
    config_utils  : YAML configuration loader
    data_utils    : Dataset download, preparation, and DataLoader creation
    model         : Model creation (ResNet, EfficientNet, MobileNet)
    training      : Training loop with MLflow logging and evaluation
    visualisation : Training curves (Plotly)
"""