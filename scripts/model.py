
#                                          LIBRARIES IMPORT
# ================================================================================================

import torch
from torchvision import models
import torch.nn as nn
from torchinfo import summary


#                                         DEVICE SELECTION
# ================================================================================================

def get_device():
    """
    Selects the best available device (CUDA, MPS, or CPU)
    Returns : 
        Device : Best device that can be used for the training
    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # For Apple M1/M2/M3
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using {device} device")
    return device


#                                       MODEL CREATION
# ================================================================================================

def create_model(model_name: str, num_classes: int, device: torch.device, 
                 freeze_base: bool = True, unfreeze_layer: str = "layer4"):
    """
    Creates a model for transfer learning or fine-tuning.
    Supports ResNet, EfficientNet and MobileNet architectures.
    The following models require GPU or longer training time:
    'resnet50', 'efficientnet_b1', 'efficientnet_b2','mobilenet_v2'

    Args:
        model_name: Model name ('resnet18', 'resnet34', 'resnet50', 
                    'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                    'mobilenet_v2')
        num_classes: Number of classes to predict
        device: Device (cuda/mps/cpu)
        freeze_base: If True, freezes all layers except the classifier
        unfreeze_layer: Layer to unfreeze for fine-tuning (ResNet only, e.g. 'layer4')

    Returns:
        model
    """

    # ---- Load pre-trained model ----
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
    elif model_name == 'efficientnet_b2':
        model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}. "
                         f"Choose from: resnet18, resnet34, resnet50, "
                         f"efficientnet_b0/b1/b2, mobilenet_v2")

    # ---- Replace the classifier layer ----
    # Each architecture has a different name for its final classification layer
    if 'resnet' in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        classifier_param_names = ["fc"]

    elif 'efficientnet' in model_name:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        classifier_param_names = ["classifier"]

    elif 'mobilenet' in model_name:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        classifier_param_names = ["classifier"]

    # ---- Freeze / unfreeze layers ----
    if freeze_base:
        # Freeze everything
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze classifier + optionally one intermediate layer (ResNet only)
        for name, param in model.named_parameters():
            is_classifier = any(c in name for c in classifier_param_names)
            is_unfreeze_layer = (unfreeze_layer in name) and ('resnet' in model_name)
            if is_classifier or is_unfreeze_layer:
                param.requires_grad = True
    # If freeze_base=False, all layers are trainable (full fine-tuning)

    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | "
          f"Trainable: {trainable_params:,} / {total_params:,} params")

    return model