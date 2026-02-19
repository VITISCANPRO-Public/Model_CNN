
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

def create_model_transfer_learning(model_name:str, num_classes: int, device: torch.device, freeze_conv: bool = True):
    """
    Creates a model for transfer learning
    
    Args:
        model name : Name of the model to train
        num_classes: Number of classes to predict
        device: Device (cuda/mps/cpu)
        freeze_conv: If True, freezes all base model parameters 
        and trains only the classification head.

    Returns:
        model, trainable_params
    """
    
    if model_name=='resnet18':
        # Load the pre-trained model
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name=='resnet34':
        # Load the pre-trained model
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

    model = model.to(device)
    
    # Modify the last layer (Classifier)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Freeze convolutional layers except the last one
    if freeze_conv:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last layer (the new one)
        for param in model.fc.parameters():
            param.requires_grad = True
    
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return model, trainable_params




def create_model_fine_tuning(model_name : str, num_classes: int, device: torch.device, unfreeze_layer: str = "layer4"):   
    """
    Creates a model for fine-tuning
    
    Args:
        model_name: Name of the model to train
        num_classes: Number of classes
        device: Device (cuda/mps/cpu)
        unfreeze_layer: Name of the layer to unfreeze ("layer4" : the last layer, by default)
    
    Returns:
        model
    """
    # Load the pre-trained model based on model_name
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Choose 'resnet18' or 'resnet34'.")
    
    # Modify the last layer (Classifier)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Freeze the entire network
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the specified layer and the final layer
    for name, param in model.named_parameters():
        if unfreeze_layer in name or "fc" in name:
            param.requires_grad = True
    
    model = model.to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Fine-tuning - Trainable parameters: {trainable_params:,}")
    
    return model