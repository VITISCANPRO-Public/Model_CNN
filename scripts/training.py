
#                                          LIBRARIES IMPORT
# ================================================================================================

import os
import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#                                        TRAINING
# ================================================================================================

def train(model_name, model: nn.Module, train_loader: torch.utils.data.DataLoader, 
          val_loader: torch.utils.data.DataLoader, criterion: nn.Module, 
          optimizer: torch.optim.Optimizer, device: torch.device, 
          epochs: int = 10, experiment_id: str = None, class_names: list = None):
    """
    Trains the model with MLFlow logging
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device
        epochs: Number of epochs
        experiment_id: MLFlow experiment ID
        class_names: Names of the classes
    
    Returns:
        history: Dictionary containing the training metrics
    """
    
    # Start MLFlow run
    with mlflow.start_run(experiment_id=experiment_id):
        
        # # Log parameters
        # params = {
        #     "optimizer": type(optimizer).__name__,
        #     "learning_rate": optimizer.param_groups[0]['lr'],
        #     "epochs": epochs,
        #     "criterion": type(criterion).__name__,
        #     "model_architecture": type(model).__name__,
        #     "training_device": str(device),
        #     "num_classes": len(class_names) if class_names else "unknown"
        # }
        # mlflow.log_params(params=params)

        # # -> Check the difference between autolog and params

        mlflow.pytorch.autolog() 
        
        # History
        history = {
            'loss': [], 
            'val_loss': [], 
            'accuracy': [], 
            'val_accuracy': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # ========== TRAINING ==========
            model.train()
            total_loss, correct = 0, 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                correct += (logits.argmax(dim=1) == labels).sum().item()
            
            train_loss = total_loss / len(train_loader)
            train_acc = correct / len(train_loader.dataset)
            
            # ========== VALIDATION ==========
            model.eval()
            val_loss, val_correct = 0, 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    loss = criterion(logits, labels)
                    
                    val_loss += loss.item()
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / len(val_loader.dataset)
            
            # Store metrics
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            
            # Print progress
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("validation_accuracy", val_acc, step=epoch)
        
        # ========== FINAL METRICS ==========
        # Collect predictions
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = logits.argmax(dim=1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        # F1 Score
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        mlflow.log_metric("final_validation_accuracy", val_acc)
        mlflow.log_metric("final_train_loss", train_loss)
        mlflow.log_metric("f1_score_weighted", f1_weighted)
        mlflow.log_metric("f1_score_macro", f1_macro)
        
        print(f"\nFinal Metrics:")
        print(f"  F1 Score (weighted): {f1_weighted:.4f}")
        print(f"  F1 Score (macro): {f1_macro:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names or range(len(cm)),
                    yticklabels=class_names or range(len(cm)))
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        
        cm_path = 'confusion_matrix.png'
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(cm_path)
        plt.close()
        
        print("Confusion matrix saved")
        
        # Log model
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="Resnet18",
            registered_model_name=f"{type(model).__name__}"
        )
        
        print("--- Metrics and model logged into MLFlow ---")
    
    return history