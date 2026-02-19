#                                          LIBRARIES IMPORT
# ================================================================================================

import os
import tempfile
import mlflow
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


#                                            EVALUATION
# ================================================================================================

def evaluate_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                   device: torch.device):
    """
    Evaluates the model on a dataset and returns predictions.

    Args:
        model: PyTorch model
        data_loader: DataLoader to evaluate on
        device: Device (cuda/mps/cpu)

    Returns:
        accuracy, y_true, y_pred
    """
    model.eval()
    y_true, y_pred = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = correct / total
    return accuracy, y_true, y_pred


def log_metrics(dataset_type: str, y_true: list, y_pred: list):
    """
    Computes and logs precision, recall, F1 score into MLflow.

    Args:
        dataset_type: 'validation' or 'test'
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary of scores
    """
    results = {}
    for avg_mode in ['weighted', 'macro']:
        results[avg_mode] = {}
        for metric_name, metric_fn in [
            ('precision', precision_score),
            ('recall', recall_score),
            ('f1', f1_score)
        ]:
            value = metric_fn(y_true, y_pred, average=avg_mode, zero_division=0)
            results[avg_mode][metric_name] = value
            mlflow.log_metric(f"{dataset_type}_{metric_name}_{avg_mode}", value)

    return results


def log_confusion_matrix(dataset_type: str, y_true: list, y_pred: list, 
                         class_names: list, model_name: str):
    """
    Generates a confusion matrix and logs it into MLflow as an artifact.

    Args:
        dataset_type: 'validation' or 'test'
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        model_name: Model name (used in the title)
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title(f'Confusion Matrix - {model_name} - {dataset_type.upper()}')
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp_dir:
        cm_path = os.path.join(tmp_dir, f'confusion_matrix_{dataset_type}.png')
        plt.savefig(cm_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(cm_path)

    plt.close()
    print(f"Confusion matrix ({dataset_type}) saved and logged.")


#                                        TRAINING
# ================================================================================================

def train(model_name: str, model: nn.Module, 
          train_loader: torch.utils.data.DataLoader,
          val_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          criterion: nn.Module, optimizer: torch.optim.Optimizer, 
          device: torch.device, epochs: int = 10, patience: int = 5,
          experiment_id: str = None, class_names: list = None):
    """
    Trains the model with MLflow logging, early stopping, 
    and final evaluation on validation and test sets.

    Args:
        model_name: Model architecture name ('resnet18' or 'resnet34')
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cuda/mps/cpu)
        epochs: Maximum number of epochs
        patience: Number of epochs without improvement before early stopping
        experiment_id: MLflow experiment ID
        class_names: List of class names

    Returns:
        history: Dictionary containing training metrics per epoch
    """

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    with mlflow.start_run(experiment_id=experiment_id):

        # Log hyperparameters explicitly
        params = {
            "model_name": model_name,
            "optimizer": type(optimizer).__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epochs": epochs,
            "patience": patience,
            "criterion": type(criterion).__name__,
            "training_device": str(device),
            "num_classes": len(class_names) if class_names else "unknown"
        }
        mlflow.log_params(params)

        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }

        # ========== TRAINING LOOP ==========
        for epoch in range(epochs):

            # --- Training phase ---
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

            # --- Validation phase ---
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

            # Store and log metrics
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("validation_accuracy", val_acc, step=epoch)

            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # --- Early stopping check ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    model.load_state_dict(best_model_state)
                    break

        mlflow.log_param("last_epoch", epoch + 1)

        # ========== FINAL EVALUATION - VALIDATION ==========
        print("\n--- Final evaluation on VALIDATION set ---")
        val_acc_final, y_true_val, y_pred_val = evaluate_model(model, val_loader, device)
        mlflow.log_metric("final_validation_accuracy", val_acc_final)
        val_scores = log_metrics("validation", y_true_val, y_pred_val)
        log_confusion_matrix("validation", y_true_val, y_pred_val, class_names, model_name)

        for avg_mode, scores in val_scores.items():
            for metric, value in scores.items():
                print(f"  {metric.capitalize()} ({avg_mode}): {value:.4f}")

        # ========== FINAL EVALUATION - TEST ==========
        print("\n--- Final evaluation on TEST set ---")
        test_acc, y_true_test, y_pred_test = evaluate_model(model, test_loader, device)
        mlflow.log_metric("final_test_accuracy", test_acc)
        test_scores = log_metrics("test", y_true_test, y_pred_test)
        log_confusion_matrix("test", y_true_test, y_pred_test, class_names, model_name)

        print(f"  Test Accuracy: {test_acc:.4f}")
        for avg_mode, scores in test_scores.items():
            for metric, value in scores.items():
                print(f"  {metric.capitalize()} ({avg_mode}): {value:.4f}")

        # ========== MODEL LOGGING ==========
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=model_name,
            registered_model_name=model_name
        )

        print("\n--- Metrics and model logged into MLflow ---")

    return history