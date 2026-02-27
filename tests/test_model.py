"""
tests/test_model.py — Unit tests for model creation and device selection.

Tests cover:
  - Device detection (CPU/CUDA/MPS)
  - Model creation for all supported architectures
  - Transfer learning mode (frozen base)
  - Fine-tuning mode (unfrozen layers)
  - Classifier output dimension matches num_classes
  - Unsupported model name raises ValueError
"""

import pytest
import torch

from scripts.model import get_device, create_model


# ──────────────────────────────────────────────
#  DEVICE
# ──────────────────────────────────────────────

class TestGetDevice:
    """Tests for device selection."""

    def test_returns_valid_device(self):
        device = get_device()
        assert str(device) in ["cpu", "cuda", "mps"]

    def test_returns_torch_device(self):
        device = get_device()
        assert isinstance(device, torch.device)


# ──────────────────────────────────────────────
#  MODEL CREATION — RESNET
# ──────────────────────────────────────────────

class TestResNetCreation:
    """Tests for ResNet model creation."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_resnet18_transfer_learning(self, device):
        model = create_model("resnet18", num_classes=7, device=device, freeze_base=True)
        assert model is not None
        # Verify output dimension
        if isinstance(model.fc, torch.nn.Sequential):
            out_features = model.fc[-1].out_features
        else:
            out_features = model.fc.out_features
        assert out_features == 7

    def test_resnet18_fine_tuning(self, device):
        model = create_model("resnet18", num_classes=7, device=device,
                             freeze_base=False, unfreeze_layer="layer4")
        assert model is not None
        # Fine-tuning adds dropout — fc is Sequential
        assert isinstance(model.fc, torch.nn.Sequential)

    def test_resnet34_transfer_learning(self, device):
        model = create_model("resnet34", num_classes=7, device=device, freeze_base=True)
        assert model is not None

    def test_resnet18_frozen_params(self, device):
        """In transfer learning, base layers should be frozen."""
        model = create_model("resnet18", num_classes=7, device=device, freeze_base=True)
        # conv1 should be frozen
        for param in model.conv1.parameters():
            assert not param.requires_grad
        # fc (classifier) should be trainable
        for param in model.fc.parameters():
            assert param.requires_grad

    def test_resnet18_finetuning_params(self, device):
        """In fine-tuning, all layers should be trainable."""
        model = create_model("resnet18", num_classes=7, device=device, freeze_base=False)
        for param in model.parameters():
            assert param.requires_grad


# ──────────────────────────────────────────────
#  MODEL CREATION — EFFICIENTNET
# ──────────────────────────────────────────────

class TestEfficientNetCreation:
    """Tests for EfficientNet model creation."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_efficientnet_b0(self, device):
        model = create_model("efficientnet_b0", num_classes=7, device=device,
                             freeze_base=False)
        assert model is not None
        out_features = model.classifier[1].out_features
        assert out_features == 7


# ──────────────────────────────────────────────
#  EDGE CASES
# ──────────────────────────────────────────────

class TestModelEdgeCases:
    """Tests for error handling and edge cases."""

    @pytest.fixture
    def device(self):
        return torch.device("cpu")

    def test_unsupported_model_raises_error(self, device):
        with pytest.raises(ValueError, match="Unsupported model"):
            create_model("vgg16", num_classes=7, device=device)

    def test_single_class(self, device):
        """Model should work even with 1 class (degenerate case)."""
        model = create_model("resnet18", num_classes=1, device=device, freeze_base=True)
        if isinstance(model.fc, torch.nn.Sequential):
            out_features = model.fc[-1].out_features
        else:
            out_features = model.fc.out_features
        assert out_features == 1

    def test_model_on_correct_device(self, device):
        model = create_model("resnet18", num_classes=7, device=device, freeze_base=True)
        # Check that parameters are on the right device
        first_param = next(model.parameters())
        assert first_param.device == device