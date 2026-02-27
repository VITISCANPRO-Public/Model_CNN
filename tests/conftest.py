"""
tests/conftest.py — Shared fixtures for all test modules.

Provides reusable test components: device selection, dummy data,
and configuration loading. Used automatically by pytest.
"""

import pytest
import torch
from PIL import Image

from scripts.config_utils import load_config


@pytest.fixture(scope="session")
def device():
    """Returns CPU device for testing (no GPU required in CI)."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def config():
    """Loads config.yml once for the entire test session."""
    return load_config("config.yml")


@pytest.fixture
def num_classes():
    """Default number of disease classes in the INRAE dataset."""
    return 7


@pytest.fixture
def dummy_image():
    """Creates a synthetic 300x400 RGB image for transform testing."""
    return Image.new("RGB", (400, 300), color=(128, 64, 32))


@pytest.fixture
def dummy_batch():
    """Creates a batch of 4 random tensors simulating preprocessed images."""
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 7, (4,))
    return images, labels