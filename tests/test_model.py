import pytest
from scripts.model import get_device, create_model_transfer_learning, create_model_fine_tuning

def test_get_device():
    device = get_device()
    assert device.lower() in [ "cpu", "cuda", "mps"], "Device in 'cpu', 'cuda', 'mps'"
