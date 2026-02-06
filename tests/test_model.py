import pytest
from scripts.model import get_device, create_model_transfer_learning, create_model_fine_tuning

def test_get_device():
    device = get_device()
    device_str = str(device).lower()
    assert device_str in [ "cpu", "cuda", "mps"], f"Device must be 'cpu', 'cuda' or 'mps', but got '{device_str}'"
