from mlopsproject.model import MyAwesomeModel
import torch 
import pytest
from tests import _PATH_DATA

# Test model behavior with a valid input
def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)  # Input tensor of the correct shape and data type
    y = model(x)
    assert y.shape == (1, 10)
