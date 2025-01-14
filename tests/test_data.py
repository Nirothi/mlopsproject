from mlopsproject.data import corrupt_mnist
from tests import _PATH_DATA
import torch
import os.path
import pytest


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"
    assert len(train) == 30000, "Train dataset did not have the correct number of samples"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()
