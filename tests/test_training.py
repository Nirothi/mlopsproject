import pytest
import torch
from mlopsproject.train import train
from mlopsproject.model import MyAwesomeModel
from mlopsproject.data import corrupt_mnist
from tests import _PATH_DATA


def test_training_loop(data, model):
    train_data, _ = corrupt_mnist()
    model = MyAwesomeModel()
    initial_params = {name: param.clone() for name, param in model.named_parameters()}
    train(model, train_data, epochs=1)
    for name, param in model.named_parameters():
        assert not torch.equal(param, initial_params[name]), f"Parameter {name} did not change after training"



