import pytest
import torch.nn as nn

from sssd.core.layers.activation import Activation


def test_activation_identity():
    activation_layer = Activation(activation=None)
    assert isinstance(activation_layer.activation_fn, nn.Identity)

    activation_layer = Activation(activation="id")
    assert isinstance(activation_layer.activation_fn, nn.Identity)

    activation_layer = Activation(activation="identity")
    assert isinstance(activation_layer.activation_fn, nn.Identity)

    activation_layer = Activation(activation="linear")
    assert isinstance(activation_layer.activation_fn, nn.Identity)


def test_activation_tanh():
    activation_layer = Activation(activation="tanh")
    assert isinstance(activation_layer.activation_fn, nn.Tanh)


def test_activation_relu():
    activation_layer = Activation(activation="relu")
    assert isinstance(activation_layer.activation_fn, nn.ReLU)


def test_activation_gelu():
    activation_layer = Activation(activation="gelu")
    assert isinstance(activation_layer.activation_fn, nn.GELU)


def test_activation_swish():
    activation_layer = Activation(activation="swish")
    assert isinstance(activation_layer.activation_fn, nn.SiLU)

    activation_layer = Activation(activation="silu")
    assert isinstance(activation_layer.activation_fn, nn.SiLU)


def test_activation_glu():
    activation_layer = Activation(activation="glu", dim=1)
    assert isinstance(activation_layer.activation_fn, nn.GLU)
    assert activation_layer.activation_fn.dim == 1


def test_activation_sigmoid():
    activation_layer = Activation(activation="sigmoid")
    assert isinstance(activation_layer.activation_fn, nn.Sigmoid)


def test_activation_not_supported():
    with pytest.raises(ValueError):
        Activation(activation="unsupported")
