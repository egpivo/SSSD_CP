import pytest
import torch.nn as nn

from sssd.core.imputers.utils import Activation, get_initializer


def test_activation_identity():
    activation_layer = Activation(activation=None)
    assert isinstance(activation_layer, nn.Identity)

    activation_layer = Activation(activation="id")
    assert isinstance(activation_layer, nn.Identity)

    activation_layer = Activation(activation="identity")
    assert isinstance(activation_layer, nn.Identity)

    activation_layer = Activation(activation="linear")
    assert isinstance(activation_layer, nn.Identity)


def test_activation_tanh():
    activation_layer = Activation(activation="tanh")
    assert isinstance(activation_layer, nn.Tanh)


def test_activation_relu():
    activation_layer = Activation(activation="relu")
    assert isinstance(activation_layer, nn.ReLU)


def test_activation_gelu():
    activation_layer = Activation(activation="gelu")
    assert isinstance(activation_layer, nn.GELU)


def test_activation_swish():
    activation_layer = Activation(activation="swish")
    assert isinstance(activation_layer, nn.SiLU)

    activation_layer = Activation(activation="silu")
    assert isinstance(activation_layer, nn.SiLU)


def test_activation_glu():
    activation_layer = Activation(activation="glu", dim=1)
    assert isinstance(activation_layer, nn.GLU)
    assert activation_layer.dim == 1


def test_activation_sigmoid():
    activation_layer = Activation(activation="sigmoid")
    assert isinstance(activation_layer, nn.Sigmoid)


def test_activation_not_implemented():
    with pytest.raises(NotImplementedError):
        Activation(activation="unsupported")


def test_get_initializer_uniform():
    initializer = get_initializer("uniform")
    assert callable(initializer)


def test_get_initializer_normal():
    initializer = get_initializer("normal")
    assert callable(initializer)


def test_get_initializer_xavier():
    initializer = get_initializer("xavier")
    assert callable(initializer)


def test_get_initializer_zero():
    initializer = get_initializer("zero")
    assert callable(initializer)


def test_get_initializer_one():
    initializer = get_initializer("one")
    assert callable(initializer)


def test_get_initializer_unsupported_activation():
    with pytest.raises(ValueError):
        get_initializer("uniform", activation="unsupported")


def test_get_initializer_unsupported_initializer():
    with pytest.raises(ValueError):
        get_initializer("unsupported")
