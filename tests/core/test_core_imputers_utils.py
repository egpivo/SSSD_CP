import pytest
import torch

from sssd.core.imputers.utils import get_initializer, power


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


def test_power_without_v():
    # Test case 1: L = 2, A is a 2x2 matrix
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    L = 2
    expected_result = torch.tensor([[7, 10], [15, 22]], dtype=torch.float32)
    result = power(L, A)
    assert torch.allclose(result, expected_result)

    # Test case 2: L = 3, A is a 3x3 matrix
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    L = 3
    expected_result = torch.tensor(
        [[468, 576, 684], [1062, 1305, 1548], [1656, 2034, 2412]], dtype=torch.float32
    )
    result = power(L, A)
    assert torch.allclose(result, expected_result)
