import pytest
import torch

from sssd.core.imputers.layers.linear import TransposedLinear


@pytest.mark.parametrize(
    "batch_size, in_features, out_features, bias",
    [
        (1, 10, 5, True),
        (3, 20, 15, False),
        (5, 8, 12, True),
    ],
)
def test_transposed_linear(batch_size, in_features, out_features, bias):
    # Create input tensor
    input_tensor = torch.randn(batch_size, in_features, 1)

    # Create module
    module = TransposedLinear(in_features, out_features, bias=bias)

    # Forward pass
    output_tensor = module(input_tensor)

    # Check output shape
    assert output_tensor.shape == (batch_size, out_features, 1)

    # Check weight shape
    assert module.weight.shape == (out_features, in_features)

    # Check bias shape (if applicable)
    if bias:
        assert module.bias.shape == (out_features, 1)
    else:
        assert module.bias is None


def test_transposed_linear_with_extra_dims():
    # Create input tensor with extra dimensions
    input_tensor = torch.randn(2, 3, 4, 5, 1)

    # Create module
    module = TransposedLinear(5, 10, bias=True)

    # Forward pass
    output_tensor = module(input_tensor)

    # Check output shape
    assert output_tensor.shape == (2, 3, 4, 10, 1)

    # Check weight shape
    assert module.weight.shape == (10, 5)

    # Check bias shape
    assert module.bias.shape == (10, 1)
