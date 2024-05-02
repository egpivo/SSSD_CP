import pytest
import torch

from sssd.core.layers.linear import LinearActivation, TransposedLinear


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
        assert module.bias == 0.0


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


def test_linear_activation_forward():
    # Test the forward pass
    in_features, out_features = 10, 5
    module = LinearActivation(in_features, out_features)
    input_tensor = torch.randn(3, in_features)
    output = module(input_tensor)
    assert output.shape == (3, out_features)


def test_linear_activation_transpose():
    # Test the is_transposed option
    in_features, out_features = 10, 5
    module = LinearActivation(in_features, out_features, is_transposed=True)
    input_tensor = torch.randn(in_features, in_features)
    output = module(input_tensor)
    assert output.shape == (out_features, in_features)


def test_linear_activation_bias():
    # Test the bias option
    in_features, out_features = 10, 5
    module = LinearActivation(in_features, out_features, use_bias=False)
    input_tensor = torch.randn(3, in_features)
    output = module(input_tensor)
    assert output.shape == (3, out_features)
    assert module.linear.bias is None


def test_linear_activation_zero_init_bias():
    # Test the zero_init_bias option
    in_features, out_features = 10, 5
    module = LinearActivation(in_features, out_features, zero_init_bias=True)
    input_tensor = torch.randn(3, in_features)
    output = module(input_tensor)
    assert output.shape == (3, out_features)
    assert not module.linear.bias.data.any()


def test_linear_activation_weight_init():
    # Test the weight_init option
    in_features, out_features = 10, 5
    module = LinearActivation(in_features, out_features, weight_init="xavier_uniform_")
    input_tensor = torch.randn(3, in_features)
    output = module(input_tensor)
    assert output.shape == (3, out_features)


def test_linear_activation_activation_fn():
    # Test the activation_fn option
    in_features, out_features = 10, 5
    module = LinearActivation(
        in_features, out_features, activation_fn="relu", apply_activation=True
    )
    input_tensor = torch.randn(3, in_features)
    output = module(input_tensor)
    assert output.shape == (3, out_features)


def test_linear_activation_weight_norm():
    # Test the use_weight_norm option
    in_features, out_features = 10, 5
    module = LinearActivation(in_features, out_features, use_weight_norm=True)
    input_tensor = torch.randn(3, in_features)
    output = module(input_tensor)
    assert output.shape == (3, out_features)
    weight_norm = module.linear.weight_g.norm(2, dim=0)
    assert hasattr(module.linear, "weight_g")
