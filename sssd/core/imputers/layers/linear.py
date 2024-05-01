import math
from typing import Callable, Union

import opt_einsum as oe
import torch
import torch.nn as nn

from sssd.core.imputers.layers.activation import Activation


class TransposedLinear(nn.Module):
    """
    Linear module on the second-to-last dimension.

    This module applies a linear transformation to the second-to-last dimension of the input tensor.
    It is useful for situations where the input tensor has a batch dimension and additional dimensions
    that need to be treated as features.

    Args:
        in_features (int): Size of the input feature dimension.
        out_features (int): Size of the output feature dimension.
        bias (bool, optional): Whether to include a bias term in the linear transformation. Default: True.

    Shapes:
        - Input: `(..., in_features, 1)`
        - Output: `(..., out_features, 1)`
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, 1))
            bound = 1 / math.sqrt(in_features)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the linear transformation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape `(..., in_features, 1)`.

        Returns:
            torch.Tensor: Output tensor of shape `(..., out_features, 1)`.
        Notes:
            - Contraction: output_tensor[b_1, b_2, ..., b_n, v, l] = sum_u(x[b_1, b_2, ..., b_n, u, l] * self.weight[v, u])
            - u: in_features; v: out_features; l = 1
        """
        return oe.contract("... u l, v u -> ... v l", x, self.weight) + self.bias


class LinearActivation(nn.Module):
    """A linear module with control over axes order, initialization, and activation.

    Args:
        in_features (int): Input dimension.
        out_features (int): Output dimension.
        use_bias (bool, optional): Whether to include bias term. Defaults to True.
        zero_init_bias (bool, optional): Whether to initialize bias to zeros. Defaults to False.
        is_transposed (bool, optional): Whether to use TransposedLinear instead of Linear. Defaults to False.
        weight_init (Union[str, Callable], optional): Weight initialization scheme. Defaults to None.
        activation (str, optional): Activation function. Defaults to None.
        apply_activation (bool, optional): Whether to apply activation as part of this module. Defaults to False.
        use_weight_norm (bool, optional): Whether to apply weight normalization. Defaults to False.
        **kwargs: Additional keyword arguments for the linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        zero_init_bias: bool = False,
        is_transposed: bool = False,
        weight_init: Union[str, Callable] = None,
        activation: str = None,
        apply_activation: bool = False,
        use_weight_norm: bool = False,
        **kwargs,
    ):
        super().__init__()

        if activation == "glu":
            out_features *= 2

        self.is_transposed = is_transposed
        if is_transposed:
            self.linear = TransposedLinear(in_features, out_features, bias=use_bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=use_bias)

        if use_bias and zero_init_bias:
            nn.init.zeros_(self.linear.bias)

        if weight_init is not None:
            if isinstance(weight_init, str):
                weight_init = getattr(nn.init, weight_init)
            weight_init(self.linear.weight)

        self.activation = (
            Activation(activation, dim=-2 if is_transposed else -1)
            if apply_activation and activation
            else None
        )

        self.use_weight_norm = use_weight_norm
        if use_weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)

    def forward(self, x):
        x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)
        return x
