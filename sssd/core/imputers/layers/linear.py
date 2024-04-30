import math

import opt_einsum as oe
import torch
import torch.nn as nn


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
