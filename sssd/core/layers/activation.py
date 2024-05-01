import torch.nn as nn


class Activation(nn.Module):
    """
    A module that applies a specified activation function.

    Args:
        activation (str, optional): The activation type. Default is None.
        dim (int, optional): The dimension for the GLU activation. Default is -1.

    Raises:
        ValueError: If the specified activation type is not supported.
    """

    def __init__(self, activation=None, dim=-1):
        super().__init__()
        self.activation = activation.lower() if activation else None
        self.dim = dim

        activation_map = {
            None: nn.Identity(),
            "id": nn.Identity(),
            "identity": nn.Identity(),
            "linear": nn.Identity(),
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "glu": nn.GLU(dim=dim),
            "sigmoid": nn.Sigmoid(),
        }

        if self.activation in activation_map:
            self.activation_fn = activation_map[self.activation]
        else:
            raise ValueError(f"Activation '{self.activation}' is not supported.")

    def forward(self, x):
        return self.activation_fn(x)
