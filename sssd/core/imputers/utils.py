from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch.nn.init as init


def Activation(activation=None, dim=-1):
    """
    Returns an activation layer based on the specified activation type.

    Args:
        activation (str, optional): The activation type. Default is None.
        dim (int, optional): The dimension for the GLU activation. Default is -1.

    Returns:
        nn.Module: The activation layer.

    Raises:
        NotImplementedError: If the specified activation type is not implemented.
    """
    activation = activation.lower() if activation else None
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

    if activation in activation_map:
        return activation_map[activation]
    else:
        raise NotImplementedError(f"Activation '{activation}' is not implemented.")


def get_initializer(name: str, activation: Optional[str] = None) -> Callable:
    """
    Returns an initializer function based on the specified name and activation type.

    Args:
        name (str): The name of the initializer. Must be one of 'uniform', 'normal', 'xavier', 'zero', 'one'.
        activation (str, optional): The activation type. Must be one of 'id', 'identity', 'linear', 'modrelu',
                                    'relu', 'tanh', 'sigmoid', 'gelu', 'swish', 'silu'. Default is None.

    Returns:
        callable: The initializer function.

    Raises:
        ValueError: If the specified activation or initializer type is not supported.
    """
    SUPPORTED_ACTIVATIONS = {
        None,
        "id",
        "identity",
        "linear",
        "modrelu",
        "relu",
        "tanh",
        "sigmoid",
        "gelu",
        "swish",
        "silu",
    }
    SUPPORTED_INITIALIZERS = {"uniform", "normal", "xavier", "zero", "one"}
    if activation not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Unsupported activation: '{activation}'")

    name = name.lower()
    activation = activation.lower() if activation else None

    nonlinearity = (
        "linear"
        if activation in {None, "id", "identity", "linear", "modrelu"}
        else activation
        if activation in {"relu", "tanh", "sigmoid"}
        else "relu"
    )

    if name not in SUPPORTED_INITIALIZERS:
        raise ValueError(f"Unsupported initializer: '{name}'")

    if name == "uniform":
        initializer = partial(init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == "normal":
        initializer = partial(init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == "xavier":
        initializer = init.xavier_normal_
    elif name == "zero":
        initializer = partial(init.constant_, val=0)
    else:  # name == "one"
        initializer = partial(init.constant_, val=1)

    return initializer
