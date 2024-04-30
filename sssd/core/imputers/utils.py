from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.init as init


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


def power(exponent: int, matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute matrix raised to the power of the exponent using the square-and-multiply algorithm.

    Args:
        exponent (int): The exponent.
        matrix (torch.Tensor): Square matrix of shape (..., N, N).

    Returns:
        torch.Tensor: The result of matrix raised to the power of the exponent.

    Raises:
        ValueError: If the input matrix is not square.
        ValueError: If the exponent is negative.
    """
    # Check if the input matrix is square
    if matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError("Input matrix must be square.")

    # Check if the exponent is non-negative
    if exponent < 0:
        raise ValueError("Exponent must be non-negative.")

    # Initialize an identity matrix of the same size as the input matrix
    result = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)

    # Compute matrix powers iteratively
    while exponent > 0:
        if exponent & 1:
            result = torch.matmul(matrix, result)
        exponent >>= 1
        if exponent > 0:
            matrix = torch.matmul(matrix, matrix)

    return result
