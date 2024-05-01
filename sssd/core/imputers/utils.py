from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.init as init
from einops import rearrange
from scipy import special as ss

Matrix = np.ndarray
MeasureArgs = Dict[str, Union[float, int]]


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
    supported_activations = {
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
    supported_initializers = {"uniform", "normal", "xavier", "zero", "one"}
    if activation not in supported_activations:
        raise ValueError(f"Unsupported activation: '{activation}'")
    if name not in supported_initializers:
        raise ValueError(f"Unsupported initializer: '{name}'")

    name = name.lower()
    activation = activation.lower() if activation else None
    nonlinearity = (
        "linear"
        if activation in {None, "id", "identity", "linear", "modrelu"}
        else activation
        if activation in {"relu", "tanh", "sigmoid"}
        else "relu"
    )

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


def embed_c2r(A: Matrix) -> Matrix:
    """
    Embed a complex-valued matrix into a real-valued matrix.

    Args:
        A (Matrix): Complex-valued matrix with shape (..., M, N).

    Returns:
        Matrix: Real-valued matrix with shape (..., M+1, N+1).

    Raises:
        ValueError: If the input matrix does not have exactly 2 dimensions.
    """
    if A.ndim != 2:
        raise ValueError(f"Expected 2 dimensions, got {A.ndim}")

    # Expand dimensions and pad
    A_expanded = rearrange(A, "... m n -> ... m () n ()")
    A_padded = np.pad(A_expanded, ((0, 0), (0, 1), (0, 0), (0, 1))) + np.pad(
        A_expanded, ((0, 0), (1, 0), (0, 0), (1, 0))
    )

    # Rearrange to final shape
    return rearrange(A_padded, "m x n y -> (m x) (n y)")


class TransitionMatrix:
    def __new__(
        cls, measure: str, N: int, **measure_args: MeasureArgs
    ) -> Tuple[Matrix, Matrix]:
        """
        Generate transition matrices for different measures.

        Args:
            measure (str): Type of measure.
            N (int): Size of the transition matrices.
            **measure_args: Additional arguments specific to each measure.

        Returns:
            Tuple[Matrix, Matrix]: Transition matrices A and B.

        Raises:
            NotImplementedError: If the specified measure is not implemented.
            ValueError: If invalid measure arguments are provided.
        """
        if measure == "lagt":
            A, B = cls._translated_laguerre(N, **measure_args)
        elif measure == "glagt":
            A, B = cls._generalized_laguerre(N, **measure_args)
        elif measure == "legt":
            A, B = cls._translated_legendre(N)
        elif measure == "legs":
            A, B = cls._scaled_legendre(N)
        elif measure == "fourier":
            A, B = cls._fourier(N)
        elif measure == "random":
            A, B = cls._random(N)
        elif measure == "diagonal":
            A, B = cls._diagonal(N)
        else:
            raise NotImplementedError(f"Measure '{measure}' is not implemented.")

        return A, B

    @staticmethod
    def _translated_laguerre(
        N: int, **measure_args: MeasureArgs
    ) -> Tuple[Matrix, Matrix]:
        """
        Generate transition matrices for Laguerre (translated) measure.
        """
        beta = measure_args.get("beta", 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = beta * np.ones((N, 1))
        return A, B

    def _generalized_laguerre(
        N: int, **measure_args: MeasureArgs
    ) -> Tuple[Matrix, Matrix]:
        """
        Generate transition matrices for Generalized Laguerre measure.
        """
        alpha = measure_args.get("alpha", 0.0)
        beta = measure_args.get("beta", 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]
        L = np.exp(
            0.5 * (ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1))
        )
        A = (1.0 / L[:, None]) * A * L[None, :]
        B = (
            (1.0 / L[:, None])
            * B
            * np.exp(-0.5 * ss.gammaln(1 - alpha))
            * beta ** ((1 - alpha) / 2)
        )
        return A, B

    @staticmethod
    def _translated_legendre(N: int) -> Tuple[Matrix, Matrix]:
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
        return A, B

    @staticmethod
    def _scaled_legendre(N: int) -> Tuple[Matrix, Matrix]:
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
        return A, B

    @staticmethod
    def _fourier(N: int) -> Tuple[Matrix, Matrix]:
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N // 2, N // 2)))
        B = embed_c2r(np.ones((N // 2, 1)))[..., :1]
        return A, B

    @staticmethod
    def _random(N: int) -> Tuple[Matrix, Matrix]:
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
        return A, B

    @staticmethod
    def _diagonal(N: int) -> Tuple[Matrix, Matrix]:
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
        return A, B


def generate_rank_correction_matrix(
    measure: str, N: int, rank: int = 1, dtype: torch.dtype = torch.float
) -> torch.Tensor:
    """
    Generate a low-rank matrix L such that A + L is normal.

    Args:
        measure (str): Type of measure.
        N (int): Size of the matrix.
        rank (int, optional): Rank of the low-rank matrix. Defaults to 1.
        dtype (torch.dtype, optional): Data type of the matrix. Defaults to torch.float.

    Returns:
        torch.Tensor: Low-rank matrix L.

    Raises:
        NotImplementedError: If the specified measure is not implemented.
    """

    if measure == "legs":
        assert rank >= 1
        P = torch.sqrt(0.5 + torch.arange(N, dtype=dtype)).unsqueeze(0)  # (1, N)
    elif measure == "legt":
        assert rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(N, dtype=dtype))  # (N,)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)  # (2, N)
    elif measure == "lagt":
        assert rank >= 1
        P = 0.5**0.5 * torch.ones(1, N, dtype=dtype)
    elif measure == "fourier":
        P = torch.ones(N, dtype=dtype)  # (N,)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)  # (2, N)
    else:
        raise NotImplementedError(f"Measure '{measure}' is not implemented.")

    d = P.size(0)
    if rank > d:
        P = torch.cat([P, torch.zeros(rank - d, N, dtype=dtype)], dim=0)  # (rank, N)
    return P
