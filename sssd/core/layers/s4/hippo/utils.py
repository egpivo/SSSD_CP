import math
from typing import Tuple

import numpy as np
import torch
from einops import rearrange, repeat
from opt_einsum import contract_expression
from opt_einsum.contract import ContractExpression

from sssd.utils.logger import setup_logger

LOGGER = setup_logger()

Matrix = np.ndarray

_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_resolve_conj = lambda x: x.conj().resolve_conj()


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


def cauchy_cpu(v: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Compute the sum of the Cauchy matrix along the second-to-last dimension.

    Args:
        v (torch.Tensor): Tensor of shape (..., N).
        z (torch.Tensor): Tensor of shape (..., L).
        w (torch.Tensor): Tensor of shape (..., N).

    Returns:
        torch.Tensor: The resulting tensor of shape (..., L) after computing
                      the sum of the Cauchy matrix along the second-to-last dimension.
    """
    # Expand dimensions of v, z, and w to align for broadcasting
    v_expanded = v.unsqueeze(-1)
    z_expanded = z.unsqueeze(-2)
    w_expanded = w.unsqueeze(-1)

    # Compute the Cauchy matrix wiht Shape: (..., N, L)
    cauchy_matrix = v_expanded / (z_expanded - w_expanded)

    # Sum over the second-to-last dimension (N) to get the result
    result = torch.sum(cauchy_matrix, dim=-2)
    return result


def cauchy_wrapper(v: torch.Tensor, z: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """ "CUDA extension for cauchy multiplication not found. Please check `install_extensions_cauchy` in envs/conda/utils.sh"""
    try:  # This module will be downloaded from s4 repo
        from sssd.core.layers.s4.hippo.cauchy import cauchy_mult

        has_cauchy_extension = True
    except ModuleNotFoundError:
        has_cauchy_extension = False

    if has_cauchy_extension and z.dtype == torch.cfloat:
        """The v, z, w are assumed to be saved on GPU"""
        return cauchy_mult(v, z, w, symmetric=True)
    else:
        return cauchy_cpu(v, z, w)


def compute_fft_transform(
    sequence_length: int, dtype: torch.dtype, device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform.

    Args:
    - sequence_length (int): The length.
    - dtype: The data type.
    - device: The device.

    Returns:
    - omega: The FFT nodes. (shape (L // 2 + 1))
    - z: The transformed nodes. (shape (L // 2 + 1)_)
    """
    # ω_2L = exp(−2jπ/(2L))
    base_omega = torch.tensor(
        np.exp(-2j * np.pi / (2 * sequence_length)), dtype=dtype, device=device
    )
    omega = base_omega ** torch.arange(sequence_length // 2 + 1, device=device)
    z = 2 * (1 - omega) / (1 + omega)
    return omega, z


def hurwitz_transformation(w_real: torch.Tensor, w_image: torch.Tensor) -> torch.Tensor:
    return torch.complex(-torch.exp(w_real), w_image)


def get_diagonal_contraction(
    batch_shape: Tuple[int, ...], H: int, N: int
) -> ContractExpression:
    """
    Utility function to create a diagonal contraction expression.

    Args:
        batch_shape (Tuple[int, ...]): The shape of the batch.
        H (int): The size of the first dimension.
        N (int): The size of the second dimension.

    Returns:
        ContractExpression: The diagonal contraction expression.
    """
    return contract_expression("h n, ... h n -> ... h n", (H, N), batch_shape + (H, N))


def get_dense_contraction(
    batch_shape: Tuple[int, ...], H: int, N: int
) -> ContractExpression:
    """
    Utility function to create a dense contraction expression.

    Args:
        batch_shape (Tuple[int, ...]): The shape of the batch.
        H (int): The size of the first dimension.
        N (int): The size of the second dimension.

    Returns:
        ContractExpression: The dense contraction expression.
    """
    return contract_expression(
        "h m n, ... h n -> ... h m", (H, N, N), batch_shape + (H, N)
    )


def get_input_contraction(
    batch_shape: Tuple[int, ...], H: int, N: int
) -> ContractExpression:
    """
    Utility function to create an input contraction expression.

    Args:
        batch_shape (Tuple[int, ...]): The shape of the batch.
        H (int): The size of the first dimension.
        N (int): The size of the second dimension.

    Returns:
        ContractExpression: The input contraction expression.
    """
    return contract_expression("h n, ... h -> ... h n", (H, N), batch_shape + (H,))


def get_output_contraction(
    batch_shape: Tuple[int, ...], H: int, N: int, C: int
) -> ContractExpression:
    """
    Utility function to create an output contraction expression.

    Args:
        batch_shape (Tuple[int, ...]): The shape of the batch.
        H (int): The size of the first dimension.
        N (int): The size of the second dimension.
        C (int): The size of the third dimension.

    Returns:
        str: The output contraction expression.
    """
    return contract_expression(
        "c h n, ... h n -> ... c h", (C, H, N), batch_shape + (H, N)
    )


def low_rank_woodbury_correction(r: torch.Tensor, rank: int) -> torch.Tensor:
    """
    Compute the Low-rank Woodbury correction.

    Args:
        r (torch.Tensor): The input tensor.
        rank (int): The rank value.

    Returns:
        torch.Tensor: The result of the Low-rank Woodbury correction.
    """
    if rank == 1:
        k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (
            1 + r[-1:, -1:, :, :]
        )
    elif rank == 2:
        r00 = r[:-rank, :-rank, :, :]
        r01 = r[:-rank, -rank:, :, :]
        r10 = r[-rank:, :-rank, :, :]
        r11 = r[-rank:, -rank:, :, :]
        det = (1 + r11[:1, :1, :, :]) * (1 + r11[1:, 1:, :, :]) - r11[
            :1, 1:, :, :
        ] * r11[1:, :1, :, :]
        s = (
            r01[:, :1, :, :] * (1 + r11[1:, 1:, :, :]) * r10[:1, :, :, :]
            + r01[:, 1:, :, :] * (1 + r11[:1, :1, :, :]) * r10[1:, :, :, :]
            - r01[:, :1, :, :] * (r11[:1, 1:, :, :]) * r10[1:, :, :, :]
            - r01[:, 1:, :, :] * (r11[1:, :1, :, :]) * r10[:1, :, :, :]
        )
        s = s / det
        k_f = r00 - s
    else:
        r00 = r[:-rank, :-rank, :, :]
        r01 = r[:-rank, -rank:, :, :]
        r10 = r[-rank:, :-rank, :, :]
        r11 = r[-rank:, -rank:, :, :]
        r11 = rearrange(r11, "a b h n -> h n a b")
        r11 = torch.linalg.inv(torch.eye(rank, device=r.device) + r11)
        r11 = rearrange(r11, "h n a b -> a b h n")
        k_f = r00 - torch.einsum("i j h n, j k h n, k l h n -> i l h n", r01, r11, r10)

    return k_f


def generate_dt(
    H: int, dtype: torch.dtype, dt_min: float, dt_max: float
) -> torch.Tensor:
    """
    Generates a tensor of shape (H,) with random values between dt_min and dt_max.

    Args:
        H (int): The size of the tensor.
        dtype (torch.dtype): The data type of the tensor.
        dt_min (float): The minimum value of the tensor.
        dt_max (float): The maximum value of the tensor.

    Returns:
        torch.Tensor: A tensor of shape (H,) with random values between dt_min and dt_max.
    """
    return torch.rand(H, dtype=dtype) * (
        math.log(dt_max) - math.log(dt_min)
    ) + math.log(dt_min)


def repeat_along_additional_dimension(
    tensor: torch.Tensor, repeat_count: int
) -> torch.Tensor:
    """
    Repeats the input tensor along an additional dimension.

    Args:
        tensor: Input tensor to be repeated
        repeat_count: Number of times to repeat the tensor along the additional dimension

    Returns:
        Repeated tensor
    """
    return repeat(tensor, "... 1 n -> ... h n", h=repeat_count)
