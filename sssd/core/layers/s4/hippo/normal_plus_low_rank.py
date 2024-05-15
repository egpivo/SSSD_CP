from typing import Tuple, Union

import opt_einsum
import torch

from sssd.core.layers.s4.hippo.trainsition_matrix import TransitionMatrix
from sssd.core.layers.s4.hippo.utils import generate_low_rank_matrix
from sssd.utils.logger import setup_logger

LOGGER = setup_logger()

CONTRACT = opt_einsum.contract


def normal_plus_low_rank(
    measure: str,
    matrix_size: int,
    correction_rank: int = 1,
    dtype: Union[type(torch.float), type(torch.cfloat)] = torch.float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normal plus Low-rank
    Return eigenvalues, correction_matrix, transformed_vector, unitary_matrix such that
    (eigenvalues - correction_matrix correction_matrix^*, transformed_vector) is unitarily equivalent
    to the original HiPPO A, B by the matrix unitary_matrix.
    i.e.,
       1. A = unitary_matrix[eigenvalues - correction_matrix correction_matrix^*]unitary_matrix^*
       2. B = unitary_matrix B

    Args:
    - measure: The type of measure used for the HiPPO matrix.
    - matrix_size: The size of the HiPPO matrix.
    - correction_rank: The rank for the correction matrix P.
    - dtype: The data type for the tensors.

    Returns:
    - w: The eigenvalues of the matrix AP.
    - P: The rank correction matrix.
    - B: The transformed input vector.
    """
    assert dtype in (
        torch.float,
        torch.cfloat,
    ), "dtype must be torch.float or torch.cfloat"
    half_size = matrix_size // 2
    if measure == "random":
        dtype = torch.cfloat if dtype == torch.float else torch.cdouble
        w = -torch.exp(torch.randn(half_size, dtype=dtype)) + 1j * torch.randn(
            matrix_size // 2, dtype=dtype
        )
        P = torch.randn(correction_rank, half_size, dtype=dtype)
        B = torch.randn(half_size, dtype=dtype)
        return w, P, B

    A, B = TransitionMatrix(measure, matrix_size)
    A = torch.as_tensor(A, dtype=dtype)
    B = torch.as_tensor(B, dtype=dtype)[:, 0]

    P = generate_low_rank_matrix(
        measure, matrix_size, rank=correction_rank, dtype=dtype
    )
    AP = A + torch.einsum("...i,...j->...ij", P, P.conj()).sum(dim=-3)
    w, V = torch.linalg.eig(AP)

    # Keep only one of each pair of complex conjugate eigenvalues
    mask = w.imag >= 0
    w = w[mask]
    V = V[:, mask]

    V_inv = V.conj().transpose(-1, -2)
    B = CONTRACT("ij, j -> i", V_inv, B.to(V.dtype))  # V^* B
    P = CONTRACT("ij, ...j -> ...i", V_inv, P.to(V.dtype))  # V^* P

    return w, P, B
