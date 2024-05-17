from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
from opt_einsum import contract

from sssd.core.layers.s4.hippo.utils import embed_c2r

Matrix = np.ndarray


@dataclass
class NormalPlusLowRankResult:
    w: torch.Tensor
    P: torch.Tensor
    B: torch.Tensor


class NormalPlusLowRank:
    def __new__(
        cls,
        measure: str,
        matrix_size: int,
        correction_rank: int = 1,
        dtype: Union[type(torch.float), type(torch.cfloat)] = torch.float,
    ) -> "BaseNormalPlusLowRank":
        for subclass_name, subclass in globals().items():
            if (
                isinstance(subclass, type)
                and issubclass(subclass, BaseNormalPlusLowRank)
                and subclass_name.endswith("NormalPlusLowRank")
                and subclass_name.lower().startswith(measure.lower())
            ):
                return subclass(matrix_size, correction_rank, dtype)
        raise ValueError(f"No subclass found for measure: {measure}")


class BaseNormalPlusLowRank:
    def __init__(
        self,
        matrix_size: int,
        correction_rank: int = 1,
        dtype: Union[type(torch.float), type(torch.cfloat)] = torch.float,
    ) -> None:
        self.matrix_size = matrix_size
        self.correction_rank = correction_rank
        self.dtype = dtype

    def compute(self) -> NormalPlusLowRankResult:
        raise NotImplementedError(
            "Method _compute_random should be implemented in subclass."
        )


class RandomNormalPlusLowRank(BaseNormalPlusLowRank):
    def compute(self) -> NormalPlusLowRankResult:
        half_size = self.matrix_size // 2
        dtype = torch.cfloat if self.dtype == torch.float else torch.cdouble
        w = -torch.exp(torch.randn(half_size, dtype=dtype)) + 1j * torch.randn(
            self.matrix_size // 2, dtype=dtype
        )
        P = torch.randn(self.correction_rank, half_size, dtype=dtype)
        B = torch.randn(half_size, dtype=dtype)
        return NormalPlusLowRankResult(w=w, P=P, B=B)


class NonRandomNormalPlusLowRank(BaseNormalPlusLowRank):
    def _generate_transition_matrix(self) -> Tuple[Matrix, Matrix]:
        return NotImplemented

    def _generate_low_rank_matrix(self) -> torch.Tensor:
        return NotImplemented

    def generate_transition_matrix(self) -> Tuple[Matrix, Matrix]:
        A, B = self._generate_transition_matrix()
        A = torch.as_tensor(A, dtype=self.dtype)
        B = torch.as_tensor(B, dtype=self.dtype)[:, 0]
        return A, B

    def generate_low_rank_matrix(self) -> torch.Tensor:
        """Return the shape (rank, N)"""
        P = self._generate_low_rank_matrix()
        d = P.size(0)
        if self.correction_rank > d:
            P = torch.cat(
                [
                    P,
                    torch.zeros(
                        self.correction_rank - d, self.matrix_size, dtype=self.dtype
                    ),
                ],
                dim=0,
            )
        return P

    def compute(self) -> NormalPlusLowRankResult:
        A, B = self.generate_transition_matrix()
        P = self.generate_low_rank_matrix()
        AP = A + torch.einsum("...i,...j->...ij", P, P.conj()).sum(dim=-3)
        w, V = torch.linalg.eig(AP)

        # Keep only one of each pair of complex conjugate eigenvalues
        mask = w.imag >= 0
        w = w[mask]
        V = V[:, mask]

        V_inv = V.conj().transpose(-1, -2)
        B = contract("ij, j -> i", V_inv, B.to(V.dtype))
        P = contract("ij, ...j -> ...i", V_inv, P.to(V.dtype))

        return NormalPlusLowRankResult(w=w, P=P, B=B)


class LegsNormalPlusLowRank(NonRandomNormalPlusLowRank):
    """scaled legendre"""

    def _generate_transition_matrix(self) -> Tuple[Matrix, Matrix]:
        q = np.arange(self.matrix_size, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
        return A, B

    def _generate_low_rank_matrix(self) -> torch.Tensor:
        assert self.correction_rank >= 1
        P = torch.sqrt(
            0.5 + torch.arange(self.matrix_size, dtype=self.dtype)
        ).unsqueeze(0)
        return P


class LegtNormalPlusLowRank(NonRandomNormalPlusLowRank):
    def _generate_transition_matrix(self) -> Tuple[Matrix, Matrix]:
        """translated legendre"""
        Q = np.arange(self.matrix_size, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
        return A, B

    def _generate_low_rank_matrix(self) -> torch.Tensor:
        assert self.correction_rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(self.matrix_size, dtype=self.dtype))  # (N,)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)  # (2, N)
        return P


class LagtNormalPlusLowRank(NonRandomNormalPlusLowRank):
    def _generate_transition_matrix(self, beta: float = 1.0) -> Tuple[Matrix, Matrix]:
        """translated laguerre"""
        A = np.eye(self.matrix_size) / 2 - np.tril(
            np.ones((self.matrix_size, self.matrix_size))
        )
        B = beta * np.ones((self.matrix_size, 1))
        return A, B

    def _generate_low_rank_matrix(self) -> torch.Tensor:
        assert self.correction_rank >= 1
        return 0.5**0.5 * torch.ones(1, self.matrix_size, dtype=self.dtype)


class FourierNormalPlusLowRank(NonRandomNormalPlusLowRank):
    def _generate_transition_matrix(self) -> Tuple[Matrix, Matrix]:
        freqs = np.arange(self.matrix_size // 2)
        d = np.stack([freqs, np.zeros(self.matrix_size // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((self.matrix_size // 2, self.matrix_size // 2)))
        B = embed_c2r(np.ones((self.matrix_size // 2, 1)))[..., :1]
        return A, B

    def _generate_low_rank_matrix(self) -> torch.Tensor:
        P = torch.ones(self.matrix_size, dtype=self.dtype)  # (N,)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        return torch.stack([P0, P1], dim=0)  # (2, N)
