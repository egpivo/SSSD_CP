from dataclasses import dataclass
from typing import Union

import torch
from opt_einsum import contract

from sssd.core.layers.s4.hippo.trainsition_matrix import TransitionMatrix


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
                return subclass(measure, matrix_size, correction_rank, dtype)
        raise ValueError(f"No subclass found for measure: {measure}")


class BaseNormalPlusLowRank:
    def __init__(
        self,
        measure: str,
        matrix_size: int,
        correction_rank: int = 1,
        dtype: Union[type(torch.float), type(torch.cfloat)] = torch.float,
    ) -> None:
        self.measure = measure
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
    def _generate_low_rank_matrix(self):
        return NotImplemented

    def generate_low_rank_matrix(self):
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
            )  # (rank, N)
        return P

    def compute(self) -> NormalPlusLowRankResult:
        A, B = TransitionMatrix(self.measure, self.matrix_size)
        A = torch.as_tensor(A, dtype=self.dtype)
        B = torch.as_tensor(B, dtype=self.dtype)[:, 0]

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
    def _generate_low_rank_matrix(self):
        assert self.correction_rank >= 1
        P = torch.sqrt(
            0.5 + torch.arange(self.matrix_size, dtype=self.dtype)
        ).unsqueeze(0)
        return P


class LegtNormalPlusLowRank(NonRandomNormalPlusLowRank):
    def _generate_low_rank_matrix(self):
        assert self.correction_rank >= 2
        P = torch.sqrt(1 + 2 * torch.arange(self.matrix_size, dtype=self.dtype))  # (N,)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        P = torch.stack([P0, P1], dim=0)  # (2, N)
        return P


class LagtNormalPlusLowRank(NonRandomNormalPlusLowRank):
    def _generate_low_rank_matrix(self):
        assert self.correction_rank >= 1
        return 0.5**0.5 * torch.ones(1, self.matrix_size, dtype=self.dtype)


class FourierNormalPlusLowRank(NonRandomNormalPlusLowRank):
    def _generate_low_rank_matrix(self):
        P = torch.ones(self.matrix_size, dtype=self.dtype)  # (N,)
        P0 = P.clone()
        P0[0::2] = 0.0
        P1 = P.clone()
        P1[1::2] = 0.0
        return torch.stack([P0, P1], dim=0)  # (2, N)

        d = P.size(0)
        if rank > d:
            P = torch.cat(
                [P, torch.zeros(rank - d, N, dtype=dtype)], dim=0
            )  # (rank, N)
        return P
