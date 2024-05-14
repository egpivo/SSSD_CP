import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from sssd.core.layers.s4.hippo.state_space import SSKernelNPLR
from sssd.core.layers.s4.hippo.utils import normal_plus_low_rank
from sssd.utils.logger import setup_logger

LOGGER = setup_logger()


def generate_dt(H, dtype, dt_min, dt_max):
    return torch.rand(H, dtype=dtype) * (
        math.log(dt_max) - math.log(dt_min)
    ) + math.log(dt_min)


class HippoSSKernel(nn.Module):
    """
    Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.
    The SSKernel is expected to support the interface forward(), default_state(), setup_step(), step().
    """

    def __init__(
        self,
        H: int,
        N: int = 64,
        L: int = 1,
        measure: str = "legs",
        rank: int = 1,
        channels: int = 1,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        trainable: Optional[Dict[str, bool]] = None,
        lr: Optional[float] = None,
        length_correction: bool = True,
        hurwitz: bool = False,
        tie_state: bool = False,
        precision: int = 1,
        resample: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        self.N = N
        self.H = H
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        self.rate = None if resample else 1.0
        self.channels = channels

        w, P, B = normal_plus_low_rank(
            measure=measure, matrix_size=self.N, correction_rank=rank, dtype=dtype
        )
        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        self.kernel = SSKernelNPLR(
            L=L,
            w=w,
            P=P,
            B=B,
            C=C,
            log_dt=generate_dt(self.H, dtype, dt_min, dt_max),
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )

    def forward(self, L: Optional[int] = None) -> torch.Tensor:
        k, _ = self.kernel(rate=self.rate, L=L)
        return k.float()

    def step(
        self, u: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u, state = self.kernel.step(u, state)
        return u.float(), state

    def default_state(self, *args, **kwargs) -> torch.Tensor:
        return self.kernel.default_state(*args, **kwargs)
