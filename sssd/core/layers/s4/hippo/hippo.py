import math

import torch
import torch.nn as nn

from sssd.core.layers.s4.hippo.state_space import SSKernelNPLR
from sssd.core.layers.s4.hippo.utils import normal_plus_low_rank
from sssd.utils.logger import setup_logger

LOGGER = setup_logger()


class HippoSSKernel(nn.Module):

    """Wrapper around SSKernel that generates A, B, C, dt according to HiPPO arguments.

    The SSKernel is expected to support the interface
    forward()
    default_state()
    setup_step()
    step()
    """

    def __init__(
        self,
        H,
        N=64,
        L=1,
        measure="legs",
        rank=1,
        channels=1,  # 1-dim to C-dim map; can think of C as having separate "heads"
        dt_min=0.001,
        dt_max=0.1,
        trainable=None,  # Dictionary of options to train various HiPPO parameters
        lr=None,  # Hook to set LR of hippo parameters differently
        length_correction=True,  # Multiply by I-A|^L after initialization; can be turned off for initialization speed
        hurwitz=False,
        tie_state=False,  # Tie parameters of HiPPO ODE across the H features
        precision=1,  # 1 (single) or 2 (double) for the kernel
        resample=False,  # If given inputs of different lengths, adjust the sampling rate. Note that L should always be provided in this case, as it assumes that L is the true underlying length of the continuous signal
        verbose=False,
    ):
        super().__init__()
        self.N = N
        self.H = H
        L = L or 1
        self.precision = precision
        dtype = torch.double if self.precision == 2 else torch.float
        cdtype = torch.cfloat if dtype == torch.float else torch.cdouble
        self.rate = None if resample else 1.0
        self.channels = channels

        # Generate dt
        log_dt = torch.rand(self.H, dtype=dtype) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        w, p, B, _ = normal_plus_low_rank(
            measure=measure, matrix_size=self.N, correction_rank=rank, dtype=dtype
        )
        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)
        self.kernel = SSKernelNPLR(
            L,
            w,
            p,
            B,
            C,
            log_dt,
            hurwitz=hurwitz,
            trainable=trainable,
            lr=lr,
            tie_state=tie_state,
            length_correction=length_correction,
            verbose=verbose,
        )

    def forward(self, L=None):
        k, _ = self.kernel(rate=self.rate, L=L)
        return k.float()

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)
