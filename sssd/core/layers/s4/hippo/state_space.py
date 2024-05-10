from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from opt_einsum import contract, contract_expression

from sssd.core.layers.s4.hippo.utils import (
    compute_fft_transform,
    hurwitz_transformation,
    power,
)
from sssd.utils.logger import setup_logger

LOGGER = setup_logger()


try:  # This module will be downloaded from s4 repo
    from sssd.core.layers.s4.hippo.cauchy import cauchy_mult

    has_cauchy_extension = True
except ModuleNotFoundError:
    LOGGER.warning(
        "CUDA extension for cauchy multiplication not found. Please check `install_extensions_cauchy` in envs/conda/utils.sh "
    )
    from sssd.core.layers.s4.hippo.utils import cauchy_cpu

    has_cauchy_extension = False


_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
_resolve_conj = lambda x: x.conj().resolve_conj()


class SSKernelNPLR(nn.Module):
    """
    Stores a representation of and computes the SSKernel function K_L(A^dt, B^dt, C) corresponding to a discretized state space, where A is Normal + Low Rank (NPLR).

    The class name stands for 'State-Space SSKernel for Normal Plus Low-Rank'.
    The parameters of this function are as follows:
    - A: (... N N) the state matrix
    - B: (... N) input matrix
    - C: (... N) output matrix
    - dt: (...) timescales / discretization step size
    - p, q: (... P N) low-rank correction to A, such that Ap=A+pq^T is a normal matrix

    The forward pass of this Module returns:
    (... L) that represents FFT SSKernel_L(A^dt, B^dt, C)
    """

    def __init__(
        self,
        L: int,
        w: torch.Tensor,
        P: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        log_dt: torch.Tensor,
        hurwitz: bool = False,
        trainable: Optional[dict] = None,
        lr: Optional[float] = None,
        tie_state: bool = False,
        length_correction: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize the SSKernelNPLR module.

        Parameters:
        - L (int): Maximum length; this module computes an SSM kernel of length L.
        - w (torch.Tensor): Vector of shape (N,) representing weights.
        - P (torch.Tensor): Matrix of shape (r, N) representing low-rank correction to A.
        - B (torch.Tensor): Vector of shape (N,) representing input matrix B.
        - C (torch.Tensor): Tensor of shape (H, C, N) representing the output matrix C.
        - log_dt (torch.Tensor): Tensor of shape (H,) representing timescale per feature.
        - hurwitz (bool): Flag to tie pq and ensure w has negative real part.
        - trainable (dict, optional): Dictionary specifying which parameters are trainable.
        - lr (float, optional): Learning rate for optimization.
        - tie_state (bool): Flag to tie all state parameters across the H hidden features.
        - length_correction (bool): Flag to multiply C by (I - dA^L) - can be turned off for slight speedup at initialization.
        - verbose (bool): Flag for verbose mode.

        Note: Tensor shape N here denotes half the true state size, because of conjugate symmetry.
        """
        super().__init__()
        self.hurwitz = hurwitz
        self.tie_state = tie_state
        self.verbose = verbose

        # Rank of low-rank correction
        self.rank = P.shape[-2]
        assert w.size(-1) == P.size(-1) == B.size(-1) == C.size(-1)
        self.H = log_dt.size(-1)
        self.N = w.size(-1)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N)))  # (H, C, N)
        H = 1 if self.tie_state else self.H
        B = repeat(B, "n -> 1 h n", h=H)
        P = repeat(P, "r n -> r h n", h=H)
        w = repeat(w, "n -> h n", h=H)

        # Cache Fourier nodes every time we set up a desired length
        self.L = L
        if self.L is not None:
            self._update_fft_nodes(self.L, dtype=C.dtype, device=C.device, cache=True)

        # Register parameters
        self.C = nn.Parameter(
            _c2r(_resolve_conj(C))
        )  # C is a regular parameter, not state

        train = False
        if trainable is True:
            trainable, train = {}, True
        elif trainable is None or trainable is False:
            trainable = {}

        self.register("log_dt", log_dt, trainable.get("dt", train), lr, 0.0)
        self.register("B", _c2r(B), trainable.get("B", train), lr, 0.0)
        self.register("P", _c2r(P), trainable.get("P", train), lr, 0.0)
        if self.hurwitz:
            log_w_real = torch.log(-w.real + 1e-3)
            w_imag = w.imag
            self.register("log_w_real", log_w_real, trainable.get("A", 0), lr, 0.0)
            self.register("w_imag", w_imag, trainable.get("A", train), lr, 0.0)
            self.Q = None
        else:
            self.register("w", _c2r(w), trainable.get("A", train), lr, 0.0)
            Q = _resolve_conj(P.clone())
            self.register("Q", _c2r(Q), trainable.get("P", train), lr, 0.0)

        if length_correction:
            self._setup_C()

    @torch.no_grad()
    def _setup_C(self, double_length=False) -> None:
        """
        Constructs the modified output matrix C~ from the current C.

        If double_length is True, it converts C for a sequence of length L to a sequence of length 2L.

        Args:
            double_length (bool): Flag to indicate if the length should be doubled.

        Returns:
            None: Modifies the matrix C in place.
        """
        # Convert C to complex form
        C_complex = _r2c(self.C)

        # Setup the state for the transformation
        self._setup_state()

        # Compute the matrix power of dA for length L
        dA_power_L = power(self.L, self.dA)

        # Multiply C by (I - dA^L) or (I + dA^L) if doubling the length
        C_conj = _conj(C_complex)
        product = contract(
            "h m n, c h n -> c h m", dA_power_L.transpose(-1, -2), C_conj
        )
        if double_length:
            product = -product  # Use (I + dA^L) for doubling the length
        C_modified = C_conj - product

        # Retain only the necessary conjugate pairs
        C_modified = C_modified[..., : self.N]

        # Update the original C with the modified values
        self.C.copy_(_c2r(C_modified))

        # If doubling the length, update L and FFT nodes accordingly
        if double_length:
            self.L *= 2
            if self.verbose:
                LOGGER.info(f"S4: Doubling length from L = {self.L} to {2 * self.L}")
            self._update_fft_nodes(
                self.L, dtype=C_complex.dtype, device=C_complex.device, cache=True
            )

    def _update_fft_nodes(
        self, L: int, dtype, device, cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate (and cache) FFT nodes and their "unprocessed" them with the bilinear transform.

        Args:
        - L (int): The length.
        - dtype: The data type.
        - device: The device.
        - cache (bool, optional): Whether to cache the results. Defaults to True.

        Returns:
        - omega: The FFT nodes. (shape (L // 2 + 1))
        - z: The transformed nodes. (shape (L // 2 + 1)_)
        """
        omega, z = compute_fft_transform(L, dtype, device)
        if cache:
            self.register_buffer("omega", _c2r(omega))
            self.register_buffer("z", _c2r(z))
        return omega, z

    def _get_complex_weights(self) -> torch.Tensor:
        """
        Retrieves the complex diagonal weights 'w' for the state matrix.

        If the 'hurwitz' flag is set, it constructs 'w' using the Hurwitz transformation.
        Otherwise, it converts the stored real-valued tensor 'w' to a complex tensor.

        Returns:
            torch.Tensor: The complex diagonal weights 'w'.
        """
        return (
            hurwitz_transformation(self.log_w_real, self.w_imag)
            if self.hurwitz
            else _r2c(self.w)
        )

    def forward(self, state=None, rate=1.0, L=None):
        """
        state: (..., s, N) extra tensor that augments B
        rate: sampling rate factor

        returns: (..., c+s, L)
        """
        # Handle sampling rate logic
        # The idea is that this kernel's length (in continuous units) is self.L, while we are asked to provide a kernel of length L at (relative) sampling rate rate
        # If either are not passed in, assume we're not asked to change the scale of our kernel
        assert not (rate is None and L is None)
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        while rate * L > self.L:
            self._setup_C(double_length=True)

        dt = torch.exp(self.log_dt) * rate
        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._get_complex_weights()

        if rate == 1.0:
            # Use cached FFT nodes
            omega, z = _r2c(self.omega), _r2c(self.z)  # (..., L // 2 + 1)
        else:
            omega, z = self._update_fft_nodes(
                int(self.L / rate), dtype=w.dtype, device=w.device, cache=False
            )

        if self.tie_state:
            B = repeat(B, "... 1 n -> ... h n", h=self.H)
            P = repeat(P, "... 1 n -> ... h n", h=self.H)
            Q = repeat(Q, "... 1 n -> ... h n", h=self.H)

        # Augment B
        if state is not None:
            # Have to "unbilinear" the state to put it into the same "type" as B
            # Compute 1/dt * (I + dt/2 A) @ state

            # Can do this without expanding (maybe minor speedup using conj symmetry in theory), but it's easier to read this way
            s = _conj(state) if state.size(-1) == self.N else state  # (B H N)
            sA = s * _conj(w) - contract(  # (B H N)
                "bhm, rhm, rhn -> bhn", s, _conj(Q), _conj(P)
            )
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., : self.N]

            B = torch.cat([s, B], dim=-3)  # (s+1, H, N)

        # Incorporate dt into A
        w = w * dt.unsqueeze(-1)  # (H N)

        # Stack B and p, C and q for convenient batching
        B = torch.cat([B, P], dim=-3)  # (s+1+r, H, N)
        C = torch.cat([C, Q], dim=-3)  # (c+r, H, N)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)  # (s+1+r, c+r, H, N)
        # w = w[None, None, ...]  # (1, 1, H, N)
        # z = z[None, None, None, ...]  # (1, 1, 1, L // 2 + 1)

        # Calculate resolvent at omega
        if has_cauchy_extension and z.dtype == torch.cfloat:
            r = cauchy_mult(v, z, w, symmetric=True)
        else:
            r = cauchy_cpu(v, z, w)
        r = r * dt[None, None, :, None]  # (S+1+R, C+R, H, L // 2 + 1)

        # Low-rank Woodbury correction
        if self.rank == 1:
            k_f = r[:-1, :-1, :, :] - r[:-1, -1:, :, :] * r[-1:, :-1, :, :] / (
                1 + r[-1:, -1:, :, :]
            )
        elif self.rank == 2:
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
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
            r00 = r[: -self.rank, : -self.rank, :, :]
            r01 = r[: -self.rank, -self.rank :, :, :]
            r10 = r[-self.rank :, : -self.rank, :, :]
            r11 = r[-self.rank :, -self.rank :, :, :]
            r11 = rearrange(r11, "a b h n -> h n a b")
            r11 = torch.linalg.inv(torch.eye(self.rank, device=r.device) + r11)
            r11 = rearrange(r11, "h n a b -> a b h n")
            k_f = r00 - torch.einsum(
                "i j h n, j k h n, k l h n -> i l h n", r01, r11, r10
            )

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f)  # (S+1, C, H, L // 2 + 1)
        # Avoid the underflow or overflow
        k = torch.nan_to_num(k)
        # Truncate to target length
        L = min(L, k.shape[-1])
        k = k[..., :L]

        if state is not None:
            k_state = k[:-1, :, :, :]  # (S, C, H, L // 2 + 1)
        else:
            k_state = None
        k_B = k[-1, :, :, :]  # (C H L // 2 + 1)
        return k_B, k_state

    def _setup_linear(self):
        """Create parameters that allow fast linear stepping of state"""
        w = self._get_complex_weights()
        B = _r2c(self.B)  # (H N)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)

        # Prepare Linear stepping
        dt = torch.exp(self.log_dt)
        D = (2.0 / dt.unsqueeze(-1) - w).reciprocal()  # (H, N)
        R = (
            torch.eye(self.rank, dtype=w.dtype, device=w.device)
            + 2 * contract("r h n, h n, s h n -> h r s", Q, D, P).real
        )  # (H r r)
        Q_D = rearrange(Q * D, "r h n -> h r n")
        R = torch.linalg.solve(R.to(Q_D), Q_D)  # (H r N)
        R = rearrange(R, "h r n -> r h n")

        self.step_params = {
            "D": D,  # (H N)
            "R": R,  # (r H N)
            "P": P,  # (r H N)
            "Q": Q,  # (r H N)
            "B": B,  # (1 H N)
            "E": 2.0 / dt.unsqueeze(-1) + w,  # (H N)
        }

    def _step_state_linear(self, u=None, state=None):
        """
        Version of the step function that has time O(N) instead of O(N^2) per step, which takes advantage of the DPLR form and bilinear discretization.

        Unfortunately, as currently implemented it's about 2x slower because it calls several sequential operations. Perhaps a fused CUDA kernel implementation would be much faster

        u: (H) input
        state: (H, N/2) state with conjugate pairs
          Optionally, the state can have last dimension N
        Returns: same shape as state
        """
        C = _r2c(self.C)  # View used for dtype/device

        if u is None:  # Special case used to find dA
            u = torch.zeros(self.H, dtype=C.dtype, device=C.device)
        if state is None:  # Special case used to find dB
            state = torch.zeros(self.H, self.N, dtype=C.dtype, device=C.device)

        step_params = self.step_params.copy()
        if (
            state.size(-1) == self.N
        ):  # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            contract_fn = lambda p, x, y: contract(
                "r h n, r h m, ... h m -> ... h n", _conj(p), _conj(x), _conj(y)
            )[
                ..., : self.N
            ]  # inner outer product
        else:
            assert state.size(-1) == 2 * self.N
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            contract_fn = lambda p, x, y: contract(
                "r h n, r h m, ... h m -> ... h n", p, x, y
            )  # inner outer product
        D = step_params["D"]  # (H N)
        E = step_params["E"]  # (H N)
        R = step_params["R"]  # (r H N)
        P = step_params["P"]  # (r H N)
        Q = step_params["Q"]  # (r H N)
        B = step_params["B"]  # (1 H N)

        new_state = E * state - contract_fn(P, Q, state)  # (B H N)
        new_state = new_state + 2.0 * B * u.unsqueeze(-1)  # (B H N)
        new_state = D * (new_state - contract_fn(P, R, new_state))

        return new_state

    def _setup_state(self):
        """Construct dA and dB for discretized state equation"""

        # Construct dA and dB by using the stepping
        self._setup_linear()
        C = _r2c(self.C)  # Just returns a view that we use for finding dtype/device

        state = torch.eye(2 * self.N, dtype=C.dtype, device=C.device).unsqueeze(
            -2
        )  # (N 1 N)
        dA = self._step_state_linear(state=state)
        dA = rearrange(dA, "n h m -> h m n")
        self.dA = dA  # (H N N)

        u = C.new_ones(self.H)
        dB = self._step_state_linear(u=u)
        dB = _conj(dB)
        self.dB = rearrange(dB, "1 h n -> h n")  # (H N)

    def _step_state(self, u, state):
        """Must be called after self.default_state() is used to construct an initial state!"""
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(
            self.dB, u
        )
        return next_state

    def setup_step(self, mode="dense"):
        """Set up dA, dB, dC discretized parameters for stepping"""
        self._setup_state()
        # Calculate original C
        dA_L = power(self.L, self.dA)
        I = torch.eye(self.dA.size(-1)).to(dA_L)
        C = _conj(_r2c(self.C))  # (H C N)

        dC = torch.linalg.solve(
            I - dA_L.transpose(-1, -2),
            C.unsqueeze(-1),
        ).squeeze(-1)
        self.dC = dC

        # Do special preprocessing for different step modes

        self._step_mode = mode

        if mode == "linear":
            # Linear case: special step function for the state, we need to handle output
            # use conjugate symmetry by default, which affects the output projection
            self.dC = 2 * self.dC[:, :, : self.N]
        elif mode == "diagonal":
            # Eigen-decomposition of the A matrix
            L, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigen-decomposition is correct
            if self.verbose:
                LOGGER.info(
                    "Diagonalization error:",
                    torch.dist(V @ torch.diag_embed(L) @ V_inv, self.dA),
                )

            # Change the parameterization to diagonalize
            self.dA = L
            self.dB = contract("h n m, h m -> h n", V_inv, self.dB)
            self.dC = contract("h n m, c h n -> c h m", V, self.dC)

        elif mode == "dense":
            pass
        else:
            raise NotImplementedError(
                "NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}"
            )

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        # Cache the tensor contractions we will later do, for efficiency
        # These are put in this function because they depend on the batch size
        if self._step_mode != "linear":
            N *= 2

            if self._step_mode == "diagonal":
                self.state_contraction = contract_expression(
                    "h n, ... h n -> ... h n",
                    (H, N),
                    batch_shape + (H, N),
                )
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = contract_expression(
                    "h m n, ... h n -> ... h m",
                    (H, N, N),
                    batch_shape + (H, N),
                )

            self.input_contraction = contract_expression(
                "h n, ... h -> ... h n",
                (H, N),  # self.dB.shape
                batch_shape + (H,),
            )

        self.output_contraction = contract_expression(
            "c h n, ... h n -> ... c h",
            (C.shape[0], H, N),  # self.dC.shape
            batch_shape + (H, N),
        )

        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        """Must have called self.setup_step() and created state with self.default_state() before calling this"""

        if self._step_mode == "linear":
            new_state = self._step_state_linear(u, state)
        else:
            new_state = self._step_state(u, state)
        y = self.output_contraction(self.dC, new_state)
        return y, new_state

    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)
