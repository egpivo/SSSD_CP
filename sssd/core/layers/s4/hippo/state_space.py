from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from opt_einsum import contract

from sssd.core.layers.s4.hippo.utils import (
    _c2r,
    _conj,
    _r2c,
    _resolve_conj,
    cauchy_wrapper,
    compute_fft_transform,
    get_dense_contraction,
    get_diagonal_contraction,
    get_input_contraction,
    get_output_contraction,
    hurwitz_transformation,
    low_rank_woodbury_correction,
    power,
    repeat_along_additional_dimension,
)
from sssd.utils.logger import setup_logger

LOGGER = setup_logger()


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

        # C is a regular parameter, not state
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))

        train = False
        if trainable is True:
            trainable, train = {}, True
        elif trainable is None or trainable is False:
            trainable = {}

        # Register parameters
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
        C_modified = C_conj + product if double_length else C_conj - product

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
        self, L: int, dtype: torch.dtype, device, cache: bool = True
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

    def _handle_sampling_rate_logic(
        self, rate: Optional[float], L: Optional[int]
    ) -> Tuple[float, int]:
        """
        Ensure either rate or L is provided. Handle sampling rate logic:
        If rate is not provided, calculate it from L. If L is not provided, calculate it from rate.
        Increase the internal length if needed.

        Args:
            rate: Sampling rate factor
            L: Target length

        Returns:
            Tuple containing the adjusted rate and target length
        """
        # Ensure either rate or L is provided
        assert not (rate is None and L is None), "Either rate or L must be provided."

        # Handle sampling rate logic
        if rate is None:
            rate = self.L / L
        if L is None:
            L = int(self.L / rate)

        # Increase the internal length if needed
        while rate * L > self.L:
            self._setup_C(double_length=True)

        return rate, L

    def forward(
        self, state: torch.Tensor = None, rate: float = 1.0, L: int = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Computes the output given an input state, sampling rate, and target length.

        Args:
            state: (..., s, N) extra tensor that augments B
            rate: sampling rate factor
            L: target length

        Returns:
            k_B: Coefficients for the output
            k_state: Optional state coefficients
        """
        rate, L = self._handle_sampling_rate_logic(rate, L)
        dt = torch.exp(self.log_dt) * rate

        B = _r2c(self.B)
        C = _r2c(self.C)
        P = _r2c(self.P)
        Q = P.conj() if self.Q is None else _r2c(self.Q)
        w = self._get_complex_weights()

        # Use cached FFT nodes if rate is 1.0, else update them
        if rate == 1.0:
            omega, z = _r2c(self.omega), _r2c(self.z)
        else:
            omega, z = self._update_fft_nodes(
                int(self.L / rate), dtype=w.dtype, device=w.device, cache=False
            )

        # Handle tied state with the pattern "... 1 n -> ... h n"
        if self.tie_state:
            B = repeat_along_additional_dimension(B, self.H)
            P = repeat_along_additional_dimension(P, self.H)
            Q = repeat_along_additional_dimension(Q, self.H)

        # Augment B with state if provided
        if state is not None:
            # Compute augmented state
            s = _conj(state) if state.size(-1) == self.N else state
            sA = s * _conj(w) - contract("bhm, rhm, rhn -> bhn", s, _conj(Q), _conj(P))
            s = s / dt.unsqueeze(-1) + sA / 2
            s = s[..., : self.N]

            # Concatenate augmented state with B
            B = torch.cat([s, B], dim=-3)

        # Incorporate dt into weights
        w = w * dt.unsqueeze(-1)

        # Stack B and P, C and Q for convenient batching
        B = torch.cat([B, P], dim=-3)
        C = torch.cat([C, Q], dim=-3)

        # Incorporate B and C batch dimensions
        v = B.unsqueeze(-3) * C.unsqueeze(-4)

        # Calculate resolvent at omega
        r = cauchy_wrapper(v, z, w)
        r = r * dt[None, None, :, None]

        # Low-rank Woodbury correction
        k_f = low_rank_woodbury_correction(r, self.rank)

        # Final correction for the bilinear transform
        k_f = k_f * 2 / (1 + omega)

        # Move from frequency to coefficients
        k = torch.fft.irfft(k_f)
        k = torch.nan_to_num(k)

        # Truncate to target length
        L = min(L, k.shape[-1])
        k = k[..., :L]

        # k_state shape: (S, C, H, L // 2 + 1)
        k_state = k[:-1, :, :, :] if state is not None else None
        k_B = k[-1, :, :, :]

        return k_B, k_state

    def _setup_linear(self) -> None:
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

    def _step_state_linear(
        self, u: Optional[torch.Tensor] = None, state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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
        if state.size(-1) == self.N:
            # Only store half of the conjugate pairs; should be true by default
            # There should be a slightly faster way using conjugate symmetry
            # inner outer product
            contract_fn = lambda p, x, y: contract(
                "r h n, r h m, ... h m -> ... h n", _conj(p), _conj(x), _conj(y)
            )[..., : self.N]
        elif state.size(-1) == 2 * self.N:
            step_params = {k: _conj(v) for k, v in step_params.items()}
            # TODO worth setting up a contract_expression in default_state if we want to use this at inference time for stepping
            # inner outer product
            contract_fn = lambda p, x, y: contract(
                "r h n, r h m, ... h m -> ... h n", p, x, y
            )
        else:
            raise ValueError(f"Invalid state size, but got {state.size(-1) }")

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

    def _setup_state(self) -> None:
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

    def _step_state(self, u: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Must be called after self.default_state() is used to construct an initial state!"""
        next_state = self.state_contraction(self.dA, state) + self.input_contraction(
            self.dB, u
        )
        return next_state

    def setup_step(self, mode: str = "dense") -> None:
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
            w, V = torch.linalg.eig(self.dA)
            V_inv = torch.linalg.inv(V)
            # Check that the eigen-decomposition is correct
            if self.verbose:
                LOGGER.info(
                    "Diagonalization check:",
                    torch.dist(V @ torch.diag_embed(w) @ V_inv, self.dA),
                )

            # Change the parameterization to diagonalize
            self.dA = w
            self.dB = contract("h n m, h m -> h n", V_inv, self.dB)
            self.dC = contract("h n m, c h n -> c h m", V, self.dC)

        elif mode == "dense":
            pass
        else:
            raise NotImplementedError(
                "NPLR Kernel step mode must be {'dense' | 'linear' | 'diagonal'}"
            )

    def default_state(self, *batch_shape: int) -> torch.Tensor:
        """
        Initialize the default state tensor.

        Args:
            *batch_shape (int): The shape of the batch.

        Returns:
            torch.Tensor: The default state tensor.
        """
        C = _r2c(self.C)
        N = C.size(-1)
        H = C.size(-2)

        if self._step_mode != "linear":
            N *= 2

            if self._step_mode == "diagonal":
                self.state_contraction = get_diagonal_contraction(batch_shape, H, N)
            else:
                # Dense (quadratic) case: expand all terms
                self.state_contraction = get_dense_contraction(batch_shape, H, N)

            self.input_contraction = get_input_contraction(batch_shape, H, N)

        self.output_contraction = get_output_contraction(batch_shape, H, N, C.shape[0])
        state = torch.zeros(*batch_shape, H, N, dtype=C.dtype, device=C.device)
        return state

    def step(
        self, u: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Must have called self.setup_step() and created state with self.default_state() before calling this"""
        new_state = (
            self._step_state_linear(u, state)
            if self._step_mode == "linear"
            else self._step_state(u, state)
        )
        y = self.output_contraction(self.dC, new_state)
        return y, new_state

    def register(
        self,
        name: str,
        tensor: torch.Tensor,
        trainable: bool = False,
        lr: Optional[float] = None,
        wd: Optional[float] = None,
    ) -> None:
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
