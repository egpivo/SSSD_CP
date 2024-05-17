from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from opt_einsum import contract

from sssd.core.layers.activation import Activation
from sssd.core.layers.linear import LinearActivation
from sssd.core.layers.s4.hippo.hippo import HippoSSKernel
from sssd.utils.logger import setup_logger

LOGGER = setup_logger()


class S4(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        l_max: int = 1,
        channels: int = 1,
        bidirectional: bool = False,
        activation: str = "gelu",
        postact: Optional[str] = None,
        initializer: Optional[Callable] = None,
        weight_norm: bool = False,
        hyper_act: Optional[str] = None,
        dropout: float = 0.0,
        transposed: bool = True,
        verbose: bool = False,
        **kernel_args,
    ) -> None:
        """
        Initializes the S4 layer.

        Args:
        d_model (int): The dimension of the input and output.
        d_state (int, optional): The dimension of the state. Defaults to 64.
        l_max (int, optional): The maximum sequence length. Defaults to 1.
        channels (int, optional): The number of channels. Defaults to 1.
        bidirectional (bool, optional): Whether to use bidirectional processing. Defaults to False.
        activation (str, optional): The activation function to use. Defaults to "gelu".
        postact (Optional[str], optional): The post-activation function to use. Defaults to None.
        initializer (Optional[Callable], optional): The initializer to use. Defaults to None.
        weight_norm (bool, optional): Whether to use weight normalization. Defaults to False.
        hyper_act (Optional[str], optional): The hyperactivation function to use. Defaults to None.
        dropout (float, optional): The dropout rate. Defaults to 0.0.
        transposed (bool, optional): Whether to use transposed axis ordering. Defaults to True.
        verbose (bool, optional): Whether to be verbose. Defaults to False.
        **kernel_args: Additional keyword arguments for the SSM Kernel.
        """
        super().__init__()
        if verbose:
            LOGGER.info(f"Constructing S4 (H, N, L) = ({d_model}, {d_state}, {l_max})")

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        # optional multiplicative modulation GLU-style
        # https://arxiv.org/abs/2002.05202
        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = HippoSSKernel(
            self.h, N=self.n, L=l_max, channels=channels, verbose=verbose, **kernel_args
        )

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            in_features=self.h * self.channels,
            out_features=self.h,
            is_transposed=self.transposed,
            initializer=initializer,
            activation=postact,
            activate=True,
            weight_norm=weight_norm,
        )

    def forward(self, u: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L)  # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, "(s c) h l -> s c h l", s=2)
            k = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        k_f = torch.fft.rfft(k, n=2 * L)  # (C H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y_f = contract(
            "bhl,chl->bchl", u_f, k_f
        )  # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2 * L)[..., :L]  # (B C H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + contract(
            "bhl,ch->bchl", u, self.D
        )  # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Optional hyper-network multiplication
        if self.hyper:
            y, yh = rearrange(y, "b (s c) h l -> s b c h l", s=2)
            y = self.hyper_activation(yh) * y

        # Reshape to flatten channels
        y = rearrange(y, "... c h l -> ... (c h) l")

        y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        y = self.output_linear(y)
        return y

    def step(
        self, u: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state)  # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, "... c h -> ... (c h)")
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(
        self, *batch_shape, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self) -> int:
        return self.h * self.n

    @property
    def d_output(self) -> int:
        return self.h

    @property
    def state_to_tensor(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda state: rearrange("... h n -> ... (h n)", state)


class S4Layer(nn.Module):
    # S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(
        self,
        features: int,
        lmax: int,
        N: int = 64,
        dropout: float = 0.0,
        bidirectional: bool = True,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.s4_layer = S4(
            d_model=features, d_state=N, l_max=lmax, bidirectional=bidirectional
        )

        self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape seq, batch, feature
        x = x.permute(
            (1, 2, 0)
        )  # batch, feature, seq (as expected from S4 with transposed=True)
        xout = self.s4_layer(x)  # batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x  # skip connection   # batch, feature, seq
        xout = xout.permute((2, 0, 1))  # seq, batch, feature
        return self.norm_layer(xout)
