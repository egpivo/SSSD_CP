import opt_einsum as oe
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sssd.core.layers.activation import Activation
from sssd.core.layers.linear import LinearActivation
from sssd.core.s4.hippo.hippo import HippoSSKernel
from sssd.utils.logger import setup_logger

contract = oe.contract


LOGGER = setup_logger()


class S4(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=1,  # Maximum length of sequence. Fine if not provided: the kernel will keep doubling in length until longer than sequence. However, this can be marginally slower if the true length is not a power of 2
        channels=1,  # maps 1-dim to C-dim
        bidirectional=False,
        # Arguments for FF
        activation="gelu",  # activation in between SS and FF
        postact=None,  # activation after FF
        initializer=None,  # initializer on FF
        weight_norm=False,  # weight normalization on FF
        hyper_act=None,  # Use a "hypernetwork" multiplication
        dropout=0.0,
        transposed=True,  # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        # SSM Kernel arguments
        **kernel_args,
    ):

        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, set l_max=1
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
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
        # self.time_transformer = get_torch_trans(heads=8, layers=1, channels=self.h)

    def forward(self, u, **kwargs):  # absorbs return_output and transformer sssd mask
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

        # ysize = b, k, l, requieres l, b, k
        # y = self.time_transformer(y.permute(2,0,1)).permute(1,2,0)

        return y, None

    def step(self, u, state):
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

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange("... h n -> ... (h n)", state)


class S4Layer(nn.Module):
    # S4 Layer that can be used as a drop-in replacement for a TransformerEncoder
    def __init__(
        self, features, lmax, N=64, dropout=0.0, bidirectional=True, layer_norm=True
    ):
        super().__init__()
        self.s4_layer = S4(
            d_model=features, d_state=N, l_max=lmax, bidirectional=bidirectional
        )

        self.norm_layer = nn.LayerNorm(features) if layer_norm else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        # x has shape seq, batch, feature
        x = x.permute(
            (1, 2, 0)
        )  # batch, feature, seq (as expected from S4 with transposed=True)
        xout, _ = self.s4_layer(x)  # batch, feature, seq
        xout = self.dropout(xout)
        xout = xout + x  # skip connection   # batch, feature, seq
        xout = xout.permute((2, 0, 1))  # seq, batch, feature
        return self.norm_layer(xout)
