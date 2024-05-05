import torch

from sssd.core.layers.s4.hippo.utils import broadcast_dims, cauchy_slow

v = torch.randn(10, 4, dtype=torch.complex64, requires_grad=True)
w = torch.randn(10, 4, dtype=torch.complex64, requires_grad=True)
z = torch.randn(
    5,
    dtype=torch.complex64,
)


def cauchy_conj(v, z, w):
    """
    >>> v = torch.randn(10, 4, dtype=torch.complex64,  requires_grad=True)
    >>> w = torch.randn(10, 4, dtype=torch.complex64,  requires_grad=True)
    >>> z = torch.randn(5, dtype=torch.complex64,)
    >>> cauchy_conj(v, z, w).shape
    torch.Size([10, 5])
    """
    if torch.cuda.is_available():

        from pykeops.torch import Genred

        expr_num = "z * ComplexReal(v) - Real2Complex(Sum(v * w))"
        expr_denom = "ComplexMult(z-w, z-Conj(w))"

        cauchy_mult = Genred(
            f"ComplexDivide({expr_num}, {expr_denom})",
            [
                "v = Vj(2)",
                "z = Vi(2)",
                "w = Vj(2)",
            ],
            reduction_op="Sum",
            axis=1,
        )

        v, z, w = broadcast_dims(v, z, w)
        v = _c2r(v)
        z = _c2r(z)
        w = _c2r(w)

        r = 2 * cauchy_mult(v, z, w, backend="GPU")
        return _r2c(r)
    else:
        # Fallback to a slower method if GPU is not available
        return cauchy_slow(v, z, w)


_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
