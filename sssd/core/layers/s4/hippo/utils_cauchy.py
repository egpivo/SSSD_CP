import torch


def cauchy_conj(v, z, w):
    if torch.cuda.is_available():
        try:
            from pykeops.torch import Genred

            has_pykeops = True
        except ImportError:
            has_pykeops = False

        if has_pykeops:
            # Use PyKeOps on GPU
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
                dtype="float32" if v.dtype == torch.cfloat else "float64",
            )

            v, z, w = _broadcast_dims(v, z, w)
            v = _c2r(v)
            z = _c2r(z)
            w = _c2r(w)

            r = 2 * cauchy_mult(v, z, w, backend="GPU")
            return _r2c(r)
        else:
            # Fallback to a slower method if PyKeOps is not available
            return cauchy_slow(v, z, w)
    else:
        # Fallback to a slower method if GPU is not available
        return cauchy_slow(v, z, w)


def cauchy_slow(v, z, w):
    """
    v, w: (..., N)
    z: (..., L)
    returns: (..., L)
    """
    cauchy_matrix = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))  # (... N L)
    return torch.sum(cauchy_matrix, dim=-2)


def _broadcast_dims(*tensors):
    max_dim = max([len(tensor.shape) for tensor in tensors])
    tensors = [
        tensor.view((1,) * (max_dim - len(tensor.shape)) + tensor.shape)
        for tensor in tensors
    ]
    return tensors


_c2r = torch.view_as_real
_r2c = torch.view_as_complex
_conj = lambda x: torch.cat([x, x.conj()], dim=-1)
