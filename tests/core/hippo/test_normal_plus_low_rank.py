import pytest
import torch

from sssd.core.layers.s4.hippo.normal_plus_low_rank import NormalPlusLowRank


@pytest.fixture
def setup_data():
    measure = "random"
    N = 10
    rank = 1
    dtype = torch.float
    return measure, N, rank, dtype


def test_nplr_shapes_and_types(setup_data):
    measure, N, rank, dtype = setup_data
    nplr = NormalPlusLowRank(measure, N, rank, dtype).compute()

    # Check if the shapes are correct
    assert nplr.w.shape == (N // 2,)
    assert nplr.P.shape == (rank, N // 2)
    assert nplr.B.shape == (N // 2,)

    # Check if the types are correct
    assert nplr.w.dtype == (torch.cfloat if dtype == torch.float else torch.cdouble)
    assert nplr.P.dtype == nplr.w.dtype
    assert nplr.B.dtype == nplr.w.dtype


def test_nplr_values(setup_data):
    torch.manual_seed(1)
    measure, N, rank, dtype = setup_data
    nplr = NormalPlusLowRank(measure, N, rank, dtype).compute()

    # Check if the imaginary parts of w are within the range [-2, 2]
    assert torch.all(
        (nplr.w.imag >= -2) & (nplr.w.imag <= 2)
    ), "Imaginary part of w should be within [-3, 3]"
    assert torch.all(torch.abs(nplr.P) <= 5), "Elements of P should be within [-5, 5]"


# Test to ensure that the eigenvalues 'w' are complex numbers
def test_nplr_w_complex(setup_data):
    _, N, _, dtype = setup_data
    nplr = NormalPlusLowRank("random", N, dtype=dtype).compute()
    assert all(
        torch.is_complex(val) for val in nplr.w
    ), "Eigenvalues w should be complex numbers"


# Test to ensure that 'P' has the correct rank
def test_nplr_p_rank(setup_data):
    _, N, rank, dtype = setup_data
    nplr = NormalPlusLowRank("random", N, correction_rank=rank, dtype=dtype).compute()
    assert nplr.P.shape[0] == rank, f"Matrix P should have rank {rank}"


# Test to check if 'nplr' function handles invalid 'measure' input
def test_nplr_invalid_measure():
    with pytest.raises(ValueError):
        NormalPlusLowRank("invalid_measure", 10).compute()
