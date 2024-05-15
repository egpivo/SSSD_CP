import pytest
import torch

from sssd.core.layers.s4.hippo.normal_plus_low_rank import (
    FourierNormalPlusLowRank,
    LagtNormalPlusLowRank,
    LegsNormalPlusLowRank,
    LegtNormalPlusLowRank,
    NormalPlusLowRank,
    RandomNormalPlusLowRank,
)


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


@pytest.fixture
def matrix_size():
    return 10


@pytest.fixture
def correction_rank():
    return 2


@pytest.fixture
def dtype():
    return torch.float


@pytest.fixture
def random_instance(matrix_size, correction_rank, dtype):
    return RandomNormalPlusLowRank("random", matrix_size, correction_rank, dtype)


@pytest.fixture
def legs_instance(matrix_size, correction_rank, dtype):
    return LegsNormalPlusLowRank("legs", matrix_size, correction_rank, dtype)


@pytest.fixture
def legt_instance(matrix_size, correction_rank, dtype):
    return LegtNormalPlusLowRank("legt", matrix_size, correction_rank, dtype)


@pytest.fixture
def lagt_instance(matrix_size, correction_rank, dtype):
    return LagtNormalPlusLowRank("lagt", matrix_size, correction_rank, dtype)


@pytest.fixture
def fourier_instance(matrix_size, correction_rank, dtype):
    return FourierNormalPlusLowRank("fourier", matrix_size, correction_rank, dtype)


def test_legs_instance(legs_instance):
    result = legs_instance.compute()
    assert isinstance(result.w, torch.Tensor)
    assert isinstance(result.P, torch.Tensor)
    assert isinstance(result.B, torch.Tensor)


def test_legt_instance(legt_instance):
    result = legt_instance.compute()
    assert isinstance(result.w, torch.Tensor)
    assert isinstance(result.P, torch.Tensor)
    assert isinstance(result.B, torch.Tensor)


def test_lagt_instance(lagt_instance):
    result = lagt_instance.compute()
    assert isinstance(result.w, torch.Tensor)
    assert isinstance(result.P, torch.Tensor)
    assert isinstance(result.B, torch.Tensor)


def test_fourier_instance(fourier_instance):
    result = fourier_instance.compute()
    assert isinstance(result.w, torch.Tensor)
    assert isinstance(result.P, torch.Tensor)
    assert isinstance(result.B, torch.Tensor)


def test_random_instance(random_instance):
    result = random_instance.compute()
    assert isinstance(result.w, torch.Tensor)
    assert isinstance(result.P, torch.Tensor)
    assert isinstance(result.B, torch.Tensor)
