import numpy as np
import pytest
import torch

from sssd.core.layers.s4.hippo.utils import (
    TransitionMatrix,
    embed_c2r,
    generate_low_rank_matrix,
    normal_plus_low_rank,
    power,
)


def test_power_without_v():
    # Test case 1: L = 2, A is a 2x2 matrix
    A = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    L = 2
    expected_result = torch.tensor([[7, 10], [15, 22]], dtype=torch.float32)
    result = power(L, A)
    assert torch.allclose(result, expected_result)

    # Test case 2: L = 3, A is a 3x3 matrix
    A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    L = 3
    expected_result = torch.tensor(
        [[468, 576, 684], [1062, 1305, 1548], [1656, 2034, 2412]], dtype=torch.float32
    )
    result = power(L, A)
    assert torch.allclose(result, expected_result)


def test_transition_lagt():
    N = 10
    beta = 1.5
    A, B = TransitionMatrix("lagt", N, beta=beta)
    assert A.shape == (N, N)
    assert B.shape == (N, 1)
    assert np.allclose(A, np.eye(N) / 2 - np.tril(np.ones((N, N))))
    assert np.allclose(B, beta * np.ones((N, 1)))


def test_transition_glagt():
    N = 10
    alpha = 0.5
    beta = 0.01
    A, B = TransitionMatrix("glagt", N, alpha=alpha, beta=beta)
    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_transition_legt():
    N = 10
    A, B = TransitionMatrix("legt", N)
    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_transition_legs():
    N = 10
    A, B = TransitionMatrix("legs", N)
    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_transition_fourier():
    N = 10
    A, B = TransitionMatrix("fourier", N)
    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_transition_random():
    N = 10
    A, B = TransitionMatrix("random", N)
    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_transition_diagonal():
    N = 10
    A, B = TransitionMatrix("diagonal", N)
    assert A.shape == (N, N)
    assert B.shape == (N, 1)


def test_transition_invalid_measure():
    N = 10
    with pytest.raises(NotImplementedError):
        TransitionMatrix("invalid_measure", N)


def test_embed_c2r():
    # Test case 1: Simple 2x2 matrix
    A = np.array([[1, 2], [3, 4]])
    expected_output = np.array([[1, 0, 2, 0], [0, 1, 0, 2], [3, 0, 4, 0], [0, 3, 0, 4]])
    output = embed_c2r(A)
    assert np.allclose(output, expected_output)

    # Test case 2: 3x3 matrix
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected_output = np.array(
        [
            [1, 0, 2, 0, 3, 0],
            [0, 1, 0, 2, 0, 3],
            [4, 0, 5, 0, 6, 0],
            [0, 4, 0, 5, 0, 6],
            [7, 0, 8, 0, 9, 0],
            [0, 7, 0, 8, 0, 9],
        ]
    )

    output = embed_c2r(A)
    assert np.allclose(output, expected_output)

    # Test case 3: Higher-dimensional input (should raise an exception)
    try:
        A = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        embed_c2r(A)
        assert (
            False
        ), "Expected a ValueError for input with more than 2 dimensions, but no exception was raised"
    except ValueError as e:
        assert (
            str(e) == "Expected 2 dimensions, got 3"
        ), f"Unexpected error message: {str(e)}"

    # Test case 4: 1D input (should raise an exception)
    try:
        A = np.array([1, 2, 3])
        embed_c2r(A)
        assert False, "Expected a ValueError for 1D input, but no exception was raised"
    except ValueError as e:
        assert (
            str(e) == "Expected 2 dimensions, got 1"
        ), f"Unexpected error message: {str(e)}"

    # Test case 5: Empty input (should raise an exception)
    try:
        A = np.array([])
        embed_c2r(A)
        assert (
            False
        ), "Expected a ValueError for empty input, but no exception was raised"
    except ValueError as e:
        assert (
            str(e) == "Expected 2 dimensions, got 1"
        ), f"Unexpected error message: {str(e)}"


@pytest.mark.parametrize(
    "measure, N, rank, expected_shape",
    [
        ("legs", 5, 1, (1, 5)),
        ("legs", 10, 2, (2, 10)),
        ("legt", 5, 2, (2, 5)),
        ("legt", 10, 3, (3, 10)),
        ("lagt", 5, 1, (1, 5)),
        ("lagt", 10, 2, (2, 10)),
        ("fourier", 5, 2, (2, 5)),
        ("fourier", 10, 3, (3, 10)),
    ],
)
def test_rank_correction(measure, N, rank, expected_shape):
    P = generate_low_rank_matrix(measure, N, rank)
    assert P.shape == expected_shape


def test_rank_correction_invalid_measure():
    with pytest.raises(NotImplementedError):
        generate_low_rank_matrix("invalid_measure", 5, 1)


def test_rank_correction_dtype():
    P = generate_low_rank_matrix("legs", 5, 1, dtype=torch.float64)
    assert P.dtype == torch.float64


# nplr
@pytest.fixture
def setup_data():
    measure = "random"
    N = 10
    rank = 1
    dtype = torch.float
    return measure, N, rank, dtype


def test_nplr_shapes_and_types(setup_data):
    measure, N, rank, dtype = setup_data
    w, P, B, V = normal_plus_low_rank(measure, N, rank, dtype)

    # Check if the shapes are correct
    assert w.shape == (N // 2,)
    assert P.shape == (rank, N // 2)
    assert B.shape == (N // 2,)
    assert V.shape == (N, N // 2)

    # Check if the types are correct
    assert w.dtype == (torch.cfloat if dtype == torch.float else torch.cdouble)
    assert P.dtype == w.dtype
    assert B.dtype == w.dtype
    assert V.dtype == w.dtype


def test_nplr_values(setup_data):
    torch.manual_seed(1)
    measure, N, rank, dtype = setup_data
    w, P, B, V = normal_plus_low_rank(measure, N, rank, dtype)

    # Check if the imaginary parts of w are within the range [-2, 2]
    assert torch.all(
        (w.imag >= -2) & (w.imag <= 2)
    ), "Imaginary part of w should be within [-3, 3]"
    assert torch.all(torch.abs(P) <= 5), "Elements of P should be within [-5, 5]"


# Test to ensure that the eigenvalues 'w' are complex numbers
def test_nplr_w_complex(setup_data):
    _, N, _, dtype = setup_data
    w, _, _, _ = normal_plus_low_rank("random", N, dtype=dtype)
    assert all(
        torch.is_complex(val) for val in w
    ), "Eigenvalues w should be complex numbers"


# Test to ensure that the matrix 'V' is unitary
def test_nplr_v_unitary(setup_data):
    _, N, _, dtype = setup_data
    _, _, _, V = normal_plus_low_rank("random", N, dtype=dtype)
    V_inv = V.conj().transpose(-1, -2)
    identity = torch.eye(
        V.shape[-1], dtype=dtype
    )  # Ensure the identity matrix matches the dimensions of V
    assert torch.allclose((V_inv @ V).real, identity), "Matrix V should be unitary"


# Test to ensure that 'P' has the correct rank
def test_nplr_p_rank(setup_data):
    _, N, rank, dtype = setup_data
    _, P, _, _ = normal_plus_low_rank("random", N, correction_rank=rank, dtype=dtype)
    assert P.shape[0] == rank, f"Matrix P should have rank {rank}"


# Test to check if 'nplr' function handles invalid 'measure' input
def test_nplr_invalid_measure():
    with pytest.raises(NotImplementedError):
        normal_plus_low_rank("invalid_measure", 10)


def test_nplr_b_transformed(setup_data):
    measure, N, rank, dtype = setup_data
    _, _, _, V = normal_plus_low_rank(measure, N, correction_rank=rank, dtype=dtype)
    _, B = TransitionMatrix(measure, N)
    B = torch.as_tensor(B, dtype=V.dtype)[:, 0]
    V_inv = V.conj().transpose(-1, -2)  # Assuming V_inv is 5 x 10
    B_transformed = torch.einsum(
        "ij,j->i", V_inv, B.to(V)
    )  # Equivalent to CONTRACT("ij, j -> i", V_inv, B)
    assert torch.all(
        B_transformed.imag.abs() <= 1e-6
    ), "Transformed B should have negligible imaginary part"
