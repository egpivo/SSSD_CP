import numpy as np
import pytest
import torch

from sssd.core.layers.s4.hippo.utils import (
    TransitionMatrix,
    cauchy_cpu,
    compute_fft_transform,
    embed_c2r,
    generate_low_rank_matrix,
    get_dense_contraction,
    get_diagonal_contraction,
    get_input_contraction,
    get_output_contraction,
    hurwitz_transformation,
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
    w, P, B = normal_plus_low_rank(measure, N, rank, dtype)

    # Check if the shapes are correct
    assert w.shape == (N // 2,)
    assert P.shape == (rank, N // 2)
    assert B.shape == (N // 2,)

    # Check if the types are correct
    assert w.dtype == (torch.cfloat if dtype == torch.float else torch.cdouble)
    assert P.dtype == w.dtype
    assert B.dtype == w.dtype


def test_nplr_values(setup_data):
    torch.manual_seed(1)
    measure, N, rank, dtype = setup_data
    w, P, B = normal_plus_low_rank(measure, N, rank, dtype)

    # Check if the imaginary parts of w are within the range [-2, 2]
    assert torch.all(
        (w.imag >= -2) & (w.imag <= 2)
    ), "Imaginary part of w should be within [-3, 3]"
    assert torch.all(torch.abs(P) <= 5), "Elements of P should be within [-5, 5]"


# Test to ensure that the eigenvalues 'w' are complex numbers
def test_nplr_w_complex(setup_data):
    _, N, _, dtype = setup_data
    w, _, _ = normal_plus_low_rank("random", N, dtype=dtype)
    assert all(
        torch.is_complex(val) for val in w
    ), "Eigenvalues w should be complex numbers"


# Test to ensure that 'P' has the correct rank
def test_nplr_p_rank(setup_data):
    _, N, rank, dtype = setup_data
    _, P, _ = normal_plus_low_rank("random", N, correction_rank=rank, dtype=dtype)
    assert P.shape[0] == rank, f"Matrix P should have rank {rank}"


# Test to check if 'nplr' function handles invalid 'measure' input
def test_nplr_invalid_measure():
    with pytest.raises(NotImplementedError):
        normal_plus_low_rank("invalid_measure", 10)


def test_cauchy_cpu():
    # Define input shapes
    shape = (3, 4)  # Example shape
    N = 4
    L = 3

    # Generate random tensors for inputs
    v = torch.randn(*shape, N)
    z = torch.randn(*shape, L)
    w = torch.randn(*shape, N)

    # Call the function
    result = cauchy_cpu(v, z, w)

    # Check if the output shape matches expectation
    assert result.shape == (*shape, L)

    # Check if the output contains valid values (e.g., not NaN or Inf)
    assert torch.isfinite(result).all()

    # Sample tensors for testing
    v = torch.tensor([1.0, 2.0, 3.0])
    z = torch.tensor([1.0, 2.0])
    w = torch.tensor([0.5, 1.5, 2.5])

    # Manually compute the Cauchy matrix and sum across the appropriate dimension
    cauchy_matrix_manual = v.unsqueeze(-1) / (z.unsqueeze(-2) - w.unsqueeze(-1))
    expected_result_manual = torch.sum(cauchy_matrix_manual, dim=-2)

    # Use the cauchy_cpu function to compute the result
    result = cauchy_cpu(v, z, w)

    # Check if the result matches the expected result
    assert torch.allclose(
        result, expected_result_manual
    ), "The result does not match the expected values"


def test_compute_fft_transform():
    # Test parameters
    sequence_length = 4
    dtype = torch.cfloat
    device = torch.device("cpu")

    # Expected results
    expected_omega = torch.tensor(
        [np.exp(-2j * 0), np.exp(-2j * np.pi / 8), np.exp(-2j * np.pi / 8) ** 2],
        dtype=dtype,
    )
    expected_z = 2 * (1 - expected_omega) / (1 + expected_omega)

    # Compute the FFT transform
    omega, z = compute_fft_transform(sequence_length, dtype, device)

    # Verify the results
    assert torch.allclose(
        omega, expected_omega, atol=1e-5
    ), "Omega values do not match expected results."
    assert torch.allclose(
        z, expected_z, atol=1e-5
    ), "Z values do not match expected results."


def test_hurwitz_transformation():
    # Test data
    log_w_real = torch.tensor([0.0, -1.0, -2.0])  # Logarithm of the real parts
    w_imag = torch.tensor([1.0, 2.0, 3.0])  # Imaginary parts

    # Expected output
    expected_w = torch.tensor([-1.0 + 1.0j, -0.3679 + 2.0j, -0.1353 + 3.0j])

    # Perform the transformation
    result_w = hurwitz_transformation(log_w_real, w_imag)

    # Assert that the result is close to the expected output
    assert torch.allclose(
        result_w, expected_w, atol=1e-4
    ), "The hurwitz_transformation function did not produce the expected output."


def test_get_dense_contraction():
    batch_shape = (2, 3)
    H = 4
    N = 5
    result = get_dense_contraction(batch_shape, H, N)
    assert result.contraction == "h m n, ... h n -> ... h m"
    assert result.contraction_list[0][2] == "cdhn,hmn->cdhm"


def test_get_input_contraction():
    batch_shape = (2, 3)
    H = 4
    N = 5
    result = get_input_contraction(batch_shape, H, N)
    assert result.contraction == "h n, ... h -> ... h n"
    assert result.contraction_list[0][2] == "bch,hn->bchn"


def test_get_output_contraction():
    batch_shape = (2, 3)
    H = 4
    N = 5
    C = 6
    result = get_output_contraction(batch_shape, H, N, C)
    assert result.contraction == "c h n, ... h n -> ... c h"
    assert result.contraction_list[0][2] == "dehn,chn->dech"


def test_get_diagonal_contraction():
    batch_shape = (2, 3)
    H = 4
    N = 5
    result = get_diagonal_contraction(batch_shape, H, N)
    assert result.contraction == "h n, ... h n -> ... h n"
    assert result.contraction_list[0][2] == "cdhn,hn->cdhn"
