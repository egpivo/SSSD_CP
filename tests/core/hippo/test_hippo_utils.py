import numpy as np
import pytest
import torch

from sssd.core.s4.hippo.utils import (
    TransitionMatrix,
    embed_c2r,
    generate_rank_correction_matrix,
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
    P = generate_rank_correction_matrix(measure, N, rank)
    assert P.shape == expected_shape


def test_rank_correction_invalid_measure():
    with pytest.raises(NotImplementedError):
        generate_rank_correction_matrix("invalid_measure", 5, 1)


def test_rank_correction_dtype():
    P = generate_rank_correction_matrix("legs", 5, 1, dtype=torch.float64)
    assert P.dtype == torch.float64
