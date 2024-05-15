import numpy as np
import pytest

from sssd.core.layers.s4.hippo.trainsition_matrix import TransitionMatrix


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
