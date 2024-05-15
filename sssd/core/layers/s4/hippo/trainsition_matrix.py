from typing import Dict, Tuple, Union

import numpy as np
from scipy import special as ss

from sssd.core.layers.s4.hippo.utils import embed_c2r

Matrix = np.ndarray
MeasureArgs = Dict[str, Union[float, int]]


class TransitionMatrix:
    def __new__(
        cls, measure: str, N: int, **measure_args: MeasureArgs
    ) -> Tuple[Matrix, Matrix]:
        """
        Generate transition matrices for different measures.

        Args:
            measure (str): Type of measure.
            N (int): Size of the transition matrices.
            **measure_args: Additional arguments specific to each measure.

        Returns:
            Tuple[Matrix, Matrix]: Transition matrices A and B.

        Raises:
            NotImplementedError: If the specified measure is not implemented.
            ValueError: If invalid measure arguments are provided.
        """
        if measure == "lagt":
            A, B = cls._translated_laguerre(N, **measure_args)
        elif measure == "glagt":
            A, B = cls._generalized_laguerre(N, **measure_args)
        elif measure == "legt":
            A, B = cls._translated_legendre(N)
        elif measure == "legs":
            A, B = cls._scaled_legendre(N)
        elif measure == "fourier":
            A, B = cls._fourier(N)
        elif measure == "random":
            A, B = cls._random(N)
        elif measure == "diagonal":
            A, B = cls._diagonal(N)
        else:
            raise NotImplementedError(f"Measure '{measure}' is not implemented.")

        return A, B

    @staticmethod
    def _translated_laguerre(
        N: int, **measure_args: MeasureArgs
    ) -> Tuple[Matrix, Matrix]:
        """
        Generate transition matrices for Laguerre (translated) measure.
        """
        beta = measure_args.get("beta", 1.0)
        A = np.eye(N) / 2 - np.tril(np.ones((N, N)))
        B = beta * np.ones((N, 1))
        return A, B

    def _generalized_laguerre(
        N: int, **measure_args: MeasureArgs
    ) -> Tuple[Matrix, Matrix]:
        """
        Generate transition matrices for Generalized Laguerre measure.
        """
        alpha = measure_args.get("alpha", 0.0)
        beta = measure_args.get("beta", 0.01)
        A = -np.eye(N) * (1 + beta) / 2 - np.tril(np.ones((N, N)), -1)
        B = ss.binom(alpha + np.arange(N), np.arange(N))[:, None]
        L = np.exp(
            0.5 * (ss.gammaln(np.arange(N) + alpha + 1) - ss.gammaln(np.arange(N) + 1))
        )
        A = (1.0 / L[:, None]) * A * L[None, :]
        B = (
            (1.0 / L[:, None])
            * B
            * np.exp(-0.5 * ss.gammaln(1 - alpha))
            * beta ** ((1 - alpha) / 2)
        )
        return A, B

    @staticmethod
    def _translated_legendre(N: int) -> Tuple[Matrix, Matrix]:
        Q = np.arange(N, dtype=np.float64)
        R = (2 * Q + 1) ** 0.5
        j, i = np.meshgrid(Q, Q)
        A = R[:, None] * np.where(i < j, (-1.0) ** (i - j), 1) * R[None, :]
        B = R[:, None]
        A = -A
        return A, B

    @staticmethod
    def _scaled_legendre(N: int) -> Tuple[Matrix, Matrix]:
        q = np.arange(N, dtype=np.float64)
        col, row = np.meshgrid(q, q)
        r = 2 * q + 1
        M = -(np.where(row >= col, r, 0) - np.diag(q))
        T = np.sqrt(np.diag(2 * q + 1))
        A = T @ M @ np.linalg.inv(T)
        B = np.diag(T)[:, None]
        B = B.copy()
        return A, B

    @staticmethod
    def _fourier(N: int) -> Tuple[Matrix, Matrix]:
        freqs = np.arange(N // 2)
        d = np.stack([freqs, np.zeros(N // 2)], axis=-1).reshape(-1)[:-1]
        A = 2 * np.pi * (np.diag(d, 1) - np.diag(d, -1))
        A = A - embed_c2r(np.ones((N // 2, N // 2)))
        B = embed_c2r(np.ones((N // 2, 1)))[..., :1]
        return A, B

    @staticmethod
    def _random(N: int) -> Tuple[Matrix, Matrix]:
        A = np.random.randn(N, N) / N
        B = np.random.randn(N, 1)
        return A, B

    @staticmethod
    def _diagonal(N: int) -> Tuple[Matrix, Matrix]:
        A = -np.diag(np.exp(np.random.randn(N)))
        B = np.random.randn(N, 1)
        return A, B
