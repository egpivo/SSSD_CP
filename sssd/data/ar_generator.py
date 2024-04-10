from typing import Union

import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess


class SeasonalityGenerator:
    """
    Generates seasonal components for time series data.

    Args:
        n_samples (int): Number of samples in the time series.
        season_period (int): Period of the seasonality component.
        seed (int, optional): Seed for random number generation. Defaults to None.

    Examples
    --------
    >>> generator = SeasonalityGenerator(n_samples=12, season_period=3, seed=42)
    >>> seasonality = generator.generate_cosine_seasonality()
    array([ 1. , -0.5, -0.5,  1. , -0.5, -0.5,  1. , -0.5, -0.5,  1. , -0.5,
       -0.5])
    """

    def __init__(self, n_samples: int, season_period: int, seed: int = None) -> None:
        """
        Initialize the seasonality generator with the number of samples, seasonality period, and seed.
        """
        self.n_samples = n_samples
        self.season_period = season_period
        self.seed = seed

    def generate_sine_seasonality(self) -> np.array:
        """
        Generate seasonality using a sine function.

        Returns:
            np.array: Seasonality component.
        """
        if self.seed:
            np.random.seed(self.seed)
        return np.sin(2 * np.pi * np.arange(self.n_samples) / self.season_period)

    def generate_cosine_seasonality(self) -> np.array:
        """
        Generate seasonality using a cosine function.

        Returns:
            np.array: Seasonality component.
        """
        if self.seed:
            np.random.seed(self.seed)
        return np.cos(2 * np.pi * np.arange(self.n_samples) / self.season_period)


class ArDataGenerator:
    """
    Generates autoregressive (AR) time series data.

    Args:
        coefficients (array_like): Coefficients of the AR process, where |coefficient| < 1.
        n_sample (int): Number of samples to generate.
        std (float, optional): Standard deviation of the generated samples. Must be greater than zero. Defaults to 1.
        seed (int, optional): Seed for random number generation. Defaults to None.
        season_period (int, optional): Period of the seasonality component. If provided, seasonality is added to the generated data. Defaults to None.
        seasonality_method (str, optional): Method to generate seasonality ("sine" or "cosine"). Defaults to "sine".
        detrend (bool, optional): Whether to remove the mean trend from the generated data. Defaults to False.

    Examples
    --------
    >>> from sssd.data.ar_generator import ArDataGenerator
    >>> ArDataGenerator([0.1, 0.2, 0.3], n_sample=3, std=1, seed=1, season_period=2).generate()
    array([1.62434536, -0.44932188, -0.2482348])
    """

    def __init__(
        self,
        coefficients: Union[list, np.ndarray],
        n_sample: int,
        std: float = 1,
        seed: int = None,
        season_period: int = None,
        seasonality_method: str = "sine",
        detrend: bool = False,
    ) -> None:
        """
        Initialize the AR data generator with given coefficients, number of samples, standard deviation, seasonality period, and seed.
        """
        self._validate_inputs(n_sample, coefficients, std, season_period)

        ar_parameters = [1] + [-1 * coefficient for coefficient in coefficients]
        self.ar_process = ArmaProcess(ar_parameters, [1, 0])
        self.n_sample = n_sample
        self.std = std
        self.seed = seed
        self.season_period = season_period
        self.seasonality_method = seasonality_method
        self.detrend = detrend

    @staticmethod
    def _validate_inputs(
        n_sample: int,
        coefficients: Union[list, np.ndarray],
        std: float,
        season_period: int,
    ) -> None:
        """
        Validate input parameters.
        """
        # Check if absolute values of coefficients are less than one
        if any(abs(coefficient) >= 1 for coefficient in coefficients):
            raise ValueError("Absolute values of coefficients must be less than one.")

        # Check if std is greater than zero
        if std <= 0:
            raise ValueError("Standard deviation (std) must be greater than zero.")

        if season_period is not None and (
            season_period <= 0 or season_period >= n_sample
        ):
            raise ValueError(
                "Seasonality period must be a positive integer less than n_sample."
            )

    def _generate_seasonality(self) -> np.ndarray:
        generator = SeasonalityGenerator(self.n_sample, self.season_period, self.seed)
        if self.seasonality_method == "sine":
            return generator.generate_sine_seasonality()
        elif self.seasonality_method == "cosine":
            return generator.generate_cosine_seasonality()

    def generate(self) -> np.ndarray:
        """
        Generate AR time series data with optional seasonality.

        Returns:
            np.ndarray: Generated time series data.
        """
        if self.seed:
            np.random.seed(self.seed)

        ar_process = self.ar_process.generate_sample(
            nsample=self.n_sample, scale=self.std
        )

        if self.season_period:
            seasonality = self._generate_seasonality()
            ar_process += seasonality

        if self.detrend:
            ar_process -= ar_process.mean(axis=0)

        return ar_process
