from typing import Union

import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess


class SeasonalityGenerator:
    """
    Generates seasonal components for time series data.

    Args:
        series_length (int): Number of samples in the time series.
        season_period (int): Period of the seasonality component.
        seed (int, optional): Seed for random number generation. Defaults to None.

    Examples
    --------
    >>> generator = SeasonalityGenerator(series_length=12, season_period=3, seed=42)
    >>> generator.generate_cosine_seasonality()
    array([ 1. , -0.5, -0.5,  1. , -0.5, -0.5,  1. , -0.5, -0.5,  1. , -0.5,
       -0.5])
    """

    def __init__(
        self, series_length: int, season_period: int, seed: int = None
    ) -> None:
        self.series_length = series_length
        self.season_period = season_period
        self.seed = seed

    def generate_sine_seasonality(self) -> np.array:
        if self.seed:
            np.random.seed(self.seed)
        return np.sin(2 * np.pi * np.arange(self.series_length) / self.season_period)

    def generate_cosine_seasonality(self) -> np.array:
        if self.seed:
            np.random.seed(self.seed)
        return np.cos(2 * np.pi * np.arange(self.series_length) / self.season_period)


class ArDataGenerator:
    """
    Generates autoregressive (AR) time series data.

    Args:
        coefficients (array_like): Coefficients of the AR process, where |coefficient| < 1.
        series_length (int): Number of samples to generate.
        std (float, optional): Standard deviation of the generated samples. Must be greater than zero. Defaults to 1.
        seed (int, optional): Seed for random number generation. Defaults to None.
        season_period (int, optional): Period of the seasonality component. If provided, seasonality is added to the generated data. Defaults to None.
        seasonality_method (str, optional): Method to generate seasonality ("sine" or "cosine"). Defaults to "sine".
        detrend (bool, optional): Whether to remove the mean trend from the generated data. Defaults to False.

    Examples
    --------
    >>> from sssd.data.generator import ArDataGenerator
    >>> ArDataGenerator([0.1, 0.2, 0.3], series_length=3, std=1, seed=1, season_period=2).generate()
    array([1.62434536, -0.44932188, -0.2482348])
    """

    def __init__(
        self,
        coefficients: Union[list, np.ndarray],
        series_length: int,
        std: float = 1,
        seed: int = None,
        season_period: int = None,
        seasonality_method: str = "sine",
        detrend: bool = False,
    ) -> None:
        ar_parameters = [1] + [-1 * coefficient for coefficient in coefficients]
        self.ar_process = ArmaProcess(ar_parameters, [1, 0])
        self.series_length = series_length
        self.std = std
        self.seed = seed
        self.season_period = season_period
        self.seasonality_method = seasonality_method
        self.detrend = detrend

        if std <= 0:
            raise ValueError(f"Please enter positive standard deviation, but got {std}")

        if season_period is not None and season_period <= 0:
            raise ValueError(
                f"Please enter positive season period, but got {season_period}"
            )

    def _generate_seasonality(self) -> np.ndarray:
        generator = SeasonalityGenerator(
            self.series_length, self.season_period, self.seed
        )
        if self.seasonality_method == "sine":
            return generator.generate_sine_seasonality()
        elif self.seasonality_method == "cosine":
            return generator.generate_cosine_seasonality()

    def generate(self) -> np.ndarray:
        if self.seed:
            np.random.seed(self.seed)

        ar_process = self.ar_process.generate_sample(
            nsample=self.series_length, scale=self.std
        )

        if self.season_period:
            seasonality = self._generate_seasonality()
            ar_process += seasonality

        if self.detrend:
            ar_process -= ar_process.mean(axis=0)

        return ar_process
