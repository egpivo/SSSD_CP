import numpy as np
from statsmodels.tools.validation import array_like
from statsmodels.tsa.arima_process import ArmaProcess


class ArDataGenerator:
    """
    Generates autoregressive (AR) time series data.

    Args:
        coefficients (array_like): Coefficients of the AR process, where |coefficient| < 1.
        n_sample (int): Number of samples to generate.
        std (float, optional): Standard deviation of the generated samples. Must be greater than zero. Defaults to 1.
        seed (int, optional): Seed for random number generation. Defaults to None.
        season_period (int, optional): Period of the seasonality component. If provided, seasonality is added to the generated data. Defaults to None.

    Examples
    --------
    >>> from sssd.data.ar_generator import ArDataGenerator
    >>> ArDataGenerator([0.1, 0.2, 0.3], n_sample=3, std=1, seed=1, season_period=12).generate()
    array([1.62434536, -0.44932188, -0.2482348])
    """

    def __init__(
        self,
        coefficients: array_like,
        n_sample: int,
        std: float = 1,
        seed: int = None,
        season_period: int = None,
    ) -> None:
        """
        Initialize the AR data generator with given coefficients, number of samples, standard deviation, and seasonality period.
        """
        self._validate_inputs(n_sample, coefficients, std, season_period)

        ar_parameters = [1] + [-1 * coefficient for coefficient in coefficients]
        self.ar_process = ArmaProcess(ar_parameters, [1, 0])
        self.n_sample = n_sample
        self.std = std
        self.seed = seed
        self.season_period = season_period

    @staticmethod
    def _validate_inputs(
        n_sample: int, coefficients: array_like, std: float, season_period: int
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

        if season_period is not None and season_period <= 0:
            raise ValueError("Seasonality period must be a positive integer.")

    def generate(self) -> np.array:
        """
        Generate AR time series data with optional seasonality.

        Returns:
            np.array: Generated time series data.
        """
        if self.seed:
            np.random.seed(self.seed)

        ar_process = self.ar_process.generate_sample(
            nsample=self.n_sample, scale=self.std
        )

        if self.season_period:
            seasonality = np.sin(
                2 * np.pi * np.arange(self.n_sample) / self.season_period
            )
            ar_process += seasonality

        return ar_process
