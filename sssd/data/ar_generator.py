import numpy as np
from statsmodels.tools.validation import array_like
from statsmodels.tsa.arima_process import ArmaProcess


class ArDataGenerator:
    """
    Generates autoregressive (AR) time series data.

    Args:
        coefficients (array_like): Coefficients of the AR process, where |coefficient| < 1
        n_sample (int): Number of samples to generate.
        seed (int, optional): Seed for random number generation. Defaults to None.

    Examples
    --------
    >>> from sssd.data.ar_generator import ArDataGenerator
    >>> ArDataGenerator([0.1, 0.2, 0.3], 3, seed=1).generate()
    array([1.62434536, -0.44932188, -0.24823487])
    """

    def __init__(
        self, coefficients: array_like, n_sample: int, seed: int = None
    ) -> None:
        """
        Initialize the AR data generator with given coefficients and number of samples.
        """
        # Check if absolute values of coefficients are less than one
        if any(abs(coefficient) >= 1 for coefficient in coefficients):
            raise ValueError("Absolute values of coefficients must be less than one.")

        ar_parameters = [1] + [-1 * coefficient for coefficient in coefficients]
        self.ar_process = ArmaProcess(ar_parameters, [1, 0])
        self.n_sample = n_sample
        self.seed = seed

    def generate(self) -> np.array:
        """
        Generate AR time series data.

        Returns:
            np.array: Generated time series data.
        """
        if self.seed:
            np.random.seed(self.seed)
        return self.ar_process.generate_sample(nsample=self.n_sample)
