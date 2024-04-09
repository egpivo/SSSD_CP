import numpy as np
from statsmodels.tools.validation import array_like
from statsmodels.tsa.arima_process import ArmaProcess


class ArDataGenerator:
    """
    Examples
    --------
    >>> from sssd.data.ar_generator import ArDataGenerator
    >>> ArDataGenerator([0.1, 0.2, 0.3], 3, seed=1).generate()
    array([1.62434536, -0.44932188, -0.24823487])
    """

    def __init__(
        self, coefficients: array_like, n_sample: int, seed: int = None
    ) -> None:
        ar_parameters = [1] + [-1 * coefficient for coefficient in coefficients]
        self.ar_process = ArmaProcess(ar_parameters, [1, 0])
        self.n_sample = n_sample
        self.seed = seed

    def generate(self) -> np.array:
        if self.seed:
            np.random.seed(self.seed)
        return self.ar_process.generate_sample(nsample=self.n_sample)
