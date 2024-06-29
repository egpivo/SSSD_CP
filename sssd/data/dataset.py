import random
from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from sssd.data.generator import ArDataGenerator


class ArDataset(Dataset):
    """Dataset class for generating autoregressive (AR) time series data.

    This class generates multiple time series based on provided autoregressive (AR)
    coefficients and other parameters.

    Args:
        coefficients (List[Union[List[float], np.ndarray]]): List of coefficients
            for each AR process. Each coefficient array must have shape (p,), where p
            is the order of the AR process.
        num_series (int): Number of time series to generate.
        series_length (int): Length of each generated time series.
        std (float, optional): Standard deviation of the generated noise. Defaults to 1.
        season_period (int, optional): Periodicity for the seasonal component.
            If not provided, no seasonality is added.
        seeds (List[int], optional): List of seeds for random number generation. Defaults to None.
        intercept (int, optional): Intercept of an AR process
            Defaults to False.

    Examples:
        >>> coefficients = [0.1, 0.2, 0.3]
        >>> num_series = 1024
        >>> series_length = 120
        >>> std = 1
        >>> season_period = 12
        >>> dataset = ArDataset(coefficients, num_series, series_length, std, season_period)
        >>> first_item = next(iter(dataset))
        >>> print("Shape of the first item:", first_item.shape)  # torch.Size([120, 1])
        >>> data = dataset._generate_data()
        >>> print("Shape of generated data:", data.shape)  # (1024, 120, 1)
    """

    def __init__(
        self,
        coefficients: Union[List[float], np.ndarray],
        num_series: int,
        series_length: int,
        std: float = 1.0,
        season_period: int = None,
        seeds: List[int] = None,
        intercept: float = 0,
    ) -> None:
        self.num_series = num_series
        self.coefficients = coefficients
        self.series_length = series_length
        self.std = std
        self.season_period = season_period
        self.seeds = seeds or [
            random.randint(0, 2**32 - 1) for _ in range(num_series)
        ]
        self.intercept = intercept
        self.data = self._generate_data()

    def _generate_data(self) -> np.ndarray:
        data = [
            ArDataGenerator(
                coefficients=self.coefficients,
                series_length=self.series_length,
                std=self.std,
                season_period=self.season_period,
                seed=self.seeds[i],
                intercept=self.intercept,
            ).generate()
            for i in range(self.num_series)
        ]
        return np.stack(data, axis=0).reshape(self.num_series, self.series_length, 1)

    def __len__(self) -> int:
        return self.num_series

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx, :, :], dtype=torch.float32)
