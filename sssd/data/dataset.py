from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from sssd.data.generator import ArDataGenerator


class ArDataset(Dataset):
    """
    Dataset class for generating autoregressive (AR) time series data.

    Args:
        coefficients (List[Union[List[float], np.ndarray]]): List of coefficients for each AR process. Each coefficient array must have shape (p,), where p is the order of the AR process.
        n_sample (int): Number of samples to generate for each AR process.
        std_list (List[float], optional): List of standard deviations for the generated samples. If not provided, defaults to 1.
        season_periods (List[int], optional): List of periods for the seasonality component for each AR process. If not provided, no seasonality is added.
        seed (int, optional): Seed for random number generation. Defaults to None.
        detrend (bool, optional): Whether to detrend the generated data. Defaults to False.

    Examples
    --------
    >>> from sssd.data.ar_dataset import ArDataset
    >>> coefficients = [[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]]
    >>> n_sample = 120
    >>> std_list = [1, 0.8]
    >>> season_periods = [12, 6]
    >>> seed = 123
    >>> dataset = ArDataset(coefficients, n_sample, std_list, season_periods, seed=seed, detrend=False)
    >>> next(iter(dataset))
    tensor([[-1.0856],
        [-0.8685]])
    >>> data = dataset._generate_data()
    >>> print("Shape of generated data:", data.shape)
    Shape of generated data: (120, 2, 1)
    """

    def __init__(
        self,
        coefficients: List[Union[List[float], np.ndarray]],
        n_sample: int,
        std_list: List[float] = None,
        season_periods: List[int] = None,
        seed: int = None,
        detrend: bool = False,
    ) -> None:
        self.n_processes = len(coefficients)
        self.coefficients = coefficients
        self.n_sample = n_sample
        self.std_list = std_list or [1] * self.n_processes
        self.season_periods = season_periods or [None] * self.n_processes
        self.seed = seed
        self.detrend = detrend

        self.data = self._generate_data()

    def _generate_data(self) -> np.ndarray:
        """
        Generate AR time series data.

        Returns:
            np.ndarray: Generated time series data with shape (n_sample, n_processes, 1).
        """

        data = []
        for coefficients, std, season_period in zip(
            self.coefficients, self.std_list, self.season_periods
        ):
            generator = ArDataGenerator(
                coefficients=coefficients,
                n_sample=self.n_sample,
                std=std,
                season_period=season_period,
                seed=self.seed,
                detrend=self.detrend,
            )
            data.append(generator.generate()[:, np.newaxis])
        return np.stack(data, axis=1).reshape(self.n_sample, self.n_processes, 1)

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Sample from the dataset as a PyTorch tensor.
        """
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)
