from typing import List, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from sssd.data.ar_generator import ArDataGenerator


class ArDataset(Dataset):
    """
    Dataset class for generating autoregressive (AR) time series data.

    Args:
        coefficients_list (List[Union[List[float], np.ndarray]]): List of coefficients for each AR process. Each coefficient array must have shape (p,), where p is the order of the AR process.
        n_sample (int): Number of samples to generate for each AR process.
        std_list (List[float], optional): List of standard deviations for the generated samples. If not provided, defaults to 1.
        season_period_list (List[int], optional): List of periods for the seasonality component for each AR process. If not provided, no seasonality is added.
        seed (int, optional): Seed for random number generation. Defaults to None.

    Examples
    --------
    >>> from sssd.data.ar_dataset import ArDataset
    >>> coefficients_list = [[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]]
    >>> n_sample = 120
    >>> std_list = [1, 0.8]
    >>> season_period_list = [12, 6]
    >>> seed = 123
    >>> dataset = ArDataset(coefficients_list, n_sample, std_list, season_period_list, seed=seed)
    >>> next(iter(dataset))
    tensor([-1.0856, -0.8685])
    """

    def __init__(
        self,
        coefficients_list: List[Union[List[float], np.ndarray]],
        n_sample: int,
        std_list: List[float] = None,
        season_period_list: List[int] = None,
        seed: int = None,
    ) -> None:
        self.num_processes = len(coefficients_list)
        self.coefficients_list = coefficients_list
        self.n_sample = n_sample
        self.std_list = std_list or [1] * self.num_processes
        self.season_period_list = season_period_list or [None] * self.num_processes
        self.seed = seed

        self.data = self._generate_data()

    def _generate_data(self) -> np.ndarray:
        """
        Generate AR time series data.

        Returns:
            np.ndarray: Generated time series data with shape (n_sample, num_processes, 1).
        """

        data = []
        for coefficients, std, season_period in zip(
            self.coefficients_list, self.std_list, self.season_period_list
        ):
            ar_gen = ArDataGenerator(
                coefficients=coefficients,
                n_sample=self.n_sample,
                std=std,
                season_period=season_period,
                seed=self.seed,
            )
            data.append(ar_gen.generate()[:, np.newaxis])
        return np.concatenate(data, axis=1)

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
