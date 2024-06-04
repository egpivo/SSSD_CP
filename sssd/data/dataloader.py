from typing import List

import torch
from torch.utils.data import DataLoader, random_split

from sssd.data.dataset import ArDataset


class ArDataLoader:
    """

    Examples
    --------
    >>> num_series = 1024
    >>> coefficients = [0.8]
    >>> series_length = 192
    >>> std = 0.1
    >>> season = 12
    >>> batch_size = 1024
    >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    >>> num_workers = 4
    >>> training_rate = 0.8
    >>> seed = 42
    >>> data_loader = ArDataLoader(
    >>> ...    coefficients,
    >>> ...    num_series,
    >>> ...    series_length,
    >>> ...    std,
    >>> ...    season,
    >>> ...    batch_size,
    >>> ...    device,
    >>> ...    num_workers,
    >>> ...    training_rate,
    >>> ...    seed,
    >>> ...)
    >>> train_loader = data_loader.train_dataloader
    >>> test_loader = data_loader.test_dataloader
    >>> print(len(data_loader.dataset))
    1024
    >>> print(data_loader.train_size)
    819
    >>> next(iter(train_loader)).shape
    torch.Size([819, 192, 1])
    """

    def __init__(
        self,
        coefficients: List[float],
        num_series: int,
        series_length: int,
        std: float,
        season: int,
        batch_size: int,
        device: torch.device,
        num_workers: int,
        training_rate: float,
        seed: int = None,
    ) -> None:
        self.dataset = ArDataset(
            coefficients, num_series, series_length, std, season, seed
        )
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        self.train_size = int(training_rate * num_series)
        self.test_size = num_series - self.train_size

    @property
    def train_dataloader(self) -> DataLoader:
        train_dataset, _ = random_split(
            dataset=self.dataset,
            lengths=[self.train_size, self.test_size],
            generator=self.generator,
        )
        return DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    @property
    def test_dataloader(self) -> DataLoader:
        _, test_dataset = random_split(
            dataset=self.dataset,
            lengths=[self.train_size, self.test_size],
            generator=self.generator,
        )
        return DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
