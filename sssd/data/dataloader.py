from typing import List

import torch
from torch.utils.data import DataLoader, random_split

from sssd.data.dataset import ArDataset


class ArDataLoader:
    def __init__(
        self,
        coefficients: List[float],
        num_series: int,
        series_length: int,
        std: float,
        intercept: float,
        season: int,
        batch_size: int,
        device: torch.device,
        num_workers: int,
        training_rate: float,
        seeds: List[int] = None,
    ) -> None:
        self.dataset = ArDataset(
            coefficients=coefficients,
            num_series=num_series,
            series_length=series_length,
            std=std,
            season_period=season,
            intercept=intercept,
            seeds=seeds,
        )
        self.batch_size = batch_size
        self.device = device
        self.num_workers = num_workers
        self.generator = torch.Generator()
        self.generator.manual_seed(seeds[0])
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
