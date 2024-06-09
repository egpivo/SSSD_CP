import pytest
import torch
from torch.utils.data import DataLoader

from sssd.data.dataloader import ArDataLoader


@pytest.fixture
def ar_dataloader():
    num_series = 1024
    coefficients = [0.8]
    series_length = 192
    std = 0.1
    season = 12
    batch_size = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 4
    training_rate = 0.8
    seeds = list(range(num_series))
    intercept = 0

    return ArDataLoader(
        coefficients=coefficients,
        num_series=num_series,
        series_length=series_length,
        std=std,
        season=season,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
        training_rate=training_rate,
        intercept=intercept,
        seeds=seeds,
    )


def test_train_dataloader(ar_dataloader):
    train_loader = ar_dataloader.train_dataloader
    assert isinstance(train_loader, DataLoader)
    assert len(train_loader.dataset) == 819
    batch = next(iter(train_loader))
    assert batch.shape[0] == ar_dataloader.batch_size


def test_test_dataloader(ar_dataloader):
    test_loader = ar_dataloader.test_dataloader
    assert isinstance(test_loader, DataLoader)
    assert len(test_loader.dataset) == 205
    batch = next(iter(test_loader))
    assert batch.shape[0] <= ar_dataloader.batch_size
