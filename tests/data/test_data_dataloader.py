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
    seed = 42

    return ArDataLoader(
        coefficients,
        num_series,
        series_length,
        std,
        season,
        batch_size,
        device,
        num_workers,
        training_rate,
        seed,
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
