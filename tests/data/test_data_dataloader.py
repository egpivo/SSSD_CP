import pytest
from torch.utils.data import DataLoader

from sssd.data.dataloader import ArDataLoader


@pytest.fixture
def ar_dataloader():
    coefficients = [[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]]
    n_sample = 100
    std_list = [1, 0.8]
    season_periods = [12, 6]
    batch_size = 10
    device = "cpu"
    num_workers = 1
    training_rate = 0.8
    seed = 42
    return ArDataLoader(
        coefficients,
        n_sample,
        std_list,
        season_periods,
        batch_size,
        device,
        num_workers,
        training_rate,
        seed,
    )


def test_train_dataloader(ar_dataloader):
    train_loader = ar_dataloader.train_dataloader
    assert isinstance(train_loader, DataLoader)
    assert len(train_loader.dataset) == int(0.8 * 100)
    batch = next(iter(train_loader))
    assert batch.shape[0] == ar_dataloader.batch_size


def test_test_dataloader(ar_dataloader):
    test_loader = ar_dataloader.test_dataloader
    assert isinstance(test_loader, DataLoader)
    assert len(test_loader.dataset) == 100 - int(0.8 * 100)
    batch = next(iter(test_loader))
    assert batch.shape[0] <= ar_dataloader.batch_size
