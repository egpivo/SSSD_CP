import numpy as np
import pytest
import torch

from sssd.data.dataset import ArDataset


@pytest.fixture
def ar_dataset():
    coefficients_list = [0.1, 0.2, 0.3]
    num_series = 1024
    series_length = 120
    std = 0.001
    season_period = 3
    seeds = list(range(num_series))

    return ArDataset(
        coefficients_list, num_series, series_length, std, season_period, seeds=seeds
    )


def test_ar_dataset_length(ar_dataset):
    assert len(ar_dataset) == 1024


def test_ar_dataset_sample(ar_dataset):
    sample = ar_dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == torch.Size([120, 1])


def test_ar_dataset_generated_data_shape(ar_dataset):
    generated_data = ar_dataset._generate_data()
    assert isinstance(generated_data, np.ndarray)
    assert generated_data.shape == (1024, 120, 1)
