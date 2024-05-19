import numpy as np
import pytest
import torch

from sssd.data.dataset import ArDataset


@pytest.fixture
def ar_dataset():
    coefficients_list = [[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]]
    n_sample = 120
    std_list = [1, 0.8]
    season_periods = [12, 6]
    seed = 123

    return ArDataset(coefficients_list, n_sample, std_list, season_periods, seed=seed)


def test_ar_dataset_length(ar_dataset):
    assert len(ar_dataset) == 120


def test_ar_dataset_sample(ar_dataset):
    sample = ar_dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == torch.Size([2, 1])


def test_ar_dataset_generated_data_shape(ar_dataset):
    generated_data = ar_dataset._generate_data()
    assert isinstance(generated_data, np.ndarray)
    assert generated_data.shape == (120, 2, 1)
