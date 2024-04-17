from unittest.mock import patch

import numpy as np
import pytest

from sssd.inference.utils import (
    adjust_PI,
    compute_E_star,
    coverage_rate,
    predict_interval,
    read_missing_k_data,
    read_multiple_imputations,
)


@pytest.fixture
def mock_folder_files():
    return ["imputation0.npy", "imputation1.npy"]


@patch("os.path.exists")
@patch("os.listdir")
def test_read_multiple_imputations(mock_listdir, mock_exists, mock_folder_files):
    mock_exists.return_value = True
    mock_listdir.return_value = mock_folder_files
    with patch("sssd.inference.utils.read_missing_k_data") as mock_read_missing_k_data:
        mock_read_missing_k_data.return_value = np.zeros((2, 3, 24))
        result = read_multiple_imputations("folder_path", 24)
        assert result is not None
        assert result.shape == (1, 2, 3, 24)


@patch("numpy.load")
def test_read_missing_k_data(mock_load):
    mock_load.return_value = np.zeros((10, 5, 100))
    result = read_missing_k_data("folder_path", "imputation0.npy", 24)
    assert result is not None
    assert result.shape == (10, 5, 24)


@pytest.fixture
def sample_data():
    L = np.zeros((2209, 1, 24))
    U = np.ones((2209, 1, 24))
    true = np.random.rand(2209, 1, 24)
    return L, U, true


def test_predict_interval(sample_data):
    L, U, true = sample_data
    alpha = 0.05
    lower_bound, upper_bound = predict_interval(np.concatenate([L, U]), alpha)
    assert lower_bound.shape == (1, 24)
    assert upper_bound.shape == (1, 24)
    assert np.all(upper_bound >= lower_bound)


def test_compute_E_star(sample_data):
    L, U, true = sample_data
    E_star = compute_E_star(L, U, true)

    # Ensure the shape of E_star is as expected
    assert E_star.shape == (1, 24)
    assert np.all(
        (E_star >= -1) & (E_star <= 1)
    ), "Some elements in E_star are out of range"


def test_adjust_PI(sample_data):
    L, U, true = sample_data
    E_star = np.random.rand(1, 24)
    adjusted_L, adjusted_U = adjust_PI(L, U, E_star)
    assert adjusted_L.shape == (2209, 1, 24)
    assert adjusted_U.shape == (2209, 1, 24)
    assert np.all(adjusted_U >= adjusted_L)


def test_coverage_rate(sample_data):
    L, U, true = sample_data
    cov_rate = coverage_rate(L, U, true)
    assert cov_rate.shape == (1, 24)
    assert np.all(cov_rate >= 0) and np.all(cov_rate <= 1)
