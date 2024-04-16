from unittest.mock import patch

import numpy as np
import pytest

from sssd.inference.utils import read_missing_k_data, read_multiple_imputations


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
