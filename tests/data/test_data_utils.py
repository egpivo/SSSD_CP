import numpy as np
import pandas as pd
import pytest
import torch

from sssd.data.utils import (
    load_and_split_training_data,
    load_testing_data,
    merge_all_time,
)


@pytest.fixture
def sample_data():
    dates = pd.date_range(start="2022-01-01", end="2022-01-03", freq="H")
    zones = ["A", "B"]
    loads = np.random.randint(1, 100, size=len(dates) * len(zones))

    df = pd.DataFrame(
        {
            "Date": np.repeat(dates, len(zones)),
            "Zone": np.tile(zones, len(dates)),
            "Load": loads,
        }
    )

    return df


# Test case to check if all zones are included
def test_merge_all_time_all_zones(sample_data):
    result = merge_all_time(sample_data)
    expected_zones = ["A", "B"]

    assert set(result["Zone"].unique()) == set(expected_zones)


# Test case to check if the output DataFrame has the correct shape
def test_merge_all_time_shape(sample_data):
    result = merge_all_time(sample_data)
    assert (
        result.shape[0]
        == len(pd.date_range(start="2022-01-01", end="2022-01-03", freq="H")) * 2
    )


# Test case to check if the output DataFrame contains the same columns as input
def test_merge_all_time_columns(sample_data):
    result = merge_all_time(sample_data)
    expected_columns = ["Date", "Zone", "Load"]

    assert all(col in result.columns for col in expected_columns)


@pytest.fixture
def test_data_file(tmp_path):
    # Create a sample testing data file
    test_data = np.random.rand(100, 10)  # 100 samples, 10 features
    test_data_path = tmp_path / "test_data.npy"
    np.save(test_data_path, test_data)
    return test_data_path


def test_load_testing_data(test_data_file):
    # Load testing data
    num_samples = 20  # Number of samples per batch
    testing_data = load_testing_data(str(test_data_file), num_samples)

    # Assert the shape of the loaded data tensor
    expected_shape = (5, 20, 10)  # 5 batches, each with 20 samples and 10 features
    assert testing_data.shape == expected_shape

    # Assert the type of the loaded data tensor
    assert isinstance(testing_data, torch.Tensor)

    # Assert the device of the loaded data tensor
    if torch.cuda.is_available():
        assert testing_data.device.type == "cuda"
    else:
        assert testing_data.device.type == "cpu"


def test_load_and_split_training_data():
    # Example training data with 100 samples and 10 features
    training_data_load = np.random.rand(100, 10)
    batch_num = 5
    # Use a smaller batch size to ensure it does not exceed the available training data
    batch_size = 25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Assert that the function raises a ValueError when batch size is larger than available data
    with pytest.raises(ValueError):
        load_and_split_training_data(training_data_load, batch_num, batch_size, device)
