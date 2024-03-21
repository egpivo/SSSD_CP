import random

import numpy as np
import pandas as pd
import torch


def merge_all_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in all time points and create rows for missing values.

    Args:
    df (DataFrame): DataFrame containing 'Date', 'Zone', and 'Load' columns.

    Returns:
    DataFrame: A DataFrame with the same columns. The number of rows is hours_df.shape[0] * 11.
    """
    # Create a DataFrame with all hourly time points
    hours_df = pd.DataFrame(
        {"Date": pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="1H")}
    )

    zones = df["Zone"].unique()
    result_all_time = pd.DataFrame()

    for zone in zones:
        # Extract data for the current zone
        load_zone = df.loc[df["Zone"] == zone]

        # Merge with hourly time points
        result = pd.merge(hours_df, load_zone, on="Date", how="left")
        result["Zone"] = zone

        result_all_time = pd.concat([result_all_time, result], axis=0)

    return result_all_time


def load_testing_data(test_data_path: str, num_samples: int) -> torch.Tensor:
    """
    Load and prepare testing data for generation.

    Args:
    - test_data_path (str): Path to the testing data file.
    - num_samples (int): Number of samples per batch.

    Returns:
    - torch.Tensor: Tensor containing the testing data prepared for generation.
    """
    # Load testing data
    testing_data = np.load(test_data_path)

    # Split testing data into batches
    testing_data_batches = np.split(testing_data, testing_data.shape[0] // num_samples)

    # Convert to numpy array and then to torch tensor
    testing_data_tensor = torch.from_numpy(np.array(testing_data_batches)).float()

    # Move tensor to CUDA device if available
    if torch.cuda.is_available():
        testing_data_tensor = testing_data_tensor.cuda()

    return testing_data_tensor


def load_and_split_training_data(
    training_data_load: np.ndarray,
    batch_num: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Load and split training data into batches.

    Args:
        training_data_load (np.ndarray): The training data to load and split.
        batch_num (int): The number of batches to create.
        batch_size (int): The size of each batch.
        device (torch.device): The device to move the data to.

    Returns:
        torch.Tensor: The training data split into batches and moved to the specified device.
    """
    total_samples = training_data_load.shape[0]
    if batch_size > total_samples:
        raise ValueError(
            "Batch size exceeds the total number of samples in the training data"
        )

    indices = random.sample(range(training_data_load.shape[0]), batch_num * batch_size)
    training_data = training_data_load[indices]
    training_data = np.split(training_data, batch_num, 0)
    training_data = np.array(training_data)
    return torch.from_numpy(training_data).to(device, dtype=torch.float32)
