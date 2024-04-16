import os

import numpy as np


def read_multiple_imputations(folder_path: str, missing_k: int) -> np.ndarray:
    """
    Read multiple imputations generated from 'inference_multiples.py'.

    Args:
        folder_path (str): The folder containing the imputation files.
        missing_k (int): The number of the last elements to be predicted.

    Returns:
        np.ndarray: An array containing imputations with shape (num_files, obs, channel, missing_k).
    """
    # Check if the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist.")

    # Get a list of files in the folder
    file_list = os.listdir(folder_path)

    # Filter out only imputation0.npy files
    npy_files = [file for file in file_list if file.endswith("imputation0.npy")]

    if not npy_files:
        raise FileNotFoundError(f"No imputation0.npy files found in '{folder_path}'.")

    # Initialize stack array
    stack_array_data = []

    # Loop through all imputation0.npy files and read them
    for npy_file in npy_files:
        # shape = (obs, channel, length) -> (1, obs, channel, length)
        array_data = read_missing_k_data(folder_path, npy_file, missing_k)
        if array_data is not None:
            array_data = np.expand_dims(
                array_data, axis=0
            )  # Add a new axis for stacking
            stack_array_data.append(array_data)

    if not stack_array_data:
        raise ValueError("No valid data found in the imputation files.")

    # Stack the arrays vertically
    stack_array_data = np.vstack(stack_array_data)
    return stack_array_data


def read_missing_k_data(folder_path: str, npy_file: str, missing_k: int) -> np.ndarray:
    """
    Read the last 'missing_k' elements of each observation from a NumPy file.

    Args:
        folder_path (str): The folder containing the file.
        npy_file (str): The file name to read.
        missing_k (int): The number of the last elements to be read.

    Returns:
        np.ndarray: An array containing the last 'missing_k' elements of each observation.
    """
    file_path = os.path.join(folder_path, npy_file)
    data = np.load(file_path)
    last_k_elements = data[:, :, (-missing_k):]
    return last_k_elements


def predict_interval(pred, beta=0.05):
    """
    goal: compute the (1-alpha) quantile of imputation ecdf, i.e, prediction interval
    output: lower bound and upper bound, shape: (obs, channel, length)
    input:
        pred = all data, shape(number of imputation files, obs, channel, length)
        beta = significance level of original prediction interval
    """
    # compute original prediction intervals
    L = np.quantile(pred, beta / 2, axis=0)
    U = np.quantile(pred, 1 - beta / 2, axis=0)

    return L, U


def compute_E_star(L, U, true, alpha=0.05):
    """
    goal: compute the (1-alpha) quantile of conformity scores, i.e, E_star
    output: E_star, shape: (channel, length)
    input:
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)
        alpha = miscoverage rate of conformal prediction
    """
    # compute the conformity scores
    E = np.maximum(L - true, true - U)

    # compute the (1-alpha) quantile of conformity scores
    CP_PAR = (1 + 1 / true.shape[0]) * (1 - alpha)
    E_star = np.quantile(E, CP_PAR, axis=0)

    return E_star


def adjust_PI(L, U, E_star):
    """
    goal: adjust prediction interval using conformal prediction
    output: adjusted lower and upper bound, shape: (obs, channel, length)
    input:
        L = lower bound to be adjusted, shape: (obs, channel, length)
        U = upper bound to be adjusted, shape: (obs, channel, length)
        E_star = scores, shape: (channel, length)
    """
    E_star_exd = np.expand_dims(E_star, axis=0)
    return L - E_star_exd, U + E_star_exd


def coverage_rate(L, U, true):
    """
    goal: compute the coverage rate, which is the proportion of [L,U] contains true data
    output: an array, shape (shape, length)
    input:
        L = lower bound, shape: (2209, 1, 24)
        U = upper bound, shape: (2209, 1, 24)
        true = true data, shape: (2209, 1, 24)
    """
    return np.sum(np.logical_and(true > L, true < U), axis=0) / true.shape[0]
