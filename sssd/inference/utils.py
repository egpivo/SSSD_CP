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


def predict_interval(
    pred: np.ndarray, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the (1-alpha) quantile prediction interval of imputation ecdf.

    Args:
        pred (np.ndarray): All data with shape (num_imputations, obs, channel, length).
        alpha (float, optional): Significance level of the prediction interval. Defaults to 0.05.

    Returns:
        tuple[np.ndarray, np.ndarray]: Lower and upper bounds of the prediction interval with shape (obs, channel, length).
    """
    # Compute original prediction intervals
    L = np.quantile(pred, alpha / 2, axis=0)
    U = np.quantile(pred, 1 - alpha / 2, axis=0)

    return L, U


def compute_E_star(
    L: np.ndarray, U: np.ndarray, true: np.ndarray, alpha: float = 0.05
) -> np.ndarray:
    """
    Compute the (1-alpha) quantile of conformity scores, i.e., E_star.

    Args:
        L (np.ndarray): Lower bound to be adjusted with shape (obs, channel, length).
        U (np.ndarray): Upper bound to be adjusted with shape (obs, channel, length).
        true (np.ndarray): True values with shape (obs, channel, length).
        alpha (float, optional): Mis-coverage rate of conformal prediction. Defaults to 0.05.

    Returns:
        np.ndarray: E_star with shape (channel, length).
    """
    # Compute the conformity scores
    E = np.maximum(L - true, true - U)

    # Compute the (1-alpha) quantile of conformity scores
    CP_PAR = (1 + 1 / true.shape[0]) * (1 - alpha)
    E_star = np.quantile(E, CP_PAR, axis=0)
    return E_star


def adjust_PI(
    L: np.ndarray, U: np.ndarray, E_star: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjust prediction interval using conformal prediction.

    Args:
        L (np.ndarray): Lower bound to be adjusted with shape (obs, channel, length).
        U (np.ndarray): Upper bound to be adjusted with shape (obs, channel, length).
        E_star (np.ndarray): Scores with shape (channel, length).

    Returns:
        tuple[np.ndarray, np.ndarray]: Adjusted lower and upper bound with shape (obs, channel, length).
    """
    E_star_exd = np.expand_dims(E_star, axis=0)
    adjusted_L = L - E_star_exd
    adjusted_U = U + E_star_exd
    return adjusted_L, adjusted_U


def coverage_rate(L: np.ndarray, U: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Compute the coverage rate, which is the proportion of [L, U] containing true data.

    Args:
        L (np.ndarray): Lower bound with shape (obs, channel, length).
        U (np.ndarray): Upper bound with shape (obs, channel, length).
        true (np.ndarray): True data with shape (obs, channel, length).

    Returns:
        np.ndarray: Coverage rate with shape (1, length).
    """
    coverage = np.sum(np.logical_and(true > L, true < U), axis=0) / true.shape[0]
    return coverage
