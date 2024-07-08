import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml


def flatten(v: List[Union[List[Any], Tuple[Any]]]) -> List[Any]:
    """
    Flatten a list of lists/tuples.

    Args:
        v (List[Union[List[Any], Tuple[Any]]]): List of lists or tuples.

    Returns:
        List[Any]: Flattened list.
    """
    return [x for y in v for x in y]


def find_max_epoch(path: str) -> int:
    """
    Find the maximum epoch/iteration in the given path, formatted as ${n_iter}.pkl (e.g., 100000.pkl).

    Args:
        path (str): Checkpoint path.

    Returns:
        int: Maximum iteration, -1 if there is no (valid) checkpoint.
    """
    files = os.listdir(path)
    epoch = -1
    for f in files:
        if f.endswith(".pkl"):
            try:
                epoch = max(epoch, int(f[:-4]))
            except ValueError:
                continue
    return epoch


def print_size(net: torch.nn.Module) -> None:
    """
    Print the number of parameters of a network.

    Args:
        net (torch.nn.Module): The network whose parameters need to be printed.
    """
    if not isinstance(net, torch.nn.Module):
        raise ValueError("The 'net' parameter must be an instance of torch.nn.Module.")

    module_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in module_parameters])
    print(f"{net.__class__.__name__} Parameters: {params / 1e6:.6f}M", flush=True)


def calc_diffusion_hyperparams(
    T: int, beta_0: float, beta_T: float, device: Optional[Union[torch.device, str]]
) -> Dict[str, torch.Tensor]:
    """
    Compute diffusion process hyperparameters.

    Args:
        T (int): Number of diffusion steps.
        beta_0 (float): Beta schedule start value.
        beta_T (float): Beta schedule end value.
        device (Union[torch.device, str]): Device to run the calculations on (e.g., 'cpu' or 'cuda').

    Returns:
        Dict[str, torch.Tensor]: A dictionary of diffusion hyperparameters including:
            T (int): Number of diffusion steps.
            Beta (torch.Tensor): Beta schedule tensor on the specified device, shape=(T,).
            Alpha (torch.Tensor): Alpha schedule tensor on the specified device, shape=(T,).
            Alpha_bar (torch.Tensor): Alpha_bar schedule tensor on the specified device, shape=(T,).
            Sigma (torch.Tensor): Sigma schedule tensor on the specified device, shape=(T,).
            These tensors are initially created on CPU and then moved to the specified device.
    """

    Beta = torch.linspace(beta_0, beta_T, T)  # Linear schedule
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t - 1]  # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (
            1 - Alpha_bar[t]
        )  # \tilde{\beta}_t = \beta_t * (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t)
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t

    return {
        "T": T,
        "Beta": Beta.to(device),
        "Alpha": Alpha.to(device),
        "Alpha_bar": Alpha_bar.to(device),
        "Sigma": Sigma.to(device),
    }


def std_normal(size: Tuple[int], device: Union[torch.device, str]) -> torch.Tensor:
    """
    Generate samples from the standard normal distribution of a specified size.

    Args:
        size (Tuple[int]): Size of the tensor to be generated.
        device (Union[torch.device, str]): Device to run the computations on (e.g., 'cpu' or 'cuda').

    Returns:
        torch.Tensor: Tensor containing samples from the standard normal distribution.
    """
    return torch.normal(0, 1, size=size).to(device)


def sampling(
    net: torch.nn.Module,
    size: Tuple[int, int, int],
    diffusion_hyperparams: Dict[str, torch.Tensor],
    cond: torch.Tensor,
    mask: torch.Tensor,
    sample_size: int = 1,
    only_generate_missing: int = 0,
    device: Union[torch.device, str] = "cpu",
) -> np.ndarray:
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t).

    Args:
        net (torch.nn.Module): The model.
        size (Tuple[int, int, int]): Size of tensor to be generated, usually (batch_size, channels=1, length).
        diffusion_hyperparams (Dict[str, torch.Tensor]): Dictionary of diffusion hyperparameters.
        cond (torch.Tensor): Conditioning tensor.
        mask (torch.Tensor): Mask tensor.
        sample_size (int, optional): Number of samples to generate for each input (default is 1).
        only_generate_missing (int, optional): Flag indicating whether to only generate missing values (default is 0).
        device (Union[torch.device, str], optional): Device to place tensors (default is 'cpu').

    Returns:
        np.ndarray: The generated samples, shape=(sample_size, *size).
    """
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    batch_size, channels, length = size
    all_samples = []

    for _ in range(sample_size):
        x = std_normal((batch_size, channels, length), device)

        with torch.no_grad():
            for t in range(T - 1, -1, -1):
                if only_generate_missing == 1:
                    x = x * (1 - mask).float() + cond * mask.float()
                diffusion_steps = (t * torch.ones((batch_size, 1))).to(
                    device
                )  # use the corresponding reverse step
                epsilon_theta = net(
                    (x, cond, mask, diffusion_steps)
                )  # predict \epsilon according to \epsilon_\theta
                # update x_{t-1} to \mu_\theta(x_t)
                x = (
                    x - (1 - Alpha[t]) / torch.sqrt(1 - Alpha_bar[t]) * epsilon_theta
                ) / torch.sqrt(Alpha[t])
                if t > 0:
                    x = x + Sigma[t] * std_normal(
                        (batch_size, channels, length), device
                    )  # add the variance term to x_{t-1}

            all_samples.append(x.cpu().numpy())

            return torch.tensor(np.stack(all_samples))


def display_current_time() -> str:
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


def generate_date_from_seq(value: int, start_date: str = "2016-10-20") -> str:
    """
    Generates the date based on the given number of observations.

    Args:
        value (int): The number of observations.
        start_date (str): The starting date in the format "YYYY-MM-DD". Default is "2016-10-20".

    Returns:
        str: The formatted date in the format "YYYY/MM/DD".
    """
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    target_date = start_date + timedelta(days=value)
    formatted_date = target_date.strftime("%Y/%m/%d")

    return formatted_date


def find_repo_root(current_path: str) -> str:
    """
    Find the root directory of the git repository.

    Parameters:
    current_path (str): The starting path to search for the repository root.

    Returns:
    str: The path to the repository root.

    Raises:
    FileNotFoundError: If the repository root cannot be found.
    """
    while not os.path.isdir(os.path.join(current_path, ".git")):
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Could not find the repository root.")
        current_path = parent_path
    return current_path


def load_yaml_file(file_path: str) -> Any:
    """
    Load a YAML file and return its contents.

    Parameters:
    file_path (str): The path to the YAML file.

    Returns:
    Any: The contents of the YAML file.

    Raises:
    FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, "rt") as f:
        return yaml.safe_load(f)
