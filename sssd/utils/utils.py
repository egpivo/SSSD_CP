import os
from datetime import datetime, timedelta
from typing import List, Tuple, Union

import numpy as np
import torch


def flatten(v: List[Union[list, tuple]]) -> List:
    """
    Flatten a list of lists/tuples.

    Args:
    v (List[Union[list, tuple]]): List of lists or tuples.

    Returns:
    List: Flattened list.
    """

    return [x for y in v for x in y]


def find_max_epoch(path):
    """
    Find maximum epoch/iteration in path, formatted ${n_iter}.pkl
    E.g. 100000.pkl

    Parameters:
    path (str): checkpoint path

    Returns:
    maximum iteration, -1 if there is no (valid) checkpoint
    """

    files = os.listdir(path)
    epoch = -1
    for f in files:
        if len(f) <= 4:
            continue
        if f[-4:] == ".pkl":
            try:
                epoch = max(epoch, int(f[:-4]))
            except:
                continue
    return epoch


def print_size(net: torch.nn.Module) -> None:
    """
    Print the number of parameters of a network.

    Parameters:
    net (torch.nn.Module): The network whose parameters need to be printed.
    """

    if not isinstance(net, torch.nn.Module):
        raise ValueError("The 'net' parameter must be an instance of torch.nn.Module.")

    module_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in module_parameters])
    print(f"{net.__class__.__name__} Parameters: {params / 1e6:.6f}M", flush=True)


def calc_diffusion_hyperparams(
    T: int, beta_0: float, beta_T: float, device: str
) -> dict:
    """
    Compute diffusion process hyperparameters.

    Args:
    T (int): Number of diffusion steps.
    beta_0 (float): Beta schedule start value.
    beta_T (float): Beta schedule end value.
    device (str): Device to run the calculations on.

    Returns:
    dict: A dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, )).
        These cpu tensors are changed to cuda tensors on each individual gpu.
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

    diffusion_hyperparams = {
        "T": T,
        "Beta": Beta.to(device),
        "Alpha": Alpha.to(device),
        "Alpha_bar": Alpha_bar.to(device),
        "Sigma": Sigma.to(device),
    }

    return diffusion_hyperparams


def std_normal(size: Tuple[int], device: str) -> torch.Tensor:
    """
    Generate the standard Gaussian variable of a certain size

    Parameters:
    size (tuple): Size of the tensor to be generated
    device (str): Device to run the computations on

    Returns:
    torch.Tensor: Tensor containing samples from standard normal distribution
    """
    return torch.normal(0, 1, size=size).to(device)


def sampling(
    net,
    size: tuple,
    diffusion_hyperparams: dict,
    cond: torch.Tensor,
    mask: torch.Tensor,
    only_generate_missing: int = 0,
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t).

    Parameters:
    net (torch.nn.Module): the wavenet model.
    size (tuple): size of tensor to be generated, usually is (number of audios to generate, channels=1, length of audio).
    diffusion_hyperparams (dict): dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams.
                                  Note, the tensors need to be cuda tensors.
    cond (torch.Tensor): conditioning tensor.
    mask (torch.Tensor): mask tensor.
    only_generate_missing (int): flag indicating whether to only generate missing values.
    device (str): device to place tensors (e.g., 'cpu' or 'cuda').

    Returns:
    torch.Tensor: the generated audio(s) in torch.Tensor, shape=size.
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Sigma = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha) == T
    assert len(Alpha_bar) == T
    assert len(Sigma) == T
    assert len(size) == 3

    print("begin sampling, total number of reverse steps = %s" % T)

    x = std_normal(size, device)

    with torch.no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing == 1:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * torch.ones((size[0], 1))).to(
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
                    size, device
                )  # add the variance term to x_{t-1}

    return x


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