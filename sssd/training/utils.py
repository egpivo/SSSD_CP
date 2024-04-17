from typing import Dict, Tuple

import torch

from sssd.utils.utils import std_normal


def training_loss(
    net: torch.nn.Module,
    loss_fn: torch.nn.Module,
    X: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    diffusion_hyperparams: Dict[str, torch.Tensor],
    only_generate_missing: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the training loss of epsilon and epsilon_theta.

    Args:
        net (torch.nn.Module): The wavenet model.
        loss_fn (torch.nn.Module): The loss function, default is nn.MSELoss().
        X (tuple): Training data tuple containing (audio, cond, mask, loss_mask).
        diffusion_hyperparams (dict): Dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams.
                                      Note, the tensors need to be cuda tensors.
        only_generate_missing (int): Flag to indicate whether to only generate missing values (default=1).
        device (str): Device to run the computations on (default="cuda").

    Returns:
        torch.Tensor: Training loss.
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    time_series, cond, mask, loss_mask = X

    # B is batch size, C=1, L is time_series length
    B, C, L = time_series.shape

    diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(device)
    z = std_normal(time_series.shape, device)
    if only_generate_missing:
        z = time_series * mask.float() + z * (1 - mask).float()

    # Compute x_t from q(x_t|x_0)
    transformed_series = (
        torch.sqrt(Alpha_bar[diffusion_steps]) * time_series
        + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
    )

    # Predict \epsilon according to \epsilon_\theta
    epsilon_theta = net(
        (
            transformed_series,
            cond,
            mask,
            diffusion_steps.view(B, 1),
        )
    )

    if only_generate_missing:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    else:
        return loss_fn(epsilon_theta, z)
