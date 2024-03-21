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
    Compute the training loss of epsilon and epsilon_theta

    Parameters:
    net (torch.nn.Module):            the wavenet model
    loss_fn (torch.nn.Module):         the loss function, default is nn.MSELoss()
    X (tuple):                         training data tuple containing (audio, cond, mask, loss_mask)
    diffusion_hyperparams (dict):      dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                       note, the tensors need to be cuda tensors
    only_generate_missing (int):       flag to indicate whether to only generate missing values (default=1)
    device (str):                      device to run the computations on (default="cuda")

    Returns:
    torch.Tensor: training loss
    """

    _dh = diffusion_hyperparams
    T, Alpha_bar = _dh["T"], _dh["Alpha_bar"]

    audio, cond, mask, loss_mask = X

    B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
    diffusion_steps = torch.randint(T, size=(B, 1, 1)).to(
        device
    )  # randomly sample diffusion steps from 1~T

    z = std_normal(audio.shape, device)
    if only_generate_missing:
        z = audio * mask.float() + z * (1 - mask).float()
    transformed_X = (
        torch.sqrt(Alpha_bar[diffusion_steps]) * audio
        + torch.sqrt(1 - Alpha_bar[diffusion_steps]) * z
    )  # compute x_t from q(x_t|x_0)
    epsilon_theta = net(
        (
            transformed_X,
            cond,
            mask,
            diffusion_steps.view(B, 1),
        )
    )  # predict \epsilon according to \epsilon_\theta

    if only_generate_missing:
        return loss_fn(epsilon_theta[loss_mask], z[loss_mask])
    else:
        return loss_fn(epsilon_theta, z)
