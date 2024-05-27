from typing import Dict, Tuple

import torch

from sssd.utils.utils import std_normal


def training_loss(
    model: torch.nn.Module,
    loss_function: torch.nn.Module,
    training_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    diffusion_parameters: Dict[str, torch.Tensor],
    generate_only_missing: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute the training loss of epsilon and epsilon_theta.

    Args:
        model (torch.nn.Module): The neural network model.
        loss_function (torch.nn.Module): The loss function, default is nn.MSELoss().
        training_data (tuple): Training data tuple containing (time_series, condition, mask, loss_mask).
        diffusion_parameters (dict): Dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams.
                                     Note, the tensors need to be cuda tensors.
        generate_only_missing (int): Flag to indicate whether to only generate missing values (default=1).
        device (str): Device to run the computations on (default="cuda").

    Returns:
        torch.Tensor: Training loss.
    """

    # Unpack diffusion hyperparameters
    T, alpha_bar = diffusion_parameters["T"], diffusion_parameters["Alpha_bar"]

    # Unpack training data
    time_series, condition, mask, loss_mask = training_data

    batch_size = time_series.shape[0]

    # Sample random diffusion steps for each batch element
    diffusion_steps = torch.randint(T, size=(batch_size, 1, 1)).to(device)

    # Generate Gaussian noise, applying mask if specified
    noise = (
        time_series * mask.float()
        + std_normal(time_series.shape, device) * (1 - mask).float()
        if generate_only_missing
        else std_normal(time_series.shape, device)
    )

    # Compute x_t from q(x_t|x_0)
    transformed_series = (
        torch.sqrt(alpha_bar[diffusion_steps]) * time_series
        + torch.sqrt(1 - alpha_bar[diffusion_steps]) * noise
    )

    # Predict epsilon according to epsilon_theta
    epsilon_theta = model(
        (transformed_series, condition, mask, diffusion_steps.view(batch_size, 1))
    )

    # Compute loss
    if generate_only_missing:
        return loss_function(epsilon_theta[loss_mask], noise[loss_mask])
    else:
        return loss_function(epsilon_theta, noise)
