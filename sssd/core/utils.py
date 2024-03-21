import random

import numpy as np
import torch


def get_mask_mnr(sample: torch.Tensor, k: int) -> torch.Tensor:
    """
    Get mask of random segments (non-missing at random) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed,
    and 1's to preserved as per time series imputers.

    Args:
    sample (torch.Tensor): Input tensor of shape [# of samples, # of channels].
    k (int): Number of segments.

    Returns:
    torch.Tensor: Mask tensor of the same shape as the input sample.
    """
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0] : s_nan[-1] + 1] = 0

    return mask


def get_mask_bm(sample: torch.Tensor, k: int) -> torch.Tensor:
    """
    Get mask of same segments (black-out missing) across channels based on k,
    where k == number of segments. Mask of sample's shape where 0's to be imputed,
    and 1's to be preserved as per time series imputers.

    Args:
    sample (torch.Tensor): Input tensor of shape [# of samples, # of channels].
    k (int): Number of segments.

    Returns:
    torch.Tensor: Mask tensor of the same shape as the input sample.
    """
    mask = torch.ones(sample.shape)
    length_index = torch.tensor(range(mask.shape[0]))
    list_of_segments_index = torch.split(length_index, k)
    s_nan = random.choice(list_of_segments_index)
    for channel in range(mask.shape[1]):
        mask[:, channel][s_nan[0] : s_nan[-1] + 1] = 0

    return mask


def get_mask_forecast(sample: torch.Tensor, k: int) -> torch.Tensor:
    """
    Get mask of same segments (black-out missing) across channels based on k.

    Args:
        sample (torch.Tensor): Tensor of shape [# of samples, # of channels].
        k (int): Number of missing values.

    Returns:
        torch.Tensor: Mask of sample's shape where 0's indicate missing values to be imputed, and 1's indicate preserved values.
    """
    mask = torch.ones_like(sample)  # Initialize mask with all ones

    # Calculate the indices of missing values
    s_nan = torch.arange(mask.shape[0] - k, mask.shape[0])

    # Apply mask for each channel
    for channel in range(mask.shape[1]):
        mask[s_nan, channel] = 0

    return mask


def get_mask_rm(sample: torch.Tensor, k: int) -> torch.Tensor:
    """
    Get mask of random points (missing at random) across channels based on k,
    where k == number of data points. Mask of sample's shape where 0's to be imputed,
    and 1's to be preserved as per time series imputers.

    Args:
    sample (torch.Tensor): Input tensor of shape [# of samples, # of channels].
    k (int): Number of data points to be masked.

    Returns:
    torch.Tensor: Mask tensor of the same shape as the input sample.
    """
    mask = torch.ones_like(sample)
    for channel in range(mask.shape[1]):
        perm = torch.randperm(mask.shape[0])
        idx = perm[:k]
        mask[idx, channel] = 0

    return mask


def calc_diffusion_step_embedding(
    diffusion_steps: torch.Tensor,
    diffusion_step_embed_dim_in: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Embed a diffusion step t into a higher dimensional space.
    E.g., the embedding vector in the 128-dimensional space is
    [sin(t * 10^(0*4/63)), ..., sin(t * 10^(63*4/63)), cos(t * 10^(0*4/63)), ..., cos(t * 10^(63*4/63))].

    Args:
    diffusion_steps (torch.Tensor): Diffusion steps for batch data, shape=(batchsize, 1).
    diffusion_step_embed_dim_in (int, optional): Dimensionality of the embedding space for discrete diffusion steps. Default is 128.
    device (str, optional): Device to run the calculations on. Default is "cpu".

    Returns:
    torch.Tensor: The embedding vectors, shape=(batchsize, diffusion_step_embed_dim_in).
    """
    assert diffusion_step_embed_dim_in % 2 == 0

    half_dim = diffusion_step_embed_dim_in // 2
    _embed = np.log(10000) / (half_dim - 1)
    _embed = torch.exp(torch.arange(half_dim) * -_embed).to(device)
    _embed = diffusion_steps * _embed
    diffusion_step_embed = torch.cat((torch.sin(_embed), torch.cos(_embed)), 1)

    return diffusion_step_embed
