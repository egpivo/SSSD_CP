import pytest
import torch

from sssd.core.utils import (
    calc_diffusion_step_embedding,
    get_mask_bm,
    get_mask_forecast,
    get_mask_mnr,
    get_mask_rm,
)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_get_mask_forecast(k):
    # Create sample tensor
    sample = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        dtype=torch.float32,
    )

    # Apply missing values mask
    mask = get_mask_forecast(sample, k)

    # Check if the mask has the correct shape
    assert mask.shape == sample.shape

    # Check if masked values are 0 and preserved values are 1
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if i >= mask.shape[0] - k:
                assert mask[i, j] == 0
            else:
                assert mask[i, j] == 1


@pytest.mark.parametrize("k", [1, 2, 3])
def test_get_mask_rm_shape(k):
    # Create sample tensor
    sample = torch.randn(10, 5)  # Example tensor of shape [10, 5]

    # Apply missing values mask
    mask = get_mask_rm(sample, k)

    # Check if the mask has the correct shape
    assert mask.shape == sample.shape


def test_get_mask_rm_valid_values():
    # Create sample tensor
    sample = torch.randn(10, 5)  # Example tensor of shape [10, 5]

    # Apply missing values mask
    mask = get_mask_rm(sample, 3)

    # Check if the masked values are either 0 or 1
    assert torch.all(torch.logical_or(mask == 0, mask == 1))


def test_get_mask_mnr_shape():
    # Create sample tensor
    sample = torch.randn(10, 5)  # Example tensor of shape [10, 5]

    # Apply missing values mask
    mask = get_mask_mnr(sample, 3)

    # Check if the mask has the correct shape
    assert mask.shape == sample.shape


def test_get_mask_bm_shape():
    # Create sample tensor
    sample = torch.randn(10, 5)  # Example tensor of shape [10, 5]

    # Apply missing values mask
    mask = get_mask_bm(sample, 3)

    # Check if the mask has the correct shape
    assert mask.shape == sample.shape


def test_get_mask_mnr_valid_values():
    # Create sample tensor
    sample = torch.randn(10, 5)  # Example tensor of shape [10, 5]

    # Apply missing values mask
    mask = get_mask_mnr(sample, 3)

    # Check if the masked values are either 0 or 1
    assert torch.all(torch.logical_or(mask == 0, mask == 1))


def test_get_mask_bm_valid_values():
    # Create sample tensor
    sample = torch.randn(10, 5)  # Example tensor of shape [10, 5]

    # Apply missing values mask
    mask = get_mask_bm(sample, 3)

    # Check if the masked values are either 0 or 1
    assert torch.all(torch.logical_or(mask == 0, mask == 1))


def test_calc_diffusion_step_embedding():
    # Define diffusion steps tensor
    diffusion_steps = torch.tensor([[1], [2], [3]], dtype=torch.long)

    # Calculate diffusion step embeddings
    embeddings = calc_diffusion_step_embedding(
        diffusion_steps, diffusion_step_embed_dim_input=128
    )

    # Check the shape of the embeddings
    assert embeddings.shape == (3, 128)

    # Check if all values are within the expected range [-1, 1]
    assert torch.all((embeddings >= -1) & (embeddings <= 1))

    # Check if the embeddings are on the correct device
    assert embeddings.device == torch.device("cpu")
