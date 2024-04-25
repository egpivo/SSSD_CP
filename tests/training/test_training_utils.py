import pytest
import torch

from sssd.core.imputers.SSSDS4Imputer import SSSDS4Imputer
from sssd.training.utils import training_loss


@pytest.fixture
def dummy_data():
    audio = torch.randn(10, 1, 100)  # Sample audio data
    cond = torch.randn(10, 1, 100)  # Sample condition data
    mask = torch.randint(2, (10, 1, 100)).float()  # Sample mask data
    loss_mask = torch.randint(
        2, (10, 1, 100)
    ).byte()  # Ensure loss_mask is of type byte
    diffusion_hyperparams = {
        "T": torch.tensor(100),  # Sample T value
        "Alpha_bar": torch.rand(100),  # Sample Alpha_bar tensor
    }
    return audio, cond, mask, loss_mask, diffusion_hyperparams


def test_training_loss(dummy_data):
    audio, cond, mask, loss_mask, diffusion_hyperparams = dummy_data
    net = SSSDS4Imputer(
        input_channels=1,
        residual_channels=1,
        skip_channels=1,
        output_channels=1,
        residual_layers=1,
        diffusion_step_embed_dim_input=4,
        diffusion_step_embed_dim_hidden=4,
        diffusion_step_embed_dim_output=4,
        s4_max_sequence_length=10,
        s4_state_dim=256,
        s4_dropout=0.1,
        s4_bidirectional=True,
        s4_use_layer_norm=True,
        device="cpu",
    )  # Initialize your neural network model
    loss_fn = torch.nn.MSELoss()  # Sample loss function
    loss = training_loss(
        net, loss_fn, (audio, cond, mask, loss_mask), diffusion_hyperparams
    )
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # Loss should be a scalar
