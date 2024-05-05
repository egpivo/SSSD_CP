import pytest
import torch

from sssd.core.layers.s4.hippo.hippo import HippoSSKernel


@pytest.fixture
def hippo_args():
    # Define default arguments to instantiate the HippoSSKernel
    return {
        "H": 10,
        "N": 64,
        "L": 1,
        "measure": "legs",
        "rank": 1,
        "channels": 1,
        "dt_min": 0.001,
        "dt_max": 0.1,
        "trainable": None,
        "lr": None,
        "length_correction": True,
        "hurwitz": False,
        "tie_state": False,
        "precision": 1,
        "resample": False,
        "verbose": False,
    }


@pytest.fixture
def hippo_kernel(hippo_args):
    # Instantiate the HippoSSKernel with the provided arguments
    return HippoSSKernel(**hippo_args)


def test_hippo_kernel_initialization(hippo_kernel):
    # Test if the HippoSSKernel is instantiated properly
    assert isinstance(
        hippo_kernel, HippoSSKernel
    ), "HippoSSKernel instantiation failed."


def test_hippo_kernel_forward(hippo_kernel):
    # Test the forward method
    L = 10  # Example sequence length
    output = hippo_kernel.forward(L=L)
    expected_shape = (hippo_kernel.channels, hippo_kernel.H, hippo_kernel.H)
    assert (
        output.shape == expected_shape
    ), f"Forward output shape mismatch: expected {expected_shape}, got {output.shape}"


def test_hippo_kernel_step(hippo_kernel, hippo_args):
    # Test the step method
    L = 10  # Example sequence length
    u = torch.randn(
        hippo_args["channels"],
        hippo_args["H"],
        L,
        dtype=torch.float if hippo_args["precision"] == 1 else torch.double,
    )
    state = hippo_kernel.default_state(u.size(0), u.size(2))
    u_step, state_step = hippo_kernel.step(u, state)

    assert (
        u_step.shape == u.shape
    ), f"Step output shape mismatch: expected {u.shape}, got {u_step.shape}"
    assert len(state_step) == len(state), "State shape mismatch after step."


def test_hippo_kernel_default_state(hippo_kernel, hippo_args):
    # Test the default_state method
    batch_size = 5
    seq_len = 10
    state = hippo_kernel.default_state(batch_size, seq_len)

    # The state should be a tuple with length depending on the internal structure of the kernel
    assert isinstance(state, tuple), "Default state is not a tuple."
    # Check if the shapes of the state elements are correct
    for s in state:
        assert (
            s.shape[0] == batch_size
        ), f"State batch size mismatch: expected {batch_size}, got {s.shape[0]}"
        assert (
            s.shape[-1] == seq_len
        ), f"State sequence length mismatch: expected {seq_len}, got {s.shape[-1]}"
