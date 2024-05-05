import pytest

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
