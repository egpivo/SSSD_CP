import pytest
import torch
from torch.testing import assert_allclose

from sssd.core.layers.s4.hippo.state_space import SSKernelNPLR

seed = 42
torch.manual_seed(seed)


@pytest.fixture
def sample_data():
    H = 2
    N = 4
    L = 3
    w = torch.randn(N, dtype=torch.cfloat)
    P = torch.randn(2, N, dtype=torch.cfloat)
    B = torch.randn(N, dtype=torch.cfloat)
    C = torch.randn(3, H, N, dtype=torch.cfloat)
    log_dt = torch.randn(H)
    return H, N, L, w, P, B, C, log_dt


def test_SSKernelNPLR(sample_data):
    H, N, L, w, P, B, C, log_dt = sample_data
    kernel = SSKernelNPLR(L, w, P, B, C, log_dt)

    # Test forward pass
    k_B, k_state = kernel(rate=1.0, L=L)
    assert k_B.shape == (3, H, L // 2 + 1)

    # Test setup step
    kernel.setup_step()
    assert kernel._step_mode == "dense"

    kernel.default_state()
    # Test step function
    u = torch.randn(H, dtype=torch.cfloat)
    state = torch.randn(H, H * N, dtype=torch.cfloat)
    y, new_state = kernel.step(u, state)
    assert y.shape == (3, H)
    assert new_state.shape == (H, N * H)
    assert_allclose(
        torch.view_as_real(torch.sum(new_state)), torch.Tensor([3.4641, 64.1905])
    )

    # Test default state
    batch_shape = (5,)
    default_state = kernel.default_state(*batch_shape)
    assert default_state.shape == batch_shape + (H, N * H)
    assert_allclose(torch.view_as_real(torch.sum(default_state)), torch.Tensor([0, 0]))

    # Test doubling length
    kernel._setup_C(double_length=True)
    assert kernel.L == L * 2
