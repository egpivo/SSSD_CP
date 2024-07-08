import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest
import torch
import yaml
from torch import nn

from sssd.core.imputers.SSSDS4Imputer import SSSDS4Imputer
from sssd.utils.utils import (
    calc_diffusion_hyperparams,
    display_current_time,
    find_max_epoch,
    find_repo_root,
    flatten,
    generate_date_from_seq,
    load_yaml_file,
    print_size,
    sampling,
    std_normal,
)


def test_generate_date_from_seq_default_start_date():
    result = generate_date_from_seq(0)
    assert result == "2016/10/20"


def test_generate_date_from_seq_custom_start_date():
    result = generate_date_from_seq(10, start_date="2022-01-01")
    assert result == "2022/01/11"


def test_generate_date_from_seq_negative_value():
    result = generate_date_from_seq(-5)
    assert result == "2016/10/15"


def test_display_current_time():
    # Call the function
    current_time = display_current_time()

    # Get the current time using datetime.now()
    expected_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Assert that the current time returned by the function matches the expected time
    assert current_time == f"Current time: {expected_time}"


def test_calc_diffusion_hyperparams():
    # Define parameters
    T = 10
    beta_0 = 0.1
    beta_T = 0.9
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Calculate diffusion hyperparameters
    diffusion_hyperparams = calc_diffusion_hyperparams(T, beta_0, beta_T, device)

    # Check if all tensors are moved to the correct device
    for key, value in diffusion_hyperparams.items():
        if key != "T":
            assert value.device == torch.device(device)

    # Check if the shapes are correct
    assert diffusion_hyperparams["Beta"].shape == (T,)
    assert diffusion_hyperparams["Alpha"].shape == (T,)
    assert diffusion_hyperparams["Alpha_bar"].shape == (T,)
    assert diffusion_hyperparams["Sigma"].shape == (T,)


def test_std_normal():
    # Define the size of the tensor
    size = (2, 3)

    # Generate the tensor on CPU
    tensor_cpu = std_normal(size, device="cpu")
    assert isinstance(tensor_cpu, torch.Tensor)
    assert tensor_cpu.size() == size
    assert tensor_cpu.device == torch.device("cpu")

    # Generate the tensor on CUDA if available
    if torch.cuda.is_available():
        tensor_cuda = std_normal(size, device="cuda")
        assert isinstance(tensor_cuda, torch.Tensor)
        assert tensor_cuda.size() == size
        assert tensor_cuda.device == torch.device("cuda")

        # Ensure tensors generated on CPU and CUDA are not equal
        assert not torch.allclose(tensor_cpu, tensor_cuda)


@pytest.fixture
def dummy_data():
    size = (10, 1, 100)  # Sample size
    cond = torch.randn(10, 1, 100)  # Sample condition data
    mask = torch.randint(2, (10, 1, 100)).float()  # Sample mask data
    diffusion_hyperparams = calc_diffusion_hyperparams(
        T=100, beta_0=0.1, beta_T=0.9, device="cpu"
    )
    return size, cond, mask, diffusion_hyperparams


def test_sampling(dummy_data):
    size, cond, mask, diffusion_hyperparams = dummy_data
    net = SSSDS4Imputer(
        input_channels=1,
        residual_channels=1,
        skip_channels=1,
        output_channels=1,
        residual_layers=1,
        diffusion_step_embed_dim_input=2,
        diffusion_step_embed_dim_hidden=1,
        diffusion_step_embed_dim_output=1,
        s4_max_sequence_length=32,
        s4_state_dim=128,
        s4_dropout=0.1,
        s4_bidirectional=False,
        s4_use_layer_norm=False,
        device="cpu",
    )
    audio = sampling(net, size, diffusion_hyperparams, cond, mask)
    assert list(audio.shape) == [1, *size]


def test_print_size():
    # Define a simple network for testing
    class SimpleNet(nn.Module):
        def __init__(self):
            super(SimpleNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    net = SimpleNet()

    # Calculate the expected number of parameters
    expected_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6

    # Redirect stdout to capture printed output
    import sys
    from io import StringIO

    saved_stdout = sys.stdout
    sys.stdout = StringIO()

    # Call the function
    print_size(net)

    # Get the printed output
    printed_output = sys.stdout.getvalue()

    # Reset stdout
    sys.stdout = saved_stdout

    # Check if the printed output contains the expected number of parameters
    assert (
        f"{net.__class__.__name__} Parameters: {expected_params:.6f}M" in printed_output
    )


def test_flatten():
    # Test with a list of lists
    input_list = [[1, 2, 3], [4, 5], [6, 7, 8]]
    expected_output = [1, 2, 3, 4, 5, 6, 7, 8]
    assert flatten(input_list) == expected_output

    # Test with a list of tuples
    input_list = [(1, 2, 3), (4, 5), (6, 7, 8)]
    expected_output = [1, 2, 3, 4, 5, 6, 7, 8]
    assert flatten(input_list) == expected_output

    # Test with a mixed list of lists and tuples
    input_list = [[1, 2, 3], (4, 5), [6, 7, 8]]
    expected_output = [1, 2, 3, 4, 5, 6, 7, 8]
    assert flatten(input_list) == expected_output

    # Test with an empty list
    input_list = []
    expected_output = []
    assert flatten(input_list) == expected_output

    # Test with a list containing an empty list
    input_list = [[]]
    expected_output = []
    assert flatten(input_list) == expected_output


@pytest.fixture
def checkpoint_files(tmpdir):
    # Create dummy checkpoint files
    filenames = ["10000.pkl", "20000.pkl", "30000.pkl"]
    for filename in filenames:
        open(os.path.join(tmpdir, filename), "w").close()
    return tmpdir, filenames


def test_find_max_epoch(checkpoint_files):
    path, filenames = checkpoint_files
    max_epoch = find_max_epoch(path)
    assert max_epoch == 30000


class TestFindRepoRoot(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.repo_root = Path(self.test_dir.name) / "repo"
        self.repo_root.mkdir(parents=True, exist_ok=True)

        # Create .git directory in the repo root
        (self.repo_root / ".git").mkdir()

    def tearDown(self):
        # Clean up the temporary directory after tests
        self.test_dir.cleanup()

    def test_find_repo_root(self):
        # Test if the function correctly finds the repo root
        start_path = self.repo_root / "subdir1" / "subdir2"
        start_path.mkdir(parents=True, exist_ok=True)
        found_root = find_repo_root(str(start_path))
        self.assertEqual(found_root, str(self.repo_root))

    def test_no_repo_root(self):
        # Test if the function raises FileNotFoundError when no repo root is found
        non_repo_dir = Path(self.test_dir.name) / "non_repo"
        non_repo_dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            find_repo_root(str(non_repo_dir))


class TestLoadYamlFile(unittest.TestCase):
    @patch("os.path.exists", return_value=False)
    def test_file_not_found(self, mock_exists):
        with self.assertRaises(FileNotFoundError):
            load_yaml_file("mock_file.yaml")

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
    key1: value1
    key2: value2
    nested:
      key3: value3
    """,
    )
    def test_load_yaml_file_valid(self, mock_open, mock_exists):
        expected_output = {
            "key1": "value1",
            "key2": "value2",
            "nested": {"key3": "value3"},
        }
        result = load_yaml_file("mock_file.yaml")
        self.assertEqual(result, expected_output)

    @patch("os.path.exists", return_value=True)
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="""
    key1: value1
    key2: value2
    nested:
      key3: value3
      key4 value4  # Missing colon, clearly erroneous
    """,
    )
    def test_load_yaml_file_invalid(self, mock_open, mock_exists):
        with self.assertRaises(yaml.YAMLError):
            load_yaml_file("mock_file.yaml")


if __name__ == "__main__":
    unittest.main()
