import logging
import os
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_squared_error

from sssd.core.model_specs import MASK_FN
from sssd.utils.logger import setup_logger
from sssd.utils.util import find_max_epoch, sampling

LOGGER = setup_logger()


class DiffusionGenerator:
    """
    Generate data based on ground truth.

    Parameters:
    -----------
    net (torch.nn.Module):         The neural network model
    device (torch.device):         The device to run the model on (e.g., 'cuda' or 'cpu')
    diffusion_hyperparams (dict):  Dictionary of diffusion hyperparameters
    local_path (str):              Local path format for the model
    testing_data (np.ndarray):     Numpy array containing testing data
    output_directory (str):        Path to save generated samples
    num_samples (int):             Number of samples to generate (default is 4)
    ckpt_path (str):               Checkpoint directory
    ckpt_iter (int or 'max'):      Pretrained checkpoint to load; 'max' selects the maximum iteration
    masking (str):                  Type of masking: 'mnr' (missing not at random), 'bm' (black-out), 'rm' (random missing)
    missing_k (int):               Number of missing time points for each channel across the length
    only_generate_missing (int):   Whether to generate only missing portions of the signal:
                                    0 (all sample diffusion), 1 (generate missing portions only)
    logger (Optional[logging.Logger]): Logger object for logging messages (default is None)
    """

    def __init__(
        self,
        net: torch.nn.Module,
        device: torch.device,
        diffusion_hyperparams: dict,
        local_path: str,
        testing_data: np.ndarray,
        output_directory: str,
        num_samples: int,
        ckpt_path: str,
        ckpt_iter: str,
        masking: str,
        missing_k: int,
        only_generate_missing: int,
        logger: Optional[logging.Logger] = None,
    ):
        self.net = net
        self.device = device
        self.diffusion_hyperparams = diffusion_hyperparams
        self.local_path = local_path
        self.testing_data = testing_data
        self.num_samples = num_samples
        self.masking = masking
        self.missing_k = missing_k
        self.only_generate_missing = only_generate_missing
        self.logger = logger or LOGGER

        self.output_directory = self._prepare_output_directory(
            output_directory, local_path, ckpt_iter
        )
        self._load_checkpoint(ckpt_path, ckpt_iter)

    def _load_checkpoint(self, ckpt_path: str, ckpt_iter: str) -> None:
        """Load a checkpoint for the given neural network model."""
        ckpt_path = os.path.join(ckpt_path, self.local_path)
        if ckpt_iter == "max":
            ckpt_iter = find_max_epoch(ckpt_path)
        model_path = os.path.join(ckpt_path, f"{ckpt_iter}.pkl")
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Successfully loaded model at iteration {ckpt_iter}")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model file not found at {model_path}") from e
        except Exception as e:
            raise Exception(f"Failed to load model: {e}")

    def _prepare_output_directory(
        self, output_directory: str, local_path: str, ckpt_iter: str
    ) -> str:
        """Prepare the output directory to save generated samples."""
        ckpt_iter_str = (
            "max"
            if ckpt_iter == "max"
            else f"imputation_multiple_{int(ckpt_iter) // 1000}k"
        )
        output_directory = os.path.join(output_directory, local_path, ckpt_iter_str)
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
        self.logger.info(f"Output directory: {output_directory}")
        return output_directory

    def _update_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """Update mask based on the given batch."""
        transposed_mask = MASK_FN[self.masking](batch[0], self.missing_k)
        return (
            transposed_mask.permute(1, 0)
            .repeat(batch.size()[0], 1, 1)
            .to(self.device, dtype=torch.float32)
        )

    def generate(self) -> list:
        """Generate samples using the given neural network model."""
        all_mses = []
        for i, batch in enumerate(self.testing_data):
            mask = self._update_mask(batch)
            batch = batch.permute(0, 2, 1)
            sample_length = batch.size(2)
            sample_channels = batch.size(1)

            generated_audio = sampling(
                self.net,
                (self.num_samples, sample_channels, sample_length),
                self.diffusion_hyperparams,
                cond=batch,
                mask=mask,
                only_generate_missing=self.only_generate_missing,
            )

            generated_audio = generated_audio.detach().cpu().numpy()
            batch = batch.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            outfile = f"imputation{i}.npy"
            np.save(os.path.join(self.output_directory, outfile), generated_audio)

            outfile = f"original{i}.npy"
            np.save(os.path.join(self.output_directory, outfile), batch)

            outfile = f"mask{i}.npy"
            np.save(os.path.join(self.output_directory, outfile), mask)

            mse = mean_squared_error(
                generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)]
            )
            all_mses.append(mse)

        return all_mses
