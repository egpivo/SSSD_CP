import argparse
import datetime
import json
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from sssd.core.model_specs import MASK_FN, MODEL_PATH_FORMAT, setup_model
from sssd.utils.logger import setup_logger
from sssd.utils.util import calc_diffusion_hyperparams, find_max_epoch, sampling

LOGGER = setup_logger()


class DiffusionGenerator:
    def __init__(
        self,
        net,
        device,
        diffusion_hyperparams,
        local_path,
        testing_data,
        output_directory,
        num_samples,
        ckpt_path,
        ckpt_iter,
        masking,
        missing_k,
        only_generate_missing,
        logger: Optional[logging.Logger] = None,
    ):
        self.net = net
        self.device = device
        self.diffusion_hyperparams = diffusion_hyperparams
        self.local_path = local_path
        self.testing_data = testing_data
        self.output_directory = self._prepare_output_directory(
            output_directory, local_path, ckpt_iter
        )
        self.num_samples = num_samples
        self.masking = masking
        self.missing_k = missing_k
        self.only_generate_missing = only_generate_missing
        self.logger = logger or LOGGER

        self._load_checkpoint(ckpt_path, ckpt_iter)

    def _load_checkpoint(self, ckpt_path, ckpt_iter):
        """Load a checkpoint for the given neural network model."""
        ckpt_path = os.path.join(ckpt_path, self.local_path)
        if ckpt_iter == "max":
            ckpt_iter = find_max_epoch(ckpt_path)
        model_path = os.path.join(ckpt_path, f"{ckpt_iter}.pkl")
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            self.net.load_state_dict(checkpoint["model_state_dict"])
            self.logger.info(f"Successfully loaded model at iteration {ckpt_iter}")
        except:
            raise Exception("No valid model found")

    def _prepare_output_directory(self, output_directory, local_path, ckpt_iter):
        """Prepare the output directory to save generated samples."""
        output_directory = os.path.join(
            output_directory,
            local_path,
            f"imputation_multiple_{int(ckpt_iter) // 1000}k",
        )
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
        self.logger.info("Output directory: %s", output_directory)
        return output_directory

    def generate(self):
        """Generate samples using the given neural network model."""
        all_mse = []
        for i, batch in enumerate(self.testing_data):
            transposed_mask = MASK_FN[self.masking](batch[0], self.missing_k)
            mask = (
                transposed_mask.permute(1, 0)
                .repeat(batch.size()[0], 1, 1)
                .to(self.device, dtype=torch.float32)
            )

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
            all_mse.append(mse)

        return all_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON file containing the configuration",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1,
        help="Number of utterances to be generated (default is 1)",
    )
    parser.add_argument(
        "-ckpt_iter",
        "--ckpt_iter",
        default="max",
        help='Which checkpoint to use; assign a number or "max"',
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse configs
    with open(args.config) as f:
        config = json.load(f)

    testing_data = np.load(config["trainset_config"]["test_data_path"])

    local_path = MODEL_PATH_FORMAT.format(
        T=config["diffusion_config"]["T"],
        beta_0=config["diffusion_config"]["beta_0"],
        beta_T=config["diffusion_config"]["beta_T"],
    )

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **config["diffusion_config"], device=device
    )

    current_time = datetime.datetime.now()
    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

    net = setup_model(config, device)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 0:
        print("Using", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    DiffusionGenerator(
        net=net,
        device=device,
        diffusion_hyperparams=diffusion_hyperparams,
        testing_data=testing_data,
        local_path=local_path,
        output_directory=config["gen_config"]["output_directory"],
        ckpt_path=config["gen_config"]["ckpt_path"],
        ckpt_iter=args.ckpt_iter,
        num_samples=args.num_samples,
        masking=config["train_config"]["masking"],
        missing_k=config["train_config"]["missing_k"],
        only_generate_missing=config["train_config"]["only_generate_missing"],
    ).generate()

    current_time = datetime.datetime.now()
    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
