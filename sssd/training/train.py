import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sssd.training.model_specs import MASK_FN, MODELS
from sssd.utils.util import (
    calc_diffusion_hyperparams,
    display_current_time,
    find_max_epoch,
    training_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_and_split_data(training_data_load, batch_num, batch_size, device):
    index = random.sample(range(training_data_load.shape[0]), batch_num * batch_size)
    training_data = training_data_load[index]
    training_data = np.split(training_data, batch_num, 0)
    training_data = np.array(training_data)
    return torch.from_numpy(training_data).to(device, dtype=torch.float32)


class DiffusionTrainer:
    """
    Train Diffusion Models

    Parameters:
    -----------
    output_directory (str):         Save model checkpoints to this path
    ckpt_iter (int or 'max'):       The pretrained checkpoint to be loaded;
                                    automatically selects the maximum iteration if 'max' is selected
    n_iters (int):                  Number of iterations to train
    iters_per_ckpt (int):           Number of iterations to save checkpoint,
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        Number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          Learning rate
    use_model (int):                0: DiffWave, 1: SSSDSA, 2: SSSDS4
    only_generate_missing (int):    0: All sample diffusion,  1: Only apply diffusion to missing portions of the signal
    masking (str):                  'mnr': Missing not at random, 'bm': Blackout missing, 'rm': Random missing
    missing_k (int):                K missing time steps for each feature across the sample length
    batch_size (int):               Size of each training batch
    """

    def __init__(
        self,
        training_data_load,
        diffusion_hyperparams,
        net,
        device,
        output_directory,
        ckpt_iter,
        n_iters,
        iters_per_ckpt,
        iters_per_logging,
        learning_rate,
        only_generate_missing,
        masking,
        missing_k,
        batch_size=80,
        **kwargs,
    ):
        self.training_data_load = training_data_load
        self.diffusion_hyperparams = diffusion_hyperparams
        self.net = nn.DataParallel(net).to(device)
        self.device = device
        self.output_directory = output_directory
        self.ckpt_iter = ckpt_iter
        self.n_iters = n_iters
        self.iters_per_ckpt = iters_per_ckpt
        self.iters_per_logging = iters_per_logging
        self.learning_rate = learning_rate
        self.only_generate_missing = only_generate_missing
        self.masking = masking
        self.missing_k = missing_k
        self.writer = SummaryWriter(f"{output_directory}/log")
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

        if self.masking not in MASK_FN:
            raise KeyError(f"Please enter a correct masking, but got {self.masking}")

    def _update_diffusion_hyperparams(self):
        for key in self.diffusion_hyperparams:
            if key != "T":
                self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].to(
                    self.device
                )

    def _load_checkpoint(self):
        if self.ckpt_iter == "max":
            self.ckpt_iter = find_max_epoch(self.output_directory)
        if self.ckpt_iter >= 0:
            try:
                model_path = os.path.join(
                    self.output_directory, f"{self.ckpt_iter}.pkl"
                )
                checkpoint = torch.load(model_path, map_location="cpu")

                self.net.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                logger.info("Successfully loaded model at iteration %s", self.ckpt_iter)
            except Exception as e:
                self.ckpt_iter = -1
                logger.error("No valid checkpoint model found. Error: %s", e)
        else:
            self.ckpt_iter = -1
            logger.info(
                "No valid checkpoint model found, start training from initialization."
            )

    def _prepare_training_data(self):
        training_size = self.training_data_load.shape[0]
        batch_num = training_size // self.batch_size
        training_data = load_and_split_data(
            self.training_data_load, batch_num, self.batch_size, self.device
        )
        logger.info("Data loaded with batch num - %s", batch_num)
        return training_data, batch_num

    def _save_model(self, n_iter):
        if n_iter > 0 and n_iter % self.iters_per_ckpt == 0:
            torch.save(
                {
                    "model_state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                os.path.join(self.output_directory, f"{n_iter}.pkl"),
            )

    def _update_mask(self, batch):
        transposed_mask = MASK_FN[self.masking](batch[0], self.missing_k)
        return (
            transposed_mask.permute(1, 0)
            .repeat(batch.size()[0], 1, 1)
            .to(self.device, dtype=torch.float32)
        )

    def _train_per_epoch(self, training_data):
        for batch in training_data:
            mask = self._update_mask(batch)
            loss_mask = ~mask.bool()

            batch = batch.permute(0, 2, 1)
            assert batch.size() == mask.size() == loss_mask.size()

            self.optimizer.zero_grad()
            loss = training_loss(
                net=self.net,
                loss_fn=nn.MSELoss(),
                X=(batch, batch, mask, loss_mask),
                diffusion_hyperparams=self.diffusion_hyperparams,
                only_generate_missing=self.only_generate_missing,
                device=self.device,
            )
            loss.backward()
            self.optimizer.step()

        return loss

    def train(self):
        self._update_diffusion_hyperparams()
        self._load_checkpoint()
        training_data, batch_num = self._prepare_training_data()

        n_iter_start = (
            self.ckpt_iter + 2 if self.ckpt_iter == -1 else self.ckpt_iter + 1
        )
        logger.info(f"Start the {n_iter_start} iteration")

        for n_iter in range(n_iter_start, self.n_iters + 1):
            if n_iter % batch_num == 0:
                training_data = load_and_split_data(
                    self.training_data_load, batch_num, self.batch_size, self.device
                )

            loss = self._train_per_epoch(training_data)

            self.writer.add_scalar("Train/Loss", loss.item(), n_iter)
            if n_iter % self.iters_per_logging == 0:
                logger.info(f"Iteration: {n_iter} \tLoss: { loss.item()}")
            self._save_model(n_iter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/SSSDS4.json",
        help="JSON file for configuration",
    )
    args = parser.parse_args()

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info("Using %s GPUs!", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = json.load(f)
    logger.info(config)

    # Build output directory
    local_path = "T{}_beta0{}_betaT{}".format(
        config["diffusion_config"]["T"],
        config["diffusion_config"]["beta_0"],
        config["diffusion_config"]["beta_T"],
    )
    output_directory = os.path.join(
        config["train_config"]["output_directory"], local_path
    )

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    logger.info("Output directory %s", output_directory)

    # Update config file values
    config["train_config"]["output_directory"] = output_directory

    # Set TensorBoard directory
    training_data_load = np.load(config["trainset_config"]["train_data_path"])
    diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])
    use_model = config["train_config"]["use_model"]
    if use_model in (0, 2):
        model_config = config["wavenet_config"]
    elif use_model == 1:
        model_config = config["sashimi_config"]
    else:
        raise KeyError(
            "Please enter correct model number, but got {}".format(use_model)
        )
    net = MODELS[use_model](**model_config, device=device).to(device)
    display_current_time()

    trainer = DiffusionTrainer(
        training_data_load, diffusion_hyperparams, net, device, **config["train_config"]
    )
    trainer.train()

    display_current_time()
