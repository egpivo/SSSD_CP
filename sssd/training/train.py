import argparse
import datetime
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from sssd.imputers.DiffWaveImputer import DiffWaveImputer
from sssd.imputers.SSSDS4Imputer import SSSDS4Imputer
from sssd.imputers.SSSDSAImputer import SSSDSAImputer
from sssd.utils.util import (
    calc_diffusion_hyperparams,
    find_max_epoch,
    get_mask_bm,
    get_mask_forecast,
    get_mask_mnr,
    get_mask_rm,
    training_loss,
)

MASK_FN = {
    "rm": get_mask_rm,
    "mnr": get_mask_mnr,
    "bm": get_mask_bm,
    "forecast": get_mask_forecast,
}

MODELS = {0: DiffWaveImputer, 1: SSSDSAImputer, 2: SSSDS4Imputer}


def load_and_split_data(training_data_load, batch_num, batch_size, device):
    index = random.sample(range(training_data_load.shape[0]), batch_num * batch_size)
    training_data = training_data_load[
        index,
    ]
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
    writer (SummaryWriter):         TensorBoard SummaryWriter for logging
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
        writer,
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
        self.writer = writer
        self.batch_size = batch_size

    def train(self):
        if self.masking not in MASK_FN:
            raise KeyError(f"Please enter a correct masking, but got {self.masking}")

        for key in self.diffusion_hyperparams:
            if key != "T":
                self.diffusion_hyperparams[key] = self.diffusion_hyperparams[key].to(
                    self.device
                )

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)

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
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                print(
                    "Successfully loaded model at iteration {}".format(self.ckpt_iter)
                )
            except Exception as e:
                self.ckpt_iter = -1
                print(f"No valid checkpoint model found. Error: {e}")
        else:
            self.ckpt_iter = -1
            print(
                "No valid checkpoint model found, start training from initialization."
            )

        training_size = self.training_data_load.shape[0]
        batch_num = training_size // self.batch_size

        training_data = load_and_split_data(
            self.training_data_load, batch_num, self.batch_size, self.device
        )

        n_iter_start = (
            self.ckpt_iter + 2 if self.ckpt_iter == -1 else self.ckpt_iter + 1
        )
        print(f"Start the {n_iter_start} iteration")

        for n_iter in range(n_iter_start, self.n_iters + 1):
            for batch in training_data:
                if n_iter % batch_num == 0:
                    training_data = load_and_split_data(
                        self.training_data_load, batch_num, self.batch_size, self.device
                    )
                transposed_mask = MASK_FN[self.masking](batch[0], self.missing_k)
                mask = (
                    transposed_mask.permute(1, 0)
                    .repeat(batch.size()[0], 1, 1)
                    .to(self.device, dtype=torch.float32)
                )
                loss_mask = ~mask.bool()
                batch = batch.permute(0, 2, 1)

                assert batch.size() == mask.size() == loss_mask.size()

                optimizer.zero_grad()
                loss = training_loss(
                    net=self.net,
                    loss_fn=nn.MSELoss(),
                    X=(batch, batch, mask, loss_mask),
                    diffusion_hyperparams=self.diffusion_hyperparams,
                    only_generate_missing=self.only_generate_missing,
                    device=self.device,
                )

                loss.backward()
                optimizer.step()

                self.writer.add_scalar("Train/Loss", loss.item(), n_iter)
                if n_iter % self.iters_per_logging == 0:
                    print(f"Iteration: {n_iter} \tLoss: {loss.item()}")

                if n_iter > 0 and n_iter % self.iters_per_ckpt == 0:
                    checkpoint_name = f"{n_iter}.pkl"
                    torch.save(
                        {
                            "model_state_dict": self.net.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        os.path.join(self.output_directory, checkpoint_name),
                    )
                    print(f"Model at iteration {n_iter} is saved")
                    current_time = datetime.datetime.now()
                    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/SSSDS4.json",
        help="JSON file for configuration",
    )

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    print(config)

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
    print("Output directory", output_directory, flush=True)

    # Update config file values
    config["train_config"]["output_directory"] = output_directory

    # Set TensorBoard directory
    writer = SummaryWriter(f"{output_directory}/log")
    config["train_config"]["writer"] = writer
    train_config = config["train_config"]  # training parameters

    training_data_load = np.load(config["trainset_config"]["train_data_path"])
    diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])

    if train_config["use_model"] in (0, 2):
        model_config = config["wavenet_config"]
    elif train_config["use_model"] == 1:
        model_config = config["sashimi_config"]
    else:
        raise KeyError(
            f"Please enter correct model number, but got {train_config['use_model']}"
        )
    net = MODELS[train_config["use_model"]](**model_config, device=device).to(device)
    current_time = datetime.datetime.now()
    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

    DiffusionTrainer(
        training_data_load, diffusion_hyperparams, net, device, **train_config
    ).train()

    current_time = datetime.datetime.now()
    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
