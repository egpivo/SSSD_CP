import argparse
import datetime
import json
import os
import random
from math import floor

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
    print_size,
    training_loss,
)

MASK_FN = {
    "rm": get_mask_rm,
    "mnr": get_mask_mnr,
    "bm": get_mask_bm,
    "forecast": get_mask_forecast,
}

MODELS = {0: DiffWaveImputer, 1: SSSDSAImputer, 2: SSSDS4Imputer}


def train(
    output_directory,
    ckpt_iter,
    n_iters,
    iters_per_ckpt,
    iters_per_logging,
    learning_rate,
    use_model,
    only_generate_missing,
    masking,
    missing_k,
    writer,
    batch_size=80,
):
    """
    Train Diffusion Models

    Parameters:
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
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Map diffusion hyperparameters to GPU
    for key in diffusion_hyperparams:
        if key != "T":
            diffusion_hyperparams[key] = diffusion_hyperparams[key].to(device)

    # Predefine model
    if use_model not in MODELS:
        raise KeyError(f"Please enter a correct model number, but got {use_model}")
    net = MODELS[use_model](**model_config, device=device).to(device)
    print_size(net)

    # Move the model to the device
    net = nn.DataParallel(net).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Load checkpoint
    if ckpt_iter == "max":
        print(output_directory)
        ckpt_iter = find_max_epoch(output_directory)
        print(ckpt_iter)
    if ckpt_iter >= 0:
        try:
            # Load checkpoint file
            model_path = os.path.join(output_directory, f"{ckpt_iter}.pkl")
            print(model_path)
            checkpoint = torch.load(model_path, map_location="cpu")

            # Feed model dict and optimizer state
            net.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            print("Successfully loaded model at iteration {}".format(ckpt_iter))
        except Exception as e:
            ckpt_iter = -1
            print(f"No valid checkpoint model found. Error: {e}")
    else:
        ckpt_iter = -1
        print("No valid checkpoint model found, start training from initialization.")

    # Custom data loading and reshaping
    training_data_load = np.load(trainset_config["train_data_path"])
    training_size = training_data_load.shape[0]
    batch_num = floor(training_size / batch_size)
    print(batch_num)

    index = random.sample(range(training_size), batch_num * batch_size)
    training_data = training_data_load[
        index,
    ]
    training_data = np.split(training_data, batch_num, 0)
    training_data = np.array(training_data)
    training_data = torch.from_numpy(training_data).to(device, dtype=torch.float32)
    print("Data loaded")

    # Training
    n_iter = ckpt_iter + 2 if ckpt_iter == -1 else ckpt_iter + 1
    print(f"Start the {n_iter} iteration")
    while n_iter < n_iters + 1:
        for batch in training_data:
            # Shuffle batch after each epoch
            if n_iter % batch_num == 0:
                index = random.sample(range(training_size), batch_num * batch_size)
                training_data = training_data_load[
                    index,
                ]
                training_data = np.split(training_data, batch_num, 0)
                training_data = np.array(training_data)
                training_data = torch.from_numpy(training_data).to(
                    device, dtype=torch.float32
                )

            if masking not in MASK_FN:
                raise KeyError(f"Please enter a correct masking, but got {masking}")
            transposed_mask = MASK_FN[masking](batch[0], missing_k)
            mask = (
                transposed_mask.permute(1, 0)
                .repeat(batch.size()[0], 1, 1)
                .to(device, dtype=torch.float32)
            )
            loss_mask = ~mask.bool()
            batch = batch.permute(0, 2, 1)

            assert batch.size() == mask.size() == loss_mask.size()

            # Back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(
                net,
                nn.MSELoss(),
                X,
                diffusion_hyperparams,
                only_generate_missing=only_generate_missing,
                device=device,
            )

            loss.backward()
            optimizer.step()

            writer.add_scalar("Train/Loss", loss.item(), n_iter)
            if n_iter % iters_per_logging == 0:
                print(f"Iteration: {n_iter} \tLoss: {loss.item()}")

                # Save checkpoint
            if n_iter > 0 and n_iter % iters_per_ckpt == 0:
                checkpoint_name = f"{n_iter}.pkl"
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    os.path.join(output_directory, checkpoint_name),
                )
                print(f"Model at iteration {n_iter} is saved")
                current_time = datetime.datetime.now()
                print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

            n_iter += 1


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

    with open(args.config) as f:
        config = json.load(f)
    print(config)

    # 建立 output directionary
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
    print("output directory", output_directory, flush=True)

    # 更新 config file 的值
    config["train_config"]["output_directory"] = output_directory

    # 設定 toesorboard 要存在哪
    writer = SummaryWriter(f"{output_directory}/log")
    config["train_config"]["writer"] = writer
    train_config = config["train_config"]  # training parameters

    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

    global diffusion_config
    diffusion_config = config["diffusion_config"]  # basic hyperparameters

    global diffusion_hyperparams
    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config
    )  # dictionary of all diffusion hyperparameters

    global model_config
    if train_config["use_model"] in (0, 2):
        model_config = config["wavenet_config"]
    elif train_config["use_model"] == 1:
        model_config = config["sashimi_config"]
    else:
        raise KeyError(
            f"Please enter correct model number, but got {train_config['use_model']}"
        )

    current_time = datetime.datetime.now()
    print("當前時間:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

    train(**train_config)

    current_time = datetime.datetime.now()
    print("當前時間:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
