import argparse
import json
import os

import numpy as np
import torch

from sssd.training.model_specs import MODELS
from sssd.training.trainer import DiffusionTrainer
from sssd.utils.logger import setup_logger
from sssd.utils.util import calc_diffusion_hyperparams, display_current_time

LOGGER = setup_logger()


def fetch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/SSSDS4.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=80,
        help="Batch size",
    )
    return parser.parse_args()


def setup_model(config: dict, device: torch.device) -> torch.nn.Module:
    use_model = config["train_config"]["use_model"]
    if use_model in (0, 2):
        model_config = config["wavenet_config"]
    elif use_model == 1:
        model_config = config["sashimi_config"]
    else:
        raise KeyError(
            "Please enter correct model number, but got {}".format(use_model)
        )
    return MODELS[use_model](**model_config, device=device).to(device)


def setup_output_directory(config: dict) -> str:
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
    LOGGER.info("Output directory %s", output_directory)
    return output_directory


def run_job(config: dict, device: torch.device, batch_size: int) -> None:
    output_directory = setup_output_directory(config)
    training_data_load = np.load(config["trainset_config"]["train_data_path"])
    diffusion_hyperparams = calc_diffusion_hyperparams(**config["diffusion_config"])
    net = setup_model(config, device)

    display_current_time()
    trainer = DiffusionTrainer(
        training_data_load=training_data_load,
        diffusion_hyperparams=diffusion_hyperparams,
        net=net,
        device=device,
        output_directory=output_directory,
        ckpt_iter=config["train_config"].get("ckpt_iter"),
        n_iters=config["train_config"].get("n_iters"),
        iters_per_ckpt=config["train_config"].get("iters_per_ckpt"),
        iters_per_logging=config["train_config"].get("iters_per_logging"),
        learning_rate=config["train_config"].get("learning_rate"),
        only_generate_missing=config["train_config"].get("only_generate_missing"),
        masking=config["train_config"].get("masking"),
        missing_k=config["train_config"].get("missing_k"),
        batch_size=batch_size,
        logger=LOGGER,
    )
    trainer.train()

    display_current_time()


if __name__ == "__main__":
    args = fetch_args()

    with open(args.config) as f:
        config = json.load(f)
    LOGGER.info(config)

    if torch.cuda.device_count() > 1:
        LOGGER.info("Using %s GPUs!", torch.cuda.device_count())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_job(config, device, args.batch_size)
