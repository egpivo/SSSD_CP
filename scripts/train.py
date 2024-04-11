import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sssd.core.model_specs import MODEL_PATH_FORMAT, setup_model
from sssd.training.trainer import DiffusionTrainer
from sssd.utils.logger import setup_logger
from sssd.utils.utils import calc_diffusion_hyperparams, display_current_time

LOGGER = setup_logger()


def fetch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/SSSDS4.json",
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


def setup_output_directory(config: dict) -> str:
    # Build output directory
    local_path = MODEL_PATH_FORMAT.format(
        T=config["diffusion_config"]["T"],
        beta_0=config["diffusion_config"]["beta_0"],
        beta_T=config["diffusion_config"]["beta_T"],
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
    dataset = TensorDataset(torch.from_numpy(training_data_load))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **config["diffusion_config"], device=device
    )
    net = setup_model(config, device)

    LOGGER.info(display_current_time())
    trainer = DiffusionTrainer(
        dataloader=dataloader,
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

    LOGGER.info(display_current_time())


if __name__ == "__main__":
    args = fetch_args()

    with open(args.config) as f:
        config = json.load(f)
    LOGGER.info(config)

    if torch.cuda.device_count() > 0:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_job(config, device, args.batch_size)
