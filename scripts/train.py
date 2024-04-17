import argparse
import json
import os
from typing import Optional

import torch

from sssd.core.model_specs import MODEL_PATH_FORMAT, setup_model
from sssd.data.utils import get_dataloader
from sssd.training.trainer import DiffusionTrainer
from sssd.utils.logger import setup_logger
from sssd.utils.utils import calc_diffusion_hyperparams, display_current_time

LOGGER = setup_logger()
NUM_WORKERS = 1


def fetch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/SSSDS4.json",
        help="JSON file for configuration",
    )
    return parser.parse_args()


def setup_output_directory(config: dict) -> str:
    # Build output directory
    local_path = MODEL_PATH_FORMAT.format(
        T=config["diffusion"]["T"],
        beta_0=config["diffusion"]["beta_0"],
        beta_T=config["diffusion"]["beta_T"],
    )
    output_directory = os.path.join(config["training"]["output_directory"], local_path)

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    LOGGER.info("Output directory %s", output_directory)
    return output_directory


def run_job(config: dict, device: Optional[torch.device, str]) -> None:
    output_directory = setup_output_directory(config)
    batch_size = config["common"]["train_batch_size"]

    dataloader = get_dataloader(
        config["data"]["train_data_path"],
        batch_size,
        device=device,
        num_workers=config["common"]["num_workers"],
    )

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **config["diffusion"], device=device
    )
    net = setup_model(config, device)

    LOGGER.info(display_current_time())
    trainer = DiffusionTrainer(
        dataloader=dataloader,
        diffusion_hyperparams=diffusion_hyperparams,
        net=net,
        device=device,
        output_directory=output_directory,
        ckpt_iter=config["training"].get("ckpt_iter"),
        n_iters=config["training"].get("n_iters"),
        iters_per_ckpt=config["training"].get("iters_per_ckpt"),
        iters_per_logging=config["training"].get("iters_per_logging"),
        learning_rate=config["training"].get("learning_rate"),
        only_generate_missing=config["training"].get("only_generate_missing"),
        masking=config["training"].get("masking"),
        missing_k=config["training"].get("missing_k"),
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

    run_job(config, device)
