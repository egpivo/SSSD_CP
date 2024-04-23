import argparse
import json
from typing import Optional, Union

import torch
import torch.nn as nn

from sssd.core.model_specs import MODEL_PATH_FORMAT, setup_model
from sssd.data.utils import get_dataloader
from sssd.inference.generator import DiffusionGenerator
from sssd.utils.logger import setup_logger
from sssd.utils.utils import calc_diffusion_hyperparams, display_current_time

LOGGER = setup_logger()


def fetch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config.json",
        help="Path to the JSON file containing the configuration",
    )
    parser.add_argument(
        "-ckpt_iter",
        "--ckpt_iter",
        default="max",
        help='Which checkpoint to use; assign a number or "max" to find the latest checkpoint',
    )
    parser.add_argument(
        "-trials",
        "--trials",
        type=int,
        default=1,
        help="Trials of inference. If replications > 1, only save imputation results",
    )
    return parser.parse_args()


def run_job(
    config: dict,
    device: Optional[Union[torch.device, str]],
    ckpt_iter: Union[str, int],
    trials: int,
) -> None:
    batch_size = config["common"]["inference_batch_size"]
    dataloader = get_dataloader(
        config["data"]["test_data_path"],
        batch_size,
        device=device,
        num_workers=config["common"]["num_workers"],
    )

    local_path = MODEL_PATH_FORMAT.format(
        T=config["diffusion"]["T"],
        beta_0=config["diffusion"]["beta_0"],
        beta_T=config["diffusion"]["beta_T"],
    )

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **config["diffusion"], device=device
    )
    LOGGER.info(display_current_time())
    net = setup_model(config, device)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 0:
        net = nn.DataParallel(net)

    data_names = ["imputation", "original", "mask"]
    directory = config["generation"]["output_directory"]

    if trials > 1:
        directory += "_{trial}"

    for trial in range(1, trials + 1):
        LOGGER.info(f"The {trial}th inference trial")
        saved_data_names = data_names if trial == 0 else data_names[0]

        mse = DiffusionGenerator(
            net=net,
            device=device,
            diffusion_hyperparams=diffusion_hyperparams,
            dataloader=dataloader,
            local_path=local_path,
            output_directory=directory.format(trial=trial) if trials > 1 else directory,
            ckpt_path=config["generation"]["ckpt_path"],
            ckpt_iter=ckpt_iter,
            batch_size=batch_size,
            masking=config["training"]["masking"],
            missing_k=config["training"]["missing_k"],
            only_generate_missing=config["training"]["only_generate_missing"],
            saved_data_names=saved_data_names,
        ).generate()

        LOGGER.info(f"Average MSE: {sum(mse) / len(mse)}")
        LOGGER.info(display_current_time())


if __name__ == "__main__":

    args = fetch_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 0:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse configs
    with open(args.config) as f:
        config = json.load(f)

    run_job(config, device, args.ckpt_iter, args.trials)
