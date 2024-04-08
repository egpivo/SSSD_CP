import argparse
import json
from typing import Union

import torch
import torch.nn as nn

from sssd.core.model_specs import MODEL_PATH_FORMAT, setup_model
from sssd.data.utils import load_testing_data
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
    device: torch.device,
    num_samples: int,
    ckpt_iter: Union[str, int],
    trials: int,
) -> None:

    testing_data = load_testing_data(
        config["trainset_config"]["test_data_path"], args.num_samples
    )
    local_path = MODEL_PATH_FORMAT.format(
        T=config["diffusion_config"]["T"],
        beta_0=config["diffusion_config"]["beta_0"],
        beta_T=config["diffusion_config"]["beta_T"],
    )

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **config["diffusion_config"], device=device
    )
    LOGGER.info(display_current_time())
    net = setup_model(config, device)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 0:
        net = nn.DataParallel(net)

    data_names = ["imputation", "original", "mask"]

    if trials > 1:
        directory = 'config["gen_config"]["output_directory"]_{trial}'
    else:
        directory = 'config["gen_config"]["output_directory"]'

    for trial in range(trials):
        saved_data_names = data_names if trial == 0 else data_names[0]

        mse = DiffusionGenerator(
            net=net,
            device=device,
            diffusion_hyperparams=diffusion_hyperparams,
            testing_data=testing_data,
            local_path=local_path,
            output_directory=directory.format(trial=trial) if trials > 1 else directory,
            ckpt_path=config["gen_config"]["ckpt_path"],
            ckpt_iter=ckpt_iter,
            num_samples=num_samples,
            masking=config["train_config"]["masking"],
            missing_k=config["train_config"]["missing_k"],
            only_generate_missing=config["train_config"]["only_generate_missing"],
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

    run_job(config, device, args.num_samples, args.ckpt_iter, args.trials)
