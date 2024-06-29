import argparse
from typing import Optional, Union

import torch
import torch.nn as nn
import yaml

from sssd.core.model_specs import MODEL_PATH_FORMAT, setup_model
from sssd.data.utils import get_dataloader
from sssd.inference.generator import DiffusionGenerator
from sssd.utils.logger import setup_logger
from sssd.utils.utils import calc_diffusion_hyperparams, display_current_time

LOGGER = setup_logger()


def fetch_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_config",
        type=str,
        default="configs/model.yaml",
        help="Model configuration",
    )
    parser.add_argument(
        "-i",
        "--inference_config",
        type=str,
        default="configs/inference_config.yaml",
        help="Inference configuration",
    )
    parser.add_argument(
        "-ckpt_iter",
        "--ckpt_iter",
        default="max",
        help='Which checkpoint to use; assign a number or "max" to find the latest checkpoint',
    )
    return parser.parse_args()


def run_job(
    model_config: dict,
    inference_config: dict,
    device: Optional[Union[torch.device, str]],
    ckpt_iter: Union[str, int],
) -> None:
    trials = inference_config.get("trials")
    batch_size = inference_config["batch_size"]
    dataloader = get_dataloader(
        inference_config["data"]["test_path"],
        batch_size,
        device=device,
    )

    local_path = MODEL_PATH_FORMAT.format(
        T=model_config["diffusion"]["T"],
        beta_0=model_config["diffusion"]["beta_0"],
        beta_T=model_config["diffusion"]["beta_T"],
    )

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **model_config["diffusion"], device=device
    )
    LOGGER.info(display_current_time())
    net = setup_model(inference_config["use_model"], model_config, device)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 0:
        net = nn.DataParallel(net)

    data_names = ["imputation", "original", "mask"]
    directory = inference_config["output_directory"]

    if trials > 1:
        directory += "_{trial}"

    for trial in range(1, trials + 1):
        LOGGER.info(f"The {trial}th inference trial")
        saved_data_names = data_names if trial == 0 else data_names[0]

        mses, mapes = DiffusionGenerator(
            net=net,
            device=device,
            diffusion_hyperparams=diffusion_hyperparams,
            dataloader=dataloader,
            local_path=local_path,
            output_directory=directory.format(trial=trial) if trials > 1 else directory,
            ckpt_path=inference_config["ckpt_path"],
            ckpt_iter=ckpt_iter,
            batch_size=batch_size,
            masking=inference_config["masking"],
            missing_k=inference_config["missing_k"],
            only_generate_missing=inference_config["only_generate_missing"],
            saved_data_names=saved_data_names,
        ).generate()

        LOGGER.info(f"Average MSE: {sum(mses) / len(mses)}")
        LOGGER.info(f"Average MAPE: {sum(mapes) / len(mapes)}")
        LOGGER.info(display_current_time())


if __name__ == "__main__":
    args = fetch_args()

    with open(args.model_config, "rt") as f:
        model_config = yaml.safe_load(f.read())
    with open(args.inference_config, "rt") as f:
        inference_config = yaml.safe_load(f.read())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 0:
        LOGGER.info(f"Using {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_job(model_config, inference_config, device, args.ckpt_iter)
