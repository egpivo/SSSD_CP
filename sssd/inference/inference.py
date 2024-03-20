import argparse
import datetime
import json
import os
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from sssd.core.model_specs import MASK_FN, MODEL_PATH_FORMAT, setup_model
from sssd.utils.util import (
    calc_diffusion_hyperparams,
    find_max_epoch,
    print_size,
    sampling,
)


def generate(
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
):

    """
    Generate data based on ground truth.

    Parameters:
    - net (torch.nn.Module): The neural network model.
    - device (torch.device): The device to run the model on (e.g., "cuda" or "cpu").
    - diffusion_hyperparams (dict): Dictionary of diffusion hyperparameters.
    - local_path (str): Local path format for the model.
    - testing_data (numpy.ndarray): Numpy array containing testing data.
    - output_directory (str): Path to save generated samples.
    - num_samples (int): Number of samples to generate (default is 4).
    - ckpt_path (str): Checkpoint directory.
    - ckpt_iter (int or 'max'): Pretrained checkpoint to load; 'max' selects the maximum iteration.
    - masking (str): Type of masking: 'mnr' (missing not at random), 'bm' (black-out), 'rm' (random missing).
    - only_generate_missing (int): Whether to generate only missing portions of the signal:
                                    0 (all sample diffusion), 1 (generate missing portions only).
    - missing_k (int): Number of missing time points for each channel across the length.
    """

    # Print network size
    print_size(net)

    # Load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, f"{ckpt_iter}.pkl")
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print("Checkpoint loaded")
        net.load_state_dict(checkpoint["model_state_dict"])
        print(f"Successfully loaded model at iteration {ckpt_iter}")
    except:
        raise Exception("No valid model found")

    # Create output directory
    output_directory = os.path.join(
        output_directory,
        local_path,
        f"imputation_multiple_{int(ckpt_iter) // 1000}k",
    )
    os.makedirs(output_directory, exist_ok=True)
    os.chmod(output_directory, 0o775)
    print("Output directory:", output_directory)

    # Load testing data
    testing_data = np.split(testing_data, testing_data.shape[0] / num_samples, 0)
    testing_data = np.array(testing_data)
    testing_data = torch.from_numpy(testing_data).float().cuda()
    print("Data loaded")

    all_mse = []

    for i, batch in enumerate(testing_data):
        transposed_mask = MASK_FN[masking](batch[0], missing_k)
        mask = (
            transposed_mask.permute(1, 0)
            .repeat(batch.size()[0], 1, 1)
            .to(device, dtype=torch.float32)
        )

        batch = batch.permute(0, 2, 1)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        sample_length = batch.size(2)
        sample_channels = batch.size(1)
        generated_audio = sampling(
            net,
            (num_samples, sample_channels, sample_length),
            diffusion_hyperparams,
            cond=batch,
            mask=mask,
            only_generate_missing=only_generate_missing,
        )

        end.record()
        torch.cuda.synchronize()

        print(
            f"Generated {num_samples} utterances at iteration {ckpt_iter} in {int(start.elapsed_time(end) / 1000)} seconds"
        )

        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        outfile = f"imputation{i}.npy"
        np.save(os.path.join(output_directory, outfile), generated_audio)

        outfile = f"original{i}.npy"
        np.save(os.path.join(output_directory, outfile), batch)

        outfile = f"mask{i}.npy"
        np.save(os.path.join(output_directory, outfile), mask)

        print(f"Saved generated samples at iteration {ckpt_iter}")

        mse = mean_squared_error(
            generated_audio[~mask.astype(bool)], batch[~mask.astype(bool)]
        )
        all_mse.append(mse)

    print("Total MSE:", mean(all_mse))


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

    generate(
        net,
        device,
        diffusion_hyperparams,
        testing_data=testing_data,
        local_path=local_path,
        output_directory=config["gen_config"]["output_directory"],
        ckpt_path=config["gen_config"]["ckpt_path"],
        ckpt_iter=args.ckpt_iter,
        num_samples=args.num_samples,
        masking=config["train_config"]["masking"],
        missing_k=config["train_config"]["missing_k"],
        only_generate_missing=config["train_config"]["only_generate_missing"],
    )

    current_time = datetime.datetime.now()
    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
