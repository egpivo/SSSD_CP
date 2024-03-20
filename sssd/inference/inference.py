import argparse
import datetime
import json
import os
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

from sssd.core.model_specs import MASK_FN, setup_model
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
    output_directory,
    num_samples,
    ckpt_path,
    data_path,
    ckpt_iter,
    use_model,
    masking,
    missing_k,
    only_generate_missing,
):

    """
    Generate data based on ground truth

    Parameters:
    output_directory (str):           save generated speeches to this path
    num_samples (int):                number of samples to generate, default is 4
    ckpt_path (str):                  checkpoint path
    ckpt_iter (int or 'max'):         the pretrained checkpoint to be loaded;
                                      automitically selects the maximum iteration if 'max' is selected
    data_path (str):                  path to NYISO, numpy array.
    use_model (int):                  0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    masking (str):                    'mnr': missing not at random, 'bm': black-out, 'rm': random missing
    only_generate_missing (int):      0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    missing_k (int)                   k missing time points for each channel across the length.
    """

    # generate experiment (local) path
    local_path = "T{}_beta0{}_betaT{}".format(
        diffusion_config["T"], diffusion_config["beta_0"], diffusion_config["beta_T"]
    )

    print_size(net)

    # load checkpoint
    ckpt_path = os.path.join(ckpt_path, local_path)
    if ckpt_iter == "max":
        ckpt_iter = find_max_epoch(ckpt_path)
    model_path = os.path.join(ckpt_path, "{}.pkl".format(ckpt_iter))
    try:
        checkpoint = torch.load(model_path, map_location="cpu")
        print("checkpoint")
        # net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        net.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded model at iteration {}".format(ckpt_iter))
    except:
        raise Exception("No valid model found")

    # Get shared output_directory ready
    output_directory = os.path.join(
        output_directory,
        local_path,
        "imputaiton_multiple_" + str(round(int(ckpt_iter) / 1000)) + "k",
    )
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
        os.chmod(output_directory, 0o775)
    print("output directory", output_directory, flush=True)

    ### Custom data loading and reshaping ###

    testing_data = np.load(trainset_config["test_data_path"])
    testing_data = np.split(
        testing_data, testing_data.shape[0] / num_samples, 0
    )  # (data, 分4組, 0) batch size = num_samples = 272 為了創造 batch，除不盡可用 np.array_split
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
            "generated {} utterances of random_digit at iteration {} in {} seconds".format(
                num_samples, ckpt_iter, int(start.elapsed_time(end) / 1000)
            )
        )

        generated_audio = generated_audio.detach().cpu().numpy()
        batch = batch.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()

        outfile = f"imputation{i}.npy"
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, generated_audio)

        outfile = f"original{i}.npy"
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, batch)

        outfile = f"mask{i}.npy"
        new_out = os.path.join(output_directory, outfile)
        np.save(new_out, mask)

        print("saved generated samples at iteration %s" % ckpt_iter)

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
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=1,
        help="Number of utterances to be generated",
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

    gen_config = config["gen_config"]
    train_config = config["train_config"]
    trainset_config = config["trainset_config"]
    diffusion_config = config["diffusion_config"]

    diffusion_hyperparams = calc_diffusion_hyperparams(
        **diffusion_config, device=device
    )

    current_time = datetime.datetime.now()
    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))

    net = setup_model(config, device)

    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 0:
        print("Using ", torch.cuda.device_count(), " GPUs!")
        net = nn.DataParallel(net)

    generate(
        net,
        device,
        diffusion_hyperparams,
        **gen_config,
        ckpt_iter=args.ckpt_iter,
        num_samples=args.num_samples,
        use_model=train_config["use_model"],
        data_path=trainset_config["test_data_path"],
        masking=train_config["masking"],
        missing_k=train_config["missing_k"],
        only_generate_missing=train_config["only_generate_missing"],
    )

    current_time = datetime.datetime.now()
    print("Current time:", current_time.strftime("%Y-%m-%d %H:%M:%S"))
