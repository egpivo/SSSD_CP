import random

import numpy as np
import torch


def load_and_split_data(
    training_data_load: np.ndarray,
    batch_num: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    index = random.sample(range(training_data_load.shape[0]), batch_num * batch_size)
    training_data = training_data_load[index]
    training_data = np.split(training_data, batch_num, 0)
    training_data = np.array(training_data)
    return torch.from_numpy(training_data).to(device, dtype=torch.float32)
