import torch

from sssd.core.imputers.DiffWaveImputer import DiffWaveImputer
from sssd.core.imputers.SSSDS4Imputer import SSSDS4Imputer
from sssd.core.imputers.SSSDSAImputer import SSSDSAImputer
from sssd.utils.util import get_mask_bm, get_mask_forecast, get_mask_mnr, get_mask_rm

MASK_FN = {
    "rm": get_mask_rm,
    "mnr": get_mask_mnr,
    "bm": get_mask_bm,
    "forecast": get_mask_forecast,
}

MODELS = {0: DiffWaveImputer, 1: SSSDSAImputer, 2: SSSDS4Imputer}
MODEL_PATH_FORMAT = "T{T}_beta0{beta_0}_betaT{beta_T}"


def setup_model(config: dict, device: torch.device) -> torch.nn.Module:
    use_model = config["train_config"]["use_model"]
    if use_model in (0, 2):
        model_config = config["wavenet_config"]
    elif use_model == 1:
        model_config = config["sashimi_config"]
    else:
        raise KeyError(f"Please enter correct model number, but got {use_model}")
    return MODELS[use_model](**model_config, device=device).to(device)
