import torch

from sssd.core.imputers.DiffWaveImputer import DiffWaveImputer
from sssd.core.imputers.SSSDS4Imputer import SSSDS4Imputer
from sssd.core.imputers.SSSDSAImputer import SSSDSAImputer
from sssd.core.utils import get_mask_bm, get_mask_forecast, get_mask_mnr, get_mask_rm

MASK_FN = {
    "rm": get_mask_rm,
    "mnr": get_mask_mnr,
    "bm": get_mask_bm,
    "forecast": get_mask_forecast,
}

MODELS = {0: DiffWaveImputer, 1: SSSDSAImputer, 2: SSSDS4Imputer}
MODEL_PATH_FORMAT = "T{T}_beta0{beta_0}_betaT{beta_T}"


def setup_model(
    use_model: int, model_config: dict, device: torch.device
) -> torch.nn.Module:
    model_settings = None
    if use_model in (0, 2):
        model_settings = model_config.get("wavenet")
    elif use_model == 1:
        model_settings = model_config.get("sashimi_config")
    else:
        raise KeyError(f"Please enter correct use-model number, but got {use_model}")
    if model_settings is None:
        raise ValueError(f"Please enter model settings in config")
    return MODELS[use_model](**model_settings, device=device).to(device)
