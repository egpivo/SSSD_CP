from sssd.imputers.DiffWaveImputer import DiffWaveImputer
from sssd.imputers.SSSDS4Imputer import SSSDS4Imputer
from sssd.imputers.SSSDSAImputer import SSSDSAImputer
from sssd.utils.util import get_mask_bm, get_mask_forecast, get_mask_mnr, get_mask_rm

MASK_FN = {
    "rm": get_mask_rm,
    "mnr": get_mask_mnr,
    "bm": get_mask_bm,
    "forecast": get_mask_forecast,
}

MODELS = {0: DiffWaveImputer, 1: SSSDSAImputer, 2: SSSDS4Imputer}
