#!/bin/bash
#
# Execute a diffusion process by training and generation
#
# Parameters:
#    - Mandatory:
#        - -c/--config: configuration path
#    - Optional:
#        - -t/--trials: number of trials in the inference stage
#        - -u/--update_conda_env: flag to update Conda environment
#
# Examples:
#    - Execute the script: ./diffusion_process.sh -c configs/config_SSSDS4-NYISO-3-mix.json
#    - Execute the script with 2 trials: ./diffusion_process.sh -c configs/config_SSSDS4-NYISO-3-mix.json -t 2
#    - Execute the script with Conda environment update: ./diffusion_process.sh -c configs/config_SSSDS4-NYISO-3-mix.json -u
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${DIR}/.."
source "${PACKAGE_BASE_PATH}/bin/color_map.sh"
source "${PACKAGE_BASE_PATH}/bin/exit_code.sh"
CONDA_ENV="sssd"

CONFIG=""
TRIALS=1
DOES_UPDATE_CONDA_ENV="false"  # Default value for updating Conda environment

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG="$2"
      shift
      ;;
    -t|--trials)
      TRIALS="$2"
      shift
      ;;
    -u|--update_conda_env)
      DOES_UPDATE_CONDA_ENV="true"
      ;;
    *)
      ;;
  esac
  shift
done

# Validate mandatory parameter
if [[ -z "$CONFIG" ]]; then
  echo "Error: Configuration file path is required."
  exit "${ERROR_EXITCODE}"
fi

# Define training and inference job commands
TRAINING_JOB_COMMANDS=(
  "${DIR}/train.py"
  --config "${CONFIG}"
)

INFERENCE_JOB_COMMANDS=(
  "${DIR}/infer.py"
  --config "${CONFIG}"
  --trials "${TRIALS}"
)

# Initialize Conda environment if specified
if [ -x "$(command -v conda)" ]; then
  if [[ "$DOES_UPDATE_CONDA_ENV" == "true" ]]; then
    echo -e "${FG_YELLOW}Updating Conda environment - sssd${FG_RESET}"
    bash "${PACKAGE_BASE_PATH}/envs/conda/build_conda_env.sh" --conda_env ${CONDA_ENV}
  else
    echo -e "${FG_GREEN}Conda environment update is not requested.${FG_RESET}"
  fi
    source activate ${CONDA_ENV}
else
  echo -e "${FG_RED}Conda is not installed.${FG_RESET}"
fi

# Execute training
echo -e "${FG_YELLOW}[Execution - Training]${FG_RESET}"
echo -e "${FG_GREEN}${TRAINING_JOB_COMMANDS[*]}${FG_RESET}"
python "${TRAINING_JOB_COMMANDS[@]}"

# Execute inference
echo -e "${FG_YELLOW}[Execution - Inference]${FG_RESET}"
echo -e "${FG_GREEN}${INFERENCE_JOB_COMMANDS[*]}${FG_RESET}"
python "${INFERENCE_JOB_COMMANDS[@]}"

echo -e "${FG_GREEN}Job completed${FG_RESET}"
