#!/bin/bash
#
# Execute a diffusion process - training
#
# Parameters:
#    - Mandatory:
#        - -m/--model_config: model configuration path
#        - -t/--training_config: training configuration path
#    - Optional:
#        - -u/--update_conda_env: flag to update Conda environment
#
# Examples:
#    - Execute the script: ./training_job.sh -m configs/model.yaml - t configs/training.yaml
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${DIR}/.."
source "${DIR}/utils.sh"
source "${PACKAGE_BASE_PATH}/bin/color_map.sh"
source "${PACKAGE_BASE_PATH}/bin/exit_code.sh"
CONDA_ENV="sssd"

MODEL_CONFIG=""
TRAINING_CONFIG=""
DOES_UPDATE_CONDA_ENV="false"  # Default value for updating Conda environment

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model_config)
      MODEL_CONFIG="$2"
      shift
      ;;
    -t|--training_config)
      TRAINING_CONFIG="$2"
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
if [[ -z "${MODEL_CONFIG}" ]]; then
  echo "Error: MODEL Configuration file path is required."
  exit "${ERROR_EXITCODE}"
fi
if [[ -z "${TRAINING_CONFIG}" ]]; then
  echo "Error: Training Configuration file path is required."
  exit "${ERROR_EXITCODE}"
fi

# Initialize Conda environment if specified
update_conda_environment ${PACKAGE_BASE_PATH} ${DOES_UPDATE_CONDA_ENV} ${CONDA_ENV}

# Define training job commands
TRAINING_JOB_COMMANDS=(
  "${DIR}/train.py"
  --model_config "${MODEL_CONFIG}"
  --training_config "${TRAINING_CONFIG}"
)

# Execute training
echo -e "${FG_YELLOW}[Execution - Training]${FG_RESET}"
echo -e "${FG_GREEN}${TRAINING_JOB_COMMANDS[*]}${FG_RESET}"
python "${TRAINING_JOB_COMMANDS[@]}"

echo -e "${FG_GREEN}Training Job completed${FG_RESET}"
