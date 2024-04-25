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
#    - Execute the script: ./training_job.sh -m configs/model.yaml -t configs/training.yaml
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${DIR}/.."

MODEL_CONFIG=""
TRAINING_CONFIG=""
DOES_UPDATE_CONDA_ENV="false"
CONDA_ENV="sssd"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model_config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    -t|--training_config)
      TRAINING_CONFIG="$2"
      shift 2
      ;;
    -u|--update_conda_env)
      DOES_UPDATE_CONDA_ENV="true"
      shift
      ;;
    --|*)
      shift
      break
      ;;
  esac
done

# Source utility functions
if [[ -f "${DIR}/utils.sh" ]]; then
  source "${DIR}/utils.sh"
else
  echo "Error: utils.sh not found" >&2
  exit "${ERROR_EXITCODE}"
fi

if [[ -f "${PACKAGE_BASE_PATH}/bin/color_map.sh" ]]; then
  source "${PACKAGE_BASE_PATH}/bin/color_map.sh"
else
  echo "Error: color_map.sh not found" >&2
  exit "${ERROR_EXITCODE}"
fi

if [[ -f "${PACKAGE_BASE_PATH}/bin/exit_code.sh" ]]; then
  source "${PACKAGE_BASE_PATH}/bin/exit_code.sh"
else
  echo "Error: exit_code.sh not found" >&2
  exit "${ERROR_EXITCODE}"
fi

# Validate mandatory parameters
if [[ -z "${MODEL_CONFIG}" ]]; then
  echo "Error: Model configuration file path is required."
  exit "${ERROR_EXITCODE}"
fi

if [[ -z "${TRAINING_CONFIG}" ]]; then
  echo "Error: Training configuration file path is required."
  exit "${ERROR_EXITCODE}"
fi

# Check if the configuration files exist
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Error: Model configuration file '${MODEL_CONFIG}' not found."
  exit "${ERROR_EXITCODE}"
fi

if [[ ! -f "${TRAINING_CONFIG}" ]]; then
  echo "Error: Training configuration file '${TRAINING_CONFIG}' not found."
  exit "${ERROR_EXITCODE}"
fi

# Initialize Conda environment if specified
update_conda_environment "${PACKAGE_BASE_PATH}" "${DOES_UPDATE_CONDA_ENV}" "${CONDA_ENV}"

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
