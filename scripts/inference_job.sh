#!/bin/bash
#
# Execute a diffusion process - inference
#
# Parameters:
#    - Mandatory:
#        - -c/--config: configuration path
#    - Optional:
#        - -u/--update_conda_env: flag to update Conda environment
#
# Examples:
#    - Execute the script: ./inference_job.sh -m configs/model.yaml - t configs/inference.yaml
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${DIR}/.."
source "${DIR}/utils.sh"
source "${PACKAGE_BASE_PATH}/bin/color_map.sh"
source "${PACKAGE_BASE_PATH}/bin/exit_code.sh"
CONDA_ENV="sssd"

MODEL_CONFIG=""
INFERENCE_CONFIG=""
DOES_UPDATE_CONDA_ENV="false"  # Default value for updating Conda environment

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model_config)
      MODEL_CONFIG="$2"
      shift
      ;;
    -i|--inference_config)
      INFERENCE_CONFIG="$2"
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

INFERENCE_JOB_COMMANDS=(
  "${DIR}/infer.py"
  --model_config "${MODEL_CONFIG}"
  --inference_config "${INFERENCE_CONFIG}"
)

# Initialize Conda environment if specified
update_conda_environment ${PACKAGE_BASE_PATH} ${DOES_UPDATE_CONDA_ENV} ${CONDA_ENV}

# Execute training
echo -e "${FG_YELLOW}[Execution - Training]${FG_RESET}"
echo -e "${FG_GREEN}${TRAINING_JOB_COMMANDS[*]}${FG_RESET}"
python "${TRAINING_JOB_COMMANDS[@]}"

# Execute inference
echo -e "${FG_YELLOW}[Execution - Inference]${FG_RESET}"
echo -e "${FG_GREEN}${INFERENCE_JOB_COMMANDS[*]}${FG_RESET}"
python "${INFERENCE_JOB_COMMANDS[@]}"

echo -e "${FG_GREEN}Job completed${FG_RESET}"
