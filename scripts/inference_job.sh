#!/bin/bash
#
# Execute a diffusion process - inference
#
# Parameters:
#    - Mandatory:
#        - -m/--model_config: model configuration path
#        - -i/--inference_config: inference configuration path
#    - Optional:
#        - -u/--update_conda_env: flag to update Conda environment
#
# Examples:
#    - Execute the script: ./inference_job.sh -m configs/model.yaml -i configs/inference.yaml
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${DIR}/.."

MODEL_CONFIG=""
INFERENCE_CONFIG=""
DOES_UPDATE_CONDA_ENV="false"
CONDA_ENV="sssd"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model_config)
      MODEL_CONFIG="$2"
      shift 2
      ;;
    -i|--inference_config)
      INFERENCE_CONFIG="$2"
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


# Check if the configuration files exist
check_file_exists "${MODEL_CONFIG}"
check_file_exists "${INFERENCE_CONFIG}"

# Initialize Conda environment if specified
if [ x"${DOES_UPDATE_CONDA_ENV}x" == "xtruex" ]; then
  update_conda_environment "${PACKAGE_BASE_PATH}" "${CONDA_ENV}"
fi
activate_conda_environment "${CONDA_ENV}"

# Define inference job commands
INFERENCE_JOB_COMMANDS=(
  "${DIR}/infer.py"
  --model_config "${MODEL_CONFIG}"
  --inference_config "${INFERENCE_CONFIG}"
)

# Execute inference
echo -e "${FG_YELLOW}[Execution - Inference]${FG_RESET}"
echo -e "${FG_GREEN}${INFERENCE_JOB_COMMANDS[*]}${FG_RESET}"
python "${INFERENCE_JOB_COMMANDS[@]}"

echo -e "${FG_GREEN}Inference Job completed${FG_RESET}"
