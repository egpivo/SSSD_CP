#!/bin/bash
#
# Execute a diffusion process - either training or inference
#
# Parameters:
#    - Mandatory:
#        - -m/--model_config: model configuration path
#    - Optional:
#        - -t/--training_config: training configuration path
#        - -i/--inference_config: inference configuration path
#        - -u/--update_conda_env: flag to update Conda environment
#
# Examples:
#    - Execute whole process: ./diffusion_process.sh -m configs/model.yaml -t configs/training.yaml -i configs/inference.yaml
#    - Execute only training process: ./diffusion_process.sh -m configs/model.yaml -t configs/training.yaml
#    - Execute only inference process: ./diffusion_process.sh -m configs/model.yaml -i configs/inference.yaml
#

DIFFUSION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${DIFFUSION_DIR}/../.."

MODEL_CONFIG="${PACKAGE_BASE_PATH}/configs/model.yaml"
TRAINING_CONFIG="${PACKAGE_BASE_PATH}/configs/training.yaml"
INFERENCE_CONFIG="${PACKAGE_BASE_PATH}/configs/inference.yaml"
DOES_UPDATE_CONDA_ENV="false"

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
if [[ -f "${PACKAGE_BASE_PATH}/bin/utils.sh" ]]; then
  source "${PACKAGE_BASE_PATH}/bin/utils.sh"
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

if [[ -f "${PACKAGE_BASE_PATH}/envs/conda/utils.sh" ]]; then
  source "${PACKAGE_BASE_PATH}/envs/conda/utils.sh"
else
  echo "Error: CONDA utils.sh not found" >&2
  exit "${ERROR_EXITCODE}"
fi

CONDA_ENV="sssd"

set -euo pipefail

# Validate mandatory parameter
check_file_exists "${MODEL_CONFIG}"


# Initialize Conda environment if specified
if [ x"${DOES_UPDATE_CONDA_ENV}x" == "xtruex" ]; then
  update_conda_environment "${PACKAGE_BASE_PATH}" "${CONDA_ENV}"
fi
# Execute training if the training config exists and the file exists
if [[ -n "${TRAINING_CONFIG}" && -f "${TRAINING_CONFIG}" ]]; then
  . "${DIFFUSION_DIR}/training_job.sh" -m "${MODEL_CONFIG}" -t "${TRAINING_CONFIG}"
fi

# Execute inference if the inference config exists and the file exists
if [[ -n "${INFERENCE_CONFIG}" && -f "${INFERENCE_CONFIG}" ]]; then
  . "${DIFFUSION_DIR}/inference_job.sh" -m "${MODEL_CONFIG}" -i "${INFERENCE_CONFIG}"
fi
