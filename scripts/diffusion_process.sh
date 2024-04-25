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
#    - Execute whole process: ./training_job.sh -m configs/model.yaml -t configs/training.yaml -i configs/inference.yaml
#    - Execute only training process: ./training_job.sh -m configs/model.yaml -t configs/training.yaml
#    - Execute only inference process: ./training_job.sh -m configs/model.yaml -i configs/inference.yaml
#
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
INFERENCE_CONFIG=""
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
    -i|--inference_config)
      INFERENCE_CONFIG="$2"
      shift
      ;;
    -u|--update_conda_env)
      DOES_UPDATE_CONDA_ENV="true"
      shift
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

# Initialize Conda environment if specified
update_conda_environment ${PACKAGE_BASE_PATH} ${DOES_UPDATE_CONDA_ENV} ${CONDA_ENV}

# Execute training if the training config exists
if [[ -n "${TRAINING_CONFIG}" ]]; then
  . ${DIR}/training_job.sh -m ${MODEL_CONFIG} -t ${TRAINING_CONFIG}
fi

# Execute inference if the inference config exists
if [[ -n "${INFERENCE_CONFIG}" ]]; then
  . ${DIR}/inference_job.sh -m ${MODEL_CONFIG} -i ${INFERENCE_CONFIG}
fi
