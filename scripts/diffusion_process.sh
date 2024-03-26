#!/bin/bash
#
# Execute a diffusion process by training and generation
#
# - Parameters
#    - Mandatory
#       - -c/--config: configuration path
#    - Optional
#       - -b/--batch_size: batch size
#       - -n/--num_samples: number of samples
#
# - Examples
#       - Execute the script: ./diffusion_process.sh -c config/config_SSSDS4-NYISO-3-mix.json
#       - Execute the script with batch size and number of samples: ./diffusion_process.sh -c config/config_SSSDS4-NYISO-3-mix.json -b 32 -n 1000
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_BASE_PATH="${DIR}/.."
source "${PACKAGE_BASE_PATH}/bin/color_map.sh"
source "${PACKAGE_BASE_PATH}/bin/exit_code.sh"
CONDA_ENV="sssd"

CONFIG=""
BATCH_SIZE=""
NUM_SAMPLES=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG="$2"
      shift
      ;;
    -b|--batch_size)
      BATCH_SIZE="$2"
      shift
      ;;
    -n|--num_samples)
      NUM_SAMPLES="$2"
      shift
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
if [[ -n "$BATCH_SIZE" ]]; then
  TRAINING_JOB_COMMANDS+=(--batch_size "$BATCH_SIZE")
fi

INFERENCE_JOB_COMMANDS=(
  "${DIR}/infer.py"
  --config "${CONFIG}"
)
if [[ -n "$NUM_SAMPLES" ]]; then
  INFERENCE_JOB_COMMANDS+=(--batch_size "$NUM_SAMPLES")
fi

# Initialize Conda environment
if [ -x "$(command -v conda)" ]; then
    echo "Conda is installed."
    . "${PACKAGE_BASE_PATH}/envs/conda/build_conda_env.sh" --conda_env ${CONDA_ENV}
  source activate ${CONDA_ENV}
else
    echo -e "${FG_RED}"Conda is not installed."${FG_RESET}"
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
