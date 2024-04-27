#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/../../bin/exit_code.sh"
source "${DIR}/../../bin/color_map.sh"

update_conda_env_path() {
  local ENV_NAME=$1
  local CONDA_BASE=$(conda info --base)
  source "${CONDA_BASE}/etc/profile.d/conda.sh"

  # Check if the environment exists
  if ! conda env list | grep -q "^${ENV_NAME}"; then
    echo -e "${FG_RED}Conda env '${ENV_NAME}' is not available${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

  # Activate the environment
  conda activate "${ENV_NAME}"

  # Install dependencies with Poetry
  . "${DIR}/../conda/build_conda_env.sh" -c "${ENV_NAME}"
  poetry install --extras notebook --no-root

  # Deactivate the environment
  conda deactivate
}

check_kernel_availability() {
  local KERNEL_NAME=$1
  local KERNEL_PATH=$(jupyter kernelspec list | grep -o "^${KERNEL_NAME} .*" | cut -d' ' -f2)

  if [ -z "${KERNEL_PATH}" ]; then
    echo -e "${FG_RED}Kernel '${KERNEL_NAME}' is not available${FG_RESET}"
    return "${ERROR_EXITCODE}"
  else
    return "${SUCCESS_EXITCODE}"
  fi
}

set_jupyter_kernel_path() {
  local KERNEL_NAME=$1
  local KERNEL_PATH

  check_kernel_availability "${KERNEL_NAME}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    echo -e "${FG_RED}Installing kernel '${KERNEL_NAME}'${FG_RESET}"
    ipython kernel install --name "${KERNEL_NAME}" --user
  fi

  KERNEL_PATH=$(jupyter kernelspec list | grep -o "^${KERNEL_NAME} .*" | cut -d' ' -f2)

  if [ -n "${KERNEL_PATH}" ]; then
    KERNEL_DIR="${KERNEL_PATH}"
  else
    echo -e "${FG_RED}Failed to get kernel path for '${KERNEL_NAME}'${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi
}

update_gpu_env() {
  local CONDA_ENV="$1"
  local CONDA_ENV_DIR

  # Find the path to the Conda environment
  find_conda_env_path "${CONDA_ENV}"
  CONDA_ENV_DIR="${CONDA_ENV_PATH}"

  # Update LD_LIBRARY_PATH for the Conda environment
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_ENV_DIR}/lib"

  # Set CUDA environment variables
  local CUDA_HOME="/usr/local/cuda"
  export CUDA_HOME
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
}
