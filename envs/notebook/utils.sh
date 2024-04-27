#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/../../bin/exit_code.sh"
source "${DIR}/../../bin/color_map.sh"

update_conda_env_path() {
  # Will return `CONDA_ENV_DIR`
  local ENV_NAME=$1

  source $(conda info --base)/etc/profile.d/conda.sh
  IFS=' ' read -r -a CONDA_INFO <<<"$(conda env list | grep "${ENV_NAME}")"
  AVAILABLE_ENV_NAME="${CONDA_INFO[0]}"

  if [[ "x${AVAILABLE_ENV_NAME}x" != "x${ENV_NAME}x" ]]; then
    echo -e "${FG_RED}Conda env '${ENV_NAME}' is not available${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

  . ${DIR}/../envs/conda/build_conda_env.sh -c "${ENV_NAME}"
  source activate "${ENV_NAME}"
  poetry install ---with notebook
  conda deactivate
}

is_jupyter_kernel_path_available() {
  # Will return `KERNEL_DIR`
  KERNEL_NAME=$1

  IFS=' ' read -r -a KERNEL_INFO <<<"$(jupyter kernelspec list | grep "${KERNEL_NAME}")"
  AVAILABLE_KERNEL="${KERNEL_INFO[0]}"

  if [[ "x${AVAILABLE_KERNEL}x" != "x${KERNEL_NAME}x" ]]; then
    echo -e "${FG_RED} '${KERNEL_NAME}' is not available${FG_RESET}"
    return ${ERROR_EXITCODE}
  else
    return ${SUCCESS_EXITCODE}
  fi
}

set_jupyter_kernel_path() {
  # Will return `KERNEL_DIR`
  KERNEL_NAME=$1

  is_jupyter_kernel_path_available "${KERNEL_NAME}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    echo -e "${FG_RED} Install kernel '${KERNEL_NAME}' now ${FG_RESET}"
    ipython kernel install --name "${KERNEL_NAME}" --user
  fi

  # Exclude the pattern `python3` to avoid fetching the wrong path
  IFS=' ' read -r -a KERNEL_INFO <<<"$(jupyter kernelspec list | grep "${KERNEL_NAME}" | grep -v "python3")"

  if [ "x${KERNEL_INFO[1]}x" == "x*x" ]; then
    KERNEL_DIR="${KERNEL_INFO[2]}"
  else
    KERNEL_DIR="${KERNEL_INFO[1]}"
  fi
}

update_gpu_env() {
  # This function is used before making connection between Tensorflow and GPUs

  local CONDA_ENV="$1"
  # will update `CONDA_ENV_DIR`
  find_conda_env_path "${CONDA_ENV}"
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"${CONDA_ENV_DIR}/lib"

  export CUDA_HOME=/usr/local/cuda
  export PATH=${CUDA_HOME}/bin:${PATH}
  export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
}
