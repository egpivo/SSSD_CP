#!/bin/bash
#
# Helper functions for building a conda environment
#
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_DIR}/conda_env_info.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"

initialize_conda() {
  source $(conda info --base)/etc/profile.d/conda.sh
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    source "/opt/miniconda2/etc/profile.d/conda.sh"
  fi
}

find_conda_env_path() {
  # Will return `CONDA_ENV_DIR`
  local ENV_NAME=$1

  initialize_conda
  IFS=' ' read -r -a CONDA_INFO <<<"$(conda env list | grep "${ENV_NAME}")"
  AVAILABLE_ENV_NAME="${CONDA_INFO[0]}"

  if [[ "x${AVAILABLE_ENV_NAME}x" != "x${ENV_NAME}x" ]]; then
    echo -e "${FG_RED}Conda Env '${ENV_NAME}' is not available${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

  if [ "x${CONDA_INFO[1]}x" == "x*x" ]; then
    CONDA_ENV_DIR="${CONDA_INFO[2]}"
  else
    CONDA_ENV_DIR="${CONDA_INFO[1]}"
  fi
}


retry_to_find_conda_env_path() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  if [ "x${PYTHON_VERSION}x" == "xx" ]; then
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
  fi

  conda create -c conda-forge -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  find_conda_env_path "${CONDA_ENV}"

  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    echo -e "${FG_RED}Unknown exception occurs from the side of Conda infra${FG_RESET}"
  fi

}

install_python_package() {
  local TARGET_PROJECT_DIR=$1
  local CONDA_ENV=$2

  pushd "${TARGET_PROJECT_DIR}"

  if [ -f "setup.py" ]; then
    initialize_conda
    source activate "${CONDA_ENV}"
    echo -e "${FG_YELLOW}Activating conda env '${CONDA_ENV}'${FG_RESET}"

    if [ -d "${PWD}"/dist/ ]; then
      FILE_COUNT=$(ls "${PWD}/dist/*" 2>/dev/null | wc -l)
      if [ "x${FILE_COUNT//[[:space:]]/}x" != "x0x" ]; then
        echo -e "${FG_YELLOW}Removing ${PWD}/dist/* files${FG_RESET}"
        rm "${PWD}/dist/*"
      fi
    fi

    echo -e "${FG_YELLOW}Installing python package${FG_RESET}"
    python "setup.py" sdist
    pip install dist/*
    rm -r dist *.egg*

    conda deactivate
  else
    echo -e "${FG_RED}No valid $(setup.py) file exists${FG_RESET}"
  fi

  popd
}
