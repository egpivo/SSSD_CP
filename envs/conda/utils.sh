#!/bin/bash
#
# Helper functions for building a conda environment
#
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_DIR}/conda_env_info.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"


initialize_conda() {
  local CONDA_BASE=$(conda info --base)
  local CONDA_DIRS=(
    "$CONDA_BASE"
    "/opt/conda"
    "/opt/miniconda"
    "/opt/miniconda2"
    "/opt/miniconda3"
  )

  local IS_CONDA_FOUND=false
  for dir in "${CONDA_DIRS[@]}"; do
    if [ -f "$dir/etc/profile.d/conda.sh" ]; then
      CONDA_BASE="$dir"
      IS_CONDA_FOUND=true
      break
    fi
  done

  if ! $IS_CONDA_FOUND; then
    echo -e "${FG_RED}No Conda environment found matching${FG_RESET}"  >&2
    return ${ERROR_EXITCODE}
  fi

  echo -e "${FG_YELLOW}Intializing conda${FG_RESET}"
  source "$CONDA_BASE/etc/profile.d/conda.sh"
}


find_conda_env_path() {
  # Will return `CONDA_ENV_DIR`
  local ENV_NAME=$1

  initialize_conda
  IFS=' ' read -r -a CONDA_INFO <<<"$(conda env list | grep "${ENV_NAME}")"

  if [ ${#CONDA_INFO[@]} -eq 0 ]; then
    echo -e "${FG_RED}No Conda environment found matching '${ENV_NAME}'${FG_RESET}"
    return "${ERROR_EXITCODE}"
  fi

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
  # DO NOT REMOVE
  echo "${CONDA_ENV_DIR}"
}


initialize_conda_env() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  conda create -c conda-forge -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
  source activate "${CONDA_ENV}"
  pip install --no-cache-dir "poetry==${POETRY_VERSION}"
  conda deactivate
}

retry_to_find_conda_env_path() {
  local CONDA_ENV=$1
  local PYTHON_VERSION=$2

  if [ "x${PYTHON_VERSION}x" == "xx" ]; then
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
  fi

  initialize_conda_env "${CONDA_ENV}" "${PYTHON_VERSION}"

  find_conda_env_path "${CONDA_ENV}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    echo -e "${FG_RED}Unknown exception occurs from the side of Conda infra${FG_RESET}"
  fi

}

install_python_package() {
  local TARGET_PROJECT_DIR=$1

  pushd "${TARGET_PROJECT_DIR}" || exit

  if [ -d "${PWD}"/dist/ ]; then
    FILE_COUNT=$(ls "${PWD}/dist/*" 2>/dev/null | wc -l)
    if [ "x${FILE_COUNT//[[:space:]]/}x" != "x0x" ]; then
      echo -e "${FG_YELLOW}Removing ${PWD}/dist/* files${FG_RESET}"
      rm "${PWD}/dist/*"
    fi
  fi

  echo -e "${FG_YELLOW}Installing python package${FG_RESET}"
  poetry lock --no-update
  poetry install

  # Install 3rd party packages
  echo -e "${FG_YELLOW}Installing 3rd party packages${FG_RESET}"
  install_extensions_cauchy

  poetry build
  if [ -d "${PWD}"/dist/ ]; then
    pip install dist/*.tar.gz
    rm -r dist
  else
    echo -e "${FG_RED}Failed to install python package${FG_RESET}"
  fi
  popd || exit
}


install_extensions_cauchy() {
  # See: https://github.com/state-spaces/s4/tree/v3.0.0
  # Check if cauchy_mult is already installed
  if pip list | grep -q "cauchy_mult"; then
    echo -e "${FG_YELLOW}cauchy_mult is already installed.${FG_RESET}"
    return 0
  fi

  # Check if the s4 directory exists and is not empty
  if [ -d "s4" ] && [ "$(ls -A s4)" ]; then
    echo -e "${FG_YELLOW}Removing existing 's4' directory.${FG_RESET}"
    rm -rf "s4" || {
      echo -e "${FG_RED}Error: Failed to remove the existing 's4' directory.${FG_RESET}"
      return 1
    }
  fi

  # Clone the repository
  git clone --depth 1 --branch v3.0.0 https://github.com/state-spaces/s4.git || {
    echo -e "${FG_RED}Error: Failed to clone the repository.${FG_RESET}"
    return 1
  }

  # Change directory
  pushd "s4/extensions/cauchy" || {
    echo -e "${FG_RED}Error: Failed to navigate to the directory.${FG_RESET}"
    return 1
  }

  # Install the package
  echo -e "${FG_YELLOW}Installing cauchy-mult package${FG_RESET}"
  python setup.py install || {
    echo -e "${FG_RED}Error: Failed to install the cauchy-mult package.${FG_RESET}"
    popd || return 1
    rm -rf "s4"
    return 1
  }

  # Clean up
  popd || {
    echo -e "${FG_RED}Error: Failed to return to the previous directory.${FG_RESET}"
    return 1
  }
  rm -rf "s4"
}
