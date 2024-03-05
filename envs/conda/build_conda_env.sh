#!/bin/bash
#
# Build Conda Env
#
# - Parameters
#    - Optional
#       - -c/--conda_env: conda env name; Default: `sssd`
# - Examples
#       1. Default conda env: sssd
#           - ./build_conda_env.sh
#       2. New conda env (will automatically install empty conda env if necessary)
#           - ./build_conda_env.sh -c sssd
#
# - Caveat
#    - If you don't have `realpath` on Mac, please install via `brew install coreutils`
#
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CONDA_DIR}/conda_env_info.sh"

source "${COLOR_MAP_PATH}"
source "${BASH_UTILS_PATH}"
source "${EXIT_CODE_PATH}"
source "${CONDA_ENV_UTILS_PATH}"

for ARG in "$@"; do
  shift
  case "${ARG}" in
  "--conda_env") set -- "$@" "-c" ;;
  *) set -- "$@" "${ARG}" ;;
  esac
done

while getopts "d:t:c:*" OPT; do
  case "${OPT}" in
  c)
    CONDA_ENV="${OPTARG}"
    ;;
  *) ;;

  esac
done


build() {
  local CONDA_ENV=$1

  if [ "x${CONDA_ENV}x" == "xx" ]; then
    CONDA_ENV="sssd"
  fi

  if [ "x${CONDA_HOME}" == "xx" ]; then
    CONDA_HOME=${CONDA_PATH}
  fi

  # Will return `CONDA_ENV_DIR`
  echo -e "${FG_YELLOW}Checking Conda Env: `${CONDA_ENV}`${FG_RESET}"
  find_conda_env_path "${CONDA_ENV}"
  # Try to build the conda env if the error code is captured
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
    retry_to_find_conda_env_path "${CONDA_ENV}" "${PYTHON_VERSION}"
    if [ "$?" == "${ERROR_EXITCODE}" ]; then
      return "${ERROR_EXITCODE}"
    fi
  fi

  # Check HDF5 dependency [TODO] wrap it to other function
  if [ conda list | grep -q hdf5 ]; then
    echo -e "${FG_GREEN}hdf5 is installed.${FG_RESET}"
  else
    echo "${FG_RED}hdf5 is not installed.${FG_RESET}"
    conda install -c conda-forge hdf5 -y
  fi


  # Installation
  echo -e "${FG_YELLOW}Installing package ${FG_RESET}"
  install_python_package "${TARGET_PROJECT_DIR}" "${CONDA_ENV}"
  echo -e "${FG_GREEN}Updated package${FG_RESET}"

  return "${SUCCESS_EXITCODE}"
}

build "${CONDA_ENV}"
