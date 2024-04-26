DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/utils.sh"


update_conda_environment() {
  local PACKAGE_BASE_PATH=$1
  local CONDA_ENV=$2

  if [ -x "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Updating Conda environment - sssd${FG_RESET}"
    bash "${PACKAGE_BASE_PATH}/envs/conda/build_conda_env.sh" --conda_env ${CONDA_ENV}
  else
    echo -e "${FG_RED}Conda is not installed.${FG_RESET}"
  fi
}


activate_conda_environment() {
  local CONDA_ENV=$1
  if [ -x "$(command -v conda)" ]; then
    source activate ${CONDA_ENV}
  else
    echo -e "${FG_RED}Conda is not installed.${FG_RESET}"
  fi
}


check_file_exists() {
  local CONFIG_FILE="$1"
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Configuration file '${CONFIG_FILE}' not found." >&2
    exit "${ERROR_EXITCODE}"
  fi
}
