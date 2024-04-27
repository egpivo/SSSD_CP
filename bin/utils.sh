DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${DIR}/../bin/color_map.sh"


update_conda_environment() {
  local PACKAGE_BASE_PATH=$1
  local CONDA_ENV=$2

  if [ -x "$(command -v conda)" ]; then
    echo -e "${FG_YELLOW}Updating Conda environment - ${CONDA_ENV}${FG_RESET}"
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
  local FILE_PATH="$1"
  if [[ ! -f "${FILE_PATH}" ]]; then
    echo "Error: File '${FILE_PATH}' not found." >&2
    exit "${ERROR_EXITCODE}"
  fi
}
