update_conda_environment() {
  local PACKAGE_BASE_PATH=$1
  local DOES_UPDATE_CONDA_ENV=$2
  local CONDA_ENV=$3

  if [ -x "$(command -v conda)" ]; then
    if [[ "$DOES_UPDATE_CONDA_ENV" == "true" ]]; then
      echo -e "${FG_YELLOW}Updating Conda environment - sssd${FG_RESET}"
      bash "${PACKAGE_BASE_PATH}/envs/conda/build_conda_env.sh" --conda_env ${CONDA_ENV}
    else
      echo -e "${FG_GREEN}Conda environment update is not requested.${FG_RESET}"
    fi
    source activate ${CONDA_ENV}
  else
    echo -e "${FG_RED}Conda is not installed.${FG_RESET}"
  fi
}
