#!/bin/bash
#
# Build Conda Environment
#
# This script builds a Conda environment based on the provided name or the default 'sssd'.
#
# Parameters:
#   -c, --conda_env       Specifies the name of the Conda environment (default: 'sssd')
#
# Examples:
#   1. Build the default Conda environment 'sssd':
#       ./build_conda_env.sh
#
#   2. Build a specific Conda environment 'my_env':
#       ./build_conda_env.sh -c my_env
#
# Note:
#   - If 'realpath' is not available on macOS, install it via 'brew install coreutils'.
#

# Directory of the script
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${CONDA_DIR}/conda_env_info.sh"
source "${CONDA_DIR}/utils.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"
set +a

# Default values for options
CONDA_ENV="sssd"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

# Function to build the Conda environment
build() {
    local CONDA_ENV="$1"

    # Check if the Conda environment exists
    CONDA_ENV_DIR="$(find_conda_env_path "${CONDA_ENV}")"
    if [ "$?" == "${ERROR_EXITCODE}" ]; then
        PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
        retry_to_find_conda_env_path "${CONDA_ENV}" "${PYTHON_VERSION}"
        if [ "$?" == "${ERROR_EXITCODE}" ]; then
            return "${ERROR_EXITCODE}"
        fi
    fi

    # Activate the Conda environment
    initialize_conda
    source activate "${CONDA_ENV}"
    echo -e "${FG_YELLOW}Activating Conda environment '${CONDA_ENV}'${FG_RESET}"

    # Check HDF5 dependency
    if conda list | grep -q hdf5; then
        echo -e "${FG_GREEN}HDF5 is installed.${FG_RESET}"
    else
        echo "${FG_RED}HDF5 is not installed.${FG_RESET}"
        conda install -c conda-forge hdf5 -y
    fi

    # Install packages
    echo -e "${FG_YELLOW}Installing packages${FG_RESET}"
    install_python_package "${PACKAGE_BASE_PATH}"
    echo -e "${FG_GREEN}Updated packages${FG_RESET}"

    # Deactivate the Conda environment
    conda deactivate

    return "${SUCCESS_EXITCODE}"
}

# Build the Conda environment
build "${CONDA_ENV}"
