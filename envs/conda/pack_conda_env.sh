#!/bin/bash
#
# Pack Conda Environment
#
# This script packs a Conda environment into a zip file.
#
# Usage:
#   pack_conda_env.sh [OPTIONS]
#
# Options:
#   -c, --conda_env       Conda environment name (default: 'sssd')
#   -t, --target_project_dir
#                         Target project directory (default: current directory)
#   -d, --destination     Destination directory for the packed environment zip file (default: current directory)
#
# Example:
#   Pack the default Conda environment 'sssd':
#       pack_conda_env.sh
#
#   Pack a specific Conda environment 'my_env' and save it to the 'packages' directory:
#       pack_conda_env.sh -c my_env -d packages
#

# Directory of the script
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load utilities and environment settings
set -a
source "${CONDA_DIR}/conda_env_info.sh"
source "${CONDA_DIR}/utils.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"
set +a

# Default values for options
CONDA_ENV="sssd"
TARGET_PROJECT_DIR="${PACKAGE_BASE_PATH}"
DESTINATION="${PACKAGE_BASE_PATH}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -t|--target_project_dir)
            TARGET_PROJECT_DIR="$2"
            shift 2
            ;;
        -d|--destination)
            DESTINATION="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done


# Function to pack Conda environment as a zip file
pack_conda_env_zip() {
    local CONDA_ENV="$1"
    local DESTINATION="$2"
    local CONDA_ENV_DIR="$3"

    echo -e "${FG_YELLOW}Packing Conda environment '${CONDA_ENV}' as '${CONDA_ENV}.zip'${FG_RESET}"

    # Navigate to the Conda environment directory
    pushd "${CONDA_ENV_DIR}" > /dev/null || exit 1

    # Create the destination directory if it doesn't exist
    mkdir -p "${DESTINATION}"

    # Pack the Conda environment as a zip file
    zip -urq "${CONDA_ENV}.zip" .

    # Move the zip file to the destination directory
    mv "${CONDA_ENV_DIR}/${CONDA_ENV}.zip" "${DESTINATION}/"
    echo -e "${FG_YELLOW}Moved '${CONDA_ENV_DIR}/${CONDA_ENV}.zip' to '${DESTINATION}'${FG_RESET}"

    # Return to the original directory
    popd > /dev/null || exit 1
}

# Will update $CONDA_ENV_DIR
find_conda_env_path "${CONDA_ENV}"
# Pack the Conda environment
pack_conda_env_zip "${CONDA_ENV}" "${DESTINATION}" "${CONDA_ENV_DIR}"
