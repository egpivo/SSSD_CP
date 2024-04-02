#!/bin/bash
#
# This script finds the path of a specified Conda environment.
#
# Usage:
#   find_conda_env_path.sh [OPTIONS]
#
# Options:
#   -c, --conda_env       Specify the name of the Conda environment to find (default: 'sssd')
#   -o, --output_file     Specify the output file path to save the Conda environment path
#
# Example:
#   Find the path of the default Conda environment 'sssd':
#       find_conda_env_path.sh
#
#   Find the path of a specific Conda environment 'my_env' and save it to a custom file:
#       find_conda_env_path.sh -c my_env -o /path/to/output.txt
#
# Note:
#   This script requires the 'conda_env_info.sh' and 'utils.sh' scripts to be available in the same directory.

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
OUTPUT_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--conda_env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -o|--output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done

# Validate the output file path
if [[ -z "$OUTPUT_FILE" ]]; then
    echo "Output file path is required."
    exit 1
fi

# Will update $CONDA_ENV_DIR
find_conda_env_path "${CONDA_ENV}"
echo ${CONDA_ENV_DIR} > ${OUTPUT_FILE}
