#!/bin/bash
#
# Start a Jupyter Lab server with the specified kernel environment and configure Spark and GPU settings.
# Parameters:
#    -k|--kernel_env: Specify the kernel environment name (default: sssd).
#    -p|--port: Specify the port number for the Jupyter Lab server (default: 8501).
#    -C|--update_conda: Set the DOES_UPDATE_CONDA flag to TRUE to update the Conda environment.
#


# Set default values
KERNEL_ENV="sssd"
PORT="8501"
DOES_UPDATE_CONDA="FALSE"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -k|--kernel_env)
            KERNEL_ENV="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -C|--update_conda)
            DOES_UPDATE_CONDA="TRUE"
            shift
            ;;
        *)
            echo "Invalid option: $1"
            exit 1
            ;;
    esac
done
# Set up environment
NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${NOTEBOOK_DIR}/utils.sh"
PACKAGE_BASE_PATH="${NOTEBOOK_DIR}/../.."
source "${PACKAGE_BASE_PATH}/bin/color_map.sh"
source "${PACKAGE_BASE_PATH}/bin/exit_code.sh"
local CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"


start_jupyter_server() {
  local PORT="$1"

  if [ "${DOES_UPDATE_CONDA}" == "TRUE" ]; then
    update_conda_env_path "${KERNEL_ENV}"
  fi

  conda activate "${KERNEL_ENV}"

  check_kernel_availability "${KERNEL_ENV}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    set_jupyter_kernel_path "${KERNEL_ENV}"
  fi

  update_gpu_env "${KERNEL_ENV}"

  jupyter lab --ip=0.0.0.0 --port "${PORT}" --no-browser --NotebookApp.token='' --NotebookApp.password='' --allow-root

  conda deactivate
}

start_jupyter_server "${PORT}"
