#!/bin/bash
#
# Start a Jupyter notebook server with Kernel environment setting to connect Spark seamlessly
#
# - Parameters
#    - Optional
#       - -k/--kernel_env: kernel env name; Default: optimus
#       - -q/--queue: queue on Yarn; default: mkplalgo-dev
#       - -x/--num_executors: number of executors; default: based on ${SPARK_CONF_DIR}
#       - -m/--executor_memory: memory per executor; default: based on ${SPARK_CONF_DIR}
#       - -p/--port: port for notebook server; default: 8000
#       - -C/--rebuild_conda: rebuild the Cond environment by adding this tag
#       - -G/--does_use_gpu: update the environment variables related to GPU driver directories
#       - -E/--install ego: install ego in the Conda environment (only valid when the flag `--rebuild_conda` is added)
# - Examples
#       1. Default conda env: optimus
#           - ./start_notebook_server.sh -K -C
#       2. New conda env
#           - ./start_notebook_server.sh -k temp -K -C
#       3. Modify Spark resource
#           - ./start_notebook_server.sh -x 20 -m 16g
#       4. Default conda env: optimus with ego
#           - ./start_notebook_server.sh -K -C -E
# - Caveats
#    - The kernel environment name should be identical to the name in Conda environment.
#    - This requires that jupyter notebook service is installed in your environment.
#
NOTEBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${NOTEBOOK_DIR}/utils.sh"
PACKAGE_BASE_PATH="${NOTEBOOK_DIR}/../.."
source "${PACKAGE_BASE_PATH}/bin/color_map.sh"
source "${PACKAGE_BASE_PATH}/bin/exit_code.sh"


for ARG in "$@"; do
  shift
  case "${ARG}" in
  "--kernel_env") set -- "$@" "-k" ;;
  "--port") set -- "$@" "-p" ;;
  "--rebuild_conda") set -- "$@" "-C" ;;
  *) set -- "$@" "${ARG}" ;;
  esac
done

while getopts "k:p:C*" OPT; do
  case "${OPT}" in
  k)
    KERNEL_ENV="${OPTARG}"
    ;;
  p)
    PORT="${OPTARG}"
    ;;
  C)
    DOES_UPDATE_CONDA="TRUE"
    ;;
  *) ;;
  esac
done

start_jupyter_server() {
  local PORT="$1"
  if [ "x${KERNEL_ENV}x" == "xx" ]; then
    KERNEL_ENV="sssd"
  fi

  update_conda_env_path "${KERNEL_ENV}"
  is_jupyter_kernel_path_available "${KERNEL_ENV}"
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    set_jupyter_kernel_path "${KERNEL_ENV}"
  fi
  update_gpu_env ${KERNEL_ENV}

  jupyter lab --ip=0.0.0.0 --port "${PORT}" --no-browser
}

if [ "x${PORT}x" == "xx" ]; then
  PORT="8000"
fi
start_jupyter_server ${PORT}
