#!/bin/bash
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda activate sssd
poetry install --no-root

${DOCKER_DIR}/../../envs/jupyter/start_jupyter_lab.sh \
  --port ${PORT}
