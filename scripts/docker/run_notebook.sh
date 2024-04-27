#!/bin/bash
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

${DOCKER_DIR}/../../envs/notebook/start_jupyter_lab.sh \
  --port ${PORT}
