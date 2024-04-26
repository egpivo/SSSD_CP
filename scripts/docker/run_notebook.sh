#!/bin/bash
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

${DOCKER_DIR}/../../envs/notebook/start_notebook_server.sh \
  --port ${PORT}
