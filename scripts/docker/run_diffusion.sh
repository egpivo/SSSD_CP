#!/bin/bash
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

${DOCKER_DIR}/../diffusion/diffusion_process.sh \
  --model_config configs/$MODEL_CONFIG \
  --training_config configs/$TRAINING_CONFIG \
  --inference_config configs/$INFERENCE_CONFIG
