#!/bin/bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-aero-hand-jax}
TAG=${1:-latest}
CUDA_VERSION=${CUDA_VERSION:-12.4.1}
MUJOCO_PLAYGROUND_REF=${MUJOCO_PLAYGROUND_REF:-main}
FULL_TAG="${IMAGE_NAME}:${TAG}"

docker build \\
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \\
  --build-arg MUJOCO_PLAYGROUND_REF="${MUJOCO_PLAYGROUND_REF}" \\
  -t "${FULL_TAG}" .

echo "Built image: ${FULL_TAG}"
