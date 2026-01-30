#!/bin/bash
set -euo pipefail

IMAGE_NAME=${IMAGE_NAME:-aero-hand-jax}
TAG=${1:-latest}
FULL_TAG="${IMAGE_NAME}:${TAG}"

docker build -t "${FULL_TAG}" .

echo "Built image: ${FULL_TAG}"
