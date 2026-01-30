#!/bin/bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <registry> <tag>" >&2
  echo "Example: $0 ghcr.io/your-org/aero-hand-jax v0.1.0" >&2
  exit 1
fi

REGISTRY_IMAGE="$1"
TAG="$2"
IMAGE_NAME=${IMAGE_NAME:-aero-hand-jax}
LOCAL_TAG="${IMAGE_NAME}:${TAG}"
FULL_TAG="${REGISTRY_IMAGE}:${TAG}"

docker tag "${LOCAL_TAG}" "${FULL_TAG}"
docker push "${FULL_TAG}"

echo "Pushed image: ${FULL_TAG}"
