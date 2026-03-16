#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-torchtitan-bench}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-docker/Dockerfile.torchtitan}"
GPU_SPEC="${GPU_SPEC:-all}"
# Path to your local torchtitan and hf_assets on the host
TORCHTITAN_PATH="${TORCHTITAN_PATH:-/m-coriander/coriander/shubham/moe-scheduling/torchtitan}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/m-coriander/coriander/shubham/moe-scheduling/hf_assets}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--gpu <all|N|device=N>] [--no-build]

Options:
  --gpu <spec>   GPU selection passed to Docker (default: all)
  --no-build     Skip docker build, only run container
  -h, --help     Show this help

Environment overrides:
  IMAGE_NAME         Docker image tag (default: torchtitan-bench)
  TORCHTITAN_PATH    Host path to torchtitan repo
  HF_ASSETS_PATH     Host path to HF tokenizer assets
EOF
}

NO_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) shift; GPU_SPEC="$1" ;;
    --no-build) NO_BUILD=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
  shift
done

if [[ "$GPU_SPEC" =~ ^[0-9]+$ ]]; then
  GPU_SPEC="device=$GPU_SPEC"
fi

if [[ $NO_BUILD -eq 0 ]]; then
  echo "[setup] Building image '$IMAGE_NAME' from '$DOCKERFILE_PATH'..."
  docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" .
fi

echo "[setup] Starting container with --gpus $GPU_SPEC..."

# docker run --rm -it \
#   --gpus "$GPU_SPEC" \
#   --ipc=host \
#   --shm-size=32g \
#   --ulimit memlock=-1 \
#   --ulimit stack=67108864 \
#   -v "$PWD":/workspace \
#   -v "$TORCHTITAN_PATH":/opt/torchtitan \
#   -v "$HF_ASSETS_PATH":/opt/hf_assets \
#   -w /workspace \
#   "$IMAGE_NAME" \
#   bash -c "cd /opt/torchtitan && pip install -e . > /dev/null 2>&1 && cd /workspace && exec bash"

docker run --rm -it \
  --gpus "$GPU_SPEC" \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace \
  -v "$HF_ASSETS_PATH":/opt/hf_assets \
  -w /workspace \
  "$IMAGE_NAME" \
  bash

