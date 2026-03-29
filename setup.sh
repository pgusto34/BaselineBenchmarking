#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-bench-env}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-docker/Dockerfile}"
GPU_SPEC="${GPU_SPEC:-all}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [--gpu <all|N|device=N>] [--no-build]

Options:
  --gpu <spec>   GPU selection passed to Docker.
                 Examples: all, 3, device=3
  --no-build     Skip docker build and only run container.
  -h, --help     Show this help message.

Environment overrides:
  IMAGE_NAME         Docker image tag (default: bench-env)
  DOCKERFILE_PATH    Dockerfile path (default: docker/Dockerfile)
  GPU_SPEC           GPU selection if --gpu is not provided (default: all)
EOF
}

NO_BUILD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --gpu"; exit 1; }
      GPU_SPEC="$1"
      ;;
    --no-build)
      NO_BUILD=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

# Normalize plain numeric GPU input like "3" to docker format "device=3".
if [[ "$GPU_SPEC" =~ ^[0-9]+$ ]]; then
  GPU_SPEC="device=$GPU_SPEC"
fi

if [[ $NO_BUILD -eq 0 ]]; then
  echo "[setup] Building image '$IMAGE_NAME' from '$DOCKERFILE_PATH'..."
  docker build -f "$DOCKERFILE_PATH" -t "$IMAGE_NAME" .
fi

RESULTS_DIR="${RESULTS_DIR:-$PWD/results}"
mkdir -p "$RESULTS_DIR"

echo "[setup] Starting container from '$IMAGE_NAME' with --gpus $GPU_SPEC..."
docker run --rm -it \
  --gpus "$GPU_SPEC" \
  --ipc=host \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace \
  -v "$RESULTS_DIR":/results \
  -w /workspace \
  "$IMAGE_NAME" \
  bash
