#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEED_DIR="$SCRIPT_DIR/deepspeed"

usage() {
  cat <<'EOF'
Usage:
  ./benchmark_deepspeed.sh [--script benchmark_single.py] [-- <script args...>]
  ./benchmark_deepspeed.sh [benchmark_single.py] [<script args...>]

Defaults:
  script = benchmark_single.py

Examples:
  ./benchmark_deepspeed.sh
  ./benchmark_deepspeed.sh benchmark_dp.py --iterations 50
  ./benchmark_deepspeed.sh --script benchmark_pp.py -- --model-config 1b --seq-len 512
EOF
}

TARGET_SCRIPT="benchmark_single.py"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --script)
      if [[ $# -lt 2 ]]; then
        echo "Error: --script requires a value"
        usage
        exit 1
      fi
      TARGET_SCRIPT="$2"
      shift 2
      ;;
    --script=*)
      TARGET_SCRIPT="${1#--script=}"
      shift
      ;;
    --)
      shift
      ;;
    *)
      TARGET_SCRIPT="$1"
      shift
      ;;
  esac
fi

# Support script names without .py suffix.
if [[ "$TARGET_SCRIPT" != *.py ]]; then
  TARGET_SCRIPT="${TARGET_SCRIPT}.py"
fi

SCRIPT_PATH="$DEEPSPEED_DIR/$TARGET_SCRIPT"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: DeepSpeed script not found: $SCRIPT_PATH"
  echo "Available scripts:"
  ls -1 "$DEEPSPEED_DIR"/benchmark*.py 2>/dev/null | xargs -n1 basename || true
  exit 1
fi

# Triton may query disk stats for autotune cache with `df` and warn if missing.
TRITON_AUTOTUNE_DIR="${TRITON_CACHE_DIR:-$HOME/.triton}/autotune"
mkdir -p "$TRITON_AUTOTUNE_DIR"

echo "Running: $SCRIPT_PATH $*"
exec python3 "$SCRIPT_PATH" "$@"
