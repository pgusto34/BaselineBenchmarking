#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="$SCRIPT_DIR/torchtitan/configs"
NGPU="${NGPU:-2}"

usage() {
  cat <<'EOF'
Usage: ./benchmark_torchtitan.sh <config_name[.toml]> [-- <extra torchtitan args...>]

Examples:
  ./benchmark_torchtitan.sh qwen3_0_6b
  ./benchmark_torchtitan.sh qwen3_30b_a3b_mini -- --training.steps=50
  NGPU=4 ./benchmark_torchtitan.sh qwen3_30b_a3b_mini

Configs live in ./configs/. Each TOML file specifies model, training,
parallelism, and other settings. Extra CLI args override TOML values.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  -h|--help) usage; exit 0 ;;
esac

CONFIG_NAME="$1"
shift

if [[ "$CONFIG_NAME" != *.toml ]]; then
  CONFIG_NAME="${CONFIG_NAME}.toml"
fi

CONFIG_PATH="$CONFIG_DIR/$CONFIG_NAME"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Error: Config not found: $CONFIG_PATH"
  echo "Available configs:"
  ls -1 "$CONFIG_DIR"/*.toml 2>/dev/null | xargs -n1 basename || echo "  (none)"
  exit 1
fi

# Skip past "--" separator if present
if [[ "${1:-}" == "--" ]]; then
  shift
fi

# --- Parse [model] section (handled specially) ---
MODULE=$(grep -E '^\s*module\s*=' "$CONFIG_PATH" | head -1 | sed 's/.*=\s*"\(.*\)"/\1/')
MODEL_CONFIG=$(grep -E '^\s*config\s*=' "$CONFIG_PATH" | head -1 | sed 's/.*=\s*"\(.*\)"/\1/')
HF_ASSETS=$(grep -E '^\s*hf_assets_path\s*=' "$CONFIG_PATH" | head -1 | sed 's/.*=\s*"\(.*\)"/\1/')
REPO_ID=$(grep -E '^\s*repo_id\s*=' "$CONFIG_PATH" | head -1 | sed 's/.*=\s*"\(.*\)"/\1/')

# --- Convert all other TOML sections to CLI flags automatically ---
TOML_ARGS=$(python3 -c "
import re
section = None
skip = {'model'}
for line in open('$CONFIG_PATH'):
    line = line.strip()
    m = re.match(r'^\[(\w+)\]', line)
    if m:
        section = m.group(1)
        continue
    if section and section not in skip and '=' in line and not line.startswith('#'):
        key, val = line.split('=', 1)
        key = key.strip()
        val = val.strip().strip('\"')
        if val == 'true':
            print(f'--{section}.{key}')
        elif val == 'false':
            print(f'--{section}.no-{key}')
        else:
            print(f'--{section}.{key}={val}')
")

echo "========================================"
echo " TorchTitan Benchmark"
echo "========================================"
echo " Config:     $CONFIG_PATH"
echo " Module:     $MODULE"
echo " Model:      $MODEL_CONFIG"
echo " GPUs:       $NGPU"
echo " TOML args:  $TOML_ARGS"
echo " Extra args: $*"
echo "========================================"

# Build the torchrun command
CMD="torchrun --nproc_per_node=$NGPU --rdzv_backend c10d --rdzv_endpoint=localhost:0"
CMD="$CMD /workspace/torchtitan/custom_configs.py"
CMD="$CMD --module $MODULE --config $MODEL_CONFIG"
[[ -n "${HF_ASSETS:-}" ]] && CMD="$CMD --hf_assets_path $HF_ASSETS"
CMD="$CMD $TOML_ARGS"

# Append extra user args (these override everything)
CMD="$CMD $*"

# Auto-download tokenizer if not present
if [[ -n "${HF_ASSETS:-}" ]] && [[ -n "${REPO_ID:-}" ]] && [[ ! -f "$HF_ASSETS/tokenizer.json" ]]; then
  echo "[setup] Tokenizer not found at $HF_ASSETS"
  echo "[setup] Downloading from $REPO_ID..."
  mkdir -p "$HF_ASSETS"
  python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${REPO_ID}', local_dir='${HF_ASSETS}', allow_patterns=['tokenizer*', 'vocab*', 'merges*', 'special_tokens*'], local_dir_use_symlinks=False)
"
  echo "[setup] Download complete."
fi

echo ""
echo "Running: $CMD"
echo ""
exec $CMD