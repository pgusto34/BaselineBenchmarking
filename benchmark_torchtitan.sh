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
  LOG_RANK=1 NGPU=2 ./benchmark_torchtitan.sh llama3_2_1b_dp_ds_like -- --training.steps=50 2>&1 | tee torchtitan_run.log | python3 torchtitan/torchtitan_avg_stats.py --skip-steps 5 --include '(tps|tflops|mfu)'

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
PIPELINE_PARALLEL_DEGREE=$(grep -E '^\s*pipeline_parallel_degree\s*=' "$CONFIG_PATH" | head -1 | sed 's/.*=\s*\([0-9]\+\).*/\1/' || true)
AUTO_DOWNLOAD_TOKENIZER="${AUTO_DOWNLOAD_TOKENIZER:-0}"

has_local_tokenizer_files() {
  local p="$1"
  [[ -f "$p/tokenizer.json" ]] \
    || [[ -f "$p/tokenizer.model" ]] \
    || [[ -f "$p/tokenizer_config.json" ]] \
    || [[ -f "$p/special_tokens_map.json" ]]
}

# If hf_assets_path points at a HuggingFace cache repo directory (models--...)
# resolve it to a concrete snapshot that contains tokenizer files.
if [[ -n "${HF_ASSETS:-}" ]] && [[ -d "$HF_ASSETS/snapshots" ]] && ! has_local_tokenizer_files "$HF_ASSETS"; then
  SNAPSHOT_PATH=""
  while IFS= read -r cand; do
    if has_local_tokenizer_files "$cand"; then
      SNAPSHOT_PATH="$cand"
    fi
  done < <(find "$HF_ASSETS/snapshots" -mindepth 1 -maxdepth 1 -type d | sort)

  if [[ -n "$SNAPSHOT_PATH" ]]; then
    HF_ASSETS="$SNAPSHOT_PATH"
    echo "[setup] Resolved hf_assets_path to snapshot: $HF_ASSETS"
  fi
fi

# When local assets are present, prefer strict offline behavior to avoid
# accidental calls to gated Hugging Face repos for auxiliary tokenizer files.
if [[ -n "${HF_ASSETS:-}" ]] && has_local_tokenizer_files "$HF_ASSETS"; then
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  echo "[setup] Local tokenizer files found; using offline mode (HF_HUB_OFFLINE=$HF_HUB_OFFLINE, TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE)."
fi

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
    # Backward-compatible alias:
    # [parallelism] reshard_after_forward=true|false
    # -> --parallelism.fsdp-reshard-after-forward=always|never
    if section == 'parallelism' and key == 'reshard_after_forward':
      if val == 'true':
        print('--parallelism.fsdp-reshard-after-forward=always')
      elif val == 'false':
        print('--parallelism.fsdp-reshard-after-forward=never')
      else:
        print(f'--parallelism.fsdp-reshard-after-forward={val}')
      continue
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

# For pipeline parallel runs, include last-stage ranks in logging by default.
if [[ -z "${LOG_RANK:-}" ]] && [[ "${PIPELINE_PARALLEL_DEGREE:-}" =~ ^[0-9]+$ ]] && (( PIPELINE_PARALLEL_DEGREE > 1 )); then
  LOG_RANKS=()
  for ((r = PIPELINE_PARALLEL_DEGREE - 1; r < NGPU; r += PIPELINE_PARALLEL_DEGREE)); do
    LOG_RANKS+=("$r")
  done
  export LOG_RANK="$(IFS=,; echo "${LOG_RANKS[*]}")"
  echo "[setup] LOG_RANK not set; using LOG_RANK=$LOG_RANK for pipeline loss visibility."
fi

# Build the torchrun command
CMD="torchrun --nproc_per_node=$NGPU --rdzv_backend c10d --rdzv_endpoint=localhost:0"
CMD="$CMD /workspace/torchtitan/custom_configs.py"
CMD="$CMD --module $MODULE --config $MODEL_CONFIG"
[[ -n "${HF_ASSETS:-}" ]] && CMD="$CMD --hf_assets_path $HF_ASSETS"
CMD="$CMD $TOML_ARGS"

# Append extra user args (these override everything)
CMD="$CMD $*"

# Auto-download tokenizer only when explicitly requested.
if [[ -n "${HF_ASSETS:-}" ]] && ! has_local_tokenizer_files "$HF_ASSETS"; then
  if [[ "$AUTO_DOWNLOAD_TOKENIZER" == "1" ]] && [[ -n "${REPO_ID:-}" ]]; then
    echo "[setup] Tokenizer not found at $HF_ASSETS"
    echo "[setup] AUTO_DOWNLOAD_TOKENIZER=1; attempting download from $REPO_ID..."
    mkdir -p "$HF_ASSETS"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${REPO_ID}', local_dir='${HF_ASSETS}', allow_patterns=['tokenizer*', 'vocab*', 'merges*', 'special_tokens*'], local_dir_use_symlinks=False)
"
    echo "[setup] Download complete."
  else
    echo "[setup] Tokenizer not found at $HF_ASSETS; skipping auto-download (set AUTO_DOWNLOAD_TOKENIZER=1 to enable)."
  fi
fi

echo ""
echo "Running: $CMD"
echo ""
exec $CMD