#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./benchmark_megatron <config_name[.toml]> [OPTIONS]

Examples:
  ./benchmark_megatron pretrain_debug_single_gpu
  ./benchmark_megatron pretrain_debug_single_gpu.toml --dry-run
  ./benchmark_megatron pretrain_llama3_toml_example
  ./benchmark_megatron pretrain_llama3_toml_example.toml --dry-run

Options:
  --dry-run    Print the resolved command without executing training

Notes:
  - Run this after entering the container via setup.sh.
  - Config is resolved as /workspace/megatron/configs/<config_name>.toml.
  - GPU count comes from distributed.gpus_per_node in the TOML.
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

CONFIG_NAME="$1"
if [[ "$CONFIG_NAME" != *.toml ]]; then
  CONFIG_NAME="${CONFIG_NAME}.toml"
fi

CONFIG_PATH="/workspace/megatron/configs/$CONFIG_NAME"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

python3 /workspace/megatron/run_megatron.py \
  --config "$CONFIG_PATH" \
  --megatron-root /opt/Megatron-LM \
  "${@:2}"
