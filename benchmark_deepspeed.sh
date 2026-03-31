#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPSPEED_DIR="$SCRIPT_DIR/deepspeed"

usage() {
  cat <<'EOF'
Usage:
  ./benchmark_deepspeed.sh [--script benchmark_single.py] [--model-config MODEL] [--pp-degree N] [--dp-degree N] [--stats-output PATH] [--num-nodes N] [--node-rank N] [--master-addr HOST] [--master-port PORT] [--hostfile PATH]
  ./benchmark_deepspeed.sh [benchmark_single.py]

Defaults:
  script = benchmark_single.py
  pp-degree = 2 (for PP scripts)
  dp-degree = 1 (for PP+DP scripts; maps to ep-size for EP scripts)

Examples:
  ./benchmark_deepspeed.sh
  ./benchmark_deepspeed.sh benchmark_pp.py --model-config 70b --pp-degree 8
  ./benchmark_deepspeed.sh --script benchmark_pp_dp.py --model-config debug --pp-degree 4 --dp-degree 2
  ./benchmark_deepspeed.sh --script benchmark_pp_ep.py --pp-degree 4 --dp-degree 2
  ./benchmark_deepspeed.sh --script benchmark_pp.py --pp-degree 4 --stats-output ./results/pp_stats.json

Options:
  --script <name>        DeepSpeed benchmark script to run
  --model-config <name>  Optional model config forwarded to benchmark script
  --pp-degree <n>        Pipeline parallelism degree (default: 2)
  --dp-degree <n>        Data parallelism degree for PP+DP scripts; expert parallelism for PP+EP (default: 1)
  --stats-output <path>  Optional path to write benchmark stats JSON (supported by benchmark_pp.py)
  --num-nodes <n>        Number of nodes for distributed launch (default: 1)
  --node-rank <n>        Rank of this node in [0, num-nodes-1] (default: 0)
  --master-addr <host>   Master node address (default: localhost)
  --master-port <port>   Master port (default: 29500)
  --hostfile <path>      Optional DeepSpeed hostfile for multi-node launch
EOF
}

TARGET_SCRIPT="benchmark_single.py"
WRAPPER_MODEL_CONFIG="debug"
PP_DEGREE=2
DP_DEGREE=1
STATS_OUTPUT=""
NUM_NODES=1
NODE_RANK=0
MASTER_ADDR="localhost"
MASTER_PORT=29500
HOSTFILE=""
SCRIPT_SET=0

while [[ $# -gt 0 ]]; do
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
      SCRIPT_SET=1
      shift 2
      ;;
    --script=*)
      TARGET_SCRIPT="${1#--script=}"
      SCRIPT_SET=1
      shift
      ;;
    --model-config)
      if [[ $# -lt 2 ]]; then
        echo "Error: --model-config requires a value"
        usage
        exit 1
      fi
      WRAPPER_MODEL_CONFIG="$2"
      shift 2
      ;;
    --model-config=*)
      WRAPPER_MODEL_CONFIG="${1#--model-config=}"
      shift
      ;;
    --pp-degree)
      if [[ $# -lt 2 ]]; then
        echo "Error: --pp-degree requires a value"
        usage
        exit 1
      fi
      PP_DEGREE="$2"
      shift 2
      ;;
    --pp-degree=*)
      PP_DEGREE="${1#--pp-degree=}"
      shift
      ;;
    --dp-degree)
      if [[ $# -lt 2 ]]; then
        echo "Error: --dp-degree requires a value"
        usage
        exit 1
      fi
      DP_DEGREE="$2"
      shift 2
      ;;
    --dp-degree=*)
      DP_DEGREE="${1#--dp-degree=}"
      shift
      ;;
    --stats-output)
      if [[ $# -lt 2 ]]; then
        echo "Error: --stats-output requires a value"
        usage
        exit 1
      fi
      STATS_OUTPUT="$2"
      shift 2
      ;;
    --stats-output=*)
      STATS_OUTPUT="${1#--stats-output=}"
      shift
      ;;
    --num-nodes)
      if [[ $# -lt 2 ]]; then
        echo "Error: --num-nodes requires a value"
        usage
        exit 1
      fi
      NUM_NODES="$2"
      shift 2
      ;;
    --num-nodes=*)
      NUM_NODES="${1#--num-nodes=}"
      shift
      ;;
    --node-rank)
      if [[ $# -lt 2 ]]; then
        echo "Error: --node-rank requires a value"
        usage
        exit 1
      fi
      NODE_RANK="$2"
      shift 2
      ;;
    --node-rank=*)
      NODE_RANK="${1#--node-rank=}"
      shift
      ;;
    --master-addr)
      if [[ $# -lt 2 ]]; then
        echo "Error: --master-addr requires a value"
        usage
        exit 1
      fi
      MASTER_ADDR="$2"
      shift 2
      ;;
    --master-addr=*)
      MASTER_ADDR="${1#--master-addr=}"
      shift
      ;;
    --master-port)
      if [[ $# -lt 2 ]]; then
        echo "Error: --master-port requires a value"
        usage
        exit 1
      fi
      MASTER_PORT="$2"
      shift 2
      ;;
    --master-port=*)
      MASTER_PORT="${1#--master-port=}"
      shift
      ;;
    --hostfile)
      if [[ $# -lt 2 ]]; then
        echo "Error: --hostfile requires a value"
        usage
        exit 1
      fi
      HOSTFILE="$2"
      shift 2
      ;;
    --hostfile=*)
      HOSTFILE="${1#--hostfile=}"
      shift
      ;;
    *)
      if [[ $SCRIPT_SET -eq 0 && "$1" != --* ]]; then
        TARGET_SCRIPT="$1"
        SCRIPT_SET=1
        shift
      else
        echo "Error: Unexpected argument: $1"
        usage
        exit 1
      fi
      ;;
  esac
done

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

# Build script arguments
SCRIPT_ARGS=("--model-config" "$WRAPPER_MODEL_CONFIG")

# Determine if the script is a multi-GPU distributed script
IS_MULTIGPU=0
TOTAL_PROCS=$PP_DEGREE
GPUS_PER_NODE=$PP_DEGREE

case "$TARGET_SCRIPT" in
  benchmark_pp.py)
    IS_MULTIGPU=1
    SCRIPT_ARGS+=("--pp-degree" "$PP_DEGREE")
    if [[ -n "$STATS_OUTPUT" ]]; then
      SCRIPT_ARGS+=("--stats-output" "$STATS_OUTPUT")
    fi
    TOTAL_PROCS=$PP_DEGREE
    ;;
  benchmark_pp_dp.py)
    IS_MULTIGPU=1
    SCRIPT_ARGS+=("--pp-degree" "$PP_DEGREE" "--dp-degree" "$DP_DEGREE")
    TOTAL_PROCS=$((PP_DEGREE * DP_DEGREE))
    ;;
  benchmark_pp_zero1.py)
    IS_MULTIGPU=1
    SCRIPT_ARGS+=("--pp-degree" "$PP_DEGREE" "--dp-degree" "$DP_DEGREE")
    TOTAL_PROCS=$((PP_DEGREE * DP_DEGREE))
    ;;
  benchmark_pp_ep.py)
    IS_MULTIGPU=1
    SCRIPT_ARGS+=("--pp-degree" "$PP_DEGREE" "--ep-size" "$DP_DEGREE")
    TOTAL_PROCS=$((PP_DEGREE * DP_DEGREE))
    ;;
esac

# Choose launcher based on script type
if [[ $IS_MULTIGPU -eq 1 ]]; then
  if [[ ! "$NUM_NODES" =~ ^[0-9]+$ ]] || [[ "$NUM_NODES" -lt 1 ]]; then
    echo "Error: --num-nodes must be an integer >= 1"
    exit 1
  fi

  if [[ ! "$NODE_RANK" =~ ^[0-9]+$ ]] || [[ "$NODE_RANK" -lt 0 ]] || [[ "$NODE_RANK" -ge "$NUM_NODES" ]]; then
    echo "Error: --node-rank must be in [0, num-nodes-1]"
    exit 1
  fi

  if [[ ! "$MASTER_PORT" =~ ^[0-9]+$ ]] || [[ "$MASTER_PORT" -lt 1 ]] || [[ "$MASTER_PORT" -gt 65535 ]]; then
    echo "Error: --master-port must be in [1, 65535]"
    exit 1
  fi

  if [[ "$NUM_NODES" -gt 1 ]]; then
    if [[ $((TOTAL_PROCS % NUM_NODES)) -ne 0 ]]; then
      echo "Error: total processes ($TOTAL_PROCS) must be divisible by num-nodes ($NUM_NODES)"
      exit 1
    fi
    if [[ -n "$HOSTFILE" ]] && [[ ! -f "$HOSTFILE" ]]; then
      echo "Error: hostfile not found: $HOSTFILE"
      exit 1
    fi
  fi

  GPUS_PER_NODE=$((TOTAL_PROCS / NUM_NODES))
  if [[ "$GPUS_PER_NODE" -lt 1 ]]; then
    echo "Error: computed gpus per node is < 1 (total processes=$TOTAL_PROCS, num-nodes=$NUM_NODES)"
    exit 1
  fi

  DS_LAUNCH_ARGS=(
    "--num_nodes" "$NUM_NODES"
    "--node_rank" "$NODE_RANK"
    "--master_addr" "$MASTER_ADDR"
    "--master_port" "$MASTER_PORT"
    "--num_gpus" "$GPUS_PER_NODE"
  )

  if [[ -n "$HOSTFILE" ]]; then
    DS_LAUNCH_ARGS+=("--hostfile" "$HOSTFILE")
  fi

  echo "Running (distributed): deepspeed ${DS_LAUNCH_ARGS[*]} $SCRIPT_PATH ${SCRIPT_ARGS[*]}"
  exec deepspeed "${DS_LAUNCH_ARGS[@]}" "$SCRIPT_PATH" "${SCRIPT_ARGS[@]}"
else
  echo "Running: python3 $SCRIPT_PATH ${SCRIPT_ARGS[*]}"
  exec python3 "$SCRIPT_PATH" "${SCRIPT_ARGS[@]}"
fi
