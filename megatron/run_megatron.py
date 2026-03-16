#!/usr/bin/env python3
"""Launch Megatron pretrain_gpt.py from a TOML config.

This wrapper keeps Megatron args in one editable TOML file and converts them
into a torchrun command.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List
import tomllib 

def _required(section: Dict[str, Any], key: str, section_name: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing required key '{section_name}.{key}' in TOML config")
    return section[key]


def _bool_flag(args: List[str], enabled: bool, flag: str) -> None:
    if enabled:
        args.append(flag)


def _value_flag(args: List[str], section: Dict[str, Any], key: str, flag: str) -> None:
    if key in section and section[key] is not None:
        args.extend([flag, str(section[key])])


def _multi_value_flag(args: List[str], section: Dict[str, Any], key: str, flag: str) -> None:
    if key not in section or section[key] is None:
        return

    value = section[key]
    if isinstance(value, (list, tuple)):
        args.append(flag)
        args.extend(str(x) for x in value)
    else:
        args.extend([flag, str(value)])


def _compute_ffn_hidden_size(model: Dict[str, Any]) -> int:
    dim = int(model["dim"])
    multiple_of = int(model.get("multiple_of", 256))

    # LLaMA-style default SwiGLU hidden dim: round_up((2/3) * (4 * dim), multiple_of)
    hidden = int((8 * dim) / 3)
    multiplier = model.get("ffn_dim_multiplier")
    if multiplier is not None:
        hidden = int(hidden * float(multiplier))

    return ((hidden + multiple_of - 1) // multiple_of) * multiple_of


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _run_and_report_avg_throughput(cmd: List[str], env: Dict[str, str]) -> None:
    pattern = re.compile(
        r"throughput per GPU \(TFLOP/s/GPU\):\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)"
    )
    throughput_values: List[float] = []

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        match = pattern.search(line)
        if match:
            throughput_values.append(float(match.group(1)))

    return_code = proc.wait()

    if throughput_values:
        avg_throughput = sum(throughput_values) / len(throughput_values)
        print(
            f"[launcher] Average throughput per GPU across {len(throughput_values)} logged iterations: "
            f"{avg_throughput:.4f} TFLOP/s/GPU"
        )
    else:
        print(
            "[launcher] No throughput lines found. Ensure --log-throughput is enabled and"
            " log_interval emits training iteration logs."
        )

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def build_command(cfg: Dict[str, Any], megatron_root: Path) -> List[str]:
    distributed = cfg.get("distributed", {})
    parallelism = cfg.get("parallelism", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    data = cfg.get("data", {})
    precision = cfg.get("precision", {})
    logging_cfg = cfg.get("logging", {})
    output = cfg.get("output", {})

    tp = int(parallelism.get("tp_size", 1))
    cp = int(parallelism.get("cp_size", 1))
    pp = int(parallelism.get("pp_size", 1))

    gpus_per_node = int(distributed.get("gpus_per_node", 1))
    num_nodes = int(distributed.get("num_nodes", 1))
    node_rank = int(distributed.get("node_rank", 0))
    master_addr = str(distributed.get("master_addr", "localhost"))
    master_port = int(distributed.get("master_port", 6000))

    micro_batch_size = int(_required(training, "micro_batch_size", "training"))
    global_batch_size = int(_required(training, "global_batch_size", "training"))
    seq_length = int(_required(training, "seq_length", "training"))
    train_iters = int(_required(training, "train_iters", "training"))
    use_dist_opt = bool(training.get("use_distributed_optimizer", True))

    dtype = str(precision.get("dtype", "fp8")).lower()
    if dtype not in {"bf16", "fp16", "fp8"}:
        raise ValueError("precision.dtype must be one of: bf16, fp16, fp8")

    n_layers = int(_required(model, "n_layers", "model"))
    dim = int(_required(model, "dim", "model"))
    n_heads = int(_required(model, "n_heads", "model"))
    n_kv_heads = model.get("n_kv_heads")
    vocab_size = int(_required(model, "vocab_size", "model"))
    rope_theta = float(model.get("rope_theta", 500000.0))

    ffn_hidden_size = int(model.get("ffn_hidden_size") or _compute_ffn_hidden_size(model))

    pretrain_script = megatron_root / "pretrain_gpt.py"
    if not pretrain_script.exists():
        raise FileNotFoundError(f"pretrain_gpt.py not found at {pretrain_script}")

    cmd: List[str] = [
        "torchrun",
        "--nproc_per_node",
        str(gpus_per_node),
        "--nnodes",
        str(num_nodes),
        "--node_rank",
        str(node_rank),
        "--master_addr",
        master_addr,
        "--master_port",
        str(master_port),
        str(pretrain_script),
        "--use-mcore-models",
        "--num-layers",
        str(n_layers),
        "--hidden-size",
        str(dim),
        "--ffn-hidden-size",
        str(ffn_hidden_size),
        "--num-attention-heads",
        str(n_heads),
        "--seq-length",
        str(seq_length),
        "--max-position-embeddings",
        str(int(training.get("max_position_embeddings", seq_length))),
        "--micro-batch-size",
        str(micro_batch_size),
        "--global-batch-size",
        str(global_batch_size),
        "--train-iters",
        str(train_iters),
        "--tensor-model-parallel-size",
        str(tp),
        "--context-parallel-size",
        str(cp),
        "--pipeline-model-parallel-size",
        str(pp),
        "--position-embedding-type",
        "rope",
        "--rotary-base",
        str(int(rope_theta)),
        "--rotary-percent",
        str(model.get("rotary_percent", 1.0)),
        "--normalization",
        "RMSNorm",
        "--attention-dropout",
        str(model.get("attention_dropout", 0.0)),
        "--hidden-dropout",
        str(model.get("hidden_dropout", 0.0)),
        "--vocab-size",
        str(vocab_size),
        "--lr",
        str(training.get("lr", 1.5e-4)),
        "--min-lr",
        str(training.get("min_lr", 1e-5)),
        "--weight-decay",
        str(training.get("weight_decay", 0.1)),
        "--adam-beta1",
        str(training.get("adam_beta1", 0.9)),
        "--adam-beta2",
        str(training.get("adam_beta2", 0.95)),
        "--clip-grad",
        str(training.get("clip_grad", 1.0)),
        "--lr-decay-style",
        str(training.get("lr_decay_style", "cosine")),
        "--eval-iters",
        str(training.get("eval_iters", logging_cfg.get("eval_iters", 1))),
        "--eval-interval",
        str(training.get("eval_interval", logging_cfg.get("eval_interval", 100))),
        "--log-interval",
        str(training.get("log_interval", logging_cfg.get("log_interval", 1))),
        "--save-interval",
        str(training.get("save_interval", logging_cfg.get("save_interval", 1000))),
        "--distributed-timeout-minutes",
        str(training.get("distributed_timeout_minutes", logging_cfg.get("distributed_timeout_minutes", 60))),
    ]

    _value_flag(cmd, model, "kv_channels", "--kv-channels")
    _value_flag(cmd, model, "init_method_std", "--init-method-std")
    _value_flag(cmd, model, "attention_backend", "--attention-backend")

    _value_flag(cmd, training, "train_samples", "--train-samples")
    _value_flag(cmd, training, "lr_decay_samples", "--lr-decay-samples")
    _value_flag(cmd, training, "lr_warmup_samples", "--lr-warmup-samples")
    _value_flag(cmd, training, "lr_warmup_iters", "--lr-warmup-iters")
    _value_flag(cmd, training, "decoupled_lr", "--decoupled-lr")
    _value_flag(cmd, training, "decoupled_min_lr", "--decoupled-min-lr")
    _value_flag(cmd, training, "empty_unused_memory_level", "--empty-unused-memory-level")
    _value_flag(cmd, training, "exit_duration_in_mins", "--exit-duration-in-mins")
    _value_flag(cmd, training, "recompute_granularity", "--recompute-granularity")
    _value_flag(cmd, training, "recompute_method", "--recompute-method")
    _value_flag(cmd, training, "recompute_num_layers", "--recompute-num-layers")
    _bool_flag(cmd, bool(training.get("distribute_saved_activations", False)), "--distribute-saved-activations")
    _multi_value_flag(cmd, training, "recompute_modules", "--recompute-modules")

    if n_kv_heads is not None:
        n_kv_heads = int(n_kv_heads)
        if n_kv_heads != n_heads:
            cmd.extend(["--group-query-attention", "--num-query-groups", str(n_kv_heads)])

    # Common architecture toggles from your existing script.
    _bool_flag(cmd, bool(model.get("swiglu", True)), "--swiglu")
    _bool_flag(cmd, bool(model.get("disable_bias_linear", True)), "--disable-bias-linear")
    _bool_flag(
        cmd,
        bool(model.get("untie_embeddings_and_output_weights", True)),
        "--untie-embeddings-and-output-weights",
    )
    _bool_flag(cmd, bool(model.get("apply_layernorm_1p", True)), "--apply-layernorm-1p")

    if dtype == "bf16":
        cmd.append("--bf16")
    elif dtype == "fp16":
        cmd.append("--fp16")
    else:
        cmd.extend(
            [
                "--bf16",
                "--fp8-format",
                str(precision.get("fp8_format", "hybrid")),
                "--fp8-amax-history-len",
                str(precision.get("fp8_amax_history_len", 1024)),
                "--fp8-amax-compute-algo",
                str(precision.get("fp8_amax_compute_algo", "max")),
            ]
        )
        if use_dist_opt:
            cmd.append("--fp8-param-gather")

    if bool(parallelism.get("sequence_parallel", True)):
        cmd.append("--sequence-parallel")

    if use_dist_opt:
        cmd.append("--use-distributed-optimizer")
    _bool_flag(cmd, bool(training.get("grad_reduce_in_bf16", False)), "--grad-reduce-in-bf16")
    _bool_flag(cmd, bool(training.get("cross_entropy_loss_fusion", False)), "--cross-entropy-loss-fusion")
    _bool_flag(cmd, bool(training.get("calculate_per_token_loss", False)), "--calculate-per-token-loss")
    _bool_flag(cmd, bool(training.get("manual_gc", False)), "--manual-gc")
    if use_dist_opt:
        _bool_flag(cmd, bool(training.get("overlap_grad_reduce", True)), "--overlap-grad-reduce")
        _bool_flag(cmd, bool(training.get("overlap_param_gather", True)), "--overlap-param-gather")

    # Data mode: mock or real data.
    use_mock = bool(data.get("use_mock_data", True))
    if use_mock:
        cmd.extend(["--mock-data", "--tokenizer-type", str(data.get("tokenizer_type", "NullTokenizer"))])
    else:
        cmd.extend(
            [
                "--data-path",
                str(_required(data, "data_path", "data")),
                "--tokenizer-type",
                str(data.get("tokenizer_type", "HuggingFaceTokenizer")),
                "--tokenizer-model",
                str(_required(data, "tokenizer_model", "data")),
            ]
        )

    data_cache_path = data.get("data_cache_path")
    if data_cache_path:
        cmd.extend(["--data-cache-path", str(data_cache_path)])

    split = str(data.get("split", "99,1,0"))
    cmd.extend(["--split", split])

    _bool_flag(
        cmd,
        bool(data.get("no_create_attention_mask_in_dataloader", True)),
        "--no-create-attention-mask-in-dataloader",
    )
    _bool_flag(cmd, bool(data.get("no_mmap_bin_files", True)), "--no-mmap-bin-files")

    if "num_workers" in data:
        cmd.extend(["--num-workers", str(data["num_workers"])])

    _value_flag(cmd, data, "tiktoken_pattern", "--tiktoken-pattern")

    save_dir = output.get("save")
    if save_dir:
        cmd.extend(["--save", str(save_dir)])

    load_dir = output.get("load")
    if load_dir:
        cmd.extend(["--load", str(load_dir)])

    tensorboard_dir = output.get("tensorboard_dir")
    if tensorboard_dir:
        cmd.extend(["--tensorboard-dir", str(tensorboard_dir)])

    _bool_flag(cmd, bool(training.get("log_throughput", logging_cfg.get("log_throughput", True))), "--log-throughput")

    if "ckpt_format" in training and training["ckpt_format"] is not None:
        cmd.extend(["--ckpt-format", str(training["ckpt_format"])])
    else:
        _value_flag(cmd, logging_cfg, "ckpt_format", "--ckpt-format")

    if bool(training.get("profile", logging_cfg.get("profile", False))):
        cmd.append("--profile")
        if "profile_step_start" in training:
            cmd.extend(["--profile-step-start", str(training["profile_step_start"])])
        elif "profile_step_start" in logging_cfg:
            cmd.extend(["--profile-step-start", str(logging_cfg["profile_step_start"])])
        if "profile_step_end" in training:
            cmd.extend(["--profile-step-end", str(training["profile_step_end"])])
        elif "profile_step_end" in logging_cfg:
            cmd.extend(["--profile-step-end", str(logging_cfg["profile_step_end"])])

    extra_args = cfg.get("extra_args", [])
    for arg in extra_args:
        cmd.append(str(arg))

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch Megatron pretrain_gpt.py from TOML")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to TOML config file",
    )
    parser.add_argument(
        "--megatron-root",
        type=Path,
        default=Path("/opt/Megatron-LM"),
        help="Path to Megatron-LM repository containing pretrain_gpt.py",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=None,
        help="Override distributed.gpus_per_node from TOML",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command and exit without running",
    )

    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.gpus_per_node is not None:
        distributed_cfg = cfg.setdefault("distributed", {})
        distributed_cfg["gpus_per_node"] = int(args.gpus_per_node)

    cmd = build_command(cfg, args.megatron_root)

    print("Resolved command:")
    print(" ".join(shlex.quote(x) for x in cmd))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    env["PYTHONPATH"] = f"{args.megatron_root}:{env.get('PYTHONPATH', '')}"

    _run_and_report_avg_throughput(cmd, env)
    return 0


if __name__ == "__main__":
    sys.exit(main())
