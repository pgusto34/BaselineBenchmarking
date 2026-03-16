#!/usr/bin/env python3
"""
Benchmark script using DeepSpeed on a single device.

Configuration:
- 1 GPU total
- No pipeline parallelism
- No data parallelism
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import deepspeed

from model import Transformer, ModelArgs, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_deepspeed_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load DeepSpeed configuration from JSON file.

    Args:
        config_path: Path to DeepSpeed config JSON file.
                    If None, uses default 'ds_config_single.json' in script directory.

    Returns:
        DeepSpeed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if config_path is None:
        # Default to ds_config_single.json in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "ds_config_single.json"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config file not found: {config_path}. "
            "Please create a ds_config_single.json file or specify a valid path."
        )

    logger.info(f"Loading DeepSpeed config from: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def benchmark_training(
    model_args: ModelArgs,
    seq_len: int,
    num_iterations: int,
    device: torch.device,
    config_path: Optional[str] = None,
) -> None:
    """
    Run training benchmark on a single device with DeepSpeed.

    Args:
        model_args: Model configuration arguments
        seq_len: Sequence length
        num_iterations: Number of training iterations
        device: Device to run on
        config_path: Path to DeepSpeed config JSON file. If None, uses default.
    """
    logger.info("Running single device benchmark (no parallelism)")

    # Create standard model (no pipeline parallelism, no data parallelism)
    logger.info("Creating model...")
    model = Transformer(model_args, seq_len, device)

    # Load DeepSpeed configuration from file
    ds_config = load_deepspeed_config(config_path)

    # Initialize DeepSpeed engine (single device)
    logger.info("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
    )

    # Create dummy data
    vocab_size = model_args.vocab_size
    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]

    logger.info(f"Starting training benchmark for {num_iterations} iterations...")
    logger.info(f"Micro batch size: {micro_batch_size}, Sequence length: {seq_len}")

    # Warmup
    logger.info("Warming up...")
    for _ in range(3):
        input_ids = torch.randint(
            0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long, device=device
        )
        labels = torch.randint(
            0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long, device=device
        )
        output = model_engine(input_ids)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, vocab_size), labels.view(-1)
        )
        model_engine.backward(loss)
        model_engine.step()

    # Benchmark
    start_time = time.time()
    total_loss = 0.0

    for iteration in range(num_iterations):
        input_ids = torch.randint(
            0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long, device=device
        )
        labels = torch.randint(
            0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long, device=device
        )

        output = model_engine(input_ids)
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, vocab_size), labels.view(-1)
        )
        model_engine.backward(loss)
        model_engine.step()

        total_loss += loss.item()

        if (iteration + 1) % 10 == 0:
            avg_loss = total_loss / (iteration + 1)
            logger.info(f"Iteration {iteration + 1}/{num_iterations}, Loss: {avg_loss:.4f}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    avg_loss = total_loss / num_iterations
    throughput = num_iterations * micro_batch_size / elapsed_time
    logger.info("=" * 60)
    logger.info("Benchmark Results:")
    logger.info(f"  Total iterations: {num_iterations}")
    logger.info(f"  Elapsed time: {elapsed_time:.2f} seconds")
    logger.info(f"  Average loss: {avg_loss:.4f}")
    logger.info(f"  Throughput: {throughput:.2f} samples/second")
    logger.info(f"  Time per iteration: {elapsed_time / num_iterations:.4f} seconds")
    logger.info("=" * 60)


def main() -> None:
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(description="DeepSpeed Single Device Benchmark")
    parser.add_argument(
        "--model-config",
        type=str,
        default="debug",
        choices=["debug", "1b", "3b", "8b"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device ID to use (default: 0)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (also reads from LOCAL_RANK env var). "
        "Ignored if --device is specified.",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON file (default: ds_config_single.json)",
    )

    args = parser.parse_args()

    # Set model configuration
    model_configs = {
        "debug": LLAMA_DEBUG,
        "1b": LLAMA_1B,
        "3b": LLAMA_3B,
        "8b": LLAMA_8B,
    }
    model_args = model_configs.get(args.model_config, LLAMA_DEBUG)

    # Set device - prioritize environment variable (set by DeepSpeed/torchrun) if available
    # Otherwise use --device argument
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else -1))
    if local_rank >= 0:
        # Use local_rank from environment/argument (when launched with deepspeed/torchrun)
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    elif torch.cuda.is_available():
        # Use --device argument (when run directly with python)
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU")

    logger.info(f"Using device: {device}")
    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Sequence length: {args.seq_len}")
    logger.info(f"Iterations: {args.iterations}")

    # Run benchmark
    benchmark_training(
        model_args=model_args,
        seq_len=args.seq_len,
        num_iterations=args.iterations,
        device=device,
        config_path=args.deepspeed_config,
    )


if __name__ == "__main__":
    main()

