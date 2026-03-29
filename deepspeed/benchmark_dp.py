#!/usr/bin/env python3
"""
Benchmark script using DeepSpeed data parallelism only.

Configuration:
- 2 GPUs total
- 2-way data parallelism (standard DDP)
- No pipeline parallelism
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
import deepspeed

from model import Transformer, ModelArgs, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B, LLAMA_70B


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_deepspeed_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load DeepSpeed configuration from JSON file.

    Args:
        config_path: Path to DeepSpeed config JSON file.
                    If None, uses default 'ds_config_dp.json' in script directory.

    Returns:
        DeepSpeed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if config_path is None:
        # Default to ds_config_dp.json in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "ds_config_dp.json"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config file not found: {config_path}. "
            "Please create a ds_config_dp.json file or specify a valid path."
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
    Run training benchmark with DeepSpeed data parallelism only.

    Args:
        model_args: Model configuration arguments
        seq_len: Sequence length
        num_iterations: Number of training iterations
        device: Device to run on
        config_path: Path to DeepSpeed config JSON file. If None, uses default.
    """
    # Initialize distributed training
    deepspeed.init_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    # Create standard model (no pipeline parallelism)
    logger.info("Creating model...")
    model = Transformer(model_args, seq_len, device)

    # Load DeepSpeed configuration from file
    ds_config = load_deepspeed_config(config_path)

    # Initialize DeepSpeed engine (standard DDP)
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

    # Synchronize before timing
    if dist.is_initialized():
        dist.barrier()

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

        if (iteration + 1) % 10 == 0 and local_rank == 0:
            avg_loss = total_loss / (iteration + 1)
            logger.info(f"Iteration {iteration + 1}/{num_iterations}, Loss: {avg_loss:.4f}")

    # Synchronize after timing
    if dist.is_initialized():
        dist.barrier()

    end_time = time.time()
    elapsed_time = end_time - start_time

    if local_rank == 0:
        avg_loss = total_loss / num_iterations
        throughput = num_iterations * micro_batch_size * world_size / elapsed_time
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
    parser = argparse.ArgumentParser(description="DeepSpeed Data Parallelism Benchmark")
    parser.add_argument(
        "--model-config",
        type=str,
        default="debug",
        choices=["debug", "1b", "3b", "8b", "70b"],
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
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (also reads from LOCAL_RANK env var)",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON file (default: ds_config_dp.json)",
    )

    args = parser.parse_args()

    # Set model configuration
    model_configs = {
        "debug": LLAMA_DEBUG,
        "1b": LLAMA_1B,
        "3b": LLAMA_3B,
        "8b": LLAMA_8B,
        "70b": LLAMA_70B,
    }
    model_args = model_configs.get(args.model_config, LLAMA_DEBUG)

    # Set device - prioritize environment variable (set by DeepSpeed/torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank if args.local_rank >= 0 else -1))
    if local_rank >= 0:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


