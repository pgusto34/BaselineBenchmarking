#!/usr/bin/env python3
"""
Benchmark script using DeepSpeed pipeline parallelism and data parallelism.

Configuration:
- 4 GPUs total
- 2-way pipeline parallelism (2 pipeline stages)
- 2-way data parallelism (2 data parallel groups per pipeline stage)
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
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe import ProcessTopology

from model import Transformer, ModelArgs, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerBlockWrapper(torch.nn.Module):
    """
    Wrapper for TransformerBlock to make it compatible with PipelineModule.
    
    PipelineModule expects layers that take a single tensor input, but TransformerBlock
    requires additional arguments (start_pos, freqs_cis, mask). This wrapper captures
    those arguments and provides a single-tensor forward interface.
    """
    
    def __init__(self, block, start_pos: int, freqs_cis: torch.Tensor, mask: torch.Tensor):
        super().__init__()
        self.block = block
        self.start_pos = start_pos
        self.freqs_cis = freqs_cis
        self.mask = mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that only takes a single tensor input."""
        return self.block(x, self.start_pos, self.freqs_cis, self.mask)


def load_deepspeed_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load DeepSpeed configuration from JSON file.

    Args:
        config_path: Path to DeepSpeed config JSON file.
                    If None, uses default 'ds_config_pp_zero1.json' in script directory.

    Returns:
        DeepSpeed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if config_path is None:
        # Default to ds_config_pp_zero1.json in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "ds_config_pp_zero1.json"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config file not found: {config_path}. "
            "Please create a ds_config_pp_zero1.json file or specify a valid path."
        )

    logger.info(f"Loading DeepSpeed config from: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def create_pipeline_model(
    model_args: ModelArgs,
    seq_len: int,
    device: torch.device,
    num_pipeline_stages: int = 2,
    num_data_parallel_groups: int = 2,
) -> PipelineModule:
    """
    Create a pipeline-parallel model using DeepSpeed PipelineModule.

    This function creates the model and splits it into pipeline stages.
    The model layers are passed directly to PipelineModule, which will
    handle the pipeline parallelism.

    Args:
        model_args: Model configuration arguments
        seq_len: Sequence length
        device: Device to place model on
        num_pipeline_stages: Number of pipeline stages (pp_degree)
        num_data_parallel_groups: Number of data parallel groups per stage (dp_degree)

    Returns:
        PipelineModule instance
    """
    # Create the full transformer model to get the required arguments
    model = Transformer(model_args, seq_len, device)

    # Extract layers in order for pipeline parallelism
    # PipelineModule expects a list of modules that will be split across stages
    layers = []

    # Add embedding layer
    layers.append(model.tok_embeddings)

    # Add transformer blocks wrapped for pipeline compatibility
    # TransformerBlock needs start_pos, freqs_cis, and mask, so we wrap them
    start_pos = 0
    for layer in model.layers:
        wrapped_layer = TransformerBlockWrapper(layer, start_pos, model.freqs_cis, model.mask)
        layers.append(wrapped_layer)

    # Add final norm and output
    layers.append(model.norm)
    layers.append(model.output)

    # Create topology for pipeline parallelism + data parallelism
    # topology defines: (pp_degree, dp_degree)
    # - pp_degree: pipeline parallelism degree (number of pipeline stages)
    # - dp_degree: data parallelism degree (number of GPUs per stage for data parallelism)
    topology = ProcessTopology(
        axes=['pipe', 'data'], dims=[num_pipeline_stages, num_data_parallel_groups]
    )

    # Create pipeline module
    # partition_method='parameters' balances parameters across stages
    # topology defines the parallelism structure for pipeline + data parallelism
    pipeline_model = PipelineModule(
        layers=layers,
        loss_fn=torch.nn.CrossEntropyLoss(),
        topology=topology,
        partition_method="parameters",  # Balance parameters across stages
    )

    return pipeline_model


def benchmark_training(
    model_args: ModelArgs,
    seq_len: int,
    num_iterations: int,
    device: torch.device,
    config_path: Optional[str] = None,
) -> None:
    """
    Run training benchmark with DeepSpeed pipeline parallelism and data parallelism.

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

    # Create pipeline model
    # 2-way pipeline parallelism (2 stages) + 2-way data parallelism (2 GPUs per stage)
    logger.info("Creating pipeline model...")
    pipeline_model = create_pipeline_model(
        model_args, seq_len, device, num_pipeline_stages=2, num_data_parallel_groups=2
    )

    # Load DeepSpeed configuration from file
    ds_config = load_deepspeed_config(config_path)

    # Initialize DeepSpeed engine
    logger.info("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=pipeline_model,
        config=ds_config,
    )

    # Create dummy data
    vocab_size = model_args.vocab_size
    micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]
    gradient_accumulation_steps = ds_config.get("gradient_accumulation_steps", 1)
 
    logger.info(f"Starting training benchmark for {num_iterations} iterations...")
    logger.info(f"Micro batch size: {micro_batch_size}, Sequence length: {seq_len}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

    def create_microbatch_iterator() -> iter:
        """
        Create an iterator that yields microbatches for gradient accumulation.
        
        Yields as many microbatches as gradient_accumulation_steps.
        """
        batches = []
        for _ in range(gradient_accumulation_steps):
            input_ids = torch.randint(
                0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long, device=device
            )
            labels = torch.zeros(
                micro_batch_size, vocab_size, dtype=torch.long, device=device
            )
            batches.append((input_ids, labels))
        return iter(batches)

    # Warmup
    logger.info("Warming up...")
    for _ in range(3):
        # For pipeline parallelism, use train_batch() instead of forward()
        # train_batch expects an iterator with microbatches for gradient accumulation
        loss = model_engine.train_batch(data_iter=create_microbatch_iterator())

    # Synchronize before timing
    if dist.is_initialized():
        dist.barrier()

    # Benchmark
    start_time = time.time()
    total_loss = 0.0

    for iteration in range(num_iterations):
        # For pipeline parallelism, use train_batch() instead of forward()
        # train_batch expects an iterator with microbatches for gradient accumulation
        loss = model_engine.train_batch(data_iter=create_microbatch_iterator())

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
    parser = argparse.ArgumentParser(
        description="DeepSpeed Pipeline + Data Parallelism Benchmark"
    )
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
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (also reads from LOCAL_RANK env var)",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON file (default: ds_config_pp_zero1.json)",
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

