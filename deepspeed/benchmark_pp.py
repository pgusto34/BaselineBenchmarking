#!/usr/bin/env python3
"""
Benchmark script using DeepSpeed pipeline parallelism only.

Configuration:
- Configurable pipeline parallelism degree (pp_degree)
- No data parallelism
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.distributed as dist
import deepspeed
from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe import ProcessTopology

from model import Transformer, ModelArgs, LLAMA_DEBUG, LLAMA_1B, LLAMA_3B, LLAMA_8B, LLAMA_70B


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global state for cleanup
_cleanup_done = False
_last_signal_received: Optional[int] = None


def classify_failure(exc: BaseException) -> str:
    """Classify failure cause to make distributed crash logs easier to triage."""
    if isinstance(exc, KeyboardInterrupt):
        return "keyboard_interrupt"

    if isinstance(exc, SystemExit):
        code = exc.code
        if isinstance(code, int) and code >= 128:
            return f"signal_{code - 128}"
        return "system_exit"

    message = str(exc).lower()
    if "out of memory" in message or "cublas_status_alloc_failed" in message or "std::bad_alloc" in message:
        return "oom"
    if "broken pipe" in message or "tcpstore" in message:
        return "store_disconnect"
    if "nccl" in message:
        return "nccl_error"
    return "unknown"


def cleanup(signum=None, frame=None):
    """Gracefully clean up distributed resources."""
    global _cleanup_done
    if _cleanup_done:
        return
    
    _cleanup_done = True
    logger.info("Starting cleanup...")
    
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed process group")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")
    
    logger.info("Cleanup complete")


def handle_signal(signum, frame):
    """Handle termination signals by cleaning up and exiting immediately."""
    global _last_signal_received
    _last_signal_received = int(signum)
    logger.warning(f"Received signal {signum}; cleaning up and exiting")
    logger.error(f"ROOT_CAUSE category=signal signal={signum}")
    cleanup(signum=signum, frame=frame)
    raise SystemExit(128 + int(signum))


# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


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
                    If None, uses default 'ds_config_pp.json' in script directory.

    Returns:
        DeepSpeed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if config_path is None:
        # Default to ds_config_pp.json in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "ds_config_pp.json"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config file not found: {config_path}. "
            "Please create a ds_config_pp.json file or specify a valid path."
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
    num_data_parallel_groups: int = 1,
) -> PipelineModule:
    """
    Create a pipeline-parallel model using DeepSpeed PipelineModule.

    Args:
        model_args: Model configuration arguments
        seq_len: Sequence length
        device: Device to place model on
        num_pipeline_stages: Number of pipeline stages
        num_data_parallel_groups: Number of data-parallel replicas per pipeline stage

    Returns:
        PipelineModule instance
    """
    # Create the full transformer model to get the required arguments
    model = Transformer(model_args, seq_len, device)

    # Extract layers in order for pipeline parallelism
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

    # Create topology for pipeline + optional data parallelism.
    topology = ProcessTopology(
        axes=['pipe', 'data'],
        dims=[num_pipeline_stages, num_data_parallel_groups],
    )

    # Create pipeline module
    pipeline_model = PipelineModule(
        layers=layers,
        loss_fn=torch.nn.CrossEntropyLoss(),
        topology=topology,
        partition_method="parameters",
    )

    return pipeline_model


def benchmark_training(
    model_args: ModelArgs,
    seq_len: int,
    num_iterations: int,
    device: torch.device,
    pp_degree: int = 2,
    config_path: Optional[str] = None,
    stats_output_path: Optional[str] = None,
) -> None:
    """
    Run training benchmark with DeepSpeed pipeline parallelism only.

    Args:
        model_args: Model configuration arguments
        seq_len: Sequence length
        num_iterations: Number of training iterations
        device: Device to run on
        pp_degree: Number of pipeline stages.
        config_path: Path to DeepSpeed config JSON file. If None, uses default.
        stats_output_path: Optional path to write benchmark statistics as JSON.
    """
    try:
        # Initialize distributed training
        deepspeed.init_distributed()
        if not dist.is_initialized():
            raise RuntimeError("Distributed backend did not initialize")

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

        if pp_degree < 1:
            raise ValueError(f"pp_degree must be >= 1, got {pp_degree}")
        if world_size % pp_degree != 0:
            raise ValueError(
                "world_size must be divisible by pp_degree. "
                f"Got world_size={world_size}, pp_degree={pp_degree}."
            )
        dp_degree = world_size // pp_degree
        logger.info(f"Derived topology: pp_degree={pp_degree}, dp_degree={dp_degree}")

        logger.info("Creating pipeline model...")
        pipeline_model = create_pipeline_model(
            model_args,
            seq_len,
            device,
            num_pipeline_stages=pp_degree,
            num_data_parallel_groups=dp_degree,
        )

        ds_config = load_deepspeed_config(config_path)

        logger.info("Initializing DeepSpeed engine...")
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=pipeline_model,
            config=ds_config,
        )

        vocab_size = model_args.vocab_size
        micro_batch_size = ds_config["train_micro_batch_size_per_gpu"]
        gradient_accumulation_steps = ds_config.get("gradient_accumulation_steps", 1)

        logger.info(f"Starting training benchmark for {num_iterations} iterations...")
        logger.info(f"Micro batch size: {micro_batch_size}, Sequence length: {seq_len}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")

        def create_microbatch_iterator() -> iter:
            """Create an iterator that yields microbatches for gradient accumulation."""
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

        logger.info("Warming up...")
        for _ in range(3):
            _ = model_engine.train_batch(data_iter=create_microbatch_iterator())

        if dist.is_initialized():
            dist.barrier()

        start_time = time.time()
        total_loss = 0.0

        for iteration in range(num_iterations):
            loss = model_engine.train_batch(data_iter=create_microbatch_iterator())
            total_loss += loss.item()

            if (iteration + 1) % 10 == 0 and local_rank == 0:
                avg_loss = total_loss / (iteration + 1)
                logger.info(f"Iteration {iteration + 1}/{num_iterations}, Loss: {avg_loss:.4f}")

        if dist.is_initialized():
            dist.barrier()

        elapsed_time = time.time() - start_time

        if local_rank == 0:
            avg_loss = total_loss / num_iterations
            throughput = num_iterations * micro_batch_size * world_size / elapsed_time
            stats = {
                "total_iterations": num_iterations,
                "elapsed_time_seconds": elapsed_time,
                "average_loss": avg_loss,
                "throughput_samples_per_second": throughput,
                "time_per_iteration_seconds": elapsed_time / num_iterations,
                "micro_batch_size": micro_batch_size,
                "sequence_length": seq_len,
                "world_size": world_size,
                "pp_degree": pp_degree,
                "dp_degree": dp_degree,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
            logger.info("=" * 60)
            logger.info("Benchmark Results:")
            logger.info(f"  Total iterations: {num_iterations}")
            logger.info(f"  Elapsed time: {elapsed_time:.2f} seconds")
            logger.info(f"  Average loss: {avg_loss:.4f}")
            logger.info(f"  Throughput: {throughput:.2f} samples/second")
            logger.info(f"  Time per iteration: {elapsed_time / num_iterations:.4f} seconds")
            logger.info("=" * 60)

            if stats_output_path:
                output_path = Path(stats_output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(stats, f, indent=2)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                logger.info(f"Wrote benchmark stats to: {output_path}")

    except BaseException as e:
        category = classify_failure(e)
        logger.error(
            "ROOT_CAUSE category=%s exception_type=%s message=%s",
            category,
            type(e).__name__,
            str(e),
        )
        logger.error(f"Benchmark failed with error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Benchmark training complete, running cleanup...")
        cleanup()


def main() -> None:
    """Main entry point for the benchmark script."""
    try:
        parser = argparse.ArgumentParser(description="DeepSpeed Pipeline Parallelism Benchmark")
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
            default=2048,
            help="Sequence length",
        )
        parser.add_argument(
            "--iterations",
            type=int,
            default=2,
            help="Number of training iterations",
        )
        parser.add_argument(
            "--pp-degree",
            type=int,
            default=2,
            help="Pipeline parallelism degree (world size must be divisible by this)",
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
            help="Path to DeepSpeed config JSON file (default: ds_config_pp.json)",
        )
        parser.add_argument(
            "--stats-output",
            type=str,
            default=None,
            help="Optional path to write benchmark statistics as JSON",
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
        logger.info(f"Pipeline degree: {args.pp_degree}")
        logger.info(f"Sequence length: {args.seq_len}")
        logger.info(f"Iterations: {args.iterations}")

        # Run benchmark
        benchmark_training(
            model_args=model_args,
            seq_len=args.seq_len,
            num_iterations=args.iterations,
            device=device,
            pp_degree=args.pp_degree,
            config_path=args.deepspeed_config,
            stats_output_path=args.stats_output,
        )

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        cleanup()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()


