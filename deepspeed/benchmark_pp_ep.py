#!/usr/bin/env python3
"""
Benchmark script using DeepSpeed pipeline parallelism and expert parallelism (MoE).

Configuration:
- 4 GPUs total
- 2-way pipeline parallelism (2 pipeline stages)
- Expert parallelism with ep_size=2 (2 expert-parallel groups)
- 8 total experts (4 experts per GPU)
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
from deepspeed.moe.layer import MoE

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
                    If None, uses default 'ds_config_pp_ep.json' in script directory.

    Returns:
        DeepSpeed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if config_path is None:
        # Default to ds_config_pp_ep.json in the same directory as this script
        script_dir = Path(__file__).parent
        config_path = script_dir / "ds_config_pp_ep.json"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"DeepSpeed config file not found: {config_path}. "
            "Please create a ds_config_pp_ep.json file or specify a valid path."
        )

    logger.info(f"Loading DeepSpeed config from: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def create_moe_feedforward(
    dim: int,
    hidden_dim: int,
    multiple_of: int,
    ffn_dim_multiplier: Optional[float],
    num_experts: int = 8,
    ep_size: int = 2,
    k: int = 1,
) -> MoE:
    """
    Create a MoE FeedForward layer using DeepSpeed MoE.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension for FFN
        multiple_of: Make hidden dim multiple of this
        ffn_dim_multiplier: Multiplier for hidden dim
        num_experts: Total number of experts
        ep_size: Expert parallelism size (number of GPUs in expert-parallel group)
        k: Number of experts to select per token (top-k)

    Returns:
        MoE layer wrapping the FeedForward module
    """
    from model import FeedForward

    # Create the expert module (FeedForward)
    # FeedForward will calculate the hidden_dim internally
    expert = FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)

    # Wrap with MoE layer
    moe_layer = MoE(
        hidden_size=dim,
        expert=expert,
        num_experts=num_experts,
        ep_size=ep_size,
        k=k,  # top-k experts per token
        use_residual=False,
    )

    return moe_layer


def create_moe_transformer_block(
    layer_id: int,
    args: ModelArgs,
    num_experts: int = 8,
    ep_size: int = 2,
) -> torch.nn.Module:
    """
    Create a TransformerBlock with MoE FeedForward layer.

    Args:
        layer_id: Layer ID
        args: Model configuration arguments
        num_experts: Total number of experts
        ep_size: Expert parallelism size

    Returns:
        TransformerBlock with MoE FeedForward
    """
    from model import Attention, RMSNorm

    class MoETransformerBlock(torch.nn.Module):
        def __init__(self, layer_id: int, args: ModelArgs, num_experts: int, ep_size: int):
            super().__init__()
            self.n_heads = args.n_heads
            self.dim = args.dim
            self.head_dim = args.dim // args.n_heads
            self.attention = Attention(args)
            
            # Create MoE FeedForward instead of regular FeedForward
            self.feed_forward = create_moe_feedforward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                ffn_dim_multiplier=args.ffn_dim_multiplier,
                num_experts=num_experts,
                ep_size=ep_size,
                k=1,  # top-1 expert selection
            )
            
            self.layer_id = layer_id
            self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        def forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
        ):
            h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
            # MoE layer returns (output, loss) tuple, we only need output
            moe_output, _ = self.feed_forward(self.ffn_norm(h))
            out = h + moe_output
            return out

    return MoETransformerBlock(layer_id, args, num_experts, ep_size)


def create_pipeline_moe_model(
    model_args: ModelArgs,
    seq_len: int,
    device: torch.device,
    num_pipeline_stages: int = 2,
    num_experts: int = 8,
    ep_size: int = 2,
) -> PipelineModule:
    """
    Create a pipeline-parallel MoE model using DeepSpeed PipelineModule.

    Args:
        model_args: Model configuration arguments
        seq_len: Sequence length
        device: Device to place model on
        num_pipeline_stages: Number of pipeline stages (pp_degree)
        num_experts: Total number of experts
        ep_size: Expert parallelism size (number of GPUs in expert-parallel group)
        world_size: Total number of GPUs (for validation)

    Returns:
        PipelineModule instance with MoE layers
    """
    # Note: Topology validation will be done by DeepSpeed's PipelineModule
    # We don't validate here because world_size may not be available at model creation time
    # DeepSpeed will validate that the topology matches the actual world size

    # Create the base transformer model to get embeddings and other components
    model = Transformer(model_args, seq_len, device)

    # Extract layers in order for pipeline parallelism
    layers = []

    # Add embedding layer
    layers.append(model.tok_embeddings)

    # Add transformer blocks with MoE FeedForward, wrapped for pipeline compatibility
    start_pos = 0
    for layer_id in range(model_args.n_layers):
        # Create MoE transformer block
        moe_block = create_moe_transformer_block(layer_id, model_args, num_experts, ep_size)
        wrapped_layer = TransformerBlockWrapper(
            moe_block, start_pos, model.freqs_cis, model.mask
        )
        layers.append(wrapped_layer)

    # Add final norm and output
    layers.append(model.norm)
    layers.append(model.output)

    # Create topology for pipeline parallelism + expert parallelism
    # For MoE with PP: expert parallelism is handled by the MoE layer via ep_size
    # The topology must account for all GPUs: [2 pipeline stages, 2 GPUs per stage] = 4 GPUs
    # With 4 GPUs, 2 pipeline stages, and ep_size=2:
    # - Stage 0: GPUs 0,1 (expert-parallel group, handled by MoE layer)
    # - Stage 1: GPUs 2,3 (expert-parallel group, handled by MoE layer)
    # Topology: [2 pipeline stages, 2 GPUs per stage] = 4 total GPUs
    # Note: ep_size=2 creates expert-parallel groups within each pipeline stage
    # The second dimension (ep_size) represents GPUs per pipeline stage for expert parallelism
    #
    # IMPORTANT: For MoE, the topology should match the structure where expert parallelism
    # groups are created within each pipeline stage. The 'data' dimension here represents
    # the number of GPUs per pipeline stage that will form expert-parallel groups.
    topology = ProcessTopology(axes=['pipe', 'data'], dims=[num_pipeline_stages, ep_size])
    
    logger.info(f"Creating PipelineModule with topology: {num_pipeline_stages} pipeline stages, {ep_size} GPUs per stage")

    # Create pipeline module
    pipeline_model = PipelineModule(
        layers=layers,
        loss_fn=torch.nn.CrossEntropyLoss(),
        topology=topology,
        partition_method="parameters",
    )

    return pipeline_model


def create_moe_param_groups(model: torch.nn.Module) -> list:
    """
    Create parameter groups for MoE optimizer.

    MoE models require separate parameter groups for expert and non-expert parameters.

    Args:
        model: The model to extract parameters from

    Returns:
        List of parameter groups for optimizer
    """
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

    parameters = {'params': [p for p in model.parameters()], 'name': 'parameters'}
    return split_params_into_different_moe_groups_for_optimizer(parameters)


def benchmark_training(
    model_args: ModelArgs,
    seq_len: int,
    num_iterations: int,
    device: torch.device,
    config_path: Optional[str] = None,
    num_experts: int = 8,
    ep_size: int = 2,
) -> None:
    """
    Run training benchmark with DeepSpeed pipeline parallelism and expert parallelism (MoE).

    Args:
        model_args: Model configuration arguments
        seq_len: Sequence length
        num_iterations: Number of training iterations
        device: Device to run on
        config_path: Path to DeepSpeed config JSON file. If None, uses default.
        num_experts: Total number of experts
        ep_size: Expert parallelism size
    """
    # Initialize distributed training
    deepspeed.init_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    logger.info(f"World size: {world_size}, Local rank: {local_rank}")
    logger.info(f"MoE configuration: {num_experts} experts, ep_size={ep_size}")

    # Create pipeline MoE model
    # 2-way pipeline parallelism + expert parallelism (ep_size=2)
    logger.info("Creating pipeline MoE model...")
    pipeline_model = create_pipeline_moe_model(
        model_args,
        seq_len,
        device,
        num_pipeline_stages=2,
        num_experts=num_experts,
        ep_size=ep_size,
    )

    # Load DeepSpeed configuration from file
    ds_config = load_deepspeed_config(config_path)

    # Create MoE parameter groups for optimizer
    moe_param_groups = create_moe_param_groups(pipeline_model)

    # Initialize DeepSpeed engine
    logger.info("Initializing DeepSpeed engine...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=pipeline_model,
        config=ds_config,
        model_parameters=moe_param_groups,
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
            labels = torch.randint(
                0, vocab_size, (micro_batch_size, seq_len), dtype=torch.long, device=device
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
        description="DeepSpeed Pipeline + Expert Parallelism (MoE) Benchmark"
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
        help="Path to DeepSpeed config JSON file (default: ds_config_pp_ep.json)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Total number of experts (default: 8)",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=2,
        help="Expert parallelism size (default: 2)",
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
    logger.info(f"MoE: {args.num_experts} experts, ep_size={args.ep_size}")

    # Run benchmark
    benchmark_training(
        model_args=model_args,
        seq_len=args.seq_len,
        num_iterations=args.iterations,
        device=device,
        config_path=args.deepspeed_config,
        num_experts=args.num_experts,
        ep_size=args.ep_size,
    )


if __name__ == "__main__":
    main()
