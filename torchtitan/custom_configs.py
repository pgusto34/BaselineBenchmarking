"""
Custom model configs for benchmarking.
Run via: torchrun ... custom_configs.py --module qwen3 --config qwen3_30b_a3b_mini
"""

# --- 1. Inject custom model architectures into qwen3_configs ---

from torchtitan.models.qwen3 import (
    Qwen3Model, Qwen3TransformerBlock, qwen3_configs, model_registry
)
from torchtitan.models.common import FeedForward, GQAttention, RoPE
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.embedding import Embedding

qwen3_configs["30B-A3B-mini"] = Qwen3Model.Config(
    vocab_size=151936,
    dim=2048,
    n_layers=8,
    tok_embeddings=Embedding.Config(),
    output=Linear.Config(),
    norm=RMSNorm.Config(eps=1e-6),
    layer=Qwen3TransformerBlock.Config(
        moe_enabled=True,
        attention_norm=RMSNorm.Config(eps=1e-6),
        ffn_norm=RMSNorm.Config(eps=1e-6),
        moe=MoE.Config(
            hidden_dim=768,
            num_experts=8,
            num_shared_experts=0,
            top_k=2,
            score_func="softmax",
            route_norm=True,
            route_scale=1.0,
            score_before_experts=False,
        ),
        feed_forward=FeedForward.Config(hidden_dim=6144),
        attention=GQAttention.Config(
            n_heads=32,
            n_kv_heads=4,
            head_dim=128,
            attn_backend="sdpa",
            rope_backend="cos_sin",
            q_norm=RMSNorm.Config(eps=1e-6),
            k_norm=RMSNorm.Config(eps=1e-6),
        ),
    ),
    rope=RoPE.Config(
        dim=128,
        max_seq_len=4096,
        theta=1000000.0,
        backend="cos_sin",
    ),
)


# --- 2. Define trainer config functions that reference the custom model ---

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer


def qwen3_30b_a3b_mini() -> Trainer.Config:
    return Trainer.Config(
        hf_assets_path="/opt/hf_assets/Qwen3-0.6B",
        metrics=MetricsProcessor.Config(log_freq=1),
        model_spec=model_registry("30B-A3B-mini"),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        optimizer=OptimizersContainer.Config(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            global_batch_size=-1,
            local_batch_size=8,
            seq_len=1024,
            steps=20,
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=2,
            pipeline_parallel_degree=2,
            pipeline_parallel_microbatch_size=4,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
    )

# Add more custom configs here as needed...


# --- 3. Inject config functions into qwen3's config_registry module ---

from torchtitan.models.qwen3 import config_registry as _cr

_cr.qwen3_30b_a3b_mini = qwen3_30b_a3b_mini


# --- 4. Run training ---

from torchtitan.train import main
main()