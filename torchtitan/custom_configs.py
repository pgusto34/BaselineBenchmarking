"""
Custom model configs for benchmarking.
Run via: torchrun ... custom_configs.py --module qwen3 --config qwen3_30b_a3b_mini
    or: torchrun ... custom_configs.py --module llama3 --config llama3_1b_dp_ds_like
    or: torchrun ... custom_configs.py --module llama3 --config llama3_2_1b_dp_ds_like
    or: torchrun ... custom_configs.py --module llama3 --config llama3_2_1b_fsdp
"""

# --- 1. Inject custom model architectures into qwen3_configs ---

from torchtitan.models.qwen3 import (
    Qwen3Model,
    Qwen3TransformerBlock,
    qwen3_configs,
    model_registry as qwen3_model_registry,
)
import dataclasses as _dc
from torchtitan.models.common import FeedForward, GQAttention, RoPE
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.embedding import Embedding

# qwen3_configs["30B-A3B-mini"] = Qwen3Model.Config(
#     vocab_size=151936,
#     dim=2048,
#     n_layers=8,
#     tok_embeddings=Embedding.Config(),
#     output=Linear.Config(),
#     norm=RMSNorm.Config(eps=1e-6),
#     layer=Qwen3TransformerBlock.Config(
#         moe_enabled=True,
#         attention_norm=RMSNorm.Config(eps=1e-6),
#         ffn_norm=RMSNorm.Config(eps=1e-6),
#         moe=MoE.Config(
#             hidden_dim=768,
#             num_experts=8,
#             num_shared_experts=0,
#             top_k=2,
#             score_func="softmax",
#             route_norm=True,
#             route_scale=1.0,
#             score_before_experts=False,
#         ),
#         feed_forward=FeedForward.Config(hidden_dim=6144),
#         attention=GQAttention.Config(
#             n_heads=32,
#             n_kv_heads=4,
#             head_dim=128,
#             attn_backend="sdpa",
#             rope_backend="cos_sin",
#             q_norm=RMSNorm.Config(eps=1e-6),
#             k_norm=RMSNorm.Config(eps=1e-6),
#         ),
#     ),
#     rope=RoPE.Config(
#         dim=128,
#         max_seq_len=4096,
#         theta=1000000.0,
#         backend="cos_sin",
#     ),
# )


# --- 2. Define trainer config functions that reference the custom model ---

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import ActivationCheckpointConfig, ParallelismConfig, TrainingConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.trainer import Trainer


# def qwen3_30b_a3b_mini() -> Trainer.Config:
#     return Trainer.Config(
#         hf_assets_path="/opt/hf_assets/Qwen3-0.6B",
#         metrics=MetricsProcessor.Config(log_freq=1),
#         model_spec=qwen3_model_registry("30B-A3B-mini"),
#         dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
#         optimizer=OptimizersContainer.Config(lr=8e-4),
#         lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
#         training=TrainingConfig(
#             global_batch_size=-1,
#             local_batch_size=8,
#             seq_len=1024,
#             steps=20,
#             mixed_precision_param="bfloat16",
#         ),
#         parallelism=ParallelismConfig(
#             data_parallel_shard_degree=-1,
#             expert_parallel_degree=2,
#             pipeline_parallel_degree=2,
#             pipeline_parallel_microbatch_size=4,
#         ),
#         checkpoint=CheckpointManager.Config(enable=False),
#         activation_checkpoint=ActivationCheckpointConfig(mode="none"),
#     )


def llama3_1b_dp_ds_like() -> Trainer.Config:
    """Llama3-1B config aligned with the DeepSpeed DP benchmark intent."""
    from torchtitan.models.llama3 import (
        llama3_configs,
        model_registry as llama3_model_registry,
    )

    if "1B_untied" not in llama3_configs:
        llama3_configs["1B_untied"] = _dc.replace(
            llama3_configs["1B"], enable_weight_tying=False
        )

    return Trainer.Config(
        hf_assets_path="/opt/hf_assets/Llama-3.1-8B",
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        model_spec=llama3_model_registry("1B_untied"),
        optimizer=OptimizersContainer.Config(
            lr=1e-4,
            weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=5),
        training=TrainingConfig(
            global_batch_size=32,
            local_batch_size=16,
            seq_len=1024,
            steps=20,
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
    )


def llama3_2_1b_dp_ds_like() -> Trainer.Config:
    """Llama3.2-1B config with DP settings matching the 3.1 DP profile."""
    from torchtitan.models.llama3 import (
        llama3_configs,
        model_registry as llama3_model_registry,
    )

    if "1B_untied" not in llama3_configs:
        llama3_configs["1B_untied"] = _dc.replace(
            llama3_configs["1B"], enable_weight_tying=False
        )

    return Trainer.Config(
        hf_assets_path="/opt/hf_assets/Llama-3.2-1B",
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        model_spec=llama3_model_registry("1B_untied"),
        optimizer=OptimizersContainer.Config(
            lr=1e-4,
            weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=5),
        training=TrainingConfig(
            global_batch_size=32,
            local_batch_size=16,
            seq_len=1024,
            steps=20,
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
    )


def llama3_2_1b_fsdp() -> Trainer.Config:
    """Llama3.2-1B config using PyTorch FSDP-style sharded data parallelism."""
    from torchtitan.models.llama3 import (
        llama3_configs,
        model_registry as llama3_model_registry,
    )

    if "1B_untied" not in llama3_configs:
        llama3_configs["1B_untied"] = _dc.replace(
            llama3_configs["1B"], enable_weight_tying=False
        )

    return Trainer.Config(
        hf_assets_path="/opt/hf_assets/Llama-3.2-1B",
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4"),
        model_spec=llama3_model_registry("1B_untied"),
        optimizer=OptimizersContainer.Config(
            lr=1e-4,
            weight_decay=0.01,
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=5),
        training=TrainingConfig(
            global_batch_size=32,
            local_batch_size=16,
            seq_len=1024,
            steps=20,
            mixed_precision_param="bfloat16",
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="none"),
    )

# Add more custom configs here as needed...


# --- 3. Inject config functions into qwen3's config_registry module ---

# from torchtitan.models.qwen3 import config_registry as _cr

# _cr.qwen3_30b_a3b_mini = qwen3_30b_a3b_mini

from torchtitan.models.llama3 import config_registry as _llama3_cr

_llama3_cr.llama3_1b_dp_ds_like = llama3_1b_dp_ds_like
_llama3_cr.llama3_2_1b_dp_ds_like = llama3_2_1b_dp_ds_like
_llama3_cr.llama3_2_1b_fsdp = llama3_2_1b_fsdp


# --- 4. Run training ---

from torchtitan.train import main
main()