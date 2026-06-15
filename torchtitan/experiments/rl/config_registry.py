# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config entry points for the RL/unified experiment.

Each function returns a complete ``RLTrainer.Config`` and is discoverable by
``ConfigManager`` via ``--module rl --config <function_name>``.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.grpo import GRPOLoss, RLTrainer
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.sum_digits import SumDigitsEnv
from torchtitan.experiments.rl.veribench_env import VeribenchEnv
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    deterministic_debug = DebugConfig(
        seed=1234,
        deterministic=True,
        deterministic_warn_only=True,
    )
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=50,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=8, global_batch_size=32, seq_len=300),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=deterministic_debug,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
            debug=deterministic_debug,
        ),
    )



def grpo_qwen3_0_6b_veribench() -> RLTrainer.Config:
    """Multi-step Qwen3-0.6B GRPO run for Veribench + CodingReward."""
    model_spec = model_registry("0.6B", attn_backend="varlen")
    model_spec.model.rope.max_seq_len = 5120
    return RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=300,
        num_prompts_per_step=4,
        num_validation_samples=8,
        env_step_concurrency=4,
        log_samples=False,
        log_rich_completions=False,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            enable_tensorboard=False,
            wandb_project="ishikori",
        ),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=32, seq_len=5120),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=VeribenchEnv.Config(
            split="medium_train",
            seed=42,
            enable_thinking=True,
            compilation_credit=0.1,
            max_len=4050,
            penalised_len=2500,
            # Turn off length penalty
            max_length_penalty=0.0,
        ),
        validation_env=VeribenchEnv.Config(
            split="medium_validation",
            seed=99,
            enable_thinking=True,
            compilation_credit=0.1,
            max_len=4050,
            penalised_len=2500,
            # Turn off length penalty
            max_length_penalty=0.0,
        ),
        trainer=PolicyTrainer.Config(
            ac_config=ActivationCheckpointConfig(mode="full"),
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(seq_len=5120),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=6,
                tensor_parallel_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=75,
                last_save_model_only=True,
                last_save_in_hf=True,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=8,
                temperature=1,
                top_p=0.95,
                max_tokens=4096,
            ),
        ),
    )

def grpo_qwen3_8b_veribench_medium() -> RLTrainer.Config:
    """Qwen3-8B GRPO multinode run for the medium Veribench subset."""
    model_spec = model_registry("8B", attn_backend="varlen")
    model_spec.model.rope.max_seq_len = 16384
    return RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-8B",
        num_steps=250,
        num_prompts_per_step=16, # This does not now define the amount of prompts per training batch, it is the number of prompts that the trainer samples at a time to reach num_target_tokens
        num_validation_samples=48,
        validation_interval=50,
        env_step_concurrency=8,
        log_samples=False,
        log_rich_completions=False,
        filter_zero_std_groups=True,
        metrics=MetricsProcessor.Config(
            enable_wandb=True,
            enable_tensorboard=False,
            wandb_project="ishikori",
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=1, global_batch_size=128, seq_len=16384),
        ),
        env=VeribenchEnv.Config(
            split="medium_train",
            seed=42,
            enable_thinking=True,
            compilation_credit=0.0,
            coding_reward="pass_rate",
            # Turn off length penalty
            max_length_penalty=0.0
        ),
        validation_env=VeribenchEnv.Config(
            split="medium_validation",
            seed=99,
            enable_thinking=True,
            compilation_credit=0.0,
            coding_reward="pass_rate",
            # Turn off length penalty
            max_length_penalty=0.0,
        ),
        trainer=PolicyTrainer.Config(
            ac_config=ActivationCheckpointConfig(mode="full"),
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="float32"),
            parallelism=ParallelismConfig(
                # DP degree is 16 here (fsdp = 8 * dp = 2)
                data_parallel_shard_degree=8,
                data_parallel_replicate_degree=2,
                tensor_parallel_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=75,
                last_save_model_only=True,
                last_save_in_hf=True,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=8,
                temperature=1,
                top_p=0.95,
                max_tokens=15000,
            ),
        ),
    )


def grpo_qwen3_8b_veribench_medium_multinode_resume() -> RLTrainer.Config:
    """Resume the Qwen3-8B medium multinode run from a saved checkpoint."""
    config = grpo_qwen3_8b_veribench_medium()
    # Edit this checkpoint path before launching a resume run.
    config.trainer.checkpoint.initial_load_path = (
        "/data/seanc/ishikori/post_training_torchtitan/outputs/"
        "28-05-26-10-35-07/checkpoint/step-75"
    )
    config.trainer.checkpoint.initial_load_model_only = False
    config.trainer.checkpoint.initial_load_in_hf = False
    config.trainer.checkpoint.load_step = -1
    return config

def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_14b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("14B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
        num_steps=10,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=8,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B under same parallelism (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL training.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_prompts_per_step=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        env=SumDigitsEnv.Config(seed=42, correctness_reward=1.0, format_reward=0.3),
        validation_env=SumDigitsEnv.Config(
            seed=99, correctness_reward=1.0, format_reward=0.3
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=OptimizersContainer.Config(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # bfloat16 is needed for trainer to align with generator dtype
            # TODO: replace bfloat16 enablement with FSDP2+TP2
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                n=group_size,
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
            debug=batch_invariant_config,
        ),
    )
