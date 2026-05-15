# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with VLLMGenerator (vLLM) and PolicyTrainer (TorchTitan)
   running on separate GPU meshes
2. Weight synchronization across meshes: trainer gathers full (unsharded) weights,
   generator reshards to match its own parallelism layout via distribute_tensor
3. Envs driven rollouts; reward and advantage computation live inline
   in the controller.

Command to run:
python3 torchtitan/experiments/rl/grpo.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import inspect
import json
import logging
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import (
    CompileConfig,
    ConfigManager,
    Configurable,
    ParallelismConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.types import (
    Completion,
    Episode,
    Step,
    TrainBatch,
    Trajectory,
)
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


async def run_env_step(env: object, completion: str) -> Step:
    """Run a sync or async env.step and return a Step."""
    result = env.step(completion)
    # Veribench step is async because it calls CodingReward, which runs the evaluator/simulator asynchronously.
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, Step):
        raise TypeError(f"env.step returned {type(result)!r}, expected type was Step")
    return result


async def run_env_steps(
    envs: list[object],
    completions: list[Completion],
    *,
    concurrency: int,
) -> list[tuple[Completion, Step]]:
    """Score completions with bounded env-step concurrency. 
    Concurrency controls how many verilog simulator jobs can run at once."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def run_one(completion: Completion) -> tuple[Completion, Step]:
        async with semaphore:
            env = envs[completion.prompt_idx]
            return completion, await run_env_step(env, completion.text)

    return await asyncio.gather(*(run_one(completion) for completion in completions))


class GRPOLoss(Configurable):
    """Clipped GRPO surrogate loss.

    Takes per-sample response logprobs (already extracted from whatever
    packing or padding format the trainer uses).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO clipping epsilon for the probability ratio."""

    def __init__(self, config: Config):
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        policy_logprobs: list[torch.Tensor],
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        per_sample_mean_lps = []
        for policy_lps in policy_logprobs:
            per_sample_mean_lps.append(policy_lps.mean())

        mean_log_ratio = torch.stack(per_sample_mean_lps)
        ratio = torch.exp(mean_log_ratio)

        unclipped_loss = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        clipped_loss = clipped_ratio * advantages
        pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

        metrics = {
            "pg_loss": pg_loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
            .float()
            .mean()
            .item(),
        }
        return pg_loss, metrics


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    In non-colocated mode, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to ``allocate(n)`` reserves the next *n* GPUs and returns a
    bootstrap callable that sets ``CUDA_VISIBLE_DEVICES`` before CUDA
    initializes in the spawned process, ensuring each mesh only sees its
    own devices.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
            import torch  # noqa: F401

        return _bootstrap


def _mean_rewards(steps: list[Step]) -> dict[str, float]:
    """Per-component mean reward across a list of Steps."""
    if not steps:
        return {}
    keys = sorted({key for step in steps for key in step.rewards})
    return {
        key: sum(step.rewards.get(key, 0.0) for step in steps) / len(steps)
        for key in keys
    }


def _format_rewards(components: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:+.3f}" for k, v in components.items())


def _format_validation(result: dict) -> str:
    return (
        f"mean_reward={result['mean_reward']:+.3f} "
        f"({_format_rewards(result['components'])})"
    )


def _sample_first_per_prompt(
    items: list[Episode] | list[Completion],
) -> list[Episode] | list[Completion]:
    """Return the same first-per-prompt samples shown by ``_log_samples``."""
    seen_prompts: set[int] = set()
    sampled = []
    for item in items:
        if item.prompt_idx in seen_prompts:
            continue
        seen_prompts.add(item.prompt_idx)
        sampled.append(item)
    return sampled


def _log_samples(items: list[Episode] | list[Completion]) -> None:
    """Log the first sample per prompt for debugging."""
    for item in _sample_first_per_prompt(items):
        reward_str = f" reward={item.reward:+.1f}" if hasattr(item, "reward") else ""
        logger.info(f"  [prompt {item.prompt_idx}]{reward_str}")
        logger.info(f"       A: {item.text[:300].replace(chr(10), ' ').strip()}")


def _preview_text(value: object, *, max_chars: int = 300) -> str:
    if value is None:
        return ""
    return str(value)[:max_chars].replace(chr(10), " ").strip()


def _step_metadata(step: Step) -> dict[str, object]:
    return step.metadata or {}


def _log_validation_samples(
    envs: list[object],
    completion_steps: list[tuple[Completion, Step]],
    *,
    max_samples: int = 2,
) -> None:
    """Log validation samples with optional env/reward metadata."""
    for completion, step in completion_steps[:max_samples]:
        env = envs[completion.prompt_idx]
        metadata = _step_metadata(step)
        logger.info(
            "  [validation prompt %s] problem_id=%s target=%s rewards=%s "
            "compilation_passed=%s func_passed=%s",
            completion.prompt_idx,
            metadata.get("problem_id", getattr(env, "problem_id", None)),
            metadata.get("target", getattr(env, "target", None)),
            step.rewards,
            metadata.get("compilation_passed"),
            metadata.get("func_passed"),
        )
        logger.info(
            "       question: %s",
            _preview_text(
                metadata.get("question", getattr(env, "question", None))
            ),
        )
        logger.info("       completion: %s", _preview_text(completion.text))
        logger.info(
            "       final_text: %s",
            _preview_text(
                metadata.get("final_text", getattr(env, "final_text", None))
            ),
        )


def _append_jsonl_record(path: str | None, record: dict[str, object]) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _log_training_step_diagnostics(
    step_idx: int,
    trajectories: list[Trajectory],
    *,
    num_prompts: int,
    completions_per_prompt: int,
    max_response_tokens: int | None = None,
    final_text_stats_path: str | None = None,
) -> None:
    steps = [trajectory.transitions[0][1] for trajectory in trajectories]
    rewards = [step.reward for step in steps]
    reward_mean = sum(rewards) / len(rewards) if rewards else 0.0
    reward_min = min(rewards) if rewards else 0.0
    reward_max = max(rewards) if rewards else 0.0
    metadata = [_step_metadata(step) for step in steps]
    compilation_passed = sum(item.get("compilation_passed") is True for item in metadata)
    func_passed = sum(item.get("func_passed") is True for item in metadata)
    format_failed = sum(item.get("format_passed") is False for item in metadata)

    grouped_rewards: dict[int, list[float]] = {}
    for trajectory, reward in zip(trajectories, rewards):
        grouped_rewards.setdefault(trajectory.sample_idx, []).append(reward)
    zero_variance_groups = sum(
        len(group) > 1 and max(group) == min(group)
        for group in grouped_rewards.values()
    )

    total_completions = len(trajectories)
    format_passed = total_completions - format_failed
    format_passed_rate = (
        format_passed / total_completions if total_completions else 0.0
    )
    completions = [trajectory.transitions[0][0] for trajectory in trajectories]
    response_lengths = [len(completion.token_ids) for completion in completions]
    prompt_lengths = [len(completion.prompt_token_ids) for completion in completions]
    sequence_lengths = [
        prompt_len + response_len
        for prompt_len, response_len in zip(prompt_lengths, response_lengths)
    ]
    prompt_length_min = min(prompt_lengths) if prompt_lengths else 0
    prompt_length_max = max(prompt_lengths) if prompt_lengths else 0
    prompt_length_mean = (
        sum(prompt_lengths) / total_completions if total_completions else 0.0
    )
    sequence_length_min = min(sequence_lengths) if sequence_lengths else 0
    sequence_length_max = max(sequence_lengths) if sequence_lengths else 0
    sequence_length_mean = (
        sum(sequence_lengths) / total_completions if total_completions else 0.0
    )
    response_length_min = min(response_lengths) if response_lengths else 0
    response_length_max = max(response_lengths) if response_lengths else 0
    response_length_mean = (
        sum(response_lengths) / total_completions if total_completions else 0.0
    )
    format_failed_flags = [
        item.get("format_passed") is False for item in metadata
    ]
    max_length_completions = (
        sum(
            is_format_failed and length >= max_response_tokens
            for is_format_failed, length in zip(format_failed_flags, response_lengths)
        )
        if max_response_tokens is not None
        else 0
    )

    logger.info(
        "Step %2d diagnostics | rollouts=%d prompts=%d completions_per_prompt=%d "
        "reward: mean=%+.3f min=%+.3f max=%+.3f | compilation_passed=%d "
        "func_passed=%d format_failed=%d max_length_completions=%d "
        "zero_variance_groups=%d",
        step_idx,
        total_completions,
        num_prompts,
        completions_per_prompt,
        reward_mean,
        reward_min,
        reward_max,
        compilation_passed,
        func_passed,
        format_failed,
        max_length_completions,
        zero_variance_groups,
    )
    _append_jsonl_record(
        final_text_stats_path,
        {
            "split": "train",
            "step": step_idx,
            "total_completions": total_completions,
            "format_passed": format_passed,
            "format_failed": format_failed,
            "max_length_completions": max_length_completions,
            "format_passed_rate": format_passed_rate,
            "format_failed_rate": 1.0 - format_passed_rate,
            "num_prompts": num_prompts,
            "completions_per_prompt": completions_per_prompt,
            "prompt_length_min": prompt_length_min,
            "prompt_length_mean": prompt_length_mean,
            "prompt_length_max": prompt_length_max,
            "sequence_length_min": sequence_length_min,
            "sequence_length_mean": sequence_length_mean,
            "sequence_length_max": sequence_length_max,
            "response_length_min": response_length_min,
            "response_length_mean": response_length_mean,
            "response_length_max": response_length_max,
            "max_length_completion_rate": (
                max_length_completions / total_completions
                if total_completions
                else 0.0
            ),
            "compilation_passed": compilation_passed,
            "func_passed": func_passed,
            "reward_mean": reward_mean,
            "reward_min": reward_min,
            "reward_max": reward_max,
        },
    )

def _log_train_batch_stats(step_idx: int, batches: list[TrainBatch]) -> None:
    summaries: list[str] = []
    for rank, batch in enumerate(batches):
        episode_count = len(batch.seq_lens)
        total_tokens = int(batch.token_ids.numel())
        prompt_total = sum(batch.prompt_lens)
        response_total = sum(batch.response_lens)
        max_seq_len = max(batch.seq_lens) if batch.seq_lens else 0
        min_seq_len = min(batch.seq_lens) if batch.seq_lens else 0
        max_prompt_len = max(batch.prompt_lens) if batch.prompt_lens else 0
        max_response_len = max(batch.response_lens) if batch.response_lens else 0
        mean_seq_len = sum(batch.seq_lens) / episode_count if episode_count else 0.0
        mean_response_len = response_total / episode_count if episode_count else 0.0
        summaries.append(
            f"rank={rank} episodes={episode_count} total_tokens={total_tokens} "
            f"seq[min/mean/max]={min_seq_len}/{mean_seq_len:.1f}/{max_seq_len} "
            f"prompt_total={prompt_total} response_total={response_total} "
            f"response_mean={mean_response_len:.1f} max_prompt={max_prompt_len} "
            f"max_response={max_response_len}"
        )
    logger.info("Step %2d train batch stats | %s", step_idx, " | ".join(summaries))


def _full_completion_record(
    *,
    split: str,
    step_idx: int | None,
    completion: Completion,
    step: Step,
) -> dict[str, object]:
    metadata = _step_metadata(step)
    return {
        "split": split,
        "step": step_idx,
        "prompt_idx": completion.prompt_idx,
        "policy_version": completion.policy_version,
        "problem_id": metadata.get("problem_id"),
        "target": metadata.get("target"),
        "rewards": step.rewards,
        "compilation_passed": metadata.get("compilation_passed"),
        "func_passed": metadata.get("func_passed"),
        "format_passed": metadata.get("format_passed"),
        "format_failed": metadata.get("format_passed") is False,
        "format_reward": metadata.get("format_reward"),
        "format_failure_reason": metadata.get("format_failure_reason"),
        "empty_response": metadata.get("empty_response"),
        "prompt_token_count": len(completion.prompt_token_ids),
        "completion_token_count": len(completion.token_ids),
        "sequence_token_count": len(completion.prompt_token_ids)
        + len(completion.token_ids),
        "question": metadata.get("question"),
        "final_text": metadata.get("final_text"),
        "extracted_code": metadata.get("extracted_code"),
        "coding_failure_reason": metadata.get("coding_failure_reason"),
        "coding_reward_log": metadata.get("coding_reward_log"),
        "completion_text": completion.text,
    }


def _append_full_completion_records(
    path: str | None,
    records: list[dict[str, object]],
) -> None:
    if not path or not records:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sampled_training_completion_records(
    trajectories: list[Trajectory],
    *,
    step_idx: int,
) -> list[dict[str, object]]:
    seen_prompts: set[int] = set()
    records = []
    for trajectory in trajectories:
        if trajectory.sample_idx in seen_prompts:
            continue
        seen_prompts.add(trajectory.sample_idx)
        completion, step = trajectory.transitions[0]
        records.append(
            _full_completion_record(
                split="train",
                step_idx=step_idx,
                completion=completion,
                step=step,
            )
        )
    return records


def _sampled_validation_completion_records(
    completion_steps: list[tuple[Completion, Step]],
    *,
    max_samples: int = 2,
) -> list[dict[str, object]]:
    return [
        _full_completion_record(
            split="validation",
            step_idx=None,
            completion=completion,
            step=step,
        )
        for completion, step in completion_steps[:max_samples]
    ]


def _log_training_completion_samples(
    step_idx: int,
    trajectories: list[Trajectory],
    *,
    max_samples: int = 2,
) -> None:
    for trajectory in trajectories[:max_samples]:
        completion, step = trajectory.transitions[0]
        metadata = _step_metadata(step)
        logger.info(
            "  [train step %s prompt %s] problem_id=%s target=%s rewards=%s "
            "compilation_passed=%s func_passed=%s",
            step_idx,
            trajectory.sample_idx,
            metadata.get("problem_id"),
            metadata.get("target"),
            step.rewards,
            metadata.get("compilation_passed"),
            metadata.get("func_passed"),
        )
        logger.info(
            "       question: %s", _preview_text(metadata.get("question"))
        )
        logger.info("       completion: %s", _preview_text(completion.text))
        logger.info(
            "       final_text: %s", _preview_text(metadata.get("final_text"))
        )


class RLTrainer(Configurable):
    """Top-level RL training orchestrator."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        num_steps: int = 10
        """Number of RL training steps."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        num_prompts_per_step: int = 5
        """Number of distinct prompts (= GRPO groups) drawn per training step.

        The total episodes per step is ``num_prompts_per_step * group_size``,
        where ``group_size`` is ``generator.sampling.n`` (completions per prompt).
        """

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for training rollouts."""

        validation_env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for validation rollouts."""

        log_samples: bool = False
        """Log first completion per episode during training and validation."""

        env_step_concurrency: int = 4
        """Maximum number of env.step calls to run concurrently."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        def __post_init__(self):
            if self.trainer.debug.batch_invariant:
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # TODO: Replace trainer dtype constraint to use mixed
                #  training enabled by FSDP.
                if self.trainer.training.dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 training dtype, "
                        f"got {self.trainer.training.dtype!r}"
                    )
                if self.generator.model_dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 generator dtype, "
                        f"got {self.generator.model_dtype!r}"
                    )
                if self.trainer.parallelism.enable_sequence_parallel:
                    raise ValueError(
                        "batch_invariant mode doesn't support SP now. "
                        "SP uses reduce-scatter which only supports Ring in NCCL "
                        "and has not been validated for determinism."
                    )

    def __init__(self, config: Config):
        self.config = config
        self.trainer = None
        self.generator = None
        self._proc_meshes = []
        output_dir = os.environ.get("TORCHTITAN_RL_OUTPUT_DIR") or config.dump_folder
        config.dump_folder = output_dir
        self.full_completion_log_path = os.path.join(
            output_dir,
            "full_sample_completions.jsonl",
        )
        self.final_text_stats_log_path = os.path.join(
            output_dir,
            "final_text_completion_stats.jsonl",
        )
        if config.log_samples:
            logger.info(
                "Full --log_samples completions will be written to %s",
                self.full_completion_log_path,
            )
        logger.info(
            "Per-step final-text completion stats will be written to %s",
            self.final_text_stats_log_path,
        )

    async def close(self):
        """Best-effort: tear down actors, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")
        for actor_name, actor in (
            ("trainer", self.trainer),
            ("generator", self.generator),
        ):
            if actor is None:
                continue
            try:
                await actor.close.call()
            except Exception:
                logger.exception("%s.close failed", actor_name)

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _get_rank_0_value(self, result, has_gpus: bool = True):
        """Extract rank 0 result, handling both single and multi-node meshes.

        Monarch actor endpoints return results from all ranks in the mesh.
        This picks out rank 0's result by indexing into the host and GPU
        dimensions as needed (multi-node meshes have an extra host dimension).
        This should be used in cases where all ranks return the same result.
        """
        kwargs = {}
        if self._multi_node:
            kwargs["hosts"] = 0
        if has_gpus:
            kwargs["gpus"] = 0
        return result.item(**kwargs)

    @staticmethod
    def _compute_world_size(p: ParallelismConfig) -> int:
        """Compute world size from all parallel dimensions."""
        dp_shard = max(p.data_parallel_shard_degree, 1)
        return (
            p.data_parallel_replicate_degree
            * dp_shard
            * p.tensor_parallel_degree
            * p.pipeline_parallel_degree
            * p.context_parallel_degree
        )

    def _shard_episodes(self, episodes: list[Episode]) -> list[list[Episode]]:
        """Round-robin partition episodes across DP ranks."""
        return [
            [episodes[i] for i in range(rank, len(episodes), self.trainer_dp_degree)]
            for rank in range(self.trainer_dp_degree)
        ]

    @staticmethod
    def _collate_episodes(episodes: list[Episode]) -> TrainBatch:
        """Pack episodes into a single varlen-packed TrainBatch."""
        all_ids: list[int] = []
        prompt_lens: list[int] = []
        response_lens: list[int] = []

        for ep in episodes:
            all_ids.extend(ep.prompt_token_ids + ep.token_ids)
            prompt_lens.append(len(ep.prompt_token_ids))
            response_lens.append(len(ep.token_ids))

        return TrainBatch(
            token_ids=torch.tensor([all_ids], dtype=torch.long),
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            seq_lens=[p + r for p, r in zip(prompt_lens, response_lens)],
            advantages=torch.tensor(
                [ep.advantage for ep in episodes],
                dtype=torch.float32,
            ),
            token_logprobs=[ep.token_logprobs for ep in episodes],
        )

    async def setup(
        self,
        *,
        host_mesh=None,
        trainer_nodes: int | None = None,
        generator_nodes: int | None = None,
        gpus_per_node: int | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Creates separate GPU meshes for trainer and generator and
        synchronizes initial weights from trainer to generator. Must be
        called before :meth:`train`.

        Args:
            host_mesh: Optional multi-node HostMesh. When provided,
                whole nodes are dedicated to trainer vs generator
                roles instead of partitioning GPUs on a single host.
            trainer_nodes: Number of nodes for the trainer (required when
                host_mesh is provided).
            generator_nodes: Number of nodes for the generator (required when
                host_mesh is provided).
            gpus_per_node: GPUs per node, assumed to be the same across all
                nodes (no heterogeneous node configurations). Required when
                host_mesh is provided.
        """
        config = self.config

        self.trainer_world_size = self._compute_world_size(config.trainer.parallelism)
        self.generator_world_size = self._compute_world_size(
            config.generator.parallelism
        )
        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        total_gpus = self.trainer_world_size + self.generator_world_size
        logger.info(
            f"{self.generator_world_size} generator GPUs + "
            f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
        )

        self._multi_node = host_mesh is not None

        if host_mesh is not None:
            # Multi-node mode: dedicate whole nodes to trainer vs generator
            if (
                trainer_nodes is None
                or generator_nodes is None
                or gpus_per_node is None
            ):
                raise ValueError(
                    "trainer_nodes, generator_nodes, and gpus_per_node are "
                    "required when host_mesh is provided"
                )
            # Validate that world sizes are evenly divisible by node counts
            assert self.trainer_world_size % trainer_nodes == 0, (
                f"trainer_world_size ({self.trainer_world_size}) must be "
                f"evenly divisible by trainer_nodes ({trainer_nodes})"
            )
            assert self.generator_world_size % generator_nodes == 0, (
                f"generator_world_size ({self.generator_world_size}) must be "
                f"evenly divisible by generator_nodes ({generator_nodes})"
            )

            # Compute GPUs per node for each role based on the config's
            # world size and number of nodes allocated to that role
            trainer_gpus_per_node = self.trainer_world_size // trainer_nodes
            generator_gpus_per_node = self.generator_world_size // generator_nodes

            trainer_host_mesh = host_mesh.slice(hosts=slice(0, trainer_nodes))
            generator_host_mesh = host_mesh.slice(
                hosts=slice(trainer_nodes, trainer_nodes + generator_nodes)
            )

            # Use Provisioner to set CUDA_VISIBLE_DEVICES so each role
            # only sees its own GPUs and doesn't conflict with other
            # processes on the node
            trainer_provisioner = Provisioner(total_gpus=gpus_per_node)
            generator_provisioner = Provisioner(total_gpus=gpus_per_node)

            trainer_mesh = trainer_host_mesh.spawn_procs(
                per_host={"gpus": trainer_gpus_per_node},
                bootstrap=trainer_provisioner.allocate(trainer_gpus_per_node),
            )
            generator_mesh = generator_host_mesh.spawn_procs(
                per_host={"gpus": generator_gpus_per_node},
                bootstrap=generator_provisioner.allocate(generator_gpus_per_node),
            )
        else:
            # Single-node mode: partition GPUs on this_host() via
            # CUDA_VISIBLE_DEVICES
            provisioner = Provisioner(total_gpus=total_gpus)
            trainer_mesh = this_host().spawn_procs(
                per_host={"gpus": self.trainer_world_size},
                bootstrap=provisioner.allocate(self.trainer_world_size),
            )
            generator_mesh = this_host().spawn_procs(
                per_host={"gpus": self.generator_world_size},
                bootstrap=provisioner.allocate(self.generator_world_size),
            )

        # Store proc meshes for cleanup
        self._proc_meshes = [trainer_mesh, generator_mesh]

        await setup_torch_elastic_env_async(trainer_mesh)
        await setup_torch_elastic_env_async(generator_mesh)

        # Spawn actors on their respective meshes
        self.trainer = trainer_mesh.spawn(
            "trainer",
            PolicyTrainer,
            config.trainer,
            model_spec=config.model_spec,
            hf_assets_path=config.hf_assets_path,
            generator_dtype=config.generator.model_dtype,
            compile_config=config.compile,
        )

        self.generator = generator_mesh.spawn(
            "generator",
            VLLMGenerator,
            config.generator,
            model_spec=config.model_spec,
            model_path=config.hf_assets_path,
            compile_config=config.compile,
            max_num_seqs=max(
                config.num_prompts_per_step * config.generator.sampling.n,
                config.num_validation_samples,
            ),
        )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # push weights from trainer
        self.trainer.push_model_state_dict.call().get()
        # pull weights for policy version 0 (initial weights)
        self.generator.pull_model_state_dict.call(0).get()

    async def _collect_rollouts(self, num_groups: int, step: int) -> list[Trajectory]:
        """Collect group rollouts: one single-use env per group, scored and returned."""
        envs = [
            self.config.env.build(step=step, group_idx=i) for i in range(num_groups)
        ]
        completions = self._get_rank_0_value(
            self.generator.generate.call([env.prompt for env in envs]).get()
        )
        completion_steps = await run_env_steps(
            envs,
            completions,
            concurrency=self.config.env_step_concurrency,
        )
        return [
            Trajectory(sample_idx=c.prompt_idx, transitions=[(c, step_result)])
            for c, step_result in completion_steps
        ]

    @staticmethod
    def _build_episodes(trajectories: list[Trajectory]) -> list[Episode]:
        """Group trajectories by sample, apply mean-baseline advantage, flatten to Episodes."""
        groups: dict[int, list[Trajectory]] = {}
        for t in trajectories:
            groups.setdefault(t.sample_idx, []).append(t)

        episodes: list[Episode] = []
        for sample_idx, group in groups.items():
            rewards = [t.total_reward for t in group]
            group_mean = sum(rewards) / len(rewards)
            for t in group:
                # Single-turn: exactly one (completion, step) per trajectory.
                c, _ = t.transitions[0]
                episodes.append(
                    Episode(
                        policy_version=c.policy_version,
                        prompt_idx=sample_idx,
                        prompt_token_ids=c.prompt_token_ids,
                        text=c.text,
                        token_ids=c.token_ids,
                        token_logprobs=c.token_logprobs,
                        reward=t.total_reward,
                        advantage=t.total_reward - group_mean,
                    )
                )
        return episodes

    async def validate(self) -> dict:
        """Run validation on held-out prompts using greedy sampling.
        TODO: investigate using pass@k."""
        num_samples = self.config.num_validation_samples
        envs = [
            self.config.validation_env.build(step=0, group_idx=i)
            for i in range(num_samples)
        ]
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
        )
        completions = self._get_rank_0_value(
            self.generator.generate.call(
                [env.prompt for env in envs], sampling_config=greedy
            ).get()
        )

        completion_steps = await run_env_steps(
            envs,
            completions,
            concurrency=self.config.env_step_concurrency,
        )
        steps = [step for _, step in completion_steps]

        if self.config.log_samples:
            _log_validation_samples(envs, completion_steps)
            _append_full_completion_records(
                self.full_completion_log_path,
                _sampled_validation_completion_records(completion_steps),
            )

        components = _mean_rewards(steps)
        return {
            "mean_reward": sum(components.values()),
            "components": components,
            "total": num_samples,
        }

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_prompts_per_step
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")
        pre_validation = await self.validate()
        logger.info(f"Pre:  {_format_validation(pre_validation)}")

        for step in range(num_steps):
            # Cancellation point for Ctrl-C (KeyboardInterrupt) handling.
            # This yields to the event loop to check for cancellation, which
            # doesn't happen with `.get` calls.
            # TODO: investigate replacing `.get()` with `await
            await asyncio.sleep(0)

            step_start = time.perf_counter()

            # --- Collect data and create episodes --- #
            trajectories = await self._collect_rollouts(num_groups, step=step)
            episodes = self._build_episodes(trajectories)
            logger.info(
                "Step %2d collected %d rollouts and built %d episodes",
                step,
                len(trajectories),
                len(episodes),
            )
            _log_training_step_diagnostics(
                step,
                trajectories,
                num_prompts=num_groups,
                completions_per_prompt=self.config.generator.sampling.n,
                max_response_tokens=self.config.generator.sampling.max_tokens,
                final_text_stats_path=self.final_text_stats_log_path,
            )

            if self.config.log_samples:
                _log_samples(episodes)
                _log_training_completion_samples(step, trajectories)
                _append_full_completion_records(
                    self.full_completion_log_path,
                    _sampled_training_completion_records(
                        trajectories,
                        step_idx=step,
                    ),
                )

            # --- Train step --- #
            batches = [
                self._collate_episodes(per_rank_episodes)
                for per_rank_episodes in self._shard_episodes(episodes)
            ]
            _log_train_batch_stats(step, batches)
            fwd_bwd_metrics = self._get_rank_0_value(
                self.trainer.forward_backward.call(batches).get()
            )
            optim_metrics = self._get_rank_0_value(self.trainer.optim_step.call().get())
            metrics = {**fwd_bwd_metrics, **optim_metrics}

            # --- Weight sync --- #
            t0 = time.perf_counter()
            self.trainer.push_model_state_dict.call().get()
            t_push = time.perf_counter() - t0
            self.generator.pull_model_state_dict.call(metrics["policy_version"]).get()
            t_sync = time.perf_counter() - t0
            logger.info(f"Weight sync: push={t_push:.3f}s, total={t_sync:.3f}s")

            # --- Logging --- #
            steps = [t.transitions[0][1] for t in trajectories]
            components = _mean_rewards(steps)
            avg_tokens = sum(len(ep.token_ids) for ep in episodes) / len(episodes)
            logger.info(
                f"Step {step:2d} | Loss: {metrics['loss']:+.4f} | "
                f"Reward: {sum(components.values()):+.3f} ({_format_rewards(components)}) | "
                f"Avg tokens: {avg_tokens:>3.0f} | "
                f"Logprob diff: mean={metrics['logprob_diff_mean']:.4e}, "
                f"max={metrics['logprob_diff_max']:.4e} | "
                f"Time: {time.perf_counter() - step_start:.1f}s"
            )

            if not math.isfinite(metrics["loss"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

        logger.info("Post-training validation")
        post_validation = await self.validate()
        logger.info(
            f"Summary:\n  Pre:  {_format_validation(pre_validation)}\n"
            f"  Post: {_format_validation(post_validation)}"
        )


async def main():
    config = ConfigManager().parse_args()
    rl_trainer = RLTrainer(config)
    try:
        await rl_trainer.setup()
        await rl_trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    except Exception:
        logger.exception("RLTrainer failed; attempting graceful shutdown...")
        raise
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
