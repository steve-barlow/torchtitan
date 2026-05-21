# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, cast

from evalsets.veribench.dataset import VeribenchSplit, load_veribench
from evalsets.veribench.eval_set import SYSTEM_PROMPT
from evalsets.veribench.evaluator import count_tests
from post_training_torchtitan.app import qwen_prompt_formatting
from post_training_torchtitan.app.grading import CodingReward, FormatReward
from torchtitan.config import Configurable
from torchtitan.experiments.rl.types import Step
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from vllm.tokenizers import TokenizerLike, get_tokenizer


class VeribenchEnv(Configurable):
    """Single-turn Veribench env backed by Qwen-formatted prompts."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tokenizer_path: str = "torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B"
        split: VeribenchSplit = "train"
        seed: int = 42
        enable_thinking: bool = True
        compilation_credit: float = 0.1

    def __init__(
        self,
        config: Config,
        *,
        step: int = 0,
        group_idx: int = 0,
        tokenizer: TokenizerLike | None = None,
    ) -> None:
        self._config = config
        self._validate_config(config)
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer(config.tokenizer_path)
        self._examples = load_veribench(config.split)
        if not self._examples:
            raise ValueError(f"VeribenchEnv selected no examples from {config.split!r} split")

        self.problem_id = self._select_problem_id(step=step, group_idx=group_idx)
        example = self._examples[self.problem_id]
        self.question = str(example["question"])
        self.num_tests = count_tests(str(example.get("testbench", "")))
        self.target = f"problem_{self.problem_id:05d}"
        self.prompt = self._format_prompt(self.question)
        self.raw_completion: str | None = None
        self.final_text: str | None = None
        self.reward_value: float | None = None
        self.reward_details: dict[str, Any] | None = None

    @staticmethod
    def _validate_config(config: Config) -> None:
        if not config.tokenizer_path:
            raise ValueError("tokenizer_path must be set")
        if config.split not in ("train", "validation", "medium"):
            raise ValueError("split must be 'train', 'validation', or 'medium'")

    def _select_problem_id(self, *, step: int = 0, group_idx: int = 0) -> int:
        if step < 0:
            raise ValueError("step must be non-negative")
        if group_idx < 0:
            raise ValueError("group_idx must be non-negative")
        available_ids = sorted(int(problem_id) for problem_id in self._examples)
        rng = random.Random(f"{self._config.split}:{self._config.seed}:{step}:{group_idx}")
        return available_ids[rng.randrange(len(available_ids))]

    def _format_prompt(self, question: str) -> str:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem:\n{question}"},
        ]
        return cast(
            str,
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._config.enable_thinking,
            ),
        )

    async def step(self, completion: str) -> Step:
        self.raw_completion = completion
        format_result = FormatReward().score(completion)
        self.final_text = format_result.code if format_result.passed else ""

        metadata: dict[str, Any] = {
            "split": self._config.split,
            "problem_id": self.problem_id,
            "target": self.target,
            "question": self.question,
            "num_tests": self.num_tests,
            "format_passed": format_result.passed,
            "format_reward": format_result.reward,
            "format_failure_reason": format_result.failure_reason,
            "final_text": self.final_text,
            "extracted_code": format_result.code,
        }
        rewards = {"format": format_result.reward}

        if not format_result.passed:
            self.reward_value = format_result.reward
            self.reward_details = metadata
            metadata.update(
                {
                    "coding_failure_reason": "format_failed",
                    "coding_reward_log": "",
                    "compilation_passed": None,
                    "func_passed": None,
                    "num_tests_passed": 0,
                    "empty_response": not bool(completion.strip()),
                }
            )
            return Step(rewards=rewards, done=True, metadata=metadata)

        reward = CodingReward(
            qwen_prompt_formatting,
            compilation_credit=self._config.compilation_credit,
        )
        coding_result = await reward.score_with_details(
            prompt=self.question,
            response=format_result.code,
            target=self.target,
        )
        rewards["coding"] = coding_result.reward
        self.reward_value = sum(rewards.values())
        self.reward_details = coding_result.details
        metadata.update(
            {
                "coding_failure_reason": _coding_failure_reason(coding_result.details),
                "coding_reward_log": coding_result.details.get("log", ""),
                "compilation_passed": coding_result.details.get("compilation_passed"),
                "func_passed": coding_result.details.get("func_passed"),
                "num_tests": coding_result.details.get("num_tests", self.num_tests),
                "num_tests_passed": coding_result.details.get("num_tests_passed", 0),
                "empty_response": bool(coding_result.details.get("empty_response", False)),
            }
        )
        return Step(rewards=rewards, done=True, metadata=metadata)


def extract_qwen_final_text(response: str, *, trim: bool = True) -> str | None:
    result = FormatReward().score(response)
    if not result.passed:
        return None
    return result.code.strip() if trim else result.code


def _coding_failure_reason(details: Mapping[str, Any]) -> str | None:
    if details.get("empty_response"):
        return "empty_response"
    if details.get("compilation_passed") is False:
        return "compilation_failed"
    if details.get("func_passed") is False:
        return "functional_failed"
    return None
