# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import statistics
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


class RewardedTrajectory(Protocol):
    sample_idx: int
    total_reward: float


T = TypeVar("T", bound=RewardedTrajectory)


@dataclass(frozen=True, slots=True)
class ZeroStdFilterResult(Generic[T]):
    kept_trajectories: list[T]
    dropped_groups: int


def group_trajectories_by_sample_idx(
    trajectories: Sequence[T],
) -> OrderedDict[int, list[T]]:
    grouped: OrderedDict[int, list[T]] = OrderedDict()
    for trajectory in trajectories:
        grouped.setdefault(trajectory.sample_idx, []).append(trajectory)
    return grouped


def num_groups_in_trajectories(trajectories: Sequence[RewardedTrajectory]) -> int:
    return len(group_trajectories_by_sample_idx(trajectories))


def reward_std(trajectories: Sequence[RewardedTrajectory]) -> float:
    rewards = [float(trajectory.total_reward) for trajectory in trajectories]
    return statistics.pstdev(rewards)


def filter_zero_std_groups(trajectories: Sequence[T]) -> ZeroStdFilterResult[T]:
    grouped = group_trajectories_by_sample_idx(trajectories)
    kept_group_ids = set()
    dropped_groups = 0

    for sample_idx, group in grouped.items():
        if reward_std(group) == 0.0:
            dropped_groups += 1
        else:
            kept_group_ids.add(sample_idx)

    return ZeroStdFilterResult(
        kept_trajectories=[
            trajectory
            for trajectory in trajectories
            if trajectory.sample_idx in kept_group_ids
        ],
        dropped_groups=dropped_groups,
    )
