"""Paper-level reward shaping and advantage normalization for RaPO."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch
from torch import Tensor


def _require_same_shape(*tensors: Tensor) -> None:
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(f"Expected tensors with the same shape, got {sorted(shapes)}")


def trajectory_drift(
    actor_token_logps: Tensor,
    anchor_token_logps: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    """Compute the detached, one-sided trajectory drift from Equation (2).

    All tensors have shape ``[num_trajectories, completion_length]``. Masked
    positions do not contribute to the length-normalized log-probability ratio.
    """

    _require_same_shape(actor_token_logps, anchor_token_logps, completion_mask)
    if actor_token_logps.ndim != 2:
        raise ValueError("Expected token log-probabilities with shape [batch, sequence]")

    mask = completion_mask.to(dtype=actor_token_logps.dtype)
    lengths = mask.sum(dim=-1)
    if torch.any(lengths <= 0):
        raise ValueError("Every trajectory must contain at least one unmasked completion token")

    mean_log_ratio = ((actor_token_logps - anchor_token_logps) * mask).sum(dim=-1) / lengths
    return mean_log_ratio.clamp_min(0).detach()


def retention_reward(drift: Tensor, alpha: float = 20.0) -> Tensor:
    """Map detached non-negative drift to the bounded reward in Equation (3)."""

    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if torch.any(drift < 0):
        raise ValueError("drift must be non-negative")
    return torch.exp(-alpha * drift.detach())


def combine_rewards(task_reward: Tensor, retention: Tensor, weight: float = 0.5) -> Tensor:
    """Combine task and retention rewards as in Equation (4)."""

    _require_same_shape(task_reward, retention)
    if weight < 0:
        raise ValueError("retention weight must be non-negative")
    return task_reward + weight * retention


@dataclass
class CrossTaskAdvantageNormalizer:
    """Persistent reward-standard-deviation EMA used by CTAN.

    The first batch initializes the EMA. Subsequent batches apply Equation (5).
    The state is plain Python data so it can be stored beside task checkpoints.
    """

    beta: float = 0.999
    epsilon: float = 1e-4
    ema_std: float | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.beta < 1:
            raise ValueError("beta must be in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.ema_std is not None:
            self._validate_std(self.ema_std)
        if self.updates < 0:
            raise ValueError("updates must be non-negative")

    @staticmethod
    def _validate_std(value: float) -> None:
        if not isfinite(value) or value < 0:
            raise ValueError("reward standard deviation must be finite and non-negative")

    def update(self, batch_std: Tensor | float) -> float:
        """Update and return the persistent EMA standard deviation."""

        if isinstance(batch_std, Tensor):
            if batch_std.numel() != 1:
                raise ValueError("batch_std must be a scalar")
            value = float(batch_std.detach().cpu().item())
        else:
            value = float(batch_std)
        self._validate_std(value)

        if self.ema_std is None:
            self.ema_std = value
        else:
            self.ema_std = self.beta * self.ema_std + (1 - self.beta) * value
        self.updates += 1
        return self.ema_std

    def advantages(
        self,
        total_rewards: Tensor,
        num_generations: int,
        batch_std: Tensor | float | None = None,
        *,
        update: bool = True,
    ) -> Tensor:
        """Compute Equation (6) and update CTAN from the current reward batch.

        Rows in ``total_rewards`` are ordered in contiguous rollout groups.
        Distributed callers pass the standard deviation of globally gathered
        rewards through ``batch_std`` while retaining local group means.
        """

        if total_rewards.ndim != 1:
            raise ValueError("total_rewards must be one-dimensional")
        if num_generations < 2:
            raise ValueError("num_generations must be at least 2")
        if total_rewards.numel() % num_generations != 0:
            raise ValueError("reward count must be divisible by num_generations")

        grouped_rewards = total_rewards.view(-1, num_generations)
        group_means = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
        if batch_std is None:
            batch_std = total_rewards.detach().float().std(correction=1)

        if update:
            scale = self.update(batch_std)
        else:
            if isinstance(batch_std, Tensor):
                if batch_std.numel() != 1:
                    raise ValueError("batch_std must be a scalar")
                current_std = float(batch_std.detach().cpu().item())
            else:
                current_std = float(batch_std)
            self._validate_std(current_std)
            scale = self.ema_std if self.ema_std is not None else current_std

        denominator = total_rewards.new_tensor(scale + self.epsilon)
        return (total_rewards - group_means) / denominator

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "epsilon": self.epsilon,
            "ema_std": self.ema_std,
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if float(state["beta"]) != self.beta or float(state["epsilon"]) != self.epsilon:
            raise ValueError("CTAN configuration does not match the saved state")
        ema_std = state["ema_std"]
        if ema_std is not None:
            ema_std = float(ema_std)
            self._validate_std(ema_std)
        updates = int(state["updates"])
        if updates < 0:
            raise ValueError("updates must be non-negative")
        self.ema_std = ema_std
        self.updates = updates
