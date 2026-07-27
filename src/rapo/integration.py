"""Stateful RaPO integration helpers for GRPO trainers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

from rapo.core import (
    CrossTaskAdvantageNormalizer,
    combine_rewards,
    retention_reward,
    trajectory_drift,
)

RAPO_STATE_NAME = "rapo_state.json"
_STATE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class RapoTrainerConfig:
    """CLI-friendly RaPO settings consumed by the Visual-RFT patch."""

    rapo_enabled: bool = False
    rapo_task_index: int = 1
    rapo_retention_alpha: float = 20.0
    rapo_retention_weight: float = 0.5
    rapo_ctan_beta: float = 0.999
    rapo_ctan_epsilon: float = 1e-4
    rapo_state_path: str | None = None
    rapo_resume_from_checkpoint: str | None = None

    def __post_init__(self) -> None:
        if self.rapo_task_index < 1:
            raise ValueError("rapo_task_index must be at least 1")
        if self.rapo_retention_alpha <= 0:
            raise ValueError("rapo_retention_alpha must be positive")
        if self.rapo_retention_weight < 0:
            raise ValueError("rapo_retention_weight must be non-negative")
        if not 0 <= self.rapo_ctan_beta < 1:
            raise ValueError("rapo_ctan_beta must be in [0, 1)")
        if self.rapo_ctan_epsilon <= 0:
            raise ValueError("rapo_ctan_epsilon must be positive")
        if not self.rapo_enabled and (
            self.rapo_state_path is not None
            or self.rapo_resume_from_checkpoint is not None
        ):
            raise ValueError("RaPO state and resume paths require rapo_enabled=True")


@dataclass(frozen=True)
class RapoBatchRewards:
    task_rewards: Tensor
    total_rewards: Tensor
    drift: Tensor | None
    retention_rewards: Tensor | None


class RapoController:
    """Combine RaPO rewards, maintain CTAN, and persist cross-task state."""

    def __init__(self, config: RapoTrainerConfig) -> None:
        if not config.rapo_enabled:
            raise ValueError("RapoController requires rapo_enabled=True")
        self.config = config
        self.normalizer = CrossTaskAdvantageNormalizer(
            beta=config.rapo_ctan_beta,
            epsilon=config.rapo_ctan_epsilon,
        )
        if config.rapo_state_path is not None:
            self.load_state_file(config.rapo_state_path)

    def build_rewards(
        self,
        actor_token_logps: Tensor,
        anchor_token_logps: Tensor,
        completion_mask: Tensor,
        task_rewards: Tensor,
    ) -> RapoBatchRewards:
        if task_rewards.ndim != 1:
            raise ValueError("task_rewards must be one-dimensional")

        if self.config.rapo_task_index == 1:
            return RapoBatchRewards(
                task_rewards=task_rewards,
                total_rewards=task_rewards,
                drift=None,
                retention_rewards=None,
            )

        drift = trajectory_drift(actor_token_logps, anchor_token_logps, completion_mask)
        if drift.shape != task_rewards.shape:
            raise ValueError(
                "task reward count must match the number of generated trajectories"
            )
        retention = retention_reward(drift, alpha=self.config.rapo_retention_alpha)
        total = combine_rewards(
            task_rewards,
            retention,
            weight=self.config.rapo_retention_weight,
        )
        return RapoBatchRewards(
            task_rewards=task_rewards,
            total_rewards=total,
            drift=drift,
            retention_rewards=retention,
        )

    def advantages(
        self,
        total_rewards: Tensor,
        num_generations: int,
        *,
        global_total_rewards: Tensor | None = None,
        update: bool = True,
    ) -> Tensor:
        normalization_rewards = (
            total_rewards if global_total_rewards is None else global_total_rewards
        )
        if normalization_rewards.ndim != 1:
            raise ValueError("global_total_rewards must be one-dimensional")
        if normalization_rewards.numel() < 2:
            raise ValueError("at least two rewards are required to compute CTAN")
        batch_std = normalization_rewards.detach().float().std(correction=1)
        return self.normalizer.advantages(
            total_rewards,
            num_generations,
            batch_std=batch_std,
            update=update,
        )

    @property
    def ctan_ema_std(self) -> float | None:
        return self.normalizer.ema_std

    def state_dict(self) -> dict[str, Any]:
        return {
            "format_version": _STATE_FORMAT_VERSION,
            "task_index": self.config.rapo_task_index,
            "settings": {
                "retention_alpha": self.config.rapo_retention_alpha,
                "retention_weight": self.config.rapo_retention_weight,
                "ctan_beta": self.config.rapo_ctan_beta,
                "ctan_epsilon": self.config.rapo_ctan_epsilon,
            },
            "normalizer": self.normalizer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["format_version"]) != _STATE_FORMAT_VERSION:
            raise ValueError("unsupported RaPO state format")

        expected = self.state_dict()["settings"]
        saved = state["settings"]
        if any(float(saved[key]) != value for key, value in expected.items()):
            raise ValueError("RaPO settings do not match the saved state")
        self.normalizer.load_state_dict(state["normalizer"])

    def save_state_file(self, path: str | Path) -> Path:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(self.state_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return state_path

    def load_state_file(self, path: str | Path) -> None:
        state_path = Path(path)
        with state_path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        self.load_state_dict(state)
