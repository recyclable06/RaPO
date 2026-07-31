"""Stateful RaPO integration helpers for GRPO trainers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from rapo.core import (
    CrossTaskAdvantageNormalizer,
    combine_rewards,
    retention_reward,
    sample_standard_deviation,
    trajectory_drift,
)

RAPO_STATE_NAME = "rapo_state.json"
_STATE_FORMAT_VERSION = 2
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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
    rapo_run_id: str | None = None
    rapo_contract_sha256: str | None = None

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
        if self.rapo_enabled:
            if not self.rapo_run_id:
                raise ValueError("rapo_run_id is required when rapo_enabled=True")
            if (
                self.rapo_contract_sha256 is None
                or _SHA256_PATTERN.fullmatch(self.rapo_contract_sha256) is None
            ):
                raise ValueError(
                    "rapo_contract_sha256 must be a lowercase SHA256 when RaPO is enabled"
                )
            if self.rapo_task_index == 1 and self.rapo_state_path is not None:
                raise ValueError("RaPO task 1 forbids prior state")
            if self.rapo_task_index >= 2 and self.rapo_state_path is None:
                raise ValueError("RaPO task 2+ requires state from the previous task")


@dataclass(frozen=True)
class RapoBatchRewards:
    task_rewards: Tensor
    total_rewards: Tensor
    drift: Tensor | None
    retention_rewards: Tensor | None


def validate_tokenized_multimodal_prompt(
    input_ids: Tensor,
    attention_mask: Tensor,
    max_prompt_length: int,
    *,
    visual_token_mask: Tensor | None = None,
) -> Tensor:
    """Fail before generation if a tokenized multimodal prompt exceeds its limit."""

    if max_prompt_length < 1:
        raise ValueError("max_prompt_length must be positive")
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must share [batch, sequence] shape")
    active_mask = attention_mask.to(dtype=torch.bool)
    if visual_token_mask is not None:
        if visual_token_mask.shape != input_ids.shape:
            raise ValueError("visual_token_mask must match input_ids")
        if torch.any(visual_token_mask.to(dtype=torch.bool) & ~active_mask):
            raise ValueError("visual tokens must be active in the prompt attention mask")
    lengths = active_mask.sum(dim=1)
    if torch.any(lengths > max_prompt_length):
        observed = int(lengths.max().detach().cpu().item())
        raise ValueError(
            "Tokenized multimodal prompt length "
            f"{observed} exceeds max_prompt_length={max_prompt_length}; "
            "visual-token truncation is not implemented"
        )
    return lengths


@dataclass
class CtanStepTransaction:
    """Accumulate global rewards and atomically update CTAN at optimizer-step end."""

    _microbatches: list[Tensor] = field(default_factory=list)

    @property
    def pending_microbatches(self) -> int:
        return len(self._microbatches)

    def stage(self, global_rewards: Tensor) -> Tensor:
        if global_rewards.ndim != 1 or global_rewards.numel() < 2:
            raise ValueError("Each global reward microbatch must contain at least two values")
        self._microbatches.append(global_rewards.detach().float().cpu().clone())
        return self.window_std()

    def window_std(self) -> Tensor:
        if not self._microbatches:
            raise ValueError("CTAN step transaction has no staged rewards")
        return sample_standard_deviation(torch.cat(self._microbatches))

    def finish(
        self,
        normalizer: CrossTaskAdvantageNormalizer,
        *,
        successful: bool,
    ) -> bool:
        if not self._microbatches:
            return False
        batch_std = self.window_std()
        self._microbatches.clear()
        if successful:
            normalizer.update(batch_std)
            return True
        return False


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
        self.step_transaction = CtanStepTransaction()
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
        training: bool = True,
    ) -> Tensor:
        normalization_rewards = (
            total_rewards if global_total_rewards is None else global_total_rewards
        )
        if normalization_rewards.ndim != 1:
            raise ValueError("global_total_rewards must be one-dimensional")
        if normalization_rewards.numel() < 2:
            raise ValueError("at least two rewards are required to compute CTAN")
        batch_std = sample_standard_deviation(normalization_rewards)
        if training:
            window_std = self.step_transaction.stage(normalization_rewards)
            scale = self.normalizer.provisional_scale(window_std)
        else:
            scale = (
                self.normalizer.ema_std
                if self.normalizer.ema_std is not None
                else float(batch_std.detach().cpu().item())
            )
        return self.normalizer.advantages(
            total_rewards,
            num_generations,
            batch_std=batch_std,
            update=False,
            scale_override=scale,
        )

    def finish_optimizer_step(self, *, successful: bool) -> bool:
        return self.step_transaction.finish(self.normalizer, successful=successful)

    @property
    def ctan_ema_std(self) -> float | None:
        return self.normalizer.ema_std

    def state_dict(self) -> dict[str, Any]:
        if self.step_transaction.pending_microbatches:
            raise ValueError("Cannot save RaPO state with an unfinished optimizer step")
        return {
            "format_version": _STATE_FORMAT_VERSION,
            "task_index": self.config.rapo_task_index,
            "run_id": self.config.rapo_run_id,
            "contract_sha256": self.config.rapo_contract_sha256,
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

        saved_task = int(state["task_index"])
        if saved_task != self.config.rapo_task_index - 1:
            raise ValueError("RaPO state must come from exactly the previous task")
        if not state.get("run_id") or _SHA256_PATTERN.fullmatch(
            str(state.get("contract_sha256", ""))
        ) is None:
            raise ValueError("RaPO state is missing its producer run or contract binding")
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
