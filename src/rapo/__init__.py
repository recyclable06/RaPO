"""Core components for the RaPO reproduction."""

from rapo.core import (
    CrossTaskAdvantageNormalizer,
    combine_rewards,
    retention_reward,
    trajectory_drift,
)
from rapo.integration import (
    RAPO_STATE_NAME,
    RapoBatchRewards,
    RapoController,
    RapoTrainerConfig,
)

__all__ = [
    "RAPO_STATE_NAME",
    "CrossTaskAdvantageNormalizer",
    "RapoBatchRewards",
    "RapoController",
    "RapoTrainerConfig",
    "combine_rewards",
    "retention_reward",
    "trajectory_drift",
]
