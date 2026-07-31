"""Public components for the RaPO reproduction, loaded on first access."""

from importlib import import_module

__all__ = [
    "RAPO_STATE_NAME",
    "CtanStepTransaction",
    "CrossTaskAdvantageNormalizer",
    "RapoBatchRewards",
    "RapoController",
    "RapoTrainerConfig",
    "combine_rewards",
    "retention_reward",
    "sample_standard_deviation",
    "sampling_point_surrogate",
    "trajectory_drift",
    "validate_tokenized_multimodal_prompt",
]

_CORE_EXPORTS = {
    "CrossTaskAdvantageNormalizer",
    "combine_rewards",
    "retention_reward",
    "sample_standard_deviation",
    "sampling_point_surrogate",
    "trajectory_drift",
}


def __getattr__(name: str):
    if name in _CORE_EXPORTS:
        value = getattr(import_module("rapo.core"), name)
    elif name in __all__:
        value = getattr(import_module("rapo.integration"), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
