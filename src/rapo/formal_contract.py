"""Machine-checkable formal and legacy experiment profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


PROFILE_SCHEMA_VERSION = 1
_LEGACY_ENVIRONMENT_KEYS = {
    "DEEPSPEED_CONFIG",
    "RAPO_CUDNN_ENABLED",
    "RAPO_FP16_INITIAL_SCALE_POWER",
}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_experiment_profile(profile: Mapping[str, Any]) -> None:
    """Validate the frozen schema without claiming hardware readiness."""

    if int(profile.get("schema_version", -1)) != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unsupported experiment profile schema")
    kind = profile.get("profile_kind")
    if kind not in {"formal", "legacy_2080ti"}:
        raise ValueError("profile_kind must be formal or legacy_2080ti")
    for key in ("profile_name", "claim_scope", "output_namespace", "hardware_gate"):
        if not isinstance(profile.get(key), str) or not profile[key]:
            raise ValueError(f"{key} must be a non-empty string")
    training = profile.get("training")
    evaluation = profile.get("evaluation")
    hardware = profile.get("hardware")
    if not isinstance(training, dict) or not isinstance(evaluation, dict):
        raise ValueError("training and evaluation profile sections are required")
    if not isinstance(hardware, dict):
        raise ValueError("hardware profile section is required")
    for key in (
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_generations",
        "max_prompt_length",
        "max_completion_length",
        "max_pixels",
        "min_pixels",
    ):
        if not isinstance(training.get(key), int) or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training.get("precision") not in {"bf16", "fp16"}:
        raise ValueError("training.precision must be bf16 or fp16")
    if training.get("attention") not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError("training.attention is unsupported")
    if evaluation.get("torch_dtype") not in {"bfloat16", "float16", "float32"}:
        raise ValueError("evaluation.torch_dtype is unsupported")
    if evaluation.get("attention") not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError("evaluation.attention is unsupported")

    if kind == "formal":
        if profile["output_namespace"] != "formal":
            raise ValueError("Formal profile requires output_namespace=formal")
        if (
            training.get("deepspeed_config")
            != "configs/deepspeed_zero3_formal_bf16.json"
        ):
            raise ValueError(
                "Formal profile requires the formal BF16 ZeRO-3 DeepSpeed config"
            )
        if profile["hardware_gate"] != "pending_hardware_gate":
            raise ValueError("Formal profile must remain pending_hardware_gate")
        if training.get("budget_kind") != "epochs":
            raise ValueError("Formal training budget must be epoch-driven")
        if training.get("num_train_epochs") != 2 or "max_steps" in training:
            raise ValueError("Formal profile requires exactly two epochs and forbids max_steps")
        if training["precision"] != "bf16" or evaluation["torch_dtype"] != "bfloat16":
            raise ValueError("Formal profile requires explicit BF16 training and evaluation")
        if (
            training["attention"] != "flash_attention_2"
            or evaluation["attention"] != "flash_attention_2"
        ):
            raise ValueError(
                "Formal profile requires FlashAttention-2 for training and evaluation"
            )
        if hardware.get("expected_world_size") != 8:
            raise ValueError("Formal profile freezes expected_world_size=8")
    else:
        if training.get("budget_kind") != "max_steps":
            raise ValueError("Legacy profile must remain max_steps-driven")
        if not isinstance(training.get("default_max_steps"), int):
            raise ValueError("Legacy profile requires default_max_steps")


def load_experiment_profile(path: str | Path) -> tuple[dict[str, Any], str]:
    profile_path = Path(path)
    with profile_path.open(encoding="utf-8") as handle:
        profile = json.load(handle)
    if not isinstance(profile, dict):
        raise ValueError("Experiment profile must be a JSON object")
    validate_experiment_profile(profile)
    return profile, _canonical_sha256(profile)


def reject_legacy_environment(environment: Mapping[str, str] | None = None) -> None:
    values = os.environ if environment is None else environment
    leaked = sorted(
        key
        for key in values
        if key.startswith("RAPO_SMOKE_") or key in _LEGACY_ENVIRONMENT_KEYS
    )
    if leaked:
        raise ValueError(
            "Formal profile rejects legacy environment variables: " + ", ".join(leaked)
        )


def build_dry_run_contract(
    profile: Mapping[str, Any],
    *,
    profile_sha256: str,
    train_samples: int,
    world_size: int,
) -> dict[str, Any]:
    validate_experiment_profile(profile)
    if profile["profile_kind"] != "formal":
        raise ValueError("Formal dry-run requires the formal profile")
    if train_samples < 1 or world_size < 1:
        raise ValueError("train_samples and world_size must be positive")
    expected_world_size = int(profile["hardware"]["expected_world_size"])
    if world_size != expected_world_size:
        raise ValueError(
            f"Formal profile requires world_size={expected_world_size}; found {world_size}"
        )
    training = profile["training"]
    epochs = int(training["num_train_epochs"])
    per_device = int(training["per_device_train_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    generations = int(training["num_generations"])
    samples_per_rank = math.ceil(train_samples / world_size)
    sampler_examples = samples_per_rank * world_size
    batches_per_rank = math.ceil(samples_per_rank / per_device)
    optimizer_steps = math.ceil(batches_per_rank / accumulation) * epochs
    presentations = sampler_examples * epochs
    return {
        "schema_version": 1,
        "profile": dict(profile),
        "profile_sha256": profile_sha256,
        "hardware_gate": profile["hardware_gate"],
        "resolved": {
            "world_size": world_size,
            "precision": training["precision"],
            "attention": training["attention"],
            "deepspeed_config": training["deepspeed_config"],
            "per_device_train_batch_size": per_device,
            "gradient_accumulation_steps": accumulation,
            "num_generations": generations,
        },
        "training_budget": {
            "budget_kind": "epochs",
            "num_train_epochs": epochs,
            "train_samples": train_samples,
            "sampler_examples_per_epoch": sampler_examples,
            "expected_sample_presentations": presentations,
            "expected_generations": presentations * generations,
            "expected_optimizer_steps": optimizer_steps,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect the formal experiment contract")
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--sha-only", action="store_true")
    args = parser.parse_args(argv)
    reject_legacy_environment()
    profile, digest = load_experiment_profile(args.profile)
    if args.sha_only:
        print(digest)
        return 0
    if args.train_samples is None or args.world_size is None:
        parser.error("--train-samples and --world-size are required unless --sha-only is used")
    payload = build_dry_run_contract(
        profile,
        profile_sha256=digest,
        train_samples=args.train_samples,
        world_size=args.world_size,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
