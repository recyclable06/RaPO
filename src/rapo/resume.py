"""Deterministic checkpoint contracts for CPU tests and production resume gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from rapo.core import CrossTaskAdvantageNormalizer


CHECKPOINT_FORMAT_VERSION = 1
CHECKPOINT_BINDING_NAME = "rapo_checkpoint_binding.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_CHECKPOINT_KEYS = {
    "format_version",
    "identity",
    "global_step",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "python_rng_state",
    "torch_rng_state",
    "ctan_state",
}


@dataclass(frozen=True)
class CheckpointIdentity:
    run_id: str
    profile_sha256: str
    run_contract_sha256: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        for name in ("profile_sha256", "run_contract_sha256"):
            if _SHA256.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA256")


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ctan: CrossTaskAdvantageNormalizer,
    global_step: int,
    identity: CheckpointIdentity,
) -> Path:
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "identity": asdict(identity),
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "ctan_state": ctan.state_dict(),
        },
        output,
    )
    return output


def _validated_payload(
    path: str | Path,
    expected_identity: CheckpointIdentity,
    ctan: CrossTaskAdvantageNormalizer,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping")
    missing = sorted(_REQUIRED_CHECKPOINT_KEYS - payload.keys())
    if missing:
        raise ValueError("Checkpoint is missing required state: " + ", ".join(missing))
    if int(payload["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format")
    if payload["identity"] != asdict(expected_identity):
        raise ValueError("Checkpoint run/profile/contract identity does not match")
    if not isinstance(payload["global_step"], int) or payload["global_step"] < 0:
        raise ValueError("Checkpoint global_step is invalid")
    if not isinstance(payload["torch_rng_state"], torch.Tensor):
        raise ValueError("Checkpoint Torch RNG state is invalid")
    probe = CrossTaskAdvantageNormalizer(beta=ctan.beta, epsilon=ctan.epsilon)
    try:
        probe.load_state_dict(payload["ctan_state"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint CTAN state is invalid") from exc
    return payload


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ctan: CrossTaskAdvantageNormalizer,
    expected_identity: CheckpointIdentity,
) -> int:
    payload = _validated_payload(path, expected_identity, ctan)
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    ctan.load_state_dict(payload["ctan_state"])
    random.setstate(payload["python_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    return int(payload["global_step"])


def _file_inventory(checkpoint: Path) -> list[dict[str, Any]]:
    inventory = []
    for path in sorted(
        (candidate for candidate in checkpoint.rglob("*") if candidate.is_file()),
        key=lambda item: item.relative_to(checkpoint).as_posix(),
    ):
        relative = path.relative_to(checkpoint).as_posix()
        if relative == CHECKPOINT_BINDING_NAME:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        inventory.append({"path": relative, "size": path.stat().st_size, "sha256": digest})
    return inventory


def _inventory_sha256(inventory: Any) -> str:
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_production_state(checkpoint: Path, *, require_ctan: bool) -> None:
    names = {path.name for path in checkpoint.rglob("*") if path.is_file()}
    required = {"trainer_state.json", "scheduler.pt"}
    missing = sorted(required - names)
    if missing:
        raise ValueError("Checkpoint is missing production state: " + ", ".join(missing))
    if not any(name.startswith("rng_state") and name.endswith(".pth") for name in names):
        raise ValueError("Checkpoint is missing RNG state")
    if not any(
        name == "optimizer.pt" or name.endswith("_optim_states.pt") for name in names
    ):
        raise ValueError("Checkpoint is missing optimizer state")
    if not any(
        name in {"model.safetensors", "pytorch_model.bin", "adapter_model.safetensors"}
        or name.endswith("_model_states.pt")
        for name in names
    ):
        raise ValueError("Checkpoint is missing model state")
    if require_ctan and "rapo_state.json" not in names:
        raise ValueError("RaPO checkpoint is missing CTAN state")


def write_checkpoint_binding(
    checkpoint_path: str | Path,
    *,
    identity: CheckpointIdentity,
    global_step: int,
    require_ctan: bool,
) -> Path:
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    _require_production_state(checkpoint, require_ctan=require_ctan)
    inventory = _file_inventory(checkpoint)
    payload = {
        "schema_version": 1,
        "identity": asdict(identity),
        "global_step": global_step,
        "require_ctan": require_ctan,
        "checkpoint_inventory_sha256": _inventory_sha256(inventory),
    }
    binding = checkpoint / CHECKPOINT_BINDING_NAME
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if binding.exists():
        if binding.read_text(encoding="utf-8") != text:
            raise ValueError("Refusing to replace a different checkpoint binding")
        return binding
    handle, temporary_name = tempfile.mkstemp(
        dir=checkpoint,
        prefix=f".{CHECKPOINT_BINDING_NAME}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as temporary:
            temporary.write(text)
        os.replace(temporary_name, binding)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()
    return binding


def validate_checkpoint_binding(
    checkpoint_path: str | Path,
    *,
    expected_identity: CheckpointIdentity,
    require_ctan: bool,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    binding = checkpoint / CHECKPOINT_BINDING_NAME
    if not binding.is_file():
        raise ValueError("Checkpoint is missing its run binding")
    payload = json.loads(binding.read_text(encoding="utf-8"))
    if payload.get("identity") != asdict(expected_identity):
        raise ValueError("Checkpoint run/profile/contract binding does not match")
    if bool(payload.get("require_ctan")) != require_ctan:
        raise ValueError("Checkpoint method state binding does not match")
    _require_production_state(checkpoint, require_ctan=require_ctan)
    if payload.get("checkpoint_inventory_sha256") != _inventory_sha256(
        _file_inventory(checkpoint)
    ):
        raise ValueError("Checkpoint contents changed after binding")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a formal resume checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--require-ctan", action="store_true")
    args = parser.parse_args(argv)
    payload = validate_checkpoint_binding(
        args.checkpoint,
        expected_identity=CheckpointIdentity(
            args.run_id, args.profile_sha256, args.contract_sha256
        ),
        require_ctan=args.require_ctan,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
