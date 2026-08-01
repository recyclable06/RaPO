"""Canonical provenance contracts for RaPO runs and exported datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_MANIFEST_SCHEMA_VERSION = 1
STAGE_BINDING_SCHEMA_VERSION = 1
STAGE_BINDING_NAME = "rapo_stage_manifest.json"
PINNED_VISUAL_RFT_COMMIT = "2ffad63b25ddd79bfe25d3e046645401201c89d6"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

_EXPECTED_REPRODUCTION_CONFIG = {
    "schema_version": 1,
    "claim_scope": "independent_reproduction",
    "ctan": {
        "batch_scope": "gradient_accumulation_window_all_ranks_all_microbatches",
        "standard_deviation": "sample",
        "correction": 1,
        "initialization": "first_successful_optimizer_step",
        "provisional_scale": "current_window_prefix",
        "commit_clock": "successful_optimizer_step_once",
        "skipped_step": "discard",
    },
    "surrogate": {
        "rollout_reuse": 1,
        "clipping": None,
        "form": "unclipped_sampling_point",
    },
    "kl": {"beta": 0.04, "provenance": "repository_choice_not_author_setting"},
}


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically and reject non-finite numeric values."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_identity(path: str | Path) -> dict[str, Any]:
    """Return a deterministic identity for one file or directory tree."""

    resolved = Path(path).resolve()
    if resolved.is_file():
        return {
            "path": str(resolved),
            "kind": "file",
            "sha256": file_sha256(resolved),
        }
    if not resolved.is_dir():
        raise ValueError(f"Artifact path does not exist: {resolved}")

    files = []
    for item in sorted(
        (candidate for candidate in resolved.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(resolved).as_posix(),
    ):
        files.append(
            {
                "path": item.relative_to(resolved).as_posix(),
                "size": item.stat().st_size,
                "sha256": file_sha256(item),
            }
        )
    return {
        "path": str(resolved),
        "kind": "directory",
        "sha256": canonical_sha256(files),
    }


def validate_reproduction_config(config: Mapping[str, Any]) -> None:
    if dict(config) != _EXPECTED_REPRODUCTION_CONFIG:
        raise ValueError(
            "Reproduction config must match the frozen independent-reproduction "
            "CTAN, single-use surrogate, and KL contract"
        )


def load_reproduction_config(path: str | Path) -> tuple[dict[str, Any], str]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Reproduction config must be a JSON object")
    validate_reproduction_config(payload)
    return payload, canonical_sha256(payload)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as temporary:
            temporary.write(text)
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def write_json_if_absent_or_equal(
    payload: Mapping[str, Any], path: str | Path
) -> Path:
    """Write an immutable JSON artifact, accepting an identical retry only."""

    output_path = Path(path)
    if output_path.exists():
        with output_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != dict(payload):
            raise ValueError(f"Refusing to overwrite different manifest: {output_path}")
        return output_path
    _atomic_write_json(output_path, payload)
    return output_path


def build_stage_binding(
    data_manifest: Mapping[str, Any], task_index: int
) -> dict[str, Any]:
    tasks = list(data_manifest.get("tasks", []))
    if task_index < 1 or task_index > len(tasks):
        raise ValueError(f"task_index must be between 1 and {len(tasks)}")
    return {
        "schema_version": STAGE_BINDING_SCHEMA_VERSION,
        "data_manifest_sha256": canonical_sha256(data_manifest),
        "task_index": task_index,
    }


def write_stage_binding(
    stage_directory: str | Path,
    data_manifest: Mapping[str, Any],
    task_index: int,
) -> Path:
    stage_path = Path(stage_directory)
    if not stage_path.is_dir():
        raise ValueError(f"Stage dataset directory does not exist: {stage_path}")
    return write_json_if_absent_or_equal(
        build_stage_binding(data_manifest, task_index),
        stage_path / STAGE_BINDING_NAME,
    )


def validate_stage_binding(
    stage_directory: str | Path,
    data_manifest: Mapping[str, Any],
    task_index: int,
) -> dict[str, Any]:
    binding_path = Path(stage_directory) / STAGE_BINDING_NAME
    if not binding_path.is_file():
        raise ValueError(f"Stage dataset is missing {STAGE_BINDING_NAME}: {stage_directory}")
    with binding_path.open(encoding="utf-8") as handle:
        binding = json.load(handle)
    expected = build_stage_binding(data_manifest, task_index)
    if binding != expected:
        raise ValueError("Stage dataset task or data-manifest binding does not match")
    return binding


def _run_git(repo: Path, *arguments: str, text: bool = True) -> str | bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout


def repository_identity(repo_root: str | Path) -> dict[str, str]:
    repo = Path(repo_root).resolve()
    commit = str(_run_git(repo, "rev-parse", "HEAD")).strip()
    diff = bytes(_run_git(repo, "diff", "--binary", "HEAD", text=False))
    untracked_output = bytes(
        _run_git(repo, "ls-files", "--others", "--exclude-standard", "-z", text=False)
    )
    untracked = []
    for raw_path in filter(None, untracked_output.split(b"\0")):
        relative_path = raw_path.decode("utf-8")
        path = repo / relative_path
        if path.is_file():
            untracked.append(
                {"path": relative_path.replace("\\", "/"), "sha256": file_sha256(path)}
            )
    diff_payload = {
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": untracked,
    }
    return {"commit": commit, "diff_sha256": canonical_sha256(diff_payload)}


def _load_json_object(path: str | Path, description: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _require_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_sha256(manifest)


def _validate_identity(identity: Mapping[str, Any], name: str) -> None:
    if identity.get("kind") not in {"file", "directory"}:
        raise ValueError(f"{name} has an invalid artifact kind")
    _require_sha256(identity.get("sha256"), f"{name}.sha256")
    actual = path_identity(identity["path"])
    if any(actual[key] != identity.get(key) for key in ("path", "kind", "sha256")):
        raise ValueError(f"{name} identity does not match the current artifact")


def validate_run_manifest(
    manifest: Mapping[str, Any], *, require_finalized: bool | None = None
) -> None:
    if int(manifest.get("schema_version", -1)) != RUN_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported run manifest schema")
    status = manifest.get("status")
    if status not in {"prepared", "finalized"}:
        raise ValueError("Run manifest status must be prepared or finalized")
    if require_finalized is True and status != "finalized":
        raise ValueError("Parent run manifest must be finalized")
    if require_finalized is False and status != "prepared":
        raise ValueError("Run manifest is already finalized")

    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Run manifest contract must be an object")
    expected_contract_sha = canonical_sha256(contract)
    if manifest.get("contract_sha256") != expected_contract_sha:
        raise ValueError("Run contract SHA256 does not match its canonical content")

    experiment_id = contract.get("experiment_id")
    run_id = contract.get("run_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ValueError("experiment_id must be non-empty")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be non-empty")
    task_index = int(contract.get("task_index", 0))
    method = contract.get("method")
    if task_index < 1 or method not in {"grpo", "rapo"}:
        raise ValueError("Run task and method are invalid")

    repository = contract.get("repository", {})
    upstream = contract.get("upstream", {})
    if not re.fullmatch(r"[0-9a-f]{40}", str(repository.get("commit", ""))):
        raise ValueError("Repository commit must be a full lowercase Git hash")
    _require_sha256(repository.get("diff_sha256"), "repository.diff_sha256")
    if upstream.get("commit") != PINNED_VISUAL_RFT_COMMIT:
        raise ValueError("Visual-RFT commit is not the pinned upstream")
    _require_sha256(upstream.get("patch_sha256"), "upstream.patch_sha256")

    _validate_identity(contract["input_model"], "input_model")
    _validate_identity(contract["data_manifest"], "data_manifest")
    _validate_identity(contract["stage_dataset"], "stage_dataset")
    _validate_identity(contract["reproduction_config"], "reproduction_config")
    config, config_sha = load_reproduction_config(contract["reproduction_config"]["path"])
    del config
    if config_sha != contract["reproduction_config"].get("canonical_sha256"):
        raise ValueError("Reproduction config canonical SHA256 does not match")

    data_manifest = _load_json_object(contract["data_manifest"]["path"], "Data manifest")
    validate_stage_binding(contract["stage_dataset"]["path"], data_manifest, task_index)

    input_state = contract.get("input_state")
    if input_state is not None:
        _validate_identity(input_state, "input_state")
    parent_reference = contract.get("parent_manifest")
    if task_index == 1:
        if parent_reference is not None or input_state is not None:
            raise ValueError("Task 1 forbids parent manifest and prior RaPO state")
    else:
        if parent_reference is None:
            raise ValueError("Task 2+ requires a parent manifest")
        _require_sha256(parent_reference.get("sha256"), "parent_manifest.sha256")
        parent = _load_json_object(parent_reference["path"], "Parent run manifest")
        if manifest_sha256(parent) != parent_reference["sha256"]:
            raise ValueError("Parent manifest SHA256 does not match")
        validate_run_manifest(parent, require_finalized=True)
        parent_contract = parent["contract"]
        parent_artifacts = parent["artifacts"]
        if task_index != int(parent_contract["task_index"]) + 1:
            raise ValueError("Parent task must be exactly task_index - 1")
        for key in ("experiment_id", "method", "repository", "upstream"):
            if contract[key] != parent_contract[key]:
                raise ValueError(f"Run chain changed {key}")
        if run_id == parent_contract["run_id"]:
            raise ValueError("Each task run_id must be unique")
        if contract["input_model"]["sha256"] != parent_artifacts["output_model"]["sha256"]:
            raise ValueError("Input model does not match the parent output model")
        if contract["data_manifest"]["sha256"] != parent_contract["data_manifest"]["sha256"]:
            raise ValueError("Run chain changed the data manifest")
        if contract["reproduction_config"] != parent_contract["reproduction_config"]:
            raise ValueError("Run chain changed the reproduction config")
        parent_state = parent_artifacts.get("output_state")
        if method == "rapo":
            if input_state is None or parent_state is None:
                raise ValueError("RaPO task 2+ requires the parent output state")
            if input_state["sha256"] != parent_state["sha256"]:
                raise ValueError("Input RaPO state does not match the parent output state")
        elif input_state is not None or parent_state is not None:
            raise ValueError("GRPO runs must not contain CTAN state")

    if method == "grpo" and input_state is not None:
        raise ValueError("GRPO runs must not contain CTAN state")

    artifacts = manifest.get("artifacts")
    if status == "prepared":
        if artifacts is not None:
            raise ValueError("Prepared run manifest must not contain output artifacts")
        return
    if not isinstance(artifacts, dict):
        raise ValueError("Finalized run manifest requires output artifacts")
    _validate_identity(artifacts["output_model"], "output_model")
    if Path(artifacts["output_model"]["path"]).resolve() != Path(
        contract["output_model_path"]
    ).resolve():
        raise ValueError("Final output model path changed from the run contract")
    output_state = artifacts.get("output_state")
    if method == "rapo":
        if output_state is None:
            raise ValueError("Finalized RaPO run requires output state")
        _validate_identity(output_state, "output_state")
        state = _load_json_object(output_state["path"], "RaPO state")
        if (
            int(state.get("task_index", 0)) != task_index
            or state.get("run_id") != run_id
            or state.get("contract_sha256") != expected_contract_sha
        ):
            raise ValueError("RaPO state task, run, or contract binding does not match")
    elif output_state is not None:
        raise ValueError("GRPO runs must not contain CTAN state")


def prepare_run_manifest(
    *,
    manifest_path: str | Path,
    experiment_id: str,
    run_id: str,
    method: str,
    task_index: int,
    repo_root: str | Path,
    upstream_root: str | Path,
    patch_path: str | Path,
    input_model: str | Path,
    output_model: str | Path,
    data_manifest_path: str | Path,
    stage_dataset: str | Path,
    reproduction_config_path: str | Path,
    input_state: str | Path | None = None,
    parent_manifest: str | Path | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    output_model = Path(output_model).resolve()
    if manifest_path == output_model or output_model in manifest_path.parents:
        raise ValueError(
            "Run manifest path must be outside the output model directory"
        )
    data_manifest = _load_json_object(data_manifest_path, "Data manifest")
    validate_stage_binding(stage_dataset, data_manifest, task_index)
    _, reproduction_sha = load_reproduction_config(reproduction_config_path)
    upstream_commit = str(
        _run_git(Path(upstream_root).resolve(), "rev-parse", "HEAD")
    ).strip()
    contract = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "task_index": task_index,
        "method": method,
        "repository": repository_identity(repo_root),
        "upstream": {
            "commit": upstream_commit,
            "patch_sha256": file_sha256(patch_path),
        },
        "input_model": path_identity(input_model),
        "output_model_path": str(output_model),
        "input_state": None if input_state is None else path_identity(input_state),
        "data_manifest": path_identity(data_manifest_path),
        "stage_dataset": path_identity(stage_dataset),
        "reproduction_config": {
            **path_identity(reproduction_config_path),
            "canonical_sha256": reproduction_sha,
        },
        "parent_manifest": None,
    }
    if parent_manifest is not None:
        parent_path = Path(parent_manifest).resolve()
        parent_payload = _load_json_object(parent_path, "Parent run manifest")
        contract["parent_manifest"] = {
            "path": str(parent_path),
            "sha256": manifest_sha256(parent_payload),
        }
    manifest = {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "status": "prepared",
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "artifacts": None,
    }
    validate_run_manifest(manifest, require_finalized=False)
    write_json_if_absent_or_equal(manifest, manifest_path)
    return manifest


def finalize_run_manifest(
    manifest_path: str | Path,
    *,
    output_model: str | Path,
    output_state: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _load_json_object(path, "Run manifest")
    if manifest.get("status") == "finalized":
        validate_run_manifest(manifest, require_finalized=True)
        return manifest
    validate_run_manifest(manifest, require_finalized=False)
    finalized = dict(manifest)
    finalized["status"] = "finalized"
    finalized["artifacts"] = {
        "output_model": path_identity(output_model),
        "output_state": None if output_state is None else path_identity(output_state),
    }
    validate_run_manifest(finalized, require_finalized=True)
    _atomic_write_json(path, finalized)
    return finalized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare or finalize a RaPO run manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-run")
    prepare.add_argument("--manifest", required=True, type=Path)
    prepare.add_argument("--experiment-id", required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--method", required=True, choices=("grpo", "rapo"))
    prepare.add_argument("--task-index", required=True, type=int)
    prepare.add_argument("--repo-root", required=True, type=Path)
    prepare.add_argument("--upstream-root", required=True, type=Path)
    prepare.add_argument("--patch", required=True, type=Path)
    prepare.add_argument("--input-model", required=True, type=Path)
    prepare.add_argument("--output-model", required=True, type=Path)
    prepare.add_argument("--data-manifest", required=True, type=Path)
    prepare.add_argument("--stage-dataset", required=True, type=Path)
    prepare.add_argument("--reproduction-config", required=True, type=Path)
    prepare.add_argument("--input-state", type=Path)
    prepare.add_argument("--parent-manifest", type=Path)
    finalize = subparsers.add_parser("finalize-run")
    finalize.add_argument("--manifest", required=True, type=Path)
    finalize.add_argument("--output-model", required=True, type=Path)
    finalize.add_argument("--output-state", type=Path)
    validate = subparsers.add_parser("validate-run")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--finalized", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "prepare-run":
        manifest = prepare_run_manifest(
            manifest_path=args.manifest,
            experiment_id=args.experiment_id,
            run_id=args.run_id,
            method=args.method,
            task_index=args.task_index,
            repo_root=args.repo_root,
            upstream_root=args.upstream_root,
            patch_path=args.patch,
            input_model=args.input_model,
            output_model=args.output_model,
            data_manifest_path=args.data_manifest,
            stage_dataset=args.stage_dataset,
            reproduction_config_path=args.reproduction_config,
            input_state=args.input_state,
            parent_manifest=args.parent_manifest,
        )
        print(manifest["contract_sha256"])
        return 0
    if args.command == "finalize-run":
        finalized = finalize_run_manifest(
            args.manifest,
            output_model=args.output_model,
            output_state=args.output_state,
        )
        print(manifest_sha256(finalized))
        return 0
    manifest = _load_json_object(args.manifest, "Run manifest")
    validate_run_manifest(
        manifest,
        require_finalized=True if args.finalized else None,
    )
    print(manifest_sha256(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
