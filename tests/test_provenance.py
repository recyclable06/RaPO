import copy
import json
from pathlib import Path

import pytest

from rapo.formal_contract import load_experiment_profile
from rapo.provenance import (
    PINNED_VISUAL_RFT_COMMIT,
    bind_resume_checkpoint,
    canonical_sha256,
    finalize_run_manifest,
    load_reproduction_config,
    manifest_sha256,
    path_identity,
    prepare_run_manifest,
    validate_reproduction_config,
    validate_run_manifest,
    write_json_if_absent_or_equal,
    write_stage_binding,
)
from rapo.resume import CheckpointIdentity, write_checkpoint_binding


REPOSITORY = {"commit": "0" * 40, "diff_sha256": "1" * 64}
UPSTREAM = {
    "commit": PINNED_VISUAL_RFT_COMMIT,
    "patch_sha256": "2" * 64,
}


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def make_directory(path: Path, name: str, content: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(content, encoding="utf-8")
    return path


def make_shared_inputs(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "independent_reproduction.json"
    config_path.write_text(
        Path("configs/independent_reproduction.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    data_manifest = {
        "schema_version": 1,
        "protocol": {"sample_seed": 0},
        "tasks": [{"task_index": 1}, {"task_index": 2}, {"task_index": 3}],
    }
    data_path = write_json(tmp_path / "data_manifest.json", data_manifest)
    stages = {}
    for task_index in (1, 2, 3):
        stage = make_directory(
            tmp_path / f"stage_{task_index}", "rows.arrow", f"task-{task_index}"
        )
        write_stage_binding(stage, data_manifest, task_index)
        stages[task_index] = stage
    return config_path, data_path, data_manifest, stages


def reproduction_identity(config_path: Path):
    _, config_sha = load_reproduction_config(config_path)
    return {**path_identity(config_path), "canonical_sha256": config_sha}


def make_contract(
    *,
    task_index,
    method,
    input_model,
    output_model,
    data_path,
    stage,
    config_path,
    input_state=None,
    parent_path=None,
    run_id=None,
):
    parent_reference = None
    if parent_path is not None:
        parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
        parent_reference = {
            "path": str(parent_path.resolve()),
            "sha256": manifest_sha256(parent_payload),
        }
    return {
        "experiment_id": "experiment-a",
        "run_id": run_id or f"run-{task_index}",
        "task_index": task_index,
        "method": method,
        "repository": copy.deepcopy(REPOSITORY),
        "upstream": copy.deepcopy(UPSTREAM),
        "input_model": path_identity(input_model),
        "output_model_path": str(output_model.resolve()),
        "input_state": None if input_state is None else path_identity(input_state),
        "data_manifest": path_identity(data_path),
        "stage_dataset": path_identity(stage),
        "reproduction_config": reproduction_identity(config_path),
        "parent_manifest": parent_reference,
    }


def prepared_manifest(contract):
    return {
        "schema_version": 1,
        "status": "prepared",
        "contract": contract,
        "contract_sha256": canonical_sha256(contract),
        "artifacts": None,
    }


def build_finalized_parent(tmp_path: Path, method="rapo", suffix="a"):
    config_path, data_path, _, stages = make_shared_inputs(tmp_path)
    base_model = make_directory(tmp_path / "base_model", "weights.bin", "base")
    output_model = make_directory(
        tmp_path / f"output_1_{suffix}", "weights.bin", f"task-1-{suffix}"
    )
    contract = make_contract(
        task_index=1,
        method=method,
        input_model=base_model,
        output_model=output_model,
        data_path=data_path,
        stage=stages[1],
        config_path=config_path,
        run_id=f"run-1-{suffix}",
    )
    manifest = prepared_manifest(contract)
    manifest_path = tmp_path / f"parent_{suffix}.json"
    write_json_if_absent_or_equal(manifest, manifest_path)
    state_path = None
    if method == "rapo":
        state_path = write_json(
            output_model / "rapo_state.json",
            {
                "task_index": 1,
                "run_id": contract["run_id"],
                "contract_sha256": manifest["contract_sha256"],
            },
        )
    finalized = finalize_run_manifest(
        manifest_path,
        output_model=output_model,
        output_state=state_path,
    )
    return finalized, manifest_path, config_path, data_path, stages


def build_task_two(tmp_path: Path, method="rapo"):
    parent, parent_path, config_path, data_path, stages = build_finalized_parent(
        tmp_path, method=method
    )
    input_state = None
    if method == "rapo":
        input_state = Path(parent["artifacts"]["output_state"]["path"])
    contract = make_contract(
        task_index=2,
        method=method,
        input_model=Path(parent["artifacts"]["output_model"]["path"]),
        output_model=tmp_path / "output_2",
        data_path=data_path,
        stage=stages[2],
        config_path=config_path,
        input_state=input_state,
        parent_path=parent_path,
    )
    return prepared_manifest(contract), parent, parent_path, config_path, data_path, stages


def test_canonical_sha256_is_independent_of_mapping_order():
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


def test_frozen_reproduction_config_is_valid_and_machine_hashable():
    payload, digest = load_reproduction_config("configs/independent_reproduction.json")

    assert payload["claim_scope"] == "independent_reproduction"
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("ctan", "batch_scope"), "per_rank_microbatch"),
        (("ctan", "correction"), 0),
        (("surrogate", "rollout_reuse"), 2),
        (("surrogate", "clipping"), 0.2),
        (("kl", "beta"), 0.1),
    ],
)
def test_wrong_reproduction_semantics_are_rejected(path, value):
    payload = json.loads(
        Path("configs/independent_reproduction.json").read_text(encoding="utf-8")
    )
    payload[path[0]][path[1]] = value

    with pytest.raises(ValueError, match="frozen independent-reproduction"):
        validate_reproduction_config(payload)


def test_task_one_rejects_prior_state(tmp_path):
    config_path, data_path, _, stages = make_shared_inputs(tmp_path)
    base_model = make_directory(tmp_path / "base", "weights.bin", "base")
    prior_state = write_json(tmp_path / "prior.json", {"state": "foreign"})
    contract = make_contract(
        task_index=1,
        method="rapo",
        input_model=base_model,
        output_model=tmp_path / "output",
        data_path=data_path,
        stage=stages[1],
        config_path=config_path,
        input_state=prior_state,
    )

    with pytest.raises(ValueError, match="Task 1 forbids"):
        validate_run_manifest(prepared_manifest(contract))


def test_task_two_accepts_exact_finalized_parent_chain(tmp_path):
    child, *_ = build_task_two(tmp_path)

    validate_run_manifest(child, require_finalized=False)


@pytest.mark.parametrize(
    "swapped_field",
    ["task", "model", "state", "data", "config", "parent"],
)
def test_task_chain_rejects_swapped_lineage_component(tmp_path, swapped_field):
    child, parent, _, config_path, data_path, stages = build_task_two(tmp_path)
    contract = child["contract"]
    if swapped_field == "task":
        contract["task_index"] = 3
        contract["stage_dataset"] = path_identity(stages[3])
    elif swapped_field == "model":
        other_model = make_directory(tmp_path / "other_model", "weights.bin", "other")
        contract["input_model"] = path_identity(other_model)
    elif swapped_field == "state":
        other_state = write_json(tmp_path / "other_state.json", {"state": "other"})
        contract["input_state"] = path_identity(other_state)
    elif swapped_field == "data":
        changed_manifest = {
            "schema_version": 1,
            "protocol": {"sample_seed": 99},
            "tasks": [{"task_index": 1}, {"task_index": 2}],
        }
        changed_path = write_json(tmp_path / "changed_data.json", changed_manifest)
        changed_stage = make_directory(
            tmp_path / "changed_stage", "rows.arrow", "changed"
        )
        write_stage_binding(changed_stage, changed_manifest, 2)
        contract["data_manifest"] = path_identity(changed_path)
        contract["stage_dataset"] = path_identity(changed_stage)
    elif swapped_field == "config":
        swapped_config = tmp_path / "swapped_config.json"
        payload, _ = load_reproduction_config(config_path)
        swapped_config.write_text(json.dumps(payload), encoding="utf-8")
        contract["reproduction_config"] = reproduction_identity(swapped_config)
    else:
        other_root = tmp_path / "other_parent_root"
        other_parent, other_path, *_ = build_finalized_parent(other_root, suffix="b")
        del other_parent
        contract["parent_manifest"] = {
            "path": str(other_path.resolve()),
            "sha256": manifest_sha256(
                json.loads(other_path.read_text(encoding="utf-8"))
            ),
        }
    child["contract_sha256"] = canonical_sha256(contract)

    with pytest.raises(ValueError):
        validate_run_manifest(child, require_finalized=False)


def test_grpo_chain_uses_same_provenance_without_ctan_state(tmp_path):
    child, parent, *_ = build_task_two(tmp_path, method="grpo")

    validate_run_manifest(parent, require_finalized=True)
    validate_run_manifest(child, require_finalized=False)
    assert parent["artifacts"]["output_state"] is None
    assert child["contract"]["input_state"] is None


def test_existing_different_run_manifest_is_rejected(tmp_path):
    path = tmp_path / "run_manifest.json"
    write_json_if_absent_or_equal({"run_id": "a"}, path)

    with pytest.raises(ValueError, match="different manifest"):
        write_json_if_absent_or_equal({"run_id": "b"}, path)


def test_prepare_rejects_manifest_inside_output_before_artifact_access(tmp_path):
    output_model = tmp_path / "artifacts" / ".." / "output"
    manifest_path = tmp_path / "output" / "run_manifest.json"

    with pytest.raises(
        ValueError, match="Run manifest path must be outside the output model directory"
    ):
        prepare_run_manifest(
            manifest_path=manifest_path,
            experiment_id="experiment-a",
            run_id="run-1",
            method="grpo",
            task_index=1,
            repo_root=tmp_path / "missing-repository",
            upstream_root=tmp_path / "missing-upstream",
            patch_path=tmp_path / "missing.patch",
            input_model=tmp_path / "missing-input-model",
            output_model=output_model,
            data_manifest_path=tmp_path / "missing-data-manifest.json",
            stage_dataset=tmp_path / "missing-stage",
            reproduction_config_path=tmp_path / "missing-config.json",
        )

    assert not manifest_path.exists()
    assert not output_model.resolve().exists()


def test_sibling_manifest_preserves_output_directory_identity(tmp_path):
    output_model = tmp_path / "output"
    manifest_path = tmp_path / "output-audit.json"
    missing_data_manifest = tmp_path / "missing-data-manifest.json"
    with pytest.raises(FileNotFoundError) as error:
        prepare_run_manifest(
            manifest_path=manifest_path,
            experiment_id="experiment-a",
            run_id="run-1",
            method="grpo",
            task_index=1,
            repo_root=tmp_path / "missing-repository",
            upstream_root=tmp_path / "missing-upstream",
            patch_path=tmp_path / "missing.patch",
            input_model=tmp_path / "missing-input-model",
            output_model=output_model,
            data_manifest_path=missing_data_manifest,
            stage_dataset=tmp_path / "missing-stage",
            reproduction_config_path=tmp_path / "missing-config.json",
        )
    assert Path(error.value.filename) == missing_data_manifest

    config_path, data_path, _, stages = make_shared_inputs(tmp_path)
    base_model = make_directory(tmp_path / "base", "weights.bin", "base")
    output_model = make_directory(output_model, "weights.bin", "trained")
    contract = make_contract(
        task_index=1,
        method="grpo",
        input_model=base_model,
        output_model=output_model,
        data_path=data_path,
        stage=stages[1],
        config_path=config_path,
    )
    write_json_if_absent_or_equal(prepared_manifest(contract), manifest_path)

    finalized = finalize_run_manifest(manifest_path, output_model=output_model)
    validate_run_manifest(finalized, require_finalized=True)

    assert finalized["artifacts"]["output_model"] == path_identity(output_model)


def test_formal_profile_is_bound_to_contract_hash_and_output_namespace(tmp_path):
    config_path, data_path, _, stages = make_shared_inputs(tmp_path)
    base_model = make_directory(tmp_path / "base", "weights.bin", "base")
    profile_path = Path("configs/formal_profile.json").resolve()
    _, profile_sha = load_experiment_profile(profile_path)
    contract = make_contract(
        task_index=1,
        method="grpo",
        input_model=base_model,
        output_model=tmp_path / "formal" / "output",
        data_path=data_path,
        stage=stages[1],
        config_path=config_path,
    )
    contract["experiment_profile"] = {
        **path_identity(profile_path),
        "canonical_sha256": profile_sha,
    }
    manifest = prepared_manifest(contract)

    validate_run_manifest(manifest, require_finalized=False)

    contract["output_model_path"] = str((tmp_path / "legacy_2080ti" / "output").resolve())
    manifest["contract_sha256"] = canonical_sha256(contract)
    with pytest.raises(ValueError, match="profile namespace"):
        validate_run_manifest(manifest, require_finalized=False)


def test_prepared_manifest_binds_exact_production_resume_checkpoint(tmp_path):
    config_path, data_path, _, stages = make_shared_inputs(tmp_path)
    base_model = make_directory(tmp_path / "base", "weights.bin", "base")
    profile_path = Path("configs/formal_profile.json").resolve()
    _, profile_sha = load_experiment_profile(profile_path)
    contract = make_contract(
        task_index=1,
        method="grpo",
        input_model=base_model,
        output_model=tmp_path / "formal" / "output",
        data_path=data_path,
        stage=stages[1],
        config_path=config_path,
        run_id="formal-run",
    )
    contract["experiment_profile"] = {
        **path_identity(profile_path),
        "canonical_sha256": profile_sha,
    }
    manifest = prepared_manifest(contract)
    manifest["resume"] = None
    manifest_path = write_json(tmp_path / "run.json", manifest)
    checkpoint = tmp_path / "formal" / "output" / "checkpoint-3"
    checkpoint.mkdir(parents=True)
    for name in (
        "scheduler.pt",
        "optimizer.pt",
        "rng_state.pth",
        "model.safetensors",
    ):
        (checkpoint / name).write_text(name, encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 3}), encoding="utf-8"
    )
    write_checkpoint_binding(
        checkpoint,
        identity=CheckpointIdentity(
            "formal-run", profile_sha, manifest["contract_sha256"]
        ),
        global_step=3,
        require_ctan=False,
    )

    bound = bind_resume_checkpoint(manifest_path, checkpoint)

    assert bound["resume"]["global_step"] == 3
    validate_run_manifest(bound, require_finalized=False)
