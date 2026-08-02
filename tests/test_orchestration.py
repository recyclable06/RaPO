import json
import math
import shutil
from pathlib import Path

import pytest

from rapo.evaluation import build_prediction_lineage
from rapo.formal_contract import load_experiment_profile
from rapo.orchestration import (
    aggregate_plan,
    build_plan,
    status_plan,
    summarize_order_metrics,
)
from rapo.provenance import (
    PINNED_VISUAL_RFT_COMMIT,
    canonical_sha256,
    load_reproduction_config,
    path_identity,
    write_stage_binding,
)
from rapo.resume import CheckpointIdentity, write_checkpoint_binding


ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "configs" / "formal_orchestration.json"
PROFILE = ROOT / "configs" / "formal_profile.json"
REPRODUCTION = ROOT / "configs" / "independent_reproduction.json"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _small_manifest(root, order_seed):
    classes = []
    tasks = []
    for task_index in range(1, 11):
        classes.append(
            {
                "wnid": f"n{order_seed}{task_index:03d}",
                "label": f"class {task_index}",
                "task_index": task_index,
                "train_images": [f"order{order_seed}/task{task_index}/train.jpg"],
                "test_images": [f"order{order_seed}/task{task_index}/test.jpg"],
            }
        )
        tasks.append(
            {
                "task_index": task_index,
                "train_size": 1,
                "test_size": task_index,
                "class_wnids": [f"n{order_seed}{task_index:03d}"],
                "class_names": [f"class {task_index}"],
                "seen_class_names": [f"class {index}" for index in range(1, task_index + 1)],
            }
        )
    manifest = {
        "schema_version": 1,
        "dataset": "ImageNet-R",
        "protocol": {
            "num_tasks": 10,
            "class_order_seed": order_seed,
            "sample_seed": 0,
        },
        "classes": classes,
        "tasks": tasks,
    }
    manifest_path = root / f"order_{order_seed}" / "manifest.json"
    _write_json(manifest_path, manifest)
    for task_index in range(1, 11):
        stage = manifest_path.parent / "visual_rft" / f"task_{task_index:02d}"
        stage.mkdir(parents=True)
        (stage / "dataset.txt").write_text(f"order={order_seed},task={task_index}\n")
        write_stage_binding(stage, manifest, task_index)
    return manifest_path


@pytest.fixture
def plan_workspace(tmp_path):
    manifests = [_small_manifest(tmp_path / "data", seed) for seed in (0, 1, 2)]
    base = tmp_path / "pinned-base"
    base.mkdir()
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    plan = build_plan(
        config_path=CONFIG,
        manifest_paths=manifests,
        pinned_base=base,
        artifact_root=artifact_root,
    )
    return plan


def _training(plan, order_seed, method, task_index):
    return next(
        node
        for node in plan["nodes"]["training"]
        if (node["order_seed"], node["method"], node["task_index"])
        == (order_seed, method, task_index)
    )


def _write_run(plan, node, *, finalized, resume=False):
    parent = None
    if node["task_index"] > 1:
        parent_node = _training(
            plan, node["order_seed"], node["method"], node["task_index"] - 1
        )
        parent_path = Path(parent_node["run_manifest"])
        parent = json.loads(parent_path.read_text(encoding="utf-8"))

    output_model = Path(node["output_model"])
    input_state = None
    if node["method"] == "rapo" and node["task_index"] > 1:
        input_state = path_identity(Path(node["input_state"]))
    _, reproduction_sha = load_reproduction_config(REPRODUCTION)
    _, profile_sha = load_experiment_profile(PROFILE)
    contract = {
        "experiment_id": node["experiment_id"],
        "run_id": node["run_id"],
        "task_index": node["task_index"],
        "method": node["method"],
        "repository": {"commit": "0" * 40, "diff_sha256": "1" * 64},
        "upstream": {"commit": PINNED_VISUAL_RFT_COMMIT, "patch_sha256": "2" * 64},
        "input_model": path_identity(Path(node["input_model"])),
        "output_model_path": str(output_model.resolve()),
        "input_state": input_state,
        "data_manifest": path_identity(Path(node["data_manifest"])),
        "stage_dataset": path_identity(Path(node["stage_dataset"])),
        "reproduction_config": {
            **path_identity(REPRODUCTION),
            "canonical_sha256": reproduction_sha,
        },
        "experiment_profile": {
            **path_identity(PROFILE),
            "canonical_sha256": profile_sha,
        },
        "parent_manifest": None,
    }
    if parent is not None:
        contract["parent_manifest"] = {
            "path": str(Path(node["parent_manifest"]).resolve()),
            "sha256": canonical_sha256(parent),
        }
    contract_sha = canonical_sha256(contract)
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "contract": contract,
        "contract_sha256": contract_sha,
        "artifacts": None,
        "resume": None,
    }

    if finalized:
        output_model.mkdir(parents=True, exist_ok=True)
        (output_model / "model.safetensors").write_text(node["run_id"], encoding="utf-8")
        output_state = None
        if node["method"] == "rapo":
            output_state = Path(node["output_state"])
            _write_json(
                output_state,
                {
                    "task_index": node["task_index"],
                    "run_id": node["run_id"],
                    "contract_sha256": contract_sha,
                    "profile_sha256": profile_sha,
                },
            )
        manifest["status"] = "finalized"
        manifest["artifacts"] = {
            "output_model": path_identity(output_model),
            "output_state": None if output_state is None else path_identity(output_state),
        }
    elif resume:
        checkpoint = output_model / "checkpoint-3"
        checkpoint.mkdir(parents=True)
        for name in ("optimizer.pt", "scheduler.pt", "rng_state_0.pth", "model.safetensors"):
            (checkpoint / name).write_text(name, encoding="utf-8")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 3}), encoding="utf-8"
        )
        if node["method"] == "rapo":
            (checkpoint / "rapo_state.json").write_text("{}", encoding="utf-8")
        binding = write_checkpoint_binding(
            checkpoint,
            identity=CheckpointIdentity(node["run_id"], profile_sha, contract_sha),
            global_step=3,
            require_ctan=node["method"] == "rapo",
        )
        manifest["resume"] = {
            "checkpoint": path_identity(checkpoint),
            "binding": path_identity(binding),
            "global_step": 3,
        }

    _write_json(Path(node["run_manifest"]), manifest)
    return manifest


def _materialize_predictions(plan):
    cutoffs = {
        ("grpo", 0): 10,
        ("grpo", 1): 5,
        ("grpo", 2): 0,
        ("rapo", 0): 8,
        ("rapo", 1): 6,
        ("rapo", 2): 4,
    }
    for node in plan["nodes"]["training"]:
        _write_run(plan, node, finalized=True)
    for node in plan["nodes"]["prediction"]:
        lineage = build_prediction_lineage(
            model_path=node["model"],
            stage_dataset=node["stage_dataset"],
            data_manifest_path=node["data_manifest"],
            profile_path=PROFILE,
            run_manifest_path=node["run_manifest"],
            torch_dtype="bfloat16",
            attention="flash_attention_2",
        )
        rows = []
        for eval_task in range(1, node["task_index"] + 1):
            correct = node["task_index"] < 10 or eval_task <= cutoffs[
                (node["method"], node["order_seed"])
            ]
            rows.append(
                {
                    "after_task": node["task_index"],
                    "eval_task": eval_task,
                    "relative_path": f"order{node['order_seed']}/task{eval_task}/test.jpg",
                    "completion": f"<answer>{'class ' + str(eval_task) if correct else 'wrong'}</answer>",
                    "target": f"<answer>class {eval_task}</answer>",
                    "lineage": lineage,
                }
            )
        output = Path(node["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


def test_plan_is_deterministic_and_has_exact_formal_counts(plan_workspace):
    plan = plan_workspace
    rebuilt = build_plan(
        config_path=CONFIG,
        manifest_paths=[item["path"] for item in plan["inputs"]["data_manifests"]],
        pinned_base=plan["inputs"]["pinned_base"]["path"],
        artifact_root=plan["artifact_root"],
    )
    assert rebuilt == plan
    assert plan["plan_sha256"] == canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    assert {kind: len(plan["nodes"][kind]) for kind in plan["nodes"]} == {
        "training": 60,
        "prediction": 60,
        "metrics": 6,
        "summary": 2,
    }
    assert sum(node["expected_cells"] for node in plan["nodes"]["metrics"]) == 330
    assert all(node["expected_cells"] == 55 for node in plan["nodes"]["metrics"])
    assert [
        node["expected_count"]
        for node in plan["nodes"]["prediction"]
        if node["method"] == "grpo" and node["order_seed"] == 0
    ] == list(range(1, 11))
    assert all(node["sample_scope"] == "manifest_full_test" for node in plan["nodes"]["prediction"])


def test_methods_are_paired_and_each_chain_uses_only_its_immediate_parent(plan_workspace):
    plan = plan_workspace
    shared = (
        "order_seed",
        "task_index",
        "data_manifest",
        "stage_dataset",
        "experiment_profile",
        "reproduction_config",
        "training_seed",
        "data_seed",
        "sample_seed",
        "training_budget",
    )
    for order_seed in (0, 1, 2):
        for task_index in range(1, 11):
            grpo = _training(plan, order_seed, "grpo", task_index)
            rapo = _training(plan, order_seed, "rapo", task_index)
            assert {key: grpo[key] for key in shared} == {key: rapo[key] for key in shared}
            if task_index == 1:
                assert grpo["input_model"] == rapo["input_model"]
                assert grpo["depends_on"] == rapo["depends_on"] == []
            else:
                assert grpo["depends_on"] == [
                    _training(plan, order_seed, "grpo", task_index - 1)["id"]
                ]
                assert rapo["depends_on"] == [
                    _training(plan, order_seed, "rapo", task_index - 1)["id"]
                ]
            assert grpo["input_state"] is None
            assert (rapo["output_state"] is not None) == True


def test_status_accepts_finalized_predecessor_and_same_run_resume_but_rejects_foreign_resume(
    plan_workspace,
):
    plan = plan_workspace
    first = _training(plan, 0, "grpo", 1)
    second = _training(plan, 0, "grpo", 2)
    _write_run(plan, first, finalized=True)
    _write_run(plan, second, finalized=False, resume=True)

    status = status_plan(plan)
    assert first["id"] in status["completed_nodes"]
    resumed = next(item for item in status["runnable"] if item["node_id"] == second["id"])
    assert resumed["resume"] is True
    assert "RAPO_RESUME_CHECKPOINT=" in resumed["command"]

    binding_path = Path(second["output_model"]) / "checkpoint-3" / "rapo_checkpoint_binding.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["identity"]["run_id"] = "foreign-run"
    _write_json(binding_path, binding)
    manifest_path = Path(second["run_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resume"]["checkpoint"] = path_identity(binding_path.parent)
    manifest["resume"]["binding"] = path_identity(binding_path)
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="binding does not match"):
        status_plan(plan)


def test_aggregate_reuses_strict_prediction_contract_and_matches_hand_calculation(plan_workspace):
    plan = plan_workspace
    _materialize_predictions(plan)
    result = aggregate_plan(plan)
    summaries = {item["method"]: item for item in result["method_summaries"]}
    assert summaries["grpo"]["last_accuracy"]["mean"] == pytest.approx(0.5)
    assert summaries["grpo"]["last_accuracy"]["std"] == pytest.approx(math.sqrt(1 / 6))
    assert summaries["grpo"]["forgetting"]["mean"] == pytest.approx((0 + 4 / 9 + 1) / 3)
    assert summaries["rapo"]["last_accuracy"]["mean"] == pytest.approx(0.6)
    assert summaries["rapo"]["last_accuracy"]["std"] == pytest.approx(math.sqrt(2 / 75))
    assert summaries["rapo"]["forgetting"]["mean"] == pytest.approx(1 / 3)
    assert all(item["population_std_ddof"] == 0 for item in summaries.values())
    assert len(result["order_metrics"]) == 6
    assert sum(len(item["prediction_artifacts"]) for item in result["order_metrics"]) == 60
    assert all(len(item["metrics"]["accuracy_matrix"]) == 10 for item in result["order_metrics"])


@pytest.mark.parametrize("corruption", ["incomplete", "duplicate", "method", "order", "hash"])
def test_summary_rejects_incomplete_duplicate_or_wrong_order_identity(plan_workspace, corruption):
    plan = plan_workspace
    rows = [
        {
            "method": method,
            "order_seed": order_seed,
            "plan_sha256": plan["plan_sha256"],
            "last_accuracy": 0.5,
            "forgetting": 0.25,
        }
        for method in ("grpo", "rapo")
        for order_seed in (0, 1, 2)
    ]
    if corruption == "incomplete":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = dict(rows[0])
    elif corruption == "method":
        rows[-1]["method"] = "foreign"
    elif corruption == "order":
        rows[-1]["order_seed"] = 7
    else:
        rows[-1]["plan_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        summarize_order_metrics(
            rows,
            plan_sha256=plan["plan_sha256"],
            methods=("grpo", "rapo"),
            order_seeds=(0, 1, 2),
        )


def test_status_rejects_wrong_artifact_and_aggregate_rejects_changed_prediction(plan_workspace):
    plan = plan_workspace
    source = _training(plan, 0, "grpo", 1)
    target = _training(plan, 1, "grpo", 1)
    _write_run(plan, source, finalized=True)
    Path(target["run_manifest"]).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source["run_manifest"], target["run_manifest"])
    with pytest.raises(ValueError, match="does not match plan"):
        status_plan(plan)

    Path(target["run_manifest"]).unlink()
    _materialize_predictions(plan)
    prediction = Path(plan["nodes"]["prediction"][0]["output"])
    rows = prediction.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0])
    payload["lineage"]["run_id"] = "foreign-run"
    rows[0] = json.dumps(payload, sort_keys=True)
    prediction.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lineage does not match"):
        aggregate_plan(plan)
