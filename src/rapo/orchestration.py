"""Restricted CPU/no-GPU orchestration contract for the formal experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rapo.evaluation import (
    aggregate_prediction_records,
    build_prediction_lineage,
    compute_continual_metrics,
)
from rapo.formal_contract import load_experiment_profile
from rapo.provenance import (
    canonical_sha256,
    file_sha256,
    load_reproduction_config,
    path_identity,
    validate_run_manifest,
    validate_stage_binding,
)


SCHEMA_VERSION = 1
DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "formal_orchestration.json"


def _read_object(path: str | Path, description: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _resolve_reference(config_path: Path, reference: Mapping[str, Any]) -> Path:
    path = Path(str(reference.get("path", "")))
    if not path.is_absolute():
        path = config_path.resolve().parent.parent / path
    return path.resolve()


def _load_config(path: str | Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config_path = Path(path).resolve()
    config = _read_object(config_path, "Orchestration config")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported orchestration config schema")
    expected = {
        "num_tasks": 10,
        "methods": ["grpo", "rapo"],
        "class_order_seeds": [0, 1, 2],
        "sample_seed": 0,
        "training_seed": 0,
        "data_seed": 0,
        "sample_scope": "manifest_full_test",
        "artifact_namespace": "formal",
        "population_std_ddof": 0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Formal orchestration requires {key}={value!r}")
    resolved = {}
    for key in ("experiment_profile", "reproduction_config"):
        reference = config.get(key)
        if not isinstance(reference, dict):
            raise ValueError(f"Orchestration config requires {key}")
        resolved[key] = _resolve_reference(config_path, reference)
        if file_sha256(resolved[key]) != str(reference.get("sha256", "")).lower():
            raise ValueError(f"Frozen {key} SHA256 does not match")
    profile, _ = load_experiment_profile(resolved["experiment_profile"])
    if profile["profile_kind"] != "formal":
        raise ValueError("Orchestration requires the formal experiment profile")
    load_reproduction_config(resolved["reproduction_config"])
    return config, resolved


def _load_manifests(
    manifest_paths: Iterable[str | Path], config: Mapping[str, Any]
) -> dict[int, tuple[Path, dict[str, Any]]]:
    manifests = {}
    for supplied_path in manifest_paths:
        path = Path(supplied_path).resolve()
        manifest = _read_object(path, "Data manifest")
        protocol = manifest.get("protocol")
        tasks = manifest.get("tasks")
        if not isinstance(protocol, dict) or not isinstance(tasks, list):
            raise ValueError("Data manifest requires protocol and tasks")
        seed = protocol.get("class_order_seed")
        if seed not in config["class_order_seeds"]:
            raise ValueError(f"Unexpected class-order seed: {seed}")
        if seed in manifests:
            raise ValueError(f"Duplicate data manifest for class-order seed {seed}")
        if protocol.get("num_tasks") != config["num_tasks"]:
            raise ValueError("Data manifest task count does not match orchestration")
        if protocol.get("sample_seed") != config["sample_seed"]:
            raise ValueError("Data manifest sample seed does not match orchestration")
        if len(tasks) != config["num_tasks"]:
            raise ValueError("Data manifest must contain exactly ten task records")
        previous_test_size = 0
        for task_index, task in enumerate(tasks, start=1):
            if not isinstance(task, dict) or task.get("task_index") != task_index:
                raise ValueError("Data manifest tasks must be ordered 1 through 10")
            test_size = task.get("test_size")
            if not isinstance(test_size, int) or test_size <= previous_test_size:
                raise ValueError("Cumulative full-test sizes must increase at every task")
            previous_test_size = test_size
            validate_stage_binding(
                path.parent / "visual_rft" / f"task_{task_index:02d}",
                manifest,
                task_index,
            )
        manifests[int(seed)] = (path, manifest)
    if set(manifests) != set(config["class_order_seeds"]):
        raise ValueError("Formal orchestration requires exactly class orders 0, 1, and 2")
    return manifests


def _quoted(value: str | Path) -> str:
    return shlex.quote(str(value))


def _training_command(node: Mapping[str, Any], reproduction_config: Path) -> str:
    variables = [
        ("GPU_IDS", "<GPU_IDS>"),
        ("VISUAL_RFT_ROOT", "<PINNED_VISUAL_RFT_ROOT>"),
        ("MODEL_PATH", node["input_model"]),
        ("DATASET_PATH", node["stage_dataset"]),
        ("OUTPUT_DIR", node["output_model"]),
        ("RUN_MANIFEST_PATH", node["run_manifest"]),
        ("EXPERIMENT_ID", node["experiment_id"]),
        ("RUN_ID", node["run_id"]),
        ("DATA_MANIFEST_PATH", node["data_manifest"]),
        ("RAPO_REPRODUCTION_CONFIG", reproduction_config),
    ]
    if node["task_index"] > 1:
        variables.append(("PARENT_MANIFEST_PATH", node["parent_manifest"]))
    if node["input_state"] is not None:
        variables.append(("RAPO_STATE_PATH", node["input_state"]))
    environment = " ".join(f"{name}={_quoted(value)}" for name, value in variables)
    return (
        f"{environment} bash scripts/run_imagenet_r_formal.sh "
        f"{node['method']} {node['task_index']}"
    )


def _prediction_command(node: Mapping[str, Any], profile: Path) -> str:
    values = [
        "python",
        "scripts/evaluate_qwen2_vl.py",
        node["model"],
        node["stage_dataset"],
        node["output"],
        "--after-task",
        str(node["task_index"]),
        "--profile",
        profile,
        "--run-manifest",
        node["run_manifest"],
        "--data-manifest",
        node["data_manifest"],
    ]
    return " ".join(_quoted(value) for value in values)


def build_plan(
    *,
    config_path: str | Path,
    manifest_paths: Iterable[str | Path],
    pinned_base: str | Path,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Build the frozen 10-task x 2-method x 3-order DAG without execution."""

    config_path = Path(config_path).resolve()
    config, references = _load_config(config_path)
    manifests = _load_manifests(manifest_paths, config)
    pinned_base = Path(pinned_base).resolve()
    artifact_root = Path(artifact_root).resolve()
    profile, profile_sha = load_experiment_profile(references["experiment_profile"])
    _, reproduction_sha = load_reproduction_config(references["reproduction_config"])
    profile_identity = {
        **path_identity(references["experiment_profile"]),
        "canonical_sha256": profile_sha,
    }
    reproduction_identity = {
        **path_identity(references["reproduction_config"]),
        "canonical_sha256": reproduction_sha,
    }
    initial_base = path_identity(pinned_base)
    training_budget = {
        key: profile["training"][key]
        for key in (
            "budget_kind",
            "num_train_epochs",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "num_generations",
        )
    }
    nodes: dict[str, list[dict[str, Any]]] = {
        "training": [],
        "prediction": [],
        "metrics": [],
        "summary": [],
    }
    for order_seed in config["class_order_seeds"]:
        manifest_path, manifest = manifests[order_seed]
        manifest_identity = path_identity(manifest_path)
        for method in config["methods"]:
            experiment_id = f"formal-order{order_seed}-{method}"
            previous = None
            for task_index in range(1, config["num_tasks"] + 1):
                run_directory = (
                    artifact_root
                    / config["artifact_namespace"]
                    / f"order_{order_seed}"
                    / method
                    / f"task_{task_index:02d}"
                )
                output_model = run_directory / "model"
                node = {
                    "id": f"order{order_seed}-{method}-task{task_index:02d}-train",
                    "kind": "training",
                    "order_seed": order_seed,
                    "method": method,
                    "task_index": task_index,
                    "experiment_id": experiment_id,
                    "run_id": f"{experiment_id}-task{task_index:02d}",
                    "depends_on": [] if previous is None else [previous["id"]],
                    "input_model": (
                        initial_base["path"] if previous is None else previous["output_model"]
                    ),
                    "output_model": str(output_model),
                    "run_manifest": str(run_directory / "run_manifest.json"),
                    "parent_manifest": None if previous is None else previous["run_manifest"],
                    "input_state": (
                        previous["output_state"]
                        if method == "rapo" and previous is not None
                        else None
                    ),
                    "output_state": (
                        str(output_model / "rapo_state.json") if method == "rapo" else None
                    ),
                    "data_manifest": manifest_identity["path"],
                    "data_manifest_sha256": manifest_identity["sha256"],
                    "stage_dataset": str(
                        manifest_path.parent / "visual_rft" / f"task_{task_index:02d}"
                    ),
                    "experiment_profile": profile_identity["path"],
                    "reproduction_config": reproduction_identity["path"],
                    "training_seed": config["training_seed"],
                    "data_seed": config["data_seed"],
                    "sample_seed": config["sample_seed"],
                    "training_budget": training_budget,
                    "rapo_parameters": (
                        {
                            "retention_alpha": 20.0,
                            "retention_weight": 0.5,
                            "ctan_beta": 0.999,
                            "ctan_epsilon": 0.0001,
                        }
                        if method == "rapo"
                        else None
                    ),
                }
                node["command"] = _training_command(
                    node, references["reproduction_config"]
                )
                nodes["training"].append(node)
                prediction = {
                    "id": f"order{order_seed}-{method}-task{task_index:02d}-predict",
                    "kind": "prediction",
                    "order_seed": order_seed,
                    "method": method,
                    "task_index": task_index,
                    "depends_on": [node["id"]],
                    "model": node["output_model"],
                    "stage_dataset": node["stage_dataset"],
                    "data_manifest": node["data_manifest"],
                    "experiment_profile": node["experiment_profile"],
                    "run_manifest": node["run_manifest"],
                    "output": str(run_directory / "predictions_full_test.jsonl"),
                    "sample_scope": config["sample_scope"],
                    "samples_per_class": None,
                    "expected_count": manifest["tasks"][task_index - 1]["test_size"],
                    "expected_cells": task_index,
                }
                prediction["command"] = _prediction_command(
                    prediction, references["experiment_profile"]
                )
                nodes["prediction"].append(prediction)
                previous = node
            nodes["metrics"].append(
                {
                    "id": f"order{order_seed}-{method}-metrics",
                    "kind": "metrics",
                    "order_seed": order_seed,
                    "method": method,
                    "depends_on": [
                        node["id"]
                        for node in nodes["prediction"]
                        if node["order_seed"] == order_seed and node["method"] == method
                    ],
                    "expected_cells": 55,
                    "output": str(
                        artifact_root
                        / config["artifact_namespace"]
                        / f"order_{order_seed}"
                        / method
                        / "continual_metrics.json"
                    ),
                }
            )
    for method in config["methods"]:
        nodes["summary"].append(
            {
                "id": f"{method}-summary",
                "kind": "summary",
                "method": method,
                "depends_on": [
                    node["id"] for node in nodes["metrics"] if node["method"] == method
                ],
                "order_seeds": list(config["class_order_seeds"]),
                "population_std_ddof": 0,
                "output": str(
                    artifact_root
                    / config["artifact_namespace"]
                    / f"{method}_three_order_summary.json"
                ),
            }
        )
    plan = {
        "schema_version": SCHEMA_VERSION,
        "configuration": path_identity(config_path),
        "protocol_name": config["protocol_name"],
        "claim_scope": config["claim_scope"],
        "artifact_root": str(artifact_root),
        "inputs": {
            "pinned_base": initial_base,
            "experiment_profile": profile_identity,
            "reproduction_config": reproduction_identity,
            "data_manifests": [
                path_identity(manifests[seed][0]) for seed in config["class_order_seeds"]
            ],
        },
        "methods": list(config["methods"]),
        "order_seeds": list(config["class_order_seeds"]),
        "num_tasks": config["num_tasks"],
        "nodes": nodes,
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def _validate_plan(plan: Mapping[str, Any]) -> None:
    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported orchestration plan schema")
    expected_hash = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    if plan.get("plan_sha256") != expected_hash:
        raise ValueError("Orchestration plan SHA256 does not match")
    expected_counts = {"training": 60, "prediction": 60, "metrics": 6, "summary": 2}
    nodes = plan.get("nodes")
    if not isinstance(nodes, dict):
        raise ValueError("Orchestration plan nodes must be an object")
    for kind, count in expected_counts.items():
        if not isinstance(nodes.get(kind), list) or len(nodes[kind]) != count:
            raise ValueError(f"Orchestration plan requires {count} {kind} nodes")
    identifiers = [node.get("id") for group in nodes.values() for node in group]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Orchestration plan contains duplicate node IDs")
    for name in ("configuration",):
        expected = plan[name]
        if path_identity(expected["path"]) != expected:
            raise ValueError(f"Orchestration plan {name} identity changed")
    for name in ("pinned_base", "experiment_profile", "reproduction_config"):
        expected = plan["inputs"][name]
        actual = path_identity(expected["path"])
        if any(actual[key] != expected[key] for key in ("path", "kind", "sha256")):
            raise ValueError(f"Orchestration plan {name} identity changed")
    for expected in plan["inputs"]["data_manifests"]:
        if path_identity(expected["path"]) != expected:
            raise ValueError("Orchestration plan data-manifest identity changed")


def _validate_run_node(node: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    validate_run_manifest(manifest, require_finalized=None)
    contract = manifest["contract"]
    expected = {
        "experiment_id": node["experiment_id"],
        "run_id": node["run_id"],
        "method": node["method"],
        "task_index": node["task_index"],
        "output_model_path": str(Path(node["output_model"]).resolve()),
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Run artifact {node['id']} does not match plan")
    path_fields = {
        "input_model": node["input_model"],
        "data_manifest": node["data_manifest"],
        "stage_dataset": node["stage_dataset"],
        "reproduction_config": node["reproduction_config"],
        "experiment_profile": node["experiment_profile"],
    }
    for key, value in path_fields.items():
        if Path(contract[key]["path"]).resolve() != Path(value).resolve():
            raise ValueError(f"Run artifact {node['id']} does not match plan")
    parent = contract.get("parent_manifest")
    expected_parent = node["parent_manifest"]
    if (parent is None) != (expected_parent is None) or (
        parent is not None
        and Path(parent["path"]).resolve() != Path(expected_parent).resolve()
    ):
        raise ValueError(f"Run artifact {node['id']} does not match plan")
    input_state = contract.get("input_state")
    expected_state = node["input_state"]
    if (input_state is None) != (expected_state is None) or (
        input_state is not None
        and Path(input_state["path"]).resolve() != Path(expected_state).resolve()
    ):
        raise ValueError(f"Run artifact {node['id']} does not match plan")
    data_manifest = _read_object(node["data_manifest"], "Data manifest")
    if data_manifest["protocol"].get("class_order_seed") != node["order_seed"]:
        raise ValueError(f"Run artifact {node['id']} does not match plan")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Prediction row {line_number} must be an object")
            records.append(value)
    return records


def _validate_prediction_node(node: Mapping[str, Any]) -> list[dict[str, int]]:
    records = _read_jsonl(Path(node["output"]))
    if len(records) != node["expected_count"]:
        raise ValueError(f"Prediction artifact {node['id']} has the wrong full-test count")
    if any(int(record.get("after_task", 0)) != node["task_index"] for record in records):
        raise ValueError(f"Prediction artifact {node['id']} has the wrong stage")
    reported = records[0].get("lineage") if records else None
    if not isinstance(reported, dict):
        raise ValueError(f"Prediction artifact {node['id']} lacks formal lineage")
    expected_lineage = build_prediction_lineage(
        model_path=node["model"],
        stage_dataset=node["stage_dataset"],
        data_manifest_path=node["data_manifest"],
        profile_path=node["experiment_profile"],
        run_manifest_path=node["run_manifest"],
        torch_dtype=str(reported.get("torch_dtype")),
        attention=str(reported.get("attention")),
    )
    data_manifest = _read_object(node["data_manifest"], "Data manifest")
    cells = aggregate_prediction_records(
        records,
        data_manifest=data_manifest,
        expected_lineage=expected_lineage,
    )
    if len(cells) != node["expected_cells"]:
        raise ValueError(f"Prediction artifact {node['id']} has the wrong cell count")
    if sum(cell["total"] for cell in cells) != node["expected_count"]:
        raise ValueError(f"Prediction artifact {node['id']} has the wrong full-test count")
    return cells


def status_plan(
    plan: Mapping[str, Any], *, plan_path: str | Path | None = None
) -> dict[str, Any]:
    """Validate current artifacts and expose only the safe runnable frontier."""

    _validate_plan(plan)
    completed: list[str] = []
    completed_set: set[str] = set()
    runnable: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    for node in plan["nodes"]["training"]:
        predecessors_ready = all(item in completed_set for item in node["depends_on"])
        manifest_path = Path(node["run_manifest"])
        if not manifest_path.is_file():
            if predecessors_ready:
                runnable.append(
                    {"node_id": node["id"], "kind": "training", "resume": False, "command": node["command"]}
                )
            else:
                blocked.append({"node_id": node["id"], "reason": "predecessor incomplete"})
            continue
        manifest = _read_object(manifest_path, "Run manifest")
        _validate_run_node(node, manifest)
        if manifest["status"] == "finalized":
            if not predecessors_ready and node["depends_on"]:
                raise ValueError(f"Finalized node {node['id']} has an incomplete predecessor")
            completed.append(node["id"])
            completed_set.add(node["id"])
            continue
        if not predecessors_ready:
            blocked.append({"node_id": node["id"], "reason": "predecessor incomplete"})
            continue
        resume = manifest.get("resume")
        output_model = Path(node["output_model"])
        if resume is not None:
            checkpoint = Path(resume["checkpoint"]["path"]).resolve()
            if checkpoint.parent != output_model.resolve() or not checkpoint.name.startswith(
                "checkpoint-"
            ):
                raise ValueError("Resume checkpoint does not belong to the planned run/task")
            runnable.append(
                {
                    "node_id": node["id"],
                    "kind": "training",
                    "resume": True,
                    "command": f"RAPO_RESUME_CHECKPOINT={_quoted(checkpoint)} {node['command']}",
                }
            )
        elif not output_model.exists() or not any(output_model.iterdir()):
            runnable.append(
                {"node_id": node["id"], "kind": "training", "resume": False, "command": node["command"]}
            )
        else:
            blocked.append({"node_id": node["id"], "reason": "incomplete run lacks valid resume"})

    all_predictions_complete = True
    for node in plan["nodes"]["prediction"]:
        if Path(node["output"]).is_file():
            if node["depends_on"][0] not in completed_set:
                raise ValueError(f"Prediction artifact {node['id']} has no finalized training node")
            _validate_prediction_node(node)
            completed.append(node["id"])
            completed_set.add(node["id"])
        else:
            all_predictions_complete = False
            if node["depends_on"][0] in completed_set:
                runnable.append(
                    {"node_id": node["id"], "kind": "prediction", "command": node["command"]}
                )
            else:
                blocked.append({"node_id": node["id"], "reason": "training incomplete"})
    if all_predictions_complete:
        runnable.append(
            {
                "node_id": "aggregate",
                "kind": "aggregate",
                "command": (
                    "python -m rapo.orchestration aggregate --plan "
                    + _quoted("<PLAN_JSON>" if plan_path is None else Path(plan_path).resolve())
                ),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "completed_nodes": completed,
        "runnable": runnable,
        "blocked_nodes": blocked,
    }


def summarize_order_metrics(
    order_metrics: Iterable[Mapping[str, Any]],
    *,
    plan_sha256: str,
    methods: Iterable[str],
    order_seeds: Iterable[int],
) -> list[dict[str, Any]]:
    """Require the exact method/order grid and compute population statistics."""

    rows = [dict(item) for item in order_metrics]
    methods = tuple(methods)
    order_seeds = tuple(order_seeds)
    expected = {(method, seed) for method in methods for seed in order_seeds}
    observed = []
    for row in rows:
        if row.get("plan_sha256") != plan_sha256:
            raise ValueError("Order metric plan SHA256 does not match")
        key = (row.get("method"), row.get("order_seed"))
        if key not in expected:
            raise ValueError("Order metric has the wrong method or order")
        observed.append(key)
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate method/order metric artifact")
    if set(observed) != expected:
        raise ValueError("Three-order method metrics are incomplete")
    summaries = []
    for method in methods:
        ordered = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: row["order_seed"],
        )
        last = [float(row["last_accuracy"]) for row in ordered]
        forgetting = [float(row["forgetting"]) for row in ordered]
        summaries.append(
            {
                "method": method,
                "order_seeds": list(order_seeds),
                "num_orders": len(order_seeds),
                "population_std_ddof": 0,
                "last_accuracy": {
                    "mean": statistics.fmean(last),
                    "std": statistics.pstdev(last),
                },
                "forgetting": {
                    "mean": statistics.fmean(forgetting),
                    "std": statistics.pstdev(forgetting),
                },
            }
        )
    return summaries


def aggregate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate all 60 full-test artifacts and compute six matrices and summaries."""

    _validate_plan(plan)
    order_metrics = []
    all_inputs = []
    for method in plan["methods"]:
        for order_seed in plan["order_seeds"]:
            predictions = sorted(
                (
                    node
                    for node in plan["nodes"]["prediction"]
                    if node["method"] == method and node["order_seed"] == order_seed
                ),
                key=lambda node: node["task_index"],
            )
            if len(predictions) != plan["num_tasks"]:
                raise ValueError("Method/order prediction artifacts are incomplete")
            cells = []
            input_artifacts = []
            for node in predictions:
                path = Path(node["output"])
                if not path.is_file():
                    raise ValueError(f"Missing prediction artifact: {path}")
                cells.extend(_validate_prediction_node(node))
                identity = {
                    "node_id": node["id"],
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
                input_artifacts.append(identity)
                all_inputs.append(identity)
            metrics = compute_continual_metrics(cells, num_tasks=plan["num_tasks"])
            manifest_identity = next(
                item
                for item in plan["inputs"]["data_manifests"]
                if _read_object(item["path"], "Data manifest")["protocol"][
                    "class_order_seed"
                ]
                == order_seed
            )
            order_metrics.append(
                {
                    "method": method,
                    "order_seed": order_seed,
                    "plan_sha256": plan["plan_sha256"],
                    "data_manifest_sha256": manifest_identity["sha256"],
                    "prediction_artifacts": input_artifacts,
                    "last_accuracy": metrics.last_accuracy,
                    "forgetting": metrics.forgetting,
                    "metrics": metrics.to_dict(),
                }
            )
    summaries = summarize_order_metrics(
        order_metrics,
        plan_sha256=plan["plan_sha256"],
        methods=plan["methods"],
        order_seeds=plan["order_seeds"],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "orchestration_inputs": plan["inputs"],
        "input_artifacts": all_inputs,
        "order_metrics": order_metrics,
        "method_summaries": summaries,
    }


def _write_or_print(payload: Mapping[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Restricted formal RaPO CPU/no-GPU orchestration"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    dry_run = subparsers.add_parser("dry-run")
    dry_run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    dry_run.add_argument("--manifest", action="append", required=True, type=Path)
    dry_run.add_argument("--pinned-base", required=True, type=Path)
    dry_run.add_argument("--artifact-root", required=True, type=Path)
    dry_run.add_argument("--output", type=Path)
    status = subparsers.add_parser("status")
    status.add_argument("--plan", required=True, type=Path)
    status.add_argument("--output", type=Path)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--plan", required=True, type=Path)
    aggregate.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "dry-run":
        payload = build_plan(
            config_path=args.config,
            manifest_paths=args.manifest,
            pinned_base=args.pinned_base,
            artifact_root=args.artifact_root,
        )
    else:
        plan = _read_object(args.plan, "Orchestration plan")
        payload = (
            status_plan(plan, plan_path=args.plan)
            if args.command == "status"
            else aggregate_plan(plan)
        )
    _write_or_print(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
