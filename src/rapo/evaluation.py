"""Evaluation utilities for class-incremental image classification."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_LABEL_SEPARATOR_PATTERN = re.compile(r"[_\-.]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")


def extract_answer(text: str) -> str | None:
    """Extract the single final-answer span required by the paper protocol."""

    matches = _ANSWER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    return matches[0].strip()


def normalize_class_name(value: str) -> str:
    """Apply the paper's classification verifier normalization."""

    normalized = _LABEL_SEPARATOR_PATTERN.sub(" ", value.lower())
    return _WHITESPACE_PATTERN.sub(" ", normalized).strip()


def classification_answer_is_correct(completion: str, target: str) -> bool:
    """Return whether a completion exactly matches the target class."""

    predicted_answer = extract_answer(completion)
    if predicted_answer is None:
        return False
    target_answer = extract_answer(target)
    if target_answer is None:
        target_answer = target.strip()
    return normalize_class_name(predicted_answer) == normalize_class_name(target_answer)


def pad_image_to_minimum_size(image: Any, minimum_size: int) -> Any:
    """Pad a PIL image so neither spatial dimension is below ``minimum_size``."""

    if minimum_size < 1:
        raise ValueError("minimum_size must be positive")
    width, height = image.size
    if width >= minimum_size and height >= minimum_size:
        return image

    from PIL import ImageOps

    horizontal_padding = max(0, minimum_size - width)
    vertical_padding = max(0, minimum_size - height)
    left = horizontal_padding // 2
    top = vertical_padding // 2
    return ImageOps.expand(
        image,
        border=(
            left,
            top,
            horizontal_padding - left,
            vertical_padding - top,
        ),
        fill=0,
    )


def resolve_evaluator_settings(
    *,
    profile_path: str | Path | None,
    torch_dtype: str | None,
    attention: str | None,
) -> dict[str, Any]:
    """Resolve explicit evaluator numerics without a formal-to-legacy fallback."""

    if profile_path is None:
        resolved_dtype = "float16" if torch_dtype is None else torch_dtype
        resolved_attention = "sdpa" if attention is None else attention
        profile = None
        profile_sha256 = None
        profile_kind = "legacy_default"
    else:
        from rapo.formal_contract import load_experiment_profile

        profile, profile_sha256 = load_experiment_profile(profile_path)
        profile_kind = profile["profile_kind"]
        configured_dtype = profile["evaluation"]["torch_dtype"]
        configured_attention = profile["evaluation"]["attention"]
        if profile_kind == "formal" and torch_dtype not in {None, configured_dtype}:
            raise ValueError("Formal evaluator dtype must match the formal profile")
        if profile_kind == "formal" and attention not in {None, configured_attention}:
            raise ValueError("Formal evaluator attention must match the formal profile")
        resolved_dtype = configured_dtype if torch_dtype is None else torch_dtype
        resolved_attention = configured_attention if attention is None else attention
    if resolved_dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("Evaluator torch dtype is unsupported")
    if resolved_attention not in {"flash_attention_2", "sdpa", "eager"}:
        raise ValueError("Evaluator attention implementation is unsupported")
    return {
        "profile": profile,
        "profile_sha256": profile_sha256,
        "profile_kind": profile_kind,
        "torch_dtype": resolved_dtype,
        "attention": resolved_attention,
    }


def validate_evaluator_runtime_support(
    *,
    torch_module: Any,
    torch_dtype: str,
    attention: str,
) -> None:
    """Fail before model loading when the requested numerical path is unavailable."""

    if not torch_module.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if torch_dtype == "bfloat16" and not torch_module.cuda.is_bf16_supported():
        raise ValueError("The selected CUDA device does not support bfloat16")
    if attention == "flash_attention_2" and importlib.util.find_spec("flash_attn") is None:
        raise ValueError("flash_attention_2 was requested but flash-attn is unavailable")
    if attention == "sdpa" and not hasattr(
        torch_module.nn.functional, "scaled_dot_product_attention"
    ):
        raise ValueError("sdpa was requested but is unavailable")


def build_prediction_lineage(
    *,
    model_path: str | Path,
    stage_dataset: str | Path,
    data_manifest_path: str | Path,
    profile_path: str | Path,
    run_manifest_path: str | Path,
    torch_dtype: str,
    attention: str,
) -> dict[str, Any]:
    """Bind evaluator inputs to one finalized training run contract."""

    from rapo.formal_contract import load_experiment_profile
    from rapo.provenance import (
        canonical_sha256,
        path_identity,
        validate_run_manifest,
    )

    run_path = Path(run_manifest_path)
    with run_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("Run manifest must be a JSON object")
    validate_run_manifest(manifest, require_finalized=True)
    contract = manifest["contract"]
    artifacts = manifest["artifacts"]
    profile, profile_sha = load_experiment_profile(profile_path)
    identities = {
        "model": path_identity(model_path),
        "stage_dataset": path_identity(stage_dataset),
        "data_manifest": path_identity(data_manifest_path),
        "profile": path_identity(profile_path),
    }
    if identities["model"] != artifacts["output_model"]:
        raise ValueError("Evaluator model does not match the finalized run")
    if identities["stage_dataset"] != contract["stage_dataset"]:
        raise ValueError("Evaluator stage dataset does not match the run contract")
    if identities["data_manifest"] != contract["data_manifest"]:
        raise ValueError("Evaluator data manifest does not match the run contract")
    profile_contract = contract.get("experiment_profile")
    if profile_contract is None or any(
        identities["profile"].get(key) != profile_contract.get(key)
        for key in ("path", "kind", "sha256")
    ):
        raise ValueError("Evaluator profile does not match the run contract")
    if profile_contract.get("canonical_sha256") != profile_sha:
        raise ValueError("Evaluator profile canonical SHA256 does not match")
    if profile["profile_kind"] == "formal" and (
        torch_dtype != profile["evaluation"]["torch_dtype"]
        or attention != profile["evaluation"]["attention"]
    ):
        raise ValueError("Formal evaluator numerics do not match the profile")
    return {
        "schema_version": 1,
        "run_id": contract["run_id"],
        "run_contract_sha256": manifest["contract_sha256"],
        "model_sha256": identities["model"]["sha256"],
        "stage_dataset_sha256": identities["stage_dataset"]["sha256"],
        "data_manifest_sha256": identities["data_manifest"]["sha256"],
        "profile_sha256": profile_sha,
        "torch_dtype": torch_dtype,
        "attention": attention,
        "lineage_sha256": canonical_sha256(
            {
                "run_id": contract["run_id"],
                "run_contract_sha256": manifest["contract_sha256"],
                "model_sha256": identities["model"]["sha256"],
                "stage_dataset_sha256": identities["stage_dataset"]["sha256"],
                "data_manifest_sha256": identities["data_manifest"]["sha256"],
                "profile_sha256": profile_sha,
                "torch_dtype": torch_dtype,
                "attention": attention,
            }
        ),
    }


@dataclass(frozen=True)
class ContinualMetrics:
    """Metrics derived from the lower-triangular continual-learning results."""

    last_accuracy: float
    forgetting: float
    accuracy_matrix: list[list[float | None]]
    correct_matrix: list[list[int | None]]
    total_matrix: list[list[int | None]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_accuracy": self.last_accuracy,
            "last_accuracy_percent": 100 * self.last_accuracy,
            "forgetting": self.forgetting,
            "forgetting_percent": 100 * self.forgetting,
            "accuracy_matrix": self.accuracy_matrix,
            "correct_matrix": self.correct_matrix,
            "total_matrix": self.total_matrix,
        }


def aggregate_prediction_records(
    records: Iterable[dict[str, Any]],
    *,
    data_manifest: Mapping[str, Any] | None = None,
    expected_lineage: Mapping[str, Any] | None = None,
) -> list[dict[str, int]]:
    """Aggregate predictions, optionally enforcing the manifest's exact key set."""

    counts: dict[tuple[int, int], list[int]] = {}
    expected_targets: dict[tuple[int, str], str] | None = None
    observed_keys: set[tuple[int, int, str]] = set()
    observed_after_tasks: set[int] = set()
    if data_manifest is not None:
        classes = data_manifest.get("classes")
        tasks = data_manifest.get("tasks")
        if not isinstance(classes, list) or not isinstance(tasks, list) or not tasks:
            raise ValueError("Data manifest must contain classes and tasks")
        expected_targets = {}
        task_count = len(tasks)
        for class_record in classes:
            try:
                eval_task = int(class_record["task_index"])
                target = f"<answer>{class_record['label']}</answer>"
                paths = class_record["test_images"]
            except KeyError as exc:
                raise ValueError(
                    f"Data manifest class is missing {exc.args[0]!r}"
                ) from exc
            if not isinstance(paths, list):
                raise ValueError("Data manifest test_images must be a list")
            for path in paths:
                key = (eval_task, str(path))
                if key in expected_targets:
                    raise ValueError(
                        f"Data manifest contains duplicate expected key {key}"
                    )
                expected_targets[key] = target

    for row_number, record in enumerate(records, start=1):
        if expected_lineage is not None and record.get("lineage") != dict(
            expected_lineage
        ):
            raise ValueError(
                f"Prediction row {row_number} lineage does not match the result contract"
            )
        try:
            after_task = int(record["after_task"])
            eval_task = int(record["eval_task"])
            completion = str(record["completion"])
            target = str(record["target"])
        except KeyError as exc:
            raise ValueError(
                f"Prediction row {row_number} is missing {exc.args[0]!r}"
            ) from exc
        if after_task < eval_task:
            raise ValueError(
                f"Prediction row {row_number} evaluates future task {eval_task} "
                f"after task {after_task}"
            )
        if expected_targets is not None and after_task > task_count:
            raise ValueError(
                f"Prediction row {row_number} has after_task={after_task} beyond the manifest"
            )
        scoring_target = target
        if expected_targets is not None:
            try:
                relative_path = str(record["relative_path"])
            except KeyError as exc:
                raise ValueError(
                    f"Prediction row {row_number} is missing 'relative_path'"
                ) from exc
            expected_key = (eval_task, relative_path)
            if expected_key not in expected_targets:
                raise ValueError(
                    "Unknown prediction key "
                    f"eval_task={eval_task}, relative_path={relative_path}"
                )
            prediction_key = (after_task, eval_task, relative_path)
            if prediction_key in observed_keys:
                raise ValueError(
                    "Duplicate prediction key "
                    f"after_task={after_task}, eval_task={eval_task}, "
                    f"relative_path={relative_path}"
                )
            scoring_target = expected_targets[expected_key]
            if target != scoring_target:
                raise ValueError(
                    f"Prediction row {row_number} target does not match the data manifest"
                )
            observed_keys.add(prediction_key)
            observed_after_tasks.add(after_task)
        correct_and_total = counts.setdefault((after_task, eval_task), [0, 0])
        correct_and_total[0] += int(
            classification_answer_is_correct(completion, scoring_target)
        )
        correct_and_total[1] += 1

    if expected_targets is not None:
        expected_prediction_keys = {
            (after_task, eval_task, relative_path)
            for after_task in observed_after_tasks
            for (eval_task, relative_path) in expected_targets
            if eval_task <= after_task
        }
        missing = sorted(expected_prediction_keys - observed_keys)
        if missing:
            formatted = ", ".join(
                f"({after},{evaluated},{path})"
                for after, evaluated, path in missing[:10]
            )
            suffix = " ..." if len(missing) > 10 else ""
            raise ValueError(f"Missing prediction keys: {formatted}{suffix}")

    return [
        {
            "after_task": after_task,
            "eval_task": eval_task,
            "correct": correct,
            "total": total,
        }
        for (after_task, eval_task), (correct, total) in sorted(counts.items())
    ]


def compute_continual_metrics(
    records: Iterable[dict[str, Any]],
    *,
    num_tasks: int | None = None,
) -> ContinualMetrics:
    """Compute Last Accuracy and Forgetting from per-task counts.

    Last Accuracy is a micro-average over all samples observed by the final
    task. Forgetting is the macro-average historical-best drop over tasks
    ``1..T-1``.
    """

    cells: dict[tuple[int, int], tuple[int, int]] = {}
    largest_task = 0
    for row_number, record in enumerate(records, start=1):
        try:
            after_task = int(record["after_task"])
            eval_task = int(record["eval_task"])
            correct = int(record["correct"])
            total = int(record["total"])
        except KeyError as exc:
            raise ValueError(f"Metric row {row_number} is missing {exc.args[0]!r}") from exc

        if after_task < 1 or eval_task < 1:
            raise ValueError("Task indices must be positive")
        if after_task < eval_task:
            raise ValueError(
                f"Metric row {row_number} evaluates future task {eval_task} "
                f"after task {after_task}"
            )
        if total <= 0 or correct < 0 or correct > total:
            raise ValueError(
                f"Metric row {row_number} has invalid correct/total counts"
            )
        key = (after_task, eval_task)
        if key in cells:
            raise ValueError(f"Duplicate metric cell after_task={after_task}, eval_task={eval_task}")
        cells[key] = (correct, total)
        largest_task = max(largest_task, after_task)

    if not cells:
        raise ValueError("At least one metric row is required")
    task_count = largest_task if num_tasks is None else num_tasks
    if task_count < 1:
        raise ValueError("num_tasks must be positive")
    if largest_task > task_count:
        raise ValueError("Metric rows contain a task index larger than num_tasks")

    expected_cells = {
        (after_task, eval_task)
        for after_task in range(1, task_count + 1)
        for eval_task in range(1, after_task + 1)
    }
    missing = sorted(expected_cells - cells.keys())
    if missing:
        formatted = ", ".join(f"({after},{evaluated})" for after, evaluated in missing)
        raise ValueError(f"Missing lower-triangular metric cells: {formatted}")

    accuracy_matrix: list[list[float | None]] = []
    correct_matrix: list[list[int | None]] = []
    total_matrix: list[list[int | None]] = []
    for after_task in range(1, task_count + 1):
        accuracy_row: list[float | None] = []
        correct_row: list[int | None] = []
        total_row: list[int | None] = []
        for eval_task in range(1, task_count + 1):
            cell = cells.get((after_task, eval_task))
            if cell is None:
                accuracy_row.append(None)
                correct_row.append(None)
                total_row.append(None)
            else:
                correct, total = cell
                accuracy_row.append(correct / total)
                correct_row.append(correct)
                total_row.append(total)
        accuracy_matrix.append(accuracy_row)
        correct_matrix.append(correct_row)
        total_matrix.append(total_row)

    final_cells = [cells[(task_count, eval_task)] for eval_task in range(1, task_count + 1)]
    final_correct = sum(correct for correct, _ in final_cells)
    final_total = sum(total for _, total in final_cells)
    last_accuracy = final_correct / final_total

    forgetting_values = []
    for eval_task in range(1, task_count):
        historical_best = max(
            cells[(after_task, eval_task)][0] / cells[(after_task, eval_task)][1]
            for after_task in range(eval_task, task_count + 1)
        )
        final_accuracy = cells[(task_count, eval_task)][0] / cells[(task_count, eval_task)][1]
        forgetting_values.append(historical_best - final_accuracy)
    forgetting = (
        sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0.0
    )

    return ContinualMetrics(
        last_accuracy=last_accuracy,
        forgetting=forgetting,
        accuracy_matrix=accuracy_matrix,
        correct_matrix=correct_matrix,
        total_matrix=total_matrix,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} must be an object")
            records.append(value)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute RaPO continual-learning metrics from JSONL results."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--num-tasks", type=int)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--stage-dataset", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    args = parser.parse_args(argv)

    records = _read_jsonl(args.input)
    if records and "completion" in records[0]:
        if args.data_manifest is None:
            raise ValueError("Prediction aggregation requires --data-manifest")
        with args.data_manifest.open(encoding="utf-8") as handle:
            data_manifest = json.load(handle)
        if not isinstance(data_manifest, dict):
            raise ValueError("Data manifest must be a JSON object")
        expected_lineage = None
        reported_lineage = records[0].get("lineage")
        if isinstance(reported_lineage, dict) and reported_lineage.get(
            "run_contract_sha256"
        ) is not None:
            missing = [
                name
                for name, value in (
                    ("--model", args.model),
                    ("--stage-dataset", args.stage_dataset),
                    ("--profile", args.profile),
                    ("--run-manifest", args.run_manifest),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "Formal prediction aggregation requires " + " and ".join(missing)
                )
            expected_lineage = build_prediction_lineage(
                model_path=args.model,
                stage_dataset=args.stage_dataset,
                data_manifest_path=args.data_manifest,
                profile_path=args.profile,
                run_manifest_path=args.run_manifest,
                torch_dtype=str(reported_lineage.get("torch_dtype")),
                attention=str(reported_lineage.get("attention")),
            )
        records = aggregate_prediction_records(
            records,
            data_manifest=data_manifest,
            expected_lineage=expected_lineage,
        )
    metrics = compute_continual_metrics(records, num_tasks=args.num_tasks)
    payload = json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
