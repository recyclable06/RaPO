"""Evaluation utilities for class-incremental image classification."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


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
) -> list[dict[str, int]]:
    """Aggregate per-example predictions into per-task correct/total counts."""

    counts: dict[tuple[int, int], list[int]] = {}
    for row_number, record in enumerate(records, start=1):
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
        correct_and_total = counts.setdefault((after_task, eval_task), [0, 0])
        correct_and_total[0] += int(
            classification_answer_is_correct(completion, target)
        )
        correct_and_total[1] += 1

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
    args = parser.parse_args(argv)

    records = _read_jsonl(args.input)
    if records and "completion" in records[0]:
        records = aggregate_prediction_records(records)
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
