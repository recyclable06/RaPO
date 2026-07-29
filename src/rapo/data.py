"""Deterministic ImageNet-R task construction for the RaPO reproduction."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rapo.evaluation import normalize_class_name


SCHEMA_VERSION = 1
IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
WNID_LABEL_PATTERN = re.compile(r"^(n\d{8})\s+(\S+)\s*$")
CLASSIFICATION_PROMPT = """Perform image classification on the given visual input.
You MUST choose exactly one class name from this list:
{class_names}
Output the reasoning process in <think> </think>
and final answer in <answer> </answer>.
Your answer MUST be one class name from the list,
character-for-character. No extra words."""


def _display_label(value: str) -> str:
    label = " ".join(value.replace("_", " ").split())
    if not label:
        raise ValueError("Class labels must not be empty")
    return label


def load_class_map(path: str | Path) -> dict[str, str]:
    """Load WNID labels from ImageNet JSON or the official ImageNet-R README."""

    text = Path(path).read_text(encoding="utf-8")
    class_map: dict[str, str] = {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            match = WNID_LABEL_PATTERN.fullmatch(line.strip())
            if match:
                wnid, label = match.groups()
                class_map[wnid] = _display_label(label)
        if not class_map:
            raise ValueError(
                "Class map is neither supported JSON nor a WNID-label text mapping"
            )
        return class_map

    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, str):
                class_map[str(key)] = _display_label(value)
            elif isinstance(value, list) and len(value) >= 2:
                class_map[str(value[0])] = _display_label(str(value[1]))
            elif isinstance(value, dict) and "wnid" in value and "label" in value:
                class_map[str(value["wnid"])] = _display_label(str(value["label"]))
            else:
                raise ValueError(f"Unsupported class-map value for key {key!r}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            if not isinstance(value, dict) or "wnid" not in value or "label" not in value:
                raise ValueError(f"Unsupported class-map item at index {index}")
            class_map[str(value["wnid"])] = _display_label(str(value["label"]))
    else:
        raise ValueError("Class map must be a JSON object or list")

    if not class_map:
        raise ValueError("Class map must contain at least one class")
    return class_map


def _stable_key(namespace: str, seed: int, value: str) -> bytes:
    payload = f"rapo-manifest-v{SCHEMA_VERSION}\0{namespace}\0{seed}\0{value}"
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _discover_images(image_root: Path) -> dict[str, list[str]]:
    if not image_root.is_dir():
        raise ValueError(f"Image root does not exist or is not a directory: {image_root}")

    images_by_class: dict[str, list[str]] = {}
    for class_directory in sorted(image_root.iterdir(), key=lambda path: path.name):
        if not class_directory.is_dir():
            continue
        relative_images = sorted(
            path.relative_to(image_root).as_posix()
            for path in class_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if relative_images:
            images_by_class[class_directory.name] = relative_images
    if not images_by_class:
        raise ValueError(f"No supported images found below {image_root}")
    return images_by_class


def build_imagenet_r_manifest(
    image_root: str | Path,
    class_map: Mapping[str, str],
    *,
    num_tasks: int = 10,
    classes_per_task: int = 20,
    shots_per_class: int = 5,
    class_order_seed: int = 0,
    sample_seed: int = 0,
) -> dict[str, Any]:
    """Build a portable manifest with deterministic class and sample orders."""

    if num_tasks < 1 or classes_per_task < 1 or shots_per_class < 1:
        raise ValueError("Task, class, and shot counts must be positive")
    expected_classes = num_tasks * classes_per_task
    image_root = Path(image_root).resolve()
    images_by_class = _discover_images(image_root)
    observed_classes = set(images_by_class)
    if len(observed_classes) != expected_classes:
        raise ValueError(
            f"Expected {expected_classes} image classes, found {len(observed_classes)}"
        )

    missing_labels = sorted(observed_classes - class_map.keys())
    if missing_labels:
        raise ValueError(f"Class map is missing WNIDs: {', '.join(missing_labels)}")

    labels = {wnid: _display_label(str(class_map[wnid])) for wnid in observed_classes}
    normalized_labels: dict[str, str] = {}
    for wnid, label in labels.items():
        normalized = normalize_class_name(label)
        previous = normalized_labels.get(normalized)
        if previous is not None:
            raise ValueError(
                f"Class labels for {previous} and {wnid} collide after normalization"
            )
        normalized_labels[normalized] = wnid

    class_order = sorted(
        observed_classes,
        key=lambda wnid: _stable_key("class-order", class_order_seed, wnid),
    )
    class_records = []
    for class_index, wnid in enumerate(class_order):
        ordered_images = sorted(
            images_by_class[wnid],
            key=lambda relative_path: _stable_key(
                f"sample-order:{wnid}", sample_seed, relative_path
            ),
        )
        if len(ordered_images) <= shots_per_class:
            raise ValueError(
                f"Class {wnid} has {len(ordered_images)} images; it needs more than "
                f"{shots_per_class} to create non-empty train and test splits"
            )
        class_records.append(
            {
                "wnid": wnid,
                "label": labels[wnid],
                "task_index": class_index // classes_per_task + 1,
                "train_images": ordered_images[:shots_per_class],
                "test_images": ordered_images[shots_per_class:],
            }
        )

    tasks = []
    for task_index in range(1, num_tasks + 1):
        task_classes = [
            record for record in class_records if record["task_index"] == task_index
        ]
        seen_classes = [
            record for record in class_records if record["task_index"] <= task_index
        ]
        tasks.append(
            {
                "task_index": task_index,
                "class_wnids": [record["wnid"] for record in task_classes],
                "class_names": [record["label"] for record in task_classes],
                "seen_class_names": [record["label"] for record in seen_classes],
                "train_size": sum(len(record["train_images"]) for record in task_classes),
                "test_size": sum(len(record["test_images"]) for record in seen_classes),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": "ImageNet-R",
        "image_root_name": image_root.name,
        "protocol": {
            "num_tasks": num_tasks,
            "classes_per_task": classes_per_task,
            "shots_per_class": shots_per_class,
            "class_order_seed": class_order_seed,
            "sample_seed": sample_seed,
            "rehearsal": False,
            "split_assumption": (
                "For each class, deterministically select the first shots_per_class "
                "images for training and use all remaining images for testing."
            ),
        },
        "classes": class_records,
        "tasks": tasks,
    }


def make_classification_prompt(class_names: Iterable[str]) -> str:
    names = list(class_names)
    if not names:
        raise ValueError("The classification prompt requires at least one class")
    return CLASSIFICATION_PROMPT.format(class_names=", ".join(names))


def visual_rft_rows(
    manifest: Mapping[str, Any],
    image_root: str | Path,
    *,
    task_index: int,
    eval_task: int | None = None,
) -> list[dict[str, Any]]:
    """Create Visual-RFT rows for current-task training or seen-task evaluation."""

    tasks = list(manifest["tasks"])
    if task_index < 1 or task_index > len(tasks):
        raise ValueError(f"task_index must be between 1 and {len(tasks)}")
    if eval_task is not None and (eval_task < 1 or eval_task > task_index):
        raise ValueError("eval_task must be an observed task")

    seen_class_names = tasks[task_index - 1]["seen_class_names"]
    prompt = make_classification_prompt(seen_class_names)
    selected_task = task_index if eval_task is None else eval_task
    image_field = "train_images" if eval_task is None else "test_images"
    root = Path(image_root).resolve()

    rows = []
    for class_record in manifest["classes"]:
        if int(class_record["task_index"]) != selected_task:
            continue
        for relative_path in class_record[image_field]:
            image_path = root / Path(relative_path)
            if not image_path.is_file():
                raise ValueError(f"Manifest image does not exist: {image_path}")
            rows.append(
                {
                    "image": str(image_path),
                    "problem": prompt,
                    "solution": f"<answer>{class_record['label']}</answer>",
                    "task_index": selected_task,
                    "wnid": class_record["wnid"],
                    "class_name": class_record["label"],
                    "relative_path": relative_path,
                }
            )
    return rows


def export_visual_rft_task(
    manifest: Mapping[str, Any],
    image_root: str | Path,
    output_directory: str | Path,
    *,
    task_index: int,
) -> Path:
    """Save one task stage as the DatasetDict consumed by Visual-RFT."""

    try:
        import datasets
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Visual-RFT export requires the optional data dependencies; "
            "install rapo-reproduction[data]."
        ) from exc

    feature_schema = datasets.Features(
        {
            "image": datasets.Image(),
            "problem": datasets.Value("string"),
            "solution": datasets.Value("string"),
            "task_index": datasets.Value("int32"),
            "wnid": datasets.Value("string"),
            "class_name": datasets.Value("string"),
            "relative_path": datasets.Value("string"),
        }
    )

    split_rows = {
        "train": visual_rft_rows(manifest, image_root, task_index=task_index),
    }
    seen_test_rows = []
    for eval_task in range(1, task_index + 1):
        seen_test_rows.extend(
            visual_rft_rows(
                manifest,
                image_root,
                task_index=task_index,
                eval_task=eval_task,
            )
        )
    split_rows["test"] = seen_test_rows

    dataset_dict = datasets.DatasetDict(
        {
            split_name: datasets.Dataset.from_list(rows, features=feature_schema)
            for split_name, rows in split_rows.items()
        }
    )
    output_path = Path(output_directory)
    dataset_dict.save_to_disk(str(output_path))
    return output_path


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic ImageNet-R tasks for the RaPO reproduction."
    )
    parser.add_argument("image_root", type=Path)
    parser.add_argument("class_map", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--classes-per-task", type=int, default=20)
    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--class-order-seed", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--export-visual-rft", action="store_true")
    parser.add_argument(
        "--export-visual-rft-task",
        action="append",
        type=int,
        default=[],
        help="Export only this task stage; repeat the option for multiple stages.",
    )
    args = parser.parse_args(argv)
    export_tasks = set(args.export_visual_rft_task)
    invalid_tasks = sorted(
        task_index
        for task_index in export_tasks
        if task_index < 1 or task_index > args.num_tasks
    )
    if invalid_tasks:
        parser.error(
            "Visual-RFT export task indices must be between 1 and "
            f"{args.num_tasks}: {invalid_tasks}"
        )
    if args.export_visual_rft:
        export_tasks.update(range(1, args.num_tasks + 1))

    manifest = build_imagenet_r_manifest(
        args.image_root,
        load_class_map(args.class_map),
        num_tasks=args.num_tasks,
        classes_per_task=args.classes_per_task,
        shots_per_class=args.shots_per_class,
        class_order_seed=args.class_order_seed,
        sample_seed=args.sample_seed,
    )
    manifest_path = write_manifest(manifest, args.output_directory / "manifest.json")
    print(f"Wrote deterministic split manifest to {manifest_path}")

    for task_index in sorted(export_tasks):
        output_path = export_visual_rft_task(
            manifest,
            args.image_root,
            args.output_directory / "visual_rft" / f"task_{task_index:02d}",
            task_index=task_index,
        )
        print(f"Wrote Visual-RFT task {task_index} dataset to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
