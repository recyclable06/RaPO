#!/usr/bin/env python3
"""Generate bounded ImageNet-R predictions from a trained Qwen2-VL model."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from datasets import DatasetDict
from rapo.evaluation import (
    build_prediction_lineage,
    pad_image_to_minimum_size,
    resolve_evaluator_settings,
    validate_evaluator_runtime_support,
)
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--after-task", type=int, required=True)
    parser.add_argument("--samples-per-class", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--min-pixels", type=int, default=3136)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument(
        "--torch-dtype", choices=("float16", "bfloat16", "float32")
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    return parser.parse_args()


def select_rows(dataset: Any, samples_per_class: int | None) -> list[dict[str, Any]]:
    if samples_per_class is not None and samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")

    metadata = dataset.select_columns(["task_index", "wnid", "relative_path"])
    indices_by_class: dict[tuple[int, str], list[tuple[str, int]]] = defaultdict(list)
    for index, row in enumerate(metadata):
        key = (int(row["task_index"]), str(row["wnid"]))
        indices_by_class[key].append((str(row["relative_path"]), index))

    selected_indices = []
    for key in sorted(indices_by_class):
        ordered = sorted(indices_by_class[key])
        selected_indices.extend(
            index
            for _, index in (
                ordered if samples_per_class is None else ordered[:samples_per_class]
            )
        )
    return [dataset[index] for index in selected_indices]


def main() -> None:
    args = parse_args()
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_path}")
    if not args.dataset_path.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.dataset_path}")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    if args.after_task < 1:
        raise SystemExit("after-task must be positive")
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise SystemExit("batch-size and max-new-tokens must be positive")
    try:
        evaluator = resolve_evaluator_settings(
            profile_path=args.profile,
            torch_dtype=args.torch_dtype,
            attention=args.attn_implementation,
        )
        validate_evaluator_runtime_support(
            torch_module=torch,
            torch_dtype=evaluator["torch_dtype"],
            attention=evaluator["attention"],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.profile is not None:
        missing = [
            name
            for name, value in (
                ("--run-manifest", args.run_manifest),
                ("--data-manifest", args.data_manifest),
            )
            if value is None
        ]
        if missing:
            raise SystemExit(
                "Formal evaluation requires " + " and ".join(missing)
            )
        if evaluator["profile_kind"] == "formal" and args.samples_per_class is not None:
            raise SystemExit("Formal evaluation requires the full manifest test set")

    if args.profile is not None:
        try:
            lineage = build_prediction_lineage(
                model_path=args.model_path,
                stage_dataset=args.dataset_path,
                data_manifest_path=args.data_manifest,
                profile_path=args.profile,
                run_manifest_path=args.run_manifest,
                torch_dtype=evaluator["torch_dtype"],
                attention=evaluator["attention"],
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        lineage = {
            "schema_version": 1,
            "run_id": None,
            "run_contract_sha256": None,
            "model_sha256": None,
            "stage_dataset_sha256": None,
            "data_manifest_sha256": None,
            "profile_sha256": evaluator["profile_sha256"],
            "torch_dtype": evaluator["torch_dtype"],
            "attention": evaluator["attention"],
            "lineage_sha256": None,
        }

    dataset = DatasetDict.load_from_disk(str(args.dataset_path))
    samples_per_class = args.samples_per_class
    if samples_per_class is None and evaluator["profile_kind"] != "formal":
        samples_per_class = 5
    rows = select_rows(dataset["test"], samples_per_class)
    if not rows:
        raise SystemExit("The test split is empty")
    if max(int(row["task_index"]) for row in rows) > args.after_task:
        raise SystemExit("The dataset contains a future task")

    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = args.min_pixels
    spatial_factor = (
        int(processor.image_processor.patch_size)
        * int(processor.image_processor.merge_size)
    )
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype_by_name[evaluator["torch_dtype"]],
        attn_implementation=evaluator["attention"],
        low_cpu_mem_usage=True,
    )
    model.eval().to("cuda:0")

    predictions = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": row["problem"]},
                    ],
                }
            ]
            for row in batch
        ]
        texts = [
            processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        images = []
        for row in batch:
            image = row["image"]
            prepared_image = pad_image_to_minimum_size(image, spatial_factor)
            if prepared_image.size != image.size:
                print(
                    f"padded {row['relative_path']} from "
                    f"{image.size[0]}x{image.size[1]} to "
                    f"{prepared_image.size[0]}x{prepared_image.size[1]}",
                    flush=True,
                )
            images.append(prepared_image)
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
            )
        prompt_length = inputs["input_ids"].shape[1]
        completions = processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for row, completion in zip(batch, completions, strict=True):
            predictions.append(
                {
                    "after_task": args.after_task,
                    "eval_task": int(row["task_index"]),
                    "completion": completion,
                    "target": row["solution"],
                    "relative_path": row["relative_path"],
                    "lineage": lineage,
                }
            )
        print(f"generated {len(predictions)}/{len(rows)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    temporary_output.replace(args.output)
    print(
        json.dumps(
            {
                "after_task": args.after_task,
                "output": str(args.output),
                "predictions": len(predictions),
                "result_contract": lineage,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
