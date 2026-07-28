#!/usr/bin/env python3
"""Load the pinned Qwen2-VL model on one GPU and report its memory footprint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import Qwen2VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="sdpa",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory not found: {args.model_path}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; set CUDA_VISIBLE_DEVICES explicitly.")

    device = torch.device("cuda:0")
    dtype = getattr(torch, args.dtype)
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)

    started_at = time.monotonic()
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    torch.cuda.synchronize(device)

    report = {
        "attention": args.attn_implementation,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "device": properties.name,
        "dtype": args.dtype,
        "free_memory_before_mib": round(free_before / 2**20),
        "load_seconds": round(time.monotonic() - started_at, 2),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20),
        "total_memory_mib": round(total_memory / 2**20),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
