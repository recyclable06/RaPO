"""Explicit runtime compatibility controls for patched training entrypoints."""

from __future__ import annotations

import os
from collections.abc import Mapping

import torch


def configure_cudnn_from_environment(
    environment: Mapping[str, str] | None = None,
) -> bool:
    """Disable cuDNN only when ``RAPO_DISABLE_CUDNN=1`` is explicitly set."""

    source = os.environ if environment is None else environment
    value = source.get("RAPO_DISABLE_CUDNN", "0").strip()
    if value not in {"0", "1"}:
        raise ValueError("RAPO_DISABLE_CUDNN must be either 0 or 1")
    if value == "1":
        torch.backends.cudnn.enabled = False
        return True
    return False
