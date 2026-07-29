import pytest
import torch

from rapo.runtime import configure_cudnn_from_environment


def test_cudnn_is_unchanged_by_default(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    assert configure_cudnn_from_environment({}) is False
    assert torch.backends.cudnn.enabled is True


def test_cudnn_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)

    assert configure_cudnn_from_environment({"RAPO_DISABLE_CUDNN": "1"}) is True
    assert torch.backends.cudnn.enabled is False


def test_invalid_cudnn_setting_is_rejected():
    with pytest.raises(ValueError, match="must be either 0 or 1"):
        configure_cudnn_from_environment({"RAPO_DISABLE_CUDNN": "true"})
