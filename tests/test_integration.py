import json

import pytest
import torch

from rapo.integration import RapoController, RapoTrainerConfig


def config(task_index: int, state_path: str | None = None) -> RapoTrainerConfig:
    return RapoTrainerConfig(
        rapo_enabled=True,
        rapo_task_index=task_index,
        rapo_ctan_beta=0.5,
        rapo_state_path=state_path,
    )


def test_task_one_uses_task_reward_without_retention():
    controller = RapoController(config(task_index=1))
    logps = torch.zeros((2, 2))
    task_rewards = torch.tensor([0.0, 1.0])

    result = controller.build_rewards(
        logps,
        logps,
        torch.ones_like(logps),
        task_rewards,
    )

    assert result.total_rewards is task_rewards
    assert result.drift is None
    assert result.retention_rewards is None


def test_task_two_adds_detached_retention_reward():
    controller = RapoController(config(task_index=2))
    actor = torch.tensor([[0.1, 0.1], [-0.2, -0.2]], requires_grad=True)
    anchor = torch.zeros_like(actor)
    task_rewards = torch.tensor([1.0, 0.0])

    result = controller.build_rewards(
        actor,
        anchor,
        torch.ones_like(actor),
        task_rewards,
    )

    expected_retention = torch.exp(torch.tensor([-2.0, 0.0]))
    torch.testing.assert_close(result.retention_rewards, expected_retention)
    torch.testing.assert_close(
        result.total_rewards,
        task_rewards + 0.5 * expected_retention,
    )
    assert not result.drift.requires_grad
    assert not result.retention_rewards.requires_grad


def test_ctan_uses_global_reward_std_but_local_group_means():
    controller = RapoController(config(task_index=1))
    local_rewards = torch.tensor([0.0, 2.0, 4.0, 6.0])
    global_rewards = torch.tensor([0.0, 2.0, 4.0, 6.0, 100.0, 102.0, 104.0, 106.0])

    advantages = controller.advantages(
        local_rewards,
        num_generations=2,
        global_total_rewards=global_rewards,
    )

    torch.testing.assert_close(
        advantages.view(-1, 2).mean(dim=1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0,
    )
    assert controller.ctan_ema_std == pytest.approx(
        global_rewards.std(correction=1).item()
    )


def test_state_round_trip_across_task_boundary(tmp_path):
    first_task = RapoController(config(task_index=1))
    first_task.normalizer.update(2.0)
    state_path = first_task.save_state_file(tmp_path / "rapo_state.json")

    second_task = RapoController(config(task_index=2, state_path=str(state_path)))

    assert second_task.ctan_ema_std == pytest.approx(2.0)
    assert second_task.normalizer.updates == 1
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["task_index"] == 1


def test_state_rejects_changed_paper_settings(tmp_path):
    source = RapoController(config(task_index=1))
    state_path = source.save_state_file(tmp_path / "rapo_state.json")

    with pytest.raises(ValueError, match="settings do not match"):
        RapoController(
            RapoTrainerConfig(
                rapo_enabled=True,
                rapo_task_index=2,
                rapo_retention_alpha=10.0,
                rapo_ctan_beta=0.5,
                rapo_state_path=str(state_path),
            )
        )


def test_state_paths_require_rapo_to_be_enabled():
    with pytest.raises(ValueError, match="require rapo_enabled"):
        RapoTrainerConfig(rapo_state_path="rapo_state.json")
