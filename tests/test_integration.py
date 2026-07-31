import json

import pytest
import torch

from rapo.integration import (
    RapoController,
    RapoTrainerConfig,
    validate_tokenized_multimodal_prompt,
)


CONTRACT_SHA256 = "a" * 64


def config(task_index: int, state_path: str | None = None) -> RapoTrainerConfig:
    return RapoTrainerConfig(
        rapo_enabled=True,
        rapo_task_index=task_index,
        rapo_ctan_beta=0.5,
        rapo_state_path=state_path,
        rapo_run_id=f"run-task-{task_index}",
        rapo_contract_sha256=CONTRACT_SHA256,
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


def test_task_two_adds_detached_retention_reward(tmp_path):
    first = RapoController(config(task_index=1))
    state_path = first.save_state_file(tmp_path / "rapo_state.json")
    controller = RapoController(config(task_index=2, state_path=str(state_path)))
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
    controller.finish_optimizer_step(successful=True)

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
    first_task.advantages(
        torch.tensor([0.0, 2.0]),
        num_generations=2,
        global_total_rewards=torch.tensor([0.0, 2.0]),
    )
    first_task.finish_optimizer_step(successful=True)
    state_path = first_task.save_state_file(tmp_path / "rapo_state.json")

    second_task = RapoController(config(task_index=2, state_path=str(state_path)))

    assert second_task.ctan_ema_std == pytest.approx(
        torch.tensor([0.0, 2.0]).std(correction=1).item()
    )
    assert second_task.normalizer.updates == 1
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["task_index"] == 1
    assert payload["run_id"] == "run-task-1"
    assert payload["contract_sha256"] == CONTRACT_SHA256


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
                rapo_run_id="run-task-2",
                rapo_contract_sha256=CONTRACT_SHA256,
            )
        )


def test_state_paths_require_rapo_to_be_enabled():
    with pytest.raises(ValueError, match="require rapo_enabled"):
        RapoTrainerConfig(rapo_state_path="rapo_state.json")


def test_task_two_requires_previous_state():
    with pytest.raises(ValueError, match="requires state from the previous task"):
        config(task_index=2)


def test_state_rejects_non_previous_task(tmp_path):
    first = RapoController(config(task_index=1))
    state_path = first.save_state_file(tmp_path / "rapo_state.json")

    with pytest.raises(ValueError, match="exactly the previous task"):
        RapoController(
            RapoTrainerConfig(
                rapo_enabled=True,
                rapo_task_index=3,
                rapo_ctan_beta=0.5,
                rapo_state_path=str(state_path),
                rapo_run_id="run-task-3",
                rapo_contract_sha256=CONTRACT_SHA256,
            )
        )


def test_prompt_over_limit_fails_before_generation_boundary():
    input_ids = torch.arange(6).view(1, 6)
    attention_mask = torch.ones_like(input_ids)

    with pytest.raises(ValueError, match="exceeds max_prompt_length"):
        validate_tokenized_multimodal_prompt(input_ids, attention_mask, 5)


def test_prompt_exactly_at_limit_is_accepted():
    input_ids = torch.arange(6).view(1, 6)
    attention_mask = torch.ones_like(input_ids)

    lengths = validate_tokenized_multimodal_prompt(input_ids, attention_mask, 6)

    torch.testing.assert_close(lengths, torch.tensor([6]))


def test_prompt_visual_tokens_must_be_inside_attention_mask():
    input_ids = torch.arange(4).view(1, 4)
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    visual_mask = torch.tensor([[0, 1, 0, 1]])

    with pytest.raises(ValueError, match="visual tokens must be active"):
        validate_tokenized_multimodal_prompt(
            input_ids,
            attention_mask,
            4,
            visual_token_mask=visual_mask,
        )


@pytest.mark.parametrize("gradient_accumulation_steps", [2, 4])
def test_ctan_commits_complete_gradient_accumulation_window_once(
    gradient_accumulation_steps,
):
    controller = RapoController(config(task_index=1))
    window_rewards = []
    for microbatch in range(gradient_accumulation_steps):
        rewards = torch.tensor([2.0 * microbatch, 2.0 * microbatch + 1.0])
        window_rewards.append(rewards)
        controller.advantages(
            rewards,
            num_generations=2,
            global_total_rewards=rewards,
        )
        assert controller.normalizer.updates == 0

    assert controller.finish_optimizer_step(successful=True)
    assert controller.normalizer.updates == 1
    assert controller.ctan_ema_std == pytest.approx(
        torch.cat(window_rewards).std(correction=1).item()
    )


def test_ctan_skipped_step_discards_pending_window():
    controller = RapoController(config(task_index=1))
    rewards = torch.tensor([0.0, 2.0])
    controller.advantages(rewards, 2, global_total_rewards=rewards)

    assert not controller.finish_optimizer_step(successful=False)
    assert controller.normalizer.updates == 0
    assert controller.ctan_ema_std is None
    assert controller.step_transaction.pending_microbatches == 0


def test_ctan_evaluation_never_stages_or_updates():
    controller = RapoController(config(task_index=1))
    rewards = torch.tensor([0.0, 2.0])

    controller.advantages(
        rewards,
        2,
        global_total_rewards=rewards,
        training=False,
    )

    assert controller.step_transaction.pending_microbatches == 0
    assert controller.normalizer.updates == 0


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_ctan_logical_rank_shards_commit_identical_global_scale(world_size):
    global_rewards = torch.arange(world_size * 2, dtype=torch.float32)
    controllers = []
    for rank in range(world_size):
        controller = RapoController(config(task_index=1))
        local_rewards = global_rewards[rank * 2 : (rank + 1) * 2]
        advantages = controller.advantages(
            local_rewards,
            2,
            global_total_rewards=global_rewards,
        )
        torch.testing.assert_close(advantages.mean(), torch.tensor(0.0))
        controller.finish_optimizer_step(successful=True)
        controllers.append(controller)

    expected = global_rewards.std(correction=1).item()
    assert {controller.normalizer.updates for controller in controllers} == {1}
    assert all(
        controller.ctan_ema_std == pytest.approx(expected)
        for controller in controllers
    )


def test_ctan_save_load_continues_successful_step_clock(tmp_path):
    first = RapoController(config(task_index=1))
    first_rewards = torch.tensor([0.0, 2.0, 4.0, 6.0])
    first.advantages(first_rewards, 2, global_total_rewards=first_rewards)
    first.finish_optimizer_step(successful=True)
    state_path = first.save_state_file(tmp_path / "rapo_state.json")
    second = RapoController(config(task_index=2, state_path=str(state_path)))
    second_rewards = torch.tensor([0.0, 4.0, 8.0, 12.0])

    second.advantages(second_rewards, 2, global_total_rewards=second_rewards)
    second.finish_optimizer_step(successful=True)

    expected = 0.5 * first_rewards.std(correction=1) + 0.5 * second_rewards.std(
        correction=1
    )
    assert second.normalizer.updates == 2
    assert second.ctan_ema_std == pytest.approx(expected.item())


def test_ctan_refuses_state_save_inside_unfinished_step(tmp_path):
    controller = RapoController(config(task_index=1))
    rewards = torch.tensor([0.0, 2.0])
    controller.advantages(rewards, 2, global_total_rewards=rewards)

    with pytest.raises(ValueError, match="unfinished optimizer step"):
        controller.save_state_file(tmp_path / "rapo_state.json")
