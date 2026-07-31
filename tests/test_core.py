import math

import pytest
import torch

from rapo.core import (
    CrossTaskAdvantageNormalizer,
    combine_rewards,
    retention_reward,
    sampling_point_surrogate,
    trajectory_drift,
)


def test_trajectory_drift_masks_normalizes_clamps_and_detaches():
    actor = torch.tensor(
        [[1.0, 2.0, 99.0], [-1.0, -3.0, 99.0]],
        requires_grad=True,
    )
    anchor = torch.zeros_like(actor)
    mask = torch.tensor([[1, 1, 0], [1, 1, 0]])

    drift = trajectory_drift(actor, anchor, mask)

    torch.testing.assert_close(drift, torch.tensor([1.5, 0.0]))
    assert not drift.requires_grad


def test_trajectory_drift_rejects_empty_trajectory():
    logps = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="at least one"):
        trajectory_drift(logps, logps, torch.zeros_like(logps))


def test_retention_reward_is_bounded_and_combines_with_task_reward():
    drift = torch.tensor([0.0, 0.5])
    retention = retention_reward(drift, alpha=2.0)
    total = combine_rewards(torch.tensor([1.0, 0.0]), retention, weight=0.5)

    torch.testing.assert_close(retention, torch.tensor([1.0, math.exp(-1.0)]))
    torch.testing.assert_close(total, torch.tensor([1.5, 0.5 * math.exp(-1.0)]))


def test_ctan_persists_ema_across_task_boundary():
    first = CrossTaskAdvantageNormalizer(beta=0.5, epsilon=1e-4)
    first.update(2.0)
    saved = first.state_dict()

    resumed = CrossTaskAdvantageNormalizer(beta=0.5, epsilon=1e-4)
    resumed.load_state_dict(saved)
    resumed.update(4.0)

    assert resumed.ema_std == pytest.approx(3.0)
    assert resumed.updates == 2


def test_ctan_advantages_preserve_group_centering():
    normalizer = CrossTaskAdvantageNormalizer(beta=0.5, epsilon=1e-4)
    rewards = torch.tensor([0.0, 2.0, 4.0, 6.0])

    advantages = normalizer.advantages(rewards, num_generations=2)

    torch.testing.assert_close(
        advantages.view(-1, 2).mean(dim=1),
        torch.zeros(2),
        atol=1e-6,
        rtol=0,
    )
    assert normalizer.ema_std == pytest.approx(rewards.std(correction=1).item())
    assert normalizer.updates == 1


def test_ctan_can_use_global_std_without_updating_during_evaluation():
    normalizer = CrossTaskAdvantageNormalizer(beta=0.5, epsilon=1e-4)
    rewards = torch.tensor([0.0, 2.0, 4.0, 6.0])

    normalizer.advantages(
        rewards,
        num_generations=2,
        batch_std=torch.tensor(10.0),
        update=False,
    )

    assert normalizer.ema_std is None
    assert normalizer.updates == 0


def test_ctan_rejects_incompatible_saved_configuration():
    source = CrossTaskAdvantageNormalizer(beta=0.5)
    source.update(1.0)
    target = CrossTaskAdvantageNormalizer(beta=0.9)

    with pytest.raises(ValueError, match="does not match"):
        target.load_state_dict(source.state_dict())


def test_sampling_point_surrogate_has_expected_local_gradient():
    logps = torch.tensor([[0.2, -0.4], [1.1, 0.3]], requires_grad=True)
    advantages = torch.tensor([2.0, -3.0])

    sampling_point_surrogate(logps, advantages).sum().backward()

    torch.testing.assert_close(
        logps.grad,
        advantages.unsqueeze(1).expand_as(logps),
    )


def test_ratio_outside_clip_range_proves_reuse_is_not_equivalent():
    old_logp = torch.tensor([0.0])
    reused_logp = torch.tensor([math.log(2.0)], requires_grad=True)
    ratio = torch.exp(reused_logp - old_logp)
    unclipped = ratio.sum()
    unclipped.backward(retain_graph=True)
    unclipped_gradient = reused_logp.grad.detach().clone()
    reused_logp.grad.zero_()

    clipped = torch.minimum(ratio, ratio.new_tensor(1.2)).sum()
    clipped.backward()

    assert unclipped_gradient.item() == pytest.approx(2.0)
    assert reused_logp.grad.item() == pytest.approx(0.0)
