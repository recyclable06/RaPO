import copy
import json
import random
from pathlib import Path

import pytest
import torch

from rapo.core import CrossTaskAdvantageNormalizer
from rapo.formal_contract import (
    build_dry_run_contract,
    load_experiment_profile,
    reject_legacy_environment,
)
from rapo.resume import (
    CheckpointIdentity,
    load_training_checkpoint,
    save_training_checkpoint,
    validate_checkpoint_binding,
    write_checkpoint_binding,
)


FORMAL_PROFILE = Path("configs/formal_profile.json")
LEGACY_PROFILE = Path("configs/legacy_2080ti_profile.json")


def test_formal_and_legacy_profiles_are_canonical_and_isolated():
    formal, formal_sha = load_experiment_profile(FORMAL_PROFILE)
    legacy, legacy_sha = load_experiment_profile(LEGACY_PROFILE)

    assert formal["profile_kind"] == "formal"
    assert formal["training"]["num_train_epochs"] == 2
    assert "max_steps" not in formal["training"]
    assert formal["training"]["attention"] == "flash_attention_2"
    assert formal["evaluation"]["attention"] == "flash_attention_2"
    assert formal["hardware_gate"] == "pending_hardware_gate"
    assert legacy["profile_kind"] == "legacy_2080ti"
    assert legacy["training"]["budget_kind"] == "max_steps"
    assert formal["output_namespace"] != legacy["output_namespace"]
    assert formal_sha != legacy_sha


def test_formal_dry_run_has_machine_checkable_two_epoch_budget():
    profile, profile_sha = load_experiment_profile(FORMAL_PROFILE)

    contract = build_dry_run_contract(
        profile,
        profile_sha256=profile_sha,
        train_samples=101,
        world_size=8,
    )

    assert contract["training_budget"] == {
        "budget_kind": "epochs",
        "num_train_epochs": 2,
        "train_samples": 101,
        "sampler_examples_per_epoch": 104,
        "expected_sample_presentations": 208,
        "expected_generations": 1664,
        "expected_optimizer_steps": 14,
    }
    assert contract["resolved"]["precision"] == "bf16"
    assert contract["resolved"]["world_size"] == 8
    assert contract["hardware_gate"] == "pending_hardware_gate"


@pytest.mark.parametrize(
    ("section", "attention"),
    [
        ("training", "sdpa"),
        ("training", "eager"),
        ("evaluation", "sdpa"),
        ("evaluation", "eager"),
    ],
)
def test_formal_attention_downgrade_fails_before_dry_run(
    tmp_path, section, attention
):
    profile, profile_sha = load_experiment_profile(FORMAL_PROFILE)
    profile[section]["attention"] = attention
    external_profile = tmp_path / "formal_profile.json"
    external_profile.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="Formal profile requires FlashAttention-2"):
        load_experiment_profile(external_profile)
    with pytest.raises(ValueError, match="Formal profile requires FlashAttention-2"):
        build_dry_run_contract(
            profile,
            profile_sha256=profile_sha,
            train_samples=101,
            world_size=8,
        )


def test_formal_environment_rejects_every_legacy_smoke_variable():
    with pytest.raises(ValueError, match="RAPO_SMOKE_PRECISION"):
        reject_legacy_environment({"RAPO_SMOKE_PRECISION": "fp16"})


def _make_training_stack(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    ctan = CrossTaskAdvantageNormalizer(beta=0.5)
    return model, optimizer, scheduler, ctan


def _run_steps(stack, start, stop):
    model, optimizer, scheduler, ctan = stack
    trace = []
    for step in range(start, stop):
        token = random.randrange(1, 8)
        target = torch.rand((1, 1))
        prediction = model(torch.tensor([[float(token)]]))
        loss = (prediction - target).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        ctan.update(torch.tensor(float(token + step) / 10.0))
        trace.append((token, float(target.item()), float(loss.detach().item())))
    return trace


def test_interrupted_cpu_training_restores_all_state_and_future_sequence(tmp_path):
    identity = CheckpointIdentity("formal-run", "a" * 64, "b" * 64)
    uninterrupted = _make_training_stack(17)
    uninterrupted_trace = _run_steps(uninterrupted, 0, 6)

    interrupted = _make_training_stack(17)
    assert _run_steps(interrupted, 0, 3) == uninterrupted_trace[:3]
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=interrupted[0],
        optimizer=interrupted[1],
        scheduler=interrupted[2],
        ctan=interrupted[3],
        global_step=3,
        identity=identity,
    )

    resumed = _make_training_stack(999)
    restored_step = load_training_checkpoint(
        checkpoint,
        model=resumed[0],
        optimizer=resumed[1],
        scheduler=resumed[2],
        ctan=resumed[3],
        expected_identity=identity,
    )
    resumed_trace = _run_steps(resumed, restored_step, 6)

    assert resumed_trace == uninterrupted_trace[3:]
    for resumed_value, expected_value in zip(
        resumed[0].state_dict().values(), uninterrupted[0].state_dict().values(), strict=True
    ):
        torch.testing.assert_close(resumed_value, expected_value, rtol=0, atol=0)
    assert resumed[1].state_dict() == uninterrupted[1].state_dict()
    assert resumed[2].state_dict() == uninterrupted[2].state_dict()
    assert resumed[3].state_dict() == uninterrupted[3].state_dict()


@pytest.mark.parametrize(
    "corruption",
    ["run_id", "profile", "contract", "missing_rng", "state"],
)
def test_resume_rejects_wrong_or_incomplete_checkpoint_before_loading(tmp_path, corruption):
    identity = CheckpointIdentity("formal-run", "a" * 64, "b" * 64)
    source = _make_training_stack(3)
    _run_steps(source, 0, 1)
    checkpoint = save_training_checkpoint(
        tmp_path / "checkpoint.pt",
        model=source[0],
        optimizer=source[1],
        scheduler=source[2],
        ctan=source[3],
        global_step=1,
        identity=identity,
    )
    payload = torch.load(checkpoint, weights_only=False)
    if corruption == "run_id":
        payload["identity"]["run_id"] = "foreign-run"
    elif corruption == "profile":
        payload["identity"]["profile_sha256"] = "c" * 64
    elif corruption == "contract":
        payload["identity"]["run_contract_sha256"] = "d" * 64
    elif corruption == "missing_rng":
        del payload["torch_rng_state"]
    else:
        payload["ctan_state"]["updates"] = -1
    torch.save(payload, checkpoint)

    target = _make_training_stack(11)
    model_before = copy.deepcopy(target[0].state_dict())
    with pytest.raises(ValueError):
        load_training_checkpoint(
            checkpoint,
            model=target[0],
            optimizer=target[1],
            scheduler=target[2],
            ctan=target[3],
            expected_identity=identity,
        )
    for key, value in target[0].state_dict().items():
        torch.testing.assert_close(value, model_before[key], rtol=0, atol=0)


def test_formal_runner_cannot_emit_max_steps():
    formal, _ = load_experiment_profile(FORMAL_PROFILE)
    script = Path("scripts/run_imagenet_r_formal.sh").read_text(encoding="utf-8")

    assert "RAPO_FORMAL_NUM_TRAIN_EPOCHS=2" in script
    assert f"RAPO_FORMAL_SAVE_STEPS={formal['training']['save_steps']}" in script
    assert "--max_steps" not in script
    assert "RAPO_SMOKE_" in script
    assert "pending_hardware_gate" in script


def _make_production_checkpoint(tmp_path, *, global_step=3, require_ctan=True):
    checkpoint = tmp_path / f"checkpoint-{global_step}"
    checkpoint.mkdir()
    for name in (
        "scheduler.pt",
        "optimizer.pt",
        "rng_state.pth",
        "model.safetensors",
    ):
        (checkpoint / name).write_text(name, encoding="utf-8")
    if require_ctan:
        (checkpoint / "rapo_state.json").write_text("rapo_state.json", encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": global_step}), encoding="utf-8"
    )
    return checkpoint


def test_production_checkpoint_binding_covers_every_required_state(tmp_path):
    checkpoint = _make_production_checkpoint(tmp_path)
    identity = CheckpointIdentity("formal-run", "a" * 64, "b" * 64)

    write_checkpoint_binding(
        checkpoint,
        identity=identity,
        global_step=3,
        require_ctan=True,
    )
    assert validate_checkpoint_binding(
        checkpoint,
        expected_identity=identity,
        require_ctan=True,
    )["global_step"] == 3

    (checkpoint / "optimizer.pt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after binding"):
        validate_checkpoint_binding(
            checkpoint,
            expected_identity=identity,
            require_ctan=True,
        )


def test_checkpoint_binding_rejects_requested_step_that_differs_from_trainer(tmp_path):
    checkpoint = _make_production_checkpoint(tmp_path, global_step=5)
    identity = CheckpointIdentity("formal-run", "a" * 64, "b" * 64)

    with pytest.raises(ValueError, match="trainer_state.json global_step"):
        write_checkpoint_binding(
            checkpoint,
            identity=identity,
            global_step=6,
            require_ctan=True,
        )

    assert not (checkpoint / "rapo_checkpoint_binding.json").exists()


def test_checkpoint_binding_revalidates_step_against_trainer_state(tmp_path):
    checkpoint = _make_production_checkpoint(tmp_path, global_step=5)
    identity = CheckpointIdentity("formal-run", "a" * 64, "b" * 64)
    binding = write_checkpoint_binding(
        checkpoint,
        identity=identity,
        global_step=5,
        require_ctan=True,
    )

    assert validate_checkpoint_binding(
        checkpoint,
        expected_identity=identity,
        require_ctan=True,
    )["global_step"] == 5

    payload = json.loads(binding.read_text(encoding="utf-8"))
    payload["global_step"] = 6
    binding.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trainer_state.json global_step"):
        validate_checkpoint_binding(
            checkpoint,
            expected_identity=identity,
            require_ctan=True,
        )


@pytest.mark.parametrize(
    "trainer_state",
    [
        None,
        [],
        {},
        {"global_step": True},
        {"global_step": 5.0},
        {"global_step": -1},
    ],
)
def test_checkpoint_binding_rejects_invalid_trainer_global_step(
    tmp_path, trainer_state
):
    checkpoint = _make_production_checkpoint(tmp_path, global_step=5)
    trainer_state_path = checkpoint / "trainer_state.json"
    if trainer_state is None:
        trainer_state_path.unlink()
    else:
        trainer_state_path.write_text(json.dumps(trainer_state), encoding="utf-8")
    identity = CheckpointIdentity("formal-run", "a" * 64, "b" * 64)

    with pytest.raises(ValueError, match="trainer_state.json"):
        write_checkpoint_binding(
            checkpoint,
            identity=identity,
            global_step=5,
            require_ctan=True,
        )
