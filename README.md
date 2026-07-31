# RaPO Reproduction

This repository is an independent reproduction of:

> Overcoming Catastrophic Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning

The first milestone is a faithful implementation of the two RaPO components:

1. trajectory-level Retention Reward;
2. Cross-Task Advantage Normalization (CTAN).

The initial experimental target is rehearsal-free, 5-shot, 10-task class-incremental
classification on ImageNet-R with Qwen2-VL-2B. OPD extensions are intentionally out
of scope until the paper reproduction is complete.

## Current status

- [x] Paper-level RaPO reward equations implemented as an independent PyTorch module.
- [x] CTAN state can be serialized across task boundaries.
- [x] Unit tests cover masking, one-sided drift truncation, stop-gradient behavior,
      reward shaping, and CTAN persistence.
- [x] Add a reviewable integration patch for the pinned Visual-RFT GRPO trainer.
- [x] Build the deterministic ImageNet-R task split and evaluator.
- [x] Freeze a separate Visual-RFT training environment and GPU smoke runbook.
- [x] Pass the reduced 8 x RTX 2080 Ti GRPO load, update, save, and reload gates.
- [x] Run a reduced two-task GRPO/RaPO experiment.
- [x] Extend the bounded comparison through task 3 with 20 samples per class.
- [x] Extend the bounded comparison through task 4 and audit the resulting gap.
- [x] Extend the bounded comparison through task 5 and trigger the focused
      implementation audit.
- [x] Complete the bounded task-6 comparison and isolate the FP16 loss-scale
      sensitivity from inference-hardware effects.
- [ ] Pass a paired task-6 BF16 stability gate on a homogeneous 8-GPU group.
- [ ] Run the full 10-task experiment.

## Environment

The development environment uses Conda through Miniforge and Python 3.10.

```bash
bash scripts/install_miniforge.sh  # only when Conda is not installed
bash scripts/bootstrap_env.sh
conda run -n rapo pytest
```

The initial environment deliberately excludes FlashAttention and the full
Visual-RFT dependency stack. This keeps CPU/unit-test setup independent of GPU
architecture. The isolated `rapo-train` specification and launch procedure are
documented in [docs/smoke_test.md](docs/smoke_test.md); do not run Visual-RFT's
moving, conflicting upstream installer directly.

## Repository policy

- Code and small experiment metadata are tracked in Git.
- Datasets, model weights, checkpoints, full logs, private keys, and internal
  server documentation are not committed.
- Remote jobs must set `CUDA_VISIBLE_DEVICES` explicitly and run inside `tmux`.
- GPU availability must be checked with `nvidia-smi` immediately before a run.

See [docs/reproduction_spec.md](docs/reproduction_spec.md) for the executable
reproduction contract and [docs/upstream.md](docs/upstream.md) for upstream
provenance. Trainer patching and task-to-task state handling are documented in
[docs/trainer_integration.md](docs/trainer_integration.md). Deterministic
ImageNet-R manifests, Visual-RFT dataset export, and continual metrics are
documented in
[docs/data_and_evaluation.md](docs/data_and_evaluation.md).
The first GPU compatibility measurements are recorded in
[docs/experiments/2026-07-29-2080ti-smoke.md](docs/experiments/2026-07-29-2080ti-smoke.md).
The first two-task comparison is recorded in
[docs/experiments/2026-07-29-2080ti-two-task.md](docs/experiments/2026-07-29-2080ti-two-task.md).
The extended 20-step comparison is recorded in
[docs/experiments/2026-07-29-2080ti-two-task-step20.md](docs/experiments/2026-07-29-2080ti-two-task-step20.md).
The three-task continuation is recorded in
[docs/experiments/2026-07-29-2080ti-three-task-step20.md](docs/experiments/2026-07-29-2080ti-three-task-step20.md).
The four-task continuation and paired-result audit are recorded in
[docs/experiments/2026-07-30-2080ti-four-task-step20.md](docs/experiments/2026-07-30-2080ti-four-task-step20.md).
The five-task continuation, compatibility investigation, and audit trigger are
recorded in
[docs/experiments/2026-07-30-2080ti-five-task-step20.md](docs/experiments/2026-07-30-2080ti-five-task-step20.md).
The resulting equation-to-trainer audit and task-6 continuation gate are
recorded in
[docs/experiments/2026-07-30-five-task-implementation-audit.md](docs/experiments/2026-07-30-five-task-implementation-audit.md).
The completed task-6 comparison, numerical-stability boundary, and explicit
pause/resume point are recorded in
[docs/experiments/2026-07-31-2080ti-six-task-stability-boundary.md](docs/experiments/2026-07-31-2080ti-six-task-stability-boundary.md).
