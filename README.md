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
- [x] Complete the independent audit and CPU-only independent acceptance of
      remediation batch 1.
- [x] Deliver remediation batch 2's strict two-epoch, profile, resume,
      evaluation, and provenance implementation.
- [x] Repair and independently revalidate the original batch-2 P1 attention
      and Trainer-step failures.
- [x] Repair the three new P1 contract failures found by the complete batch-2
      re-acceptance and pass the final independent CPU acceptance suite.
- [x] Receive leader release for batch 2's local CPU scope, with the remaining
      non-standard binding-type observation retained as P2 backlog.
- [x] Implement and independently accept the audit-driven CPU/no-GPU
      orchestration gates.
- [ ] Pass a paired task-6 BF16 stability gate on a homogeneous 8-GPU group.
- [ ] Run the full 10-task experiment.

Batch 1 acceptance covers local CPU semantics, lineage, and failure detection.
Batch 2's original P1 failures were repaired at `90f36f6` and revalidated at
`dc63735`. The three later P1 profile/binding failures were repaired at
`4ab6337`; final independent CPU acceptance at `1d74a1e` reran the 47/80/103
suites with 0 skipped, static gates, formal dry-run, five direct production-API
regressions, and the unchanged fixed-upstream gate. All blocking controls
passed. One non-standard writer call can still emit a string-valued CTAN
binding, but the standard Trainer supplies a boolean and resume validation
fails closed; the leader retained this as non-blocking P2 backlog with no
further batch-2 repair in
[docs/remediation/batch2/BLOCKED.md](docs/remediation/batch2/BLOCKED.md).
The leader released batch 2's local CPU scope after that acceptance. Batch 3
then added the frozen 10-task x 2-method x 3-order CPU/no-GPU orchestration
contract at `680d3d1` and recorded its delivery at `6874490`. Independent
acceptance at exact target `6874490` reran the 10/113-test suites with 0
skipped, all static gates, deterministic DAG/status/aggregate controls, manual
mean/population-std checks, and three negative artifact/recovery probes. It
technically passed with no blocking issue and no new P2 observation; details
are in
[docs/remediation/batch3/PROGRESS.md](docs/remediation/batch3/PROGRESS.md).
This is not yet leader release for batch 3, GPU evidence, a rerun of training,
or completion of the paper reproduction. The next milestone is the leader's
batch-3 local CPU release decision; any later BF16/GPU gate requires a separate
approved goal. The authoritative current checkpoint remains
[AGENTS.md](AGENTS.md), and the dated independent findings remain in
[docs/audit/2026-07-31-independent-audit.md](docs/audit/2026-07-31-independent-audit.md).

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
