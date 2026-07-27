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
- [ ] Integrate the module into the pinned Visual-RFT GRPO trainer.
- [ ] Build the deterministic ImageNet-R task split and evaluator.
- [ ] Run a reduced two-task GRPO/RaPO experiment.
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
architecture. Training dependencies will be added after the core implementation
is verified.

## Repository policy

- Code and small experiment metadata are tracked in Git.
- Datasets, model weights, checkpoints, full logs, private keys, and internal
  server documentation are not committed.
- Remote jobs must set `CUDA_VISIBLE_DEVICES` explicitly and run inside `tmux`.
- GPU availability must be checked with `nvidia-smi` immediately before a run.

See [docs/reproduction_spec.md](docs/reproduction_spec.md) for the executable
reproduction contract and [docs/upstream.md](docs/upstream.md) for upstream
provenance.
