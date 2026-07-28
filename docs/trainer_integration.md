# Visual-RFT trainer integration

The first trainer integration targets Visual-RFT commit
`2ffad63b25ddd79bfe25d3e046645401201c89d6`. The repository remains an
external Apache-2.0 dependency; this project stores a small, reviewable patch
instead of vendoring the full upstream tree.

## Prepare the upstream checkout

```bash
git clone https://github.com/Liuziyu77/Visual-RFT.git external/Visual-RFT
git -C external/Visual-RFT checkout --detach 2ffad63b25ddd79bfe25d3e046645401201c89d6
bash scripts/apply_visual_rft_patch.sh external/Visual-RFT
```

The patch changes only:

- `src/virft/src/open_r1/trainer/grpo_trainer.py`;
- `src/virft/src/open_r1/grpo_classification.py`.

With `rapo_enabled=false`, the original GRPO reward normalization path is
unchanged. With `rapo_enabled=true`, the trainer:

1. uses CTAN for every task;
2. adds Retention Reward from task 2 onward;
3. treats the frozen GRPO reference model as the previous-task anchor;
4. computes CTAN batch standard deviation from rewards gathered across ranks;
5. avoids CTAN updates while the model is in evaluation mode;
6. writes `rapo_state.json` into Trainer checkpoints and final model output.

When gradient checkpointing is enabled, the patch also marks model input
embeddings as requiring gradients. This is required by PyTorch's reentrant
checkpoint implementation; without it, a run can finish with varied rewards
but a zero gradient norm.

The patch also replaces Visual-RFT's classification substring comparison with
the paper's exact-match verifier. The final class is extracted from one
`<answer>` span, lower-cased, and normalized by mapping underscores, hyphens,
and periods to spaces. A label such as `cat` therefore no longer receives a
false positive for `catfish`.

## Required task sequencing

Task 1 starts from `Qwen/Qwen2-VL-2B-Instruct` and does not pass a RaPO state
file. Task `t >= 2` must:

- start `model_name_or_path` from the final checkpoint of task `t-1`;
- pass the previous task's `rapo_state.json` through `rapo_state_path`;
- keep the paper-locked RaPO settings unchanged.

Example RaPO flags for task 2:

```yaml
rapo_enabled: true
rapo_task_index: 2
rapo_retention_alpha: 20.0
rapo_retention_weight: 0.5
rapo_ctan_beta: 0.999
rapo_ctan_epsilon: 0.0001
rapo_state_path: outputs/task_01/rapo_state.json
rapo_resume_from_checkpoint: null
```

When `rapo_resume_from_checkpoint` is set and `rapo_state_path` is unset, the
patched classification entrypoint loads
`<checkpoint>/rapo_state.json` automatically.

## Deliberate compatibility limits

- RaPO with the vLLM trainer is rejected because that trainer has a separate
  rollout and reward path.
- Task 2+ with PEFT is rejected because disabling the current adapter restores
  the base model, not necessarily the frozen task `t-1` policy.
- FlashAttention and the complete Visual-RFT dependency stack are not part of
  the core Conda environment yet. They are introduced at the later GPU
  compatibility gate.
