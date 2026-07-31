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
4. stages rewards from every rank and microbatch in one gradient-accumulation
   window and uses sample standard deviation (`correction=1`);
5. commits CTAN once from the complete window after a successful optimizer
   step, discards skipped steps, and never stages evaluation rewards;
6. rejects a tokenized multimodal prompt above `max_prompt_length` before
   generation instead of truncating an unproved visual-token boundary;
7. writes task, run ID, immutable run-contract SHA, and CTAN state into
   `rapo_state.json` at checkpoints and final model output.

These CTAN scope, first-success initialization, single-use unclipped surrogate,
and KL `beta=0.04` choices are frozen in
`configs/independent_reproduction.json`. They are repository decisions for an
independent reproduction, not paper facts or author settings.

Model-family dispatch reads `AutoConfig.model_type` from the checkpoint rather
than guessing from the directory name. This is required for task 2+, because a
saved task directory need not contain `Qwen2-VL` in its path.

The patch also replaces Visual-RFT's classification substring comparison with
the paper's exact-match verifier. The final class is extracted from one
`<answer>` span, lower-cased, and normalized by mapping underscores, hyphens,
and periods to spaces. A label such as `cat` therefore no longer receives a
false positive for `catfish`.

Dataset conversation preprocessing runs inside
`training_args.main_process_first`. On a distributed launch, rank 0 therefore
writes the Hugging Face `map` cache before the other ranks load it. This avoids
concurrent cache-file creation on shared filesystems without changing the
mapped examples.

## Required task sequencing

Task 1 starts from `Qwen/Qwen2-VL-2B-Instruct` and does not pass a RaPO state
or parent manifest. Task `t >= 2` must:

- start `model_name_or_path` from the final checkpoint of task `t-1`;
- pass the previous task's `rapo_state.json` through `rapo_state_path`;
- pass the finalized task `t-1` run manifest through `PARENT_MANIFEST_PATH`;
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
rapo_run_id: order0-rapo-task02
rapo_contract_sha256: <printed by provenance prepare-run>
rapo_resume_from_checkpoint: null
```

Batch 1 validates cross-task state continuity only: a task accepts state from
exactly `t-1`. Same-task interrupted/resumed equivalence remains a later batch
and must not be claimed from the legacy `rapo_resume_from_checkpoint` field.

## Run-manifest gate

The launcher requires `RUN_MANIFEST_PATH`, `EXPERIMENT_ID`, `RUN_ID`, and
`DATA_MANIFEST_PATH`; task 2+ also requires `PARENT_MANIFEST_PATH`. Before
`torchrun`, it invokes `python -m rapo.provenance prepare-run` to verify the
repository and diff, fixed Visual-RFT commit and exact patch, input model,
RaPO input state, data manifest, task-stage dataset, reproduction config, and
parent chain. The prepared contract is immutable. After model/state saving,
`finalize-run` binds their actual SHA256 identities. A retry may reuse only
identical manifest content.

## Deliberate compatibility limits

- RaPO with the vLLM trainer is rejected because that trainer has a separate
  rollout and reward path.
- Task 2+ with PEFT is rejected because disabling the current adapter restores
  the base model, not necessarily the frozen task `t-1` policy.
- FlashAttention and the complete Visual-RFT dependency stack are not part of
  the core Conda environment yet. They are introduced at the later GPU
  compatibility gate.
