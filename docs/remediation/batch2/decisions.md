# Batch 2 formal-contract decisions

These are repository choices for an independent reproduction, not paper facts
or author settings.

- `formal_profile.json` is canonical, epoch-driven, exactly two epochs, and isolated under output namespace `formal`; `legacy_2080ti_profile.json` remains step-driven under `legacy_2080ti`.
- Formal BF16, FlashAttention-2, world size 8, and ZeRO-3 are declared requirements with status `pending_hardware_gate`; configuration presence is not hardware evidence.
- Formal runner input rejects `RAPO_SMOKE_*`, generic `DEEPSPEED_CONFIG`, cuDNN workaround, and FP16 loss-scale injection. Smoke and 2080 Ti wrappers remain non-formal.
- Formal budget counts use a non-dropping distributed sampler: `ceil(samples/world_size) * world_size` presentations per epoch and `ceil(per-rank batches/GA)` optimizer steps per epoch.
- Formal checkpoints are requested every five successful Trainer steps. A checkpoint is resumable only when model, optimizer, scheduler, Trainer/global step, Python/Torch RNG, and (for RaPO) CTAN are all present and bound to run ID, profile SHA, and immutable run-contract SHA.
- Binding a resume checkpoint changes only the prepared manifest's `resume` record; it does not change the original contract or its SHA. A second different checkpoint cannot replace it.
- Same-task RaPO resume requires checkpoint state from the current task with the same run/contract/profile. Cross-task startup continues to require exactly task `t-1` state and parent lineage.
- Prediction identity is `(after_task, eval_task, relative_path)`; the expected per-cell set and target come from the frozen data manifest. Duplicate, missing, unknown, and target-mismatched rows fail.
- No-profile evaluator calls preserve the legacy FP16/SDPA and five-samples-per-class defaults. An explicit profile is recorded; formal evaluation requires BF16/FlashAttention-2, full manifest test data, and verified model/stage/data/profile/run lineage without fallback.
- CPU logical-rank tests cover 1/2/8 shards and freeze CTAN initialization at the first complete successful optimizer-step window. They do not claim NCCL execution.
- The leader released batch 2's local CPU scope after independent acceptance at `1d74a1ef5901b6f437cd7cf7ca19f1a56ad858c5`; this closes batch 2 only and does not authorize GPU, training, Task 7, or the formal 10-task chain.
- A direct non-standard `write_checkpoint_binding(..., require_ctan="false")` call remains a P2 backlog observation. The standard Trainer passes a boolean and resume validation fails closed, so no further batch-2 repair is authorized for this item.
