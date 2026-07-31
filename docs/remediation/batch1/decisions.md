# Batch 1 independent-reproduction decisions

These settings are repository choices for an independent reproduction. They are not paper facts or author settings.

- CTAN `sigma_batch` is the sample standard deviation (`correction=1`) over every rank and every microbatch in one gradient-accumulation window.
- A microbatch may normalize with the provisional EMA implied by rewards accumulated so far in that window.
- CTAN commits exactly once after a successful optimizer step. A skipped step discards its pending window, and evaluation never stages or commits rewards.
- The first successful optimizer step initializes the EMA directly from its complete-window sample standard deviation.
- Rollouts are single-use. The policy surrogate is the current unclipped sampling-point form; policy reuse is forbidden because an out-of-range likelihood ratio would require an explicit clipping rule that this repository does not invent.
- KL `beta=0.04` is a repository choice. The paper and author implementation do not establish it as an author setting.
- A prepared run contract is immutable and has a canonical SHA256. Finalization binds the saved output model and, for RaPO, its state without creating a manifest/state hash cycle.
- Task 1 forbids a parent manifest and prior CTAN state. Task `t >= 2` accepts only a finalized task `t-1` parent with matching method, experiment, repository, upstream patch, model, state, data manifest, and reproduction config.
- Existing data manifests and stage bindings are immutable: an identical retry is allowed; different content, seed, split, or task binding fails fast.
