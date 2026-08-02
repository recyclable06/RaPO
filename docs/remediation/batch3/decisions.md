# Batch 3 decisions

Only decisions inside the approved batch-3 remediation scope are recorded
here. This document does not constitute independent acceptance or release.

## D-B3-001 — restricted plan, not a general scheduler

The standard entry supports only `dry-run`, `status`, and `aggregate` for the
frozen 10-task x 2-method x 3-order formal protocol. It emits commands for the
existing formal launcher and evaluator but does not execute, queue, parallelize,
retry, or otherwise schedule them.

## D-B3-002 — full-test and chain identity

Every cumulative prediction count comes directly from the matching manifest's
`tasks[T-1].test_size`. Each task 2+ training node has exactly one predecessor:
task T-1 of the same method and class order. A finalized manifest or resume
checkpoint is accepted only through the existing provenance and checkpoint
identity validators plus the planned method/order/task/path binding.

## D-B3-003 — aggregate identity and standard deviation

Aggregation reuses the existing prediction-lineage, exact-key/full-count, and
continual-metrics functions for every stage. The six method/order results must
form the exact `{grpo, rapo} x {0, 1, 2}` grid and bind the plan plus all 60
prediction artifact SHA256 values.

The cross-order standard deviation is the population standard deviation,
`ddof=0`. This is an **independent reproduction choice**, not a paper fact; the
paper requires reporting across three random orders but does not specify the
standard-deviation denominator.

## D-B3-004 — claim boundary

Passing these CPU/no-GPU checks establishes only the orchestration contract and
its fail-closed artifact handling. It does not establish GPU readiness, actual
BF16/FlashAttention/NCCL execution, completed training or inference, Task 7,
the formal 10-task experiment, or paper-level reproduction.
