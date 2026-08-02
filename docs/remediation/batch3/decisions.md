# Batch 3 decisions

Sections D-B3-001 through D-B3-004 record decisions inside the approved batch-3
remediation scope. D-B3-005 records the later independent acceptance outcome.
This document does not constitute leader release.

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

## D-B3-005 — independent acceptance outcome and next gate

Independent CPU acceptance at exact target
`687449031a92562c5fd2db93fbfb0283bba9aaae` technically passed the approved
local CPU/no-GPU orchestration scope with no blocking issue and no new P2
observation. The conclusion is supported by live 10/113-test reruns, static
gates, an independently built legal three-manifest chain, hand-checked
population statistics, and three bounded negative artifact/recovery probes.

The acceptance conversation did not create or close a formal finding and did
not release the batch. The leader must separately decide final release for the
batch-3 local CPU scope. Even after such release, BF16/GPU work requires a new
approved goal and remains gated by `pending_hardware_gate`; the formal chain
must start from the pinned base at Task 1 rather than continue the legacy Task
6 artifacts.
