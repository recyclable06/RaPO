# BLOCKED

## Repair executor status (2026-08-02)

No implementation blocker was encountered while applying the approved minimal
repair at `90f36f6d41023866dd5f994b7d0168ff5b4c1397`. `B2-ACC-001` and
`B2-ACC-002` remain formally open until a new independent acceptance
conversation reruns the complete batch-2 protocol; executor tests cannot close
them. The `pending_hardware_gate` and all downstream GPU/formal-chain gates
remain unchanged.

## Open CPU acceptance blockers (2026-08-02)

Batch 2 did not pass independent CPU acceptance at
`3661b6755faf39c84e9ba368b2d23cd232cd0fd7`. The disclosed suites and static
checks passed, but the following P1 contract failures were reproduced with
repository-external inputs:

1. **`B2-ACC-001` (P1): formal attention requirement is not enforced by
   profile validation.**
   The legal `formal_profile.json` control loaded as FlashAttention-2. A copy
   with only `training.attention` changed to `sdpa` was also accepted by
   `load_experiment_profile` and `build_dry_run_contract`, which then emitted
   `resolved.attention=sdpa`. The frozen decision requires formal BF16,
   FlashAttention-2, world size 8, and ZeRO-3; the invalid profile must fail
   before formal execution.
2. **`B2-ACC-002` (P1): resume binding does not cross-check Trainer global
   step.** A legal
   checkpoint with `trainer_state.json.global_step=5` and binding step 5
   passed. A second checkpoint differing only by binding step 6 was also
   accepted by `write_checkpoint_binding` and `validate_checkpoint_binding`.
   A resumable checkpoint must bind the actual Trainer/global-step state, not
   only a caller-supplied integer.

The independent manifest-side pollution control passed: two classes declaring
the same `(eval_task, relative_path)` were rejected before aggregation. This
does not close either blocker above.

Next lifecycle step: an independent consulting conversation may draft a
minimal repair goal from these versioned failures; the leader must approve it
before a new remediation conversation changes code. After repair, a new
independent acceptance conversation must rerun the entire batch-2 acceptance,
not only the two failing probes.

## Pending external gate

`pending_hardware_gate`: BF16 operations, FlashAttention-2 import and
forward/backward, homogeneous eight-rank NCCL, ZeRO-3 behavior, memory margin,
checkpoint/reload on the target stack, and numerical stability were not run.
They remain batch-4 work and are not implied by the formal JSON, mocks, CPU
tests, or dry-run.

No GPU, training, inference, SSH, remote write, dependency installation,
deletion, or push was authorized or performed. Formal Task 1, Task 7, BF16
gate, and the 10-task chain remain unapproved.

## Verification setup observations

The fixed-upstream patch check had two setup-only failures before one complete
green run: nested PowerShell/WSL quoting failed, then a Windows Git worktree
pointer was unreadable inside WSL. A third new WSL checkout from a verified
local object source succeeded at the pinned commit with official origin, real
script apply, diff check, exact two-file scope, and reverse-check. The failed
paths were not reused as accepted evidence and were not deleted.

Independent acceptance later repeated the fixed-upstream check in another new
external checkout. A direct official clone first timed out on GitHub port 443;
a fresh `--no-hardlinks` clone from a clean local object source, with no object
alternates and official origin restored, then passed apply, diff, exact scope,
reverse-check, and Python compilation. Neither checkout history substitutes
for the two failed CPU contract probes.
