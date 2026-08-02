# BLOCKED

## Final B2-HIDDEN repair executor status (2026-08-02)

无（整改执行层面）。Implementation commit
`4ab6337a504d24ad4a27fe991dfdd353afd9745e` passed the approved executor CPU
self-checks at 47/80/103 tests with 0 skipped and preserved the formal dry-run
budget and numerical-path declarations. The historical P1 entries below are
retained as independent-acceptance evidence; this executor does not close them
or declare batch 2 accepted. A new independent acceptance conversation must
rerun the complete protocol before any release decision.

## Current CPU acceptance blockers after repair revalidation (2026-08-02)

Batch 2 did not pass complete independent CPU re-acceptance at
`dc637359e4df1d18846541e8a6655686ae097e8f`. The 44/77/100-test suites, static
gates, formal dry-run, fixed-upstream patch checks, and independent
`B2-ACC-001/002` legal/illegal controls passed. Those original two failures are
not current blockers at this target. Three new P1 contract failures remain:

1. **`B2-HIDDEN-001` (P1): formal output namespace is not isolated.** A legal
   formal control used namespace `formal`. An external copy differing only by
   `output_namespace="legacy_2080ti"` was accepted by profile loading and the
   formal dry-run. A formal run must not enter the frozen legacy namespace.
2. **`B2-HIDDEN-002` (P1): formal DeepSpeed configuration can downgrade to the
   legacy path.** A legal formal control resolved
   `configs/deepspeed_zero3_formal_bf16.json`. An external copy differing only
   by `training.deepspeed_config="configs/deepspeed_zero3_cpu_offload.json"`
   was accepted. The frozen formal BF16 ZeRO-3 requirement must fail before use
   when a legacy FP16 CPU-offload config is selected.
3. **`B2-HIDDEN-003` (P1): RaPO method-state binding accepts a string as the
   CTAN requirement.** A legal checkpoint binding with boolean
   `require_ctan=true` passed. Changing only that field to string `"false"`
   still passed validation because the current check coerces truthiness instead
   of requiring a JSON boolean. The checkpoint inventory and Trainer step were
   unchanged.

Next lifecycle step: an independent consulting conversation may draft one
minimal repair goal from these versioned failures; the leader must approve it
before a new remediation conversation changes code. After repair, a new
independent acceptance conversation must rerun the entire batch-2 protocol,
not only these three probes.

## Repair executor status before re-acceptance (2026-08-02)

No implementation blocker was encountered while applying the approved minimal
repair at `90f36f6d41023866dd5f994b7d0168ff5b4c1397`. At that point
`B2-ACC-001/002` remained open pending independent re-acceptance; the later
`dc63735` re-acceptance above independently verified their legal and illegal
controls. The `pending_hardware_gate` and all downstream GPU/formal-chain gates
remain unchanged.

## Historical CPU acceptance blockers before repair (2026-08-02)

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
did not close either historical blocker in this subsection.

These two failures motivated repair `90f36f6` and are retained as historical
evidence. Their later independent controls passed; the current lifecycle step
is governed by `B2-HIDDEN-001/002/003` above.

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
