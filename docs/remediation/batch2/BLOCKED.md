# BLOCKED

## Released local CPU status (2026-08-02)

No blocking CPU acceptance issue was reproduced at
`1d74a1ef5901b6f437cd7cf7ca19f1a56ad858c5`. The final independent acceptance
reran the 47/80/103 suites with 0 skipped, static/dry-run gates, and direct
legal/illegal controls for `B2-ACC-001/002` and
`B2-HIDDEN-001/002/003`; all approved blocking criteria behaved as required.
The acceptor did not create or close formal findings; this section records the
current technical acceptance result while preserving the historical failures
below. The leader approved final release of this local CPU scope; batch 2 is
closed.

### Non-blocking P2 observation

A direct non-standard Python call can pass string `"false"` as
`write_checkpoint_binding(..., require_ctan=...)` and produce an unusable
binding. The standard patched Trainer passes the boolean expression
`self.controller is not None`, and `validate_checkpoint_binding` rejects the
string before resume. No frozen launcher/CLI or normally generated standard
artifact path was shown to alter training, resume continuity, or a formal
result. The leader retained this as non-blocking P2 backlog and explicitly
declined further batch-2 repair for it.

### Release boundary

The release is limited to batch 2's local CPU contract and alarm behavior.
`pending_hardware_gate`, GPU execution, BF16/FlashAttention/NCCL behavior,
training, Task 7, and the formal 10-task chain remain out of scope and
unapproved.

## Historical final B2-HIDDEN repair executor status (2026-08-02)

无（整改执行层面）。Implementation commit
`4ab6337a504d24ad4a27fe991dfdd353afd9745e` passed the approved executor CPU
self-checks at 47/80/103 tests with 0 skipped and preserved the formal dry-run
budget and numerical-path declarations. The historical P1 entries below are
retained as independent-acceptance evidence; this executor does not close them
or declare batch 2 accepted. At that time the required next step was a new
independent acceptance conversation rerunning the complete protocol; that step
was later completed at `1d74a1e` and is recorded above.

## Historical CPU acceptance blockers at `dc63735` (2026-08-02)

Batch 2 did not pass complete independent CPU re-acceptance at
`dc637359e4df1d18846541e8a6655686ae097e8f`. The 44/77/100-test suites, static
gates, formal dry-run, fixed-upstream patch checks, and independent
`B2-ACC-001/002` legal/illegal controls passed. Those original two failures are
not blockers at this historical target. Three new P1 contract failures were
present at `dc63735`:

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

The required next step at that historical target was a leader-approved minimal
repair followed by a new independent acceptance conversation rerunning the
entire batch-2 protocol. That sequence was completed by repair `4ab6337` and
final acceptance at `1d74a1e`; the current status is recorded at the top.

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
