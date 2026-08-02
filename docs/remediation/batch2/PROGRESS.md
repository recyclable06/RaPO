# Batch 2 progress

## Leader release and knowledge closeout (2026-08-02)

1. The leader approved final release of batch 2's local CPU scope based on the
   independent acceptance of target
   `1d74a1ef5901b6f437cd7cf7ca19f1a56ad858c5`; blocking issues are zero and
   batch 2 is closed.
2. The direct non-standard writer call accepting string `require_ctan` remains
   a non-blocking P2 backlog item. The standard Trainer passes a boolean and
   resume validation fails closed; the leader explicitly decided not to run
   another batch-2 repair for this observation.
3. This release does not cover GPU, actual BF16/FlashAttention/NCCL execution,
   training, Task 7, the formal 10-task chain, or paper-level reproduction.
4. The next milestone is the bounded batch-3 CPU/no-GPU orchestration work
   recorded in `AGENTS.md`, with one main remediation and at most one repair
   for a research-critical blocker.
5. Repository-external acceptance probes remain temporary review evidence;
   versioned records preserve their criteria and results without depending on
   those paths remaining available.

## Final independent CPU acceptance after B2-HIDDEN repair (2026-08-02)

All executor self-checks and prior acceptance logs were treated as
`recorded-but-not-rerun` except for the fixed-upstream gate explicitly carried
forward under the approved immutable-input rule.

1. Target: `1d74a1ef5901b6f437cd7cf7ca19f1a56ad858c5`; implementation
   `4ab6337a504d24ad4a27fe991dfdd353afd9745e`; baseline
   `d6a3f22d2e6ac442036128f50e8349aeb1c33a64`. The worktree was clean before
   and after. The implementation range changed only
   `src/rapo/{formal_contract,resume}.py` and
   `tests/{test_formal_contract,test_provenance}.py`; the delivery range
   changed only batch-2 `PROGRESS.md` and `BLOCKED.md`.
2. Environment: Python 3.10.20, Torch 2.5.1+cpu, pytest 8.4.2, Git 2.51.0,
   GNU Bash 5.2.21, and WSL 2.6.1.0. WSL had no `python` command, so the
   stdlib-only formal dry-run used Python 3.12.3 as explicit `python3`.
3. Disclosed checks rerun: targeted files 47 passed; complete batch-2 suite 80
   passed; full suite 103 passed, 0 skipped. Compileall, all three launcher
   syntax checks, current diff check, and `d6a3f22..1d74a1e` diff check
   returned 0.
4. Formal dry-run returned exactly two epochs, 104 sampler examples per epoch,
   208 sample presentations, 1,664 generations, 14 optimizer steps, world size
   8, BF16, FlashAttention-2, formal ZeRO-3, and `pending_hardware_gate`.
5. Five existing blocker classes were independently reconstructed with direct
   production API calls and repository-external inputs. Legal controls passed;
   the representative attention, namespace, DeepSpeed, Trainer-step, and
   string `require_ctan` controls all failed before use or binding. The failed
   Trainer-step write left no reusable binding.
6. The single new negative probe found that a direct non-standard
   `write_checkpoint_binding(..., require_ctan="false")` call can write an
   unusable string-valued binding. The patched Trainer's standard path passes
   the boolean expression `self.controller is not None`, and validation rejects
   the malformed binding before resume. This is a non-blocking P2 observation,
   not a standard-path blocker or a formal finding created by the acceptor.
7. The fixed patch and apply script were unchanged from `dc63735` and matched
   SHA256 `A8425A7E8907089A01D7698C9E8D144E068417D98AED1D58DB8164279D1352B2`
   and `3D29ECFB96051EE3703B68E1BED1083555F3AB6B61B0EED99629886BF810E828`;
   the Visual-RFT pin remained `2ffad63b25ddd79bfe25d3e046645401201c89d6`.
   The real apply/diff/reverse/compile result from the immediately preceding
   `dc63735` independent acceptance is therefore `carried-forward-unchanged`.
   One initial PowerShell range interpolation attempt returned Git usage exit
   129; the corrected explicit range returned 0 and showed no changed inputs.
8. Verdict: **batch 2 passed final independent CPU technical acceptance with
   one non-blocking P2 observation**. The leader still owns final batch release.
   No GPU, training, inference, SSH, network, installation, deletion, or push
   was performed. This does not authorize Task 7, BF16 hardware gates, or the
   formal 10-task chain.

After the verdict and clean-worktree check, the leader explicitly invoked
`neat-freak` in the same acceptance conversation. The role exception is limited
to synchronizing docs/rules and read-only residue inventory; it does not change
business code, tests, configs, formal findings, generated memory, evidence, or
the acceptance verdict.

## B2-HIDDEN-001/002/003 final repair (2026-08-02)

1. Goal: add one representative regression per approved P1 and enforce only the frozen formal namespace, formal DeepSpeed path, and JSON-boolean RaPO CTAN binding.
2. Order: verify clean `d6a3f22` baseline -> add three failing regressions -> make three centralized checks -> rerun all CPU gates -> commit code/tests then delivery records.
3. Task 0 live rerun: branch `codex/rapo-b02-formal-contract`, clean `d6a3f22d2e6ac442036128f50e8349aeb1c33a64`, 100 passed/0 skipped, and diff check empty.
4. Largest risk: accidentally broadening validation beyond the three frozen standard-path contracts or changing canonical formal dry-run semantics.

### Red baseline

```text
TARGETED RED: 3 failed, 44 passed in 9.63s
FAILED test_formal_namespace_downgrade_fails_before_dry_run: invalid namespace was accepted
FAILED test_formal_deepspeed_downgrade_fails_before_dry_run: legacy DeepSpeed path was accepted
FAILED test_rapo_binding_rejects_non_boolean_require_ctan: string "false" was accepted
IMPLEMENTATION DIFF: git diff -- src/rapo/formal_contract.py src/rapo/resume.py returned no output
```

### Green implementation attempt 1

```text
TARGETED GREEN: 47 passed in 9.49s
Implementation: exact formal namespace and DeepSpeed path checks plus strict JSON-boolean require_ctan comparison.
BATCH-2 GREEN: 80 passed in 9.57s
FULL GREEN: 103 passed in 9.81s, 0 skipped
COMPILEALL: exit 0
LAUNCHER SYNTAX formal: exit 0
LAUNCHER SYNTAX smoke: exit 0
LAUNCHER SYNTAX 2080ti: exit 0
FORMAL DRY-RUN setup attempt 1: stopped before contract generation because the documented `RAPO_CPU_PYTHON=python` was absent in the current bash environment (`python: command not found`); no GPU/training/inference ran.
FORMAL DRY-RUN attempt 2: exit 0 with WSL `/usr/bin/python3` 3.12.3 substituted only because the documented `python` command was absent; the stdlib-only runner emitted 2 epochs, 104 sampler examples/epoch, 208 presentations, 1,664 generations, 14 optimizer steps, BF16, FlashAttention-2, formal ZeRO-3, world size 8, and `pending_hardware_gate`.
DIFF CHECK before implementation commit: exit 0 (Git emitted only LF-to-CRLF working-copy notices).
```

### Final repair delivery self-check

Implementation commit: `4ab6337a504d24ad4a27fe991dfdd353afd9745e`.

```text
RED targeted: 3 failed, 44 passed in 9.63s; implementation diff was empty
GREEN targeted: 47 passed in 9.49s
GREEN batch-2: 80 passed in 9.57s
GREEN full: 103 passed in 9.81s, 0 skipped
compileall and formal/smoke/2080ti bash syntax: exit 0
formal dry-run: 2 epochs, 208 presentations, 1,664 generations, 14 optimizer steps; BF16, FlashAttention-2, formal ZeRO-3, pending_hardware_gate
git diff --check: exit 0
baseline-to-implementation paths: src/rapo/formal_contract.py, src/rapo/resume.py, tests/test_formal_contract.py, tests/test_provenance.py
```

The fixed-upstream patch, apply script, pin, and configs were not modified. Per
the approved goal, the fixed-upstream patch was not cloned or reapplied, so no
patch-apply result is claimed by this repair. No GPU, training, inference, SSH,
remote write, dependency installation, deletion, or push was performed.
Current remediation-executor blocker: none. These are executor self-checks,
not an independent batch-2 acceptance or experiment-release verdict. The next
release point is a new independent acceptance conversation rerunning the full
batch-2 protocol.

## Independent CPU re-acceptance after repair (2026-08-02)

The independent acceptance Codex treated the repair self-check and every prior
acceptance log as `recorded-but-not-rerun` until reproducing them at the target.

1. Target: `dc637359e4df1d18846541e8a6655686ae097e8f` on
   `codex/rapo-b02-formal-contract`; the worktree was clean before and after.
   The `d98e0ab..90f36f6` implementation diff contained only
   `src/rapo/{formal_contract,resume}.py` and
   `tests/{test_formal_contract,test_provenance}.py`; the
   `90f36f6..dc63735` delivery diff contained only batch-2 `PROGRESS.md` and
   `BLOCKED.md`. All six frozen file SHA256 values and both canonical profile
   SHA256 values matched the approved target.
2. Environment: Python 3.10.20, Torch 2.5.1+cpu, pytest 8.4.2 from
   `D:\anaconda3\envs\rapo-b01\python.exe`.
3. Disclosed checks rerun: targeted files 44 passed; complete batch-2 suite 77
   passed; full suite 100 passed, 0 skipped. Compileall, all three launcher
   syntax checks, current diff check, and `621daf9..dc63735` diff check returned
   0.
4. Formal dry-run returned exactly two epochs, 104 sampler examples per epoch,
   208 sample presentations, 1,664 generations, 14 optimizer steps, BF16,
   FlashAttention-2, formal ZeRO-3, and `pending_hardware_gate`.
5. The public `B2-ACC-001/002` regressions were independently reconstructed
   without pytest. Legal controls passed; training/evaluation attention
   downgrades, requested/bound Trainer-step mismatches, missing Trainer state,
   and boolean/float/negative/non-object Trainer steps all failed before use.
   Failed binding writes left no reusable binding.
6. A fresh repository-external Visual-RFT checkout at pinned commit
   `2ffad63b25ddd79bfe25d3e046645401201c89d6` passed the real patch apply,
   upstream diff check, exact two-file scope, reverse-check, and patched Python
   compilation. Two official clones failed at GitHub port 443 and one
   Windows-native checkout was rejected as non-clean by WSL before patching;
   none was reused as accepted evidence. A later fresh WSL checkout under the
   external Windows temporary evidence root completed the full check and was
   preserved for review.
7. Three new repository-external negative probes had legal controls pass but
   all three invalid variants were accepted: `B2-HIDDEN-001` allowed a formal
   profile to use output namespace `legacy_2080ti`; `B2-HIDDEN-002` allowed a
   formal profile to select the legacy FP16 CPU-offload DeepSpeed config; and
   `B2-HIDDEN-003` accepted string `"false"` as a RaPO binding's
   `require_ctan` value.
8. Verdict: **batch 2 did not pass independent CPU re-acceptance**. The three
   P1 alarm failures override the green disclosed suites, original regressions,
   and patch checks. This verdict covers only local CPU contracts and failure
   detection; it does not authorize or claim GPU, BF16, FlashAttention, NCCL,
   Task 7, training, inference, or paper-level reproduction.

## B2-ACC-001/002 repair (2026-08-02)

> Historical executor record. The current independent verdict is the
> re-acceptance section above.

1. Goal: fail before formal execution on either attention downgrade and before binding/resume on any Trainer global-step mismatch.
2. Order: verify `d98e0ab` baseline -> add failing regressions -> make the smallest centralized validators -> rerun all gates -> commit code/tests then delivery records.
3. Baseline live rerun: clean `d98e0abf7fbdc9609e6f25e8eb80af20cbd48734`, 88 passed/0 skipped, diff check empty, and all six repair-goal frozen SHA256 values matched.
4. Largest risk: accepting caller or binding metadata without cross-checking the live `trainer_state.json`, especially during later resume validation.

### Repair delivery self-check

Implementation commit: `90f36f6d41023866dd5f994b7d0168ff5b4c1397`.

```text
RED targeted files: 11 failed, 33 passed in 9.75s
  - all four formal training/evaluation sdpa/eager downgrades were accepted
  - requested step 6 vs Trainer step 5, tampered binding step 6, and five malformed Trainer-step payloads were accepted
GREEN targeted files: 44 passed in 11.33s
GREEN disclosed batch-2 suite: 77 passed in 10.98s (historical suite was 65)
GREEN full suite: 100 passed in 11.16s, 0 skipped (repair baseline was 88)
compileall: exit 0
git diff --check: exit 0
```

Formal and legacy canonical SHA256 remain
`b7a661a4585a3423bfa89bcf9fe999862c0a15b2f5c532203bb77c62648d38f3`
and `e7fce6058500e04102008c32fa79d4d54c111945300213510dd97bf835b5ba56`.
The six frozen file SHA256 values matched the repair goal exactly. All five
pre-delivery changed paths were allowlisted; the implementation commit contains
only the two implementation and two test files. The fixed-upstream patch was
not modified or reapplied, so no prior patch-apply output is claimed here.

No GPU, training, inference, SSH, remote write, dependency installation,
deletion, or push was performed. These are executor self-checks, not an
independent batch-2 acceptance verdict. The next release point is a new
independent acceptance conversation rerunning the complete batch-2 protocol.

1. Goal: establish a machine-checkable formal experiment contract for strict two epochs, exact resume, complete evaluation, and isolated numerical profiles.
2. Order: baseline gate -> profiles/formal runner -> dynamic CPU resume -> evaluation contract -> full verification -> code commit -> delivery-record commit.
3. Largest risk: resume equivalence can look green while omitting optimizer, scheduler, RNG, global-step, or CTAN state; every component and the next token/loss sequence must be compared.
4. Task 0 live rerun: clean `621daf9c782397f1f576eb3c9e1f6af24d095c3c`, 67 passed/0 skipped, compileall and diff check returned 0, and all four frozen SHA256 values matched.

## Red-to-green evidence

```text
RED formal/profile/resume collection: ModuleNotFoundError: No module named 'rapo.formal_contract' (1 collection error).
RED prediction integrity: 5 failed, 5 passed; aggregate_prediction_records rejected the new data_manifest argument.
RED same-task CTAN resume: 1 failed, 20 deselected; RapoTrainerConfig had no profile binding, then task 1 rejected checkpoint state.
GREEN disclosed batch-2 suite: 65 passed in 9.27s.
GREEN full suite: 88 passed in 9.24s, 0 skipped (baseline was 67 passed).
REVERSE legacy leak: exit nonzero, "Formal runner rejects legacy environment variable RAPO_SMOKE_PRECISION."
```

The deterministic CPU test runs the same initialized `torch.nn.Linear`, SGD
optimizer, StepLR scheduler, Python RNG, Torch RNG, global step, and CTAN in an
uninterrupted six-step path and a three-step-save/resume-six-step path. The
next token/target/loss triples and final model/optimizer/scheduler/CTAN states
match exactly. Separate corruptions of run ID, profile SHA, contract SHA,
missing RNG, invalid CTAN, and a post-binding optimizer file all fail before a
continued step.

## Final implementation self-check

```text
commit: 91d3c7f09c3f715de7bd4521b961cd05ae09d4e0
formal profile canonical SHA256: b7a661a4585a3423bfa89bcf9fe999862c0a15b2f5c532203bb77c62648d38f3
legacy profile canonical SHA256: e7fce6058500e04102008c32fa79d4d54c111945300213510dd97bf835b5ba56
pytest batch-2 disclosed suite: 65 passed in 9.27s
pytest full suite: 88 passed in 9.24s, 0 skipped
compileall: exit 0
bash -n formal/smoke/2080ti runners: exit 0
git diff --check: exit 0
patched upstream Python syntax compile: exit 0
```

Formal CPU dry-run with 100 samples and eight logical ranks returned exactly
two epochs, 104 sampler examples/epoch, 208 sample presentations, 1,664
generations, 14 optimizer steps, BF16, FlashAttention-2, formal ZeRO-3, and
`pending_hardware_gate`. It did not load a model, use CUDA, train, or infer.

The fixed Visual-RFT verification used a fresh external WSL checkout at
`2ffad63b25ddd79bfe25d3e046645401201c89d6` with the official origin. The real
apply script, upstream `diff --check`, exact two-file allowlist, and reverse
check all returned 0. Two earlier attempts stopped before patch application:
one had nested-shell quote parsing failure and one used a Windows worktree
pointer unreadable by WSL. The third fresh checkout was the accepted run. A
later optional WSL `py_compile` could not find the already-expired `/tmp`
checkout; the same patched files in the external Windows checkout then
compiled successfully. No temporary path was deleted.

Relative to `621daf9c782397f1f576eb3c9e1f6af24d095c3c`, every changed path is in the
approved batch-2 allowlist. Frozen SHA256 values remain:

```text
AGENTS.md C867630C71982F4F1A9BAB647E48BB08923D0447ED263BBA31C0C0485E27C6D7
audit C019B7683C0606B7CC47B4264BF9C15E1F92481C839E994E0A925614C149841A
batch1 decisions 5062C61EA49C64D473712398E0DB73F9598EBBEF6F9F36690CB909D3C8CFB42E
independent config 50EE4DEA8231C6A46907B43C53EEBDB2BA4BEFECC9438EB9C2B0CCBCD1375B43
```

No GPU, training, inference, SSH, remote write, dependency installation,
deletion, or push was performed. These are executor self-checks, not an
independent acceptance verdict.

## Independent CPU acceptance (2026-08-02)

> Historical first-acceptance record, superseded as current state by the
> repair and complete re-acceptance sections above.

The independent acceptance Codex treated all executor results above as
`recorded-but-not-rerun` until reproducing them at the specified target.

1. Target: `3661b6755faf39c84e9ba368b2d23cd232cd0fd7` on
   `codex/rapo-b02-formal-contract`; the worktree was clean before and after
   acceptance. The baseline-to-implementation and implementation-to-delivery
   diffs matched the approved allowlists, and all frozen file and canonical
   profile SHA256 values matched.
2. Environment: Python 3.10.20, Torch 2.5.1+cpu, pytest 8.4.2 from
   `D:\anaconda3\envs\rapo-b01\python.exe`.
3. Disclosed checks rerun: batch-2 suite 65 passed in 32.18s; full suite 88
   passed in 21.97s, 0 skipped. Compileall, all three launcher syntax checks,
   current and baseline diff checks returned 0.
4. Formal dry-run returned 2 epochs, 104 sampler examples/epoch, 208 sample
   presentations, 1,664 generations, 14 optimizer steps, BF16,
   FlashAttention-2, formal ZeRO-3, and `pending_hardware_gate`.
5. A new external Visual-RFT checkout at pinned commit
   `2ffad63b25ddd79bfe25d3e046645401201c89d6` passed the real patch apply,
   upstream diff check, exact two-file scope, reverse-check, and patched Python
   compilation. An earlier official clone attempt timed out before checkout
   and produced no accepted checkout; it was not reused, and no manual cleanup
   was performed.
6. Three independent negative probes used repository-external inputs. A
   manifest-side duplicate test key was rejected before aggregation as
   expected. Two P1 probes failed: `B2-ACC-001` accepted a formal profile with
   only training attention changed to `sdpa` during profile loading and
   dry-run; `B2-ACC-002` accepted a checkpoint with
   `trainer_state.json.global_step=5` and binding `global_step=6`.
7. Verdict: **batch 2 did not pass independent CPU acceptance**. Green
   disclosed tests and patch checks do not override the two reproducible P1
   alarm failures. Details and the required lifecycle next step are in
   `BLOCKED.md`.
8. This verdict covers only local CPU contracts and failure detection. No GPU,
   training, inference, SSH, remote write, dependency installation, deletion,
   or push occurred. Repository-external acceptance inputs remain preserved
   for user review and are not permanent versioned artifacts.
