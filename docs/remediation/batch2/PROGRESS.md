# Batch 2 progress

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
