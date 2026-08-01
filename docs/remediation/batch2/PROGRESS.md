# Batch 2 progress

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
