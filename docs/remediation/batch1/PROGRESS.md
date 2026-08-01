# Batch 1 progress

1. Goal: close token-boundary, CTAN-clock, and artifact-lineage semantics before GPU work.
2. Order: baseline -> reproduction contract -> CTAN transaction -> provenance/export -> acceptance -> commit.
3. Largest risk: pinned Visual-RFT patch integration cannot be runtime-tested until Torch is available.
4. Task 0 partial: branch, HEAD, clean tree, both specification hashes, 29 test functions, and compile match the handoff.
5. Task 0 resolved: `rapo-b01` is Python 3.10.20, NumPy 1.26.4, Torch 2.5.1+cpu, pytest 8.4.2, and editable RaPO.
6. Environment choice: CPU Torch is acceptance-only after cu124 timeouts; it is not GPU-readiness or formal-training evidence.
7. Task 1 complete: frozen config/decisions reject wrong scope, correction, reuse, clipping, and KL choices.
8. Task 2 complete in code: prompt fail-fast and CTAN successful-step transaction cover GA2/4, skip/eval, ranks, and state continuity.
9. Task 3 complete in code: canonical prepared/finalized manifests and stage bindings reject task/model/state/data/config/parent swaps.
10. Historical executor self-check at `907345e`: 65 passed/0 skipped, compile/bash/diff 0, reverse swaps nonzero, clean patch apply/diff 0, hashes and allowlist intact. This was not independent acceptance.

## Batch 1 manifest self-hash repair (2026-08-01)

1. Goal: reject a run manifest inside the output-model directory before any artifact write or training while preserving full-directory identity checks.
2. Order: verified baseline -> regression red -> centralized prepare gate -> docs -> regression/full/patch checks -> separate code and report commits.
3. Baseline rerun: `907345e20bf613a88b32c4451bdc803e320d725f`, only `AGENTS.md` untracked, 65 passed/0 skipped.
4. Largest risk: path containment must use resolved path semantics without changing manifest schema or model-directory hashing.
5. Regression red rerun: overlap test failed at `provenance.py:393` with `FileNotFoundError` before any location gate; sibling-prefix and existing identity flow passed.
6. Regression green: centralized resolved-path containment gate added before reads/writes; both overlap fail-fast and sibling-prefix identity tests passed (2 passed).
7. CPU self-test green: provenance 19 passed; full suite 67 passed/0 skipped; compile, launcher syntax, and project diff checks returned 0.
8. Fixed patch green: clean WSL checkout at `2ffad63b25ddd79bfe25d3e046645401201c89d6`; apply and upstream `diff --check` returned 0.
9. Code/test/user-doc commit: `10e43c229515ec905bc8a9cd6aca0b4d8568adb3`; launcher inherits the centralized gate without modification.
10. Delivery-time blocker: none. These were executor self-tests only; independent acceptance was still pending at delivery and was later completed as recorded below.

## Independent CPU acceptance (2026-08-01)

Earlier attempts remain historical only: the first was a role-mixed
pre-diagnostic that found the manifest self-hash defect, while the second
stopped on a stale target that expected `fb29128` and an uncommitted
`AGENTS.md`. Neither has formal acceptance effect, and neither result was
inherited below.

1. Acceptance target: `655f88269ed75c0bcab3844a4412313f9342239d`; Windows Git reported a clean worktree before and after acceptance.
2. Environment: Python 3.10.20, Torch 2.5.1+cpu, pytest 8.4.2. This environment is CPU acceptance-only.
3. Disclosed checks rerun by the independent acceptance Codex: provenance 19 passed, full suite 67 passed, 0 skipped; compileall, launcher Bash syntax, current diff check, and `42b1a14..655f882` diff check all returned 0.
4. Fixed Visual-RFT patch: a new clean checkout at `2ffad63b25ddd79bfe25d3e046645401201c89d6` was created outside the repository from a verified Git object source whose origin is the official Visual-RFT URL. Apply, upstream `diff --check`, the two-file allowlist, and reverse-check all passed; no previously patched worktree was reused.
5. Three independently designed negative checks covered a resolved-path alias at the manifest/output boundary, non-finite CTAN input plus a successful/skipped/successful transaction sequence, and post-finalization parent artifact drift. Invalid inputs raised stable errors and paired legal inputs succeeded.
6. Verdict: batch 1 passed independent technical acceptance for its local CPU scope. No P0/P1 acceptance failure was found.
7. Non-claims: this does not establish CUDA/GPU readiness, rerun any training or inference, close other audit findings, complete the paper reproduction, or authorize BF16/Task 7/formal 10-task work.
8. Acceptance checkout, caches, and probe inputs were external system-temporary data. They are not permanent artifacts and their paths are intentionally not part of the acceptance contract.
9. Repository state after acceptance remained at the target HEAD with no tracked or untracked change and no repository-root `visual-rft-clean/`.
