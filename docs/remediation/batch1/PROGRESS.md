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
10. Acceptance green: 65 passed/0 skipped, compile/bash/diff 0, reverse swaps nonzero, clean patch apply/diff 0, hashes and allowlist intact.

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
10. Current blocker: none. These are executor self-tests only; independent acceptance must rerun the full disclosed checks and hidden negative probes.
