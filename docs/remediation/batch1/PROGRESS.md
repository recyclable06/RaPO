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
