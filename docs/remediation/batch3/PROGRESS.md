# Batch 3 remediation progress

## Start snapshot (2026-08-02)

- Role: remediation executor only; this conversation does not perform review,
  independent acceptance, or final release.
- Approved base: `0bfb849e5eb7bd713fe7bc117eeea35b5a378b89`.
- Working branch: `codex/rapo-b03-orchestration`.
- Starting worktree: clean; the repository had no batch-3 orchestration code,
  configuration, tests, or remediation directory.
- CPU environment: `D:\anaconda3\envs\rapo-b01\python.exe`, Python 3.10.20,
  PyTorch 2.5.1+cpu, pytest 8.4.2.
- Live baseline: `python -m pytest -ra` reported `103 passed` and no skips;
  `python -m compileall -q src tests scripts` and `git diff --check` exited 0.
- The approved SHA256 values for `AGENTS.md`, the independent audit,
  `configs/formal_profile.json`, and `configs/independent_reproduction.json`
  all matched the live files exactly.
- Scope: only the approved CPU/no-GPU orchestration closure for AUD-012 and
  AUD-015. No GPU, training, inference, network, install, deletion, or push is
  authorized.

## Red checkpoint

- Tests were added before any orchestration implementation or configuration.
- Command: `python -m pytest tests/test_orchestration.py -ra`.
- Result: collection failed with `ModuleNotFoundError: No module named
  'rapo.orchestration'`; pytest reported one collection error and exited 1.
- The red suite fixes deterministic node counts, six 55-cell matrices,
  manifest-derived full-test counts, method pairing, immediate-parent chains,
  finalized/resume status behavior, wrong artifact rejection, strict prediction
  aggregation, and hand-calculated three-order mean/population-std results.
- Implementation diff at the red checkpoint: `git diff --
  configs/formal_orchestration.json src/rapo/orchestration.py
  docs/data_and_evaluation.md` produced no output; none of those implementation
  files existed or had changed. Only the new test and batch-3 tracking documents
  were present as untracked files.

## Green checkpoint

- Directed suite: `python -m pytest tests/test_orchestration.py -ra` reported
  `10 passed`.
- Full suite: `python -m pytest -ra` reported `113 passed` and no skips (the
  original 103 plus 10 collected batch-3 cases).
- During convergence an earlier full run reported `111 passed, 2 failed`
  because prediction nodes did not yet carry the corrected profile path; the
  missing field was added and both the directed and complete suites were rerun
  in full to the results above.
- `python -m compileall -q src tests scripts` exited 0.
- `bash -n` exited 0 for `run_imagenet_r_smoke.sh`,
  `run_imagenet_r_formal.sh`, and `run_imagenet_r_2080ti_smoke.sh`.
- `git diff --check` exited 0 (Git emitted only its Windows LF/CRLF working-copy
  warning for `docs/data_and_evaluation.md`).
- A real `python -m rapo.orchestration` CLI exercise used ten-stage small
  manifests for orders 0/1/2 below
  `C:\Users\Administrator\AppData\Local\Temp\rapo-b03-cli-b093maip`.
  The directory is retained and was not deleted.
- Two identical dry-runs produced plan SHA256
  `efc3f5f6d02b96c462db19da55e05509d0ae175043cdd428a7e800dbd08e1aca`
  and byte-identical JSON. The plan contained 60 training, 60 prediction, six
  metrics, and two summary nodes; six matrices bound 330 cells. The small
  order-0 cumulative expected counts were exactly `[1,2,3,4,5,6,7,8,9,10]`,
  copied from its manifest rather than a bounded subset.
- Fresh `status` exposed six independent task-1 training frontier commands;
  after materializing valid small artifacts it recognized 120 validated
  training/prediction nodes and emitted an aggregate command using the actual
  plan path. Legal `aggregate` produced six order metrics, two method summaries,
  and 60 input prediction hashes.
- Critical illegal controls failed closed: a foreign order run manifest made
  `status` exit 1 with `Run artifact ... does not match plan`; changed
  prediction lineage made `aggregate` exit 1 with `lineage does not match the
  result contract`. The directed suite separately exercises a legal same-run
  checkpoint resume and rejects a foreign checkpoint binding.
- Hand-calculated summaries matched: GRPO Last Accuracy mean/std
  `0.5/0.408248290463863`, RaPO `0.6/0.16329931618554522`; both declare
  population `ddof=0`.

## Pre-commit checkpoint

- `git status --porcelain=v1 -uall` resolved to exactly the seven approved
  whitelist paths; the outside-whitelist count was zero. `AGENTS.md` remained
  unchanged at its approved SHA256.
- The first cached scope contained only
  `configs/formal_orchestration.json`, `src/rapo/orchestration.py`, and
  `tests/test_orchestration.py`; cached `diff --check` exited 0.
- Code/config/test commit:
  `680d3d1302845b5052edcab002e809f35fada7b4` (`feat: add formal experiment
  orchestration`).
- The documentation paths remained unstaged for the required separate second
  commit. No GPU, training, inference, network, install, deletion, or push was
  executed.

## Open items

- `BLOCKED`: none.
- New non-blocking P2 observations: none.
- The known batch-2 string `require_ctan` P2 was not modified.
- Any repair round, GPU gate, or formal experiment remains subject to separate
  user approval and independent acceptance.
