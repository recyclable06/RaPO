# Five-task implementation audit (2026-07-30)

## Trigger and scope

The bounded order-0 comparison favored GRPO after both tasks 4 and 5. This
activated the predeclared implementation-versus-paper audit before extending
the chain. The audit covers the RaPO equations, trainer data flow, frozen
anchor, CTAN state, classification reward, task construction, evaluation, and
the reduced 2080 Ti launch configuration.

Status: **no material RaPO implementation mismatch found through task 5**.
This result supports continuing the engineering chain, but it does not remove
the disclosed reproduction uncertainties or turn the five-task result into a
paper-level comparison.

## Equation-to-code audit

The paper PDF was checked directly at equations (2) through (6) and Appendix
B.2. The implementation matches the stated operations:

| Paper operation | Implementation check |
| --- | --- |
| Eq. (2): mean actor-minus-anchor token log-probability over generated tokens | `trajectory_drift` applies the completion mask and divides by each trajectory length. |
| Eq. (2): one-sided `max(..., 0)` | The mean log-ratio is clamped at zero. |
| Eq. (3): detached `exp(-alpha * drift)` | Drift and retention reward are detached; `alpha=20`. |
| Eq. (4): `R_task + lambda * R_ret` before advantage computation | The controller shapes rewards before calling CTAN; `lambda=0.5`. |
| Eq. (5): persistent reward-standard-deviation EMA | The normalizer uses `beta=0.999`, is updated on every rollout microbatch, and is serialized across task boundaries. |
| Eq. (6): within-group mean and CTAN denominator | The numerator retains the local prompt's eight-rollout group mean; the denominator uses the globally gathered current reward standard deviation and persistent EMA. |

The exact-match classification verifier lower-cases class names and maps
underscores, hyphens, and periods to spaces. The format verifier requires the
complete `<think>...</think><answer>...</answer>` structure. Their sum is used
as `R_task`, matching the paper's classification reward definition.

Unit tests separately exercise masking, length normalization, truncation,
detachment, reward addition, group centering, state persistence, and global
standard-deviation handling.

## Trainer and task-transition audit

For task `t >= 2`, each method starts from its own task-`t-1` final model. With
ZeRO-3 active, the trainer constructs a separate frozen reference model from
that same input model directory before training begins. RaPO uses its
per-token log probabilities as the preceding-task anchor, while the existing
GRPO KL term uses the same fixed reference model. Retention reward is inserted
before group-relative advantage computation.

The final task-5 RaPO state contains `task_index=5`, `updates=190`, and
`ema_std=0.5882913156399567`. This is the expected continuation of 38 CTAN
updates per bounded task. All ranks receive the same globally gathered reward
standard deviation, so their in-memory CTAN states evolve identically; rank 0
writes the final serialized state.

The deterministic dataset exports contain only the current task's 100 training
examples, use all cumulative seen classes in the prompt, and expose no future
class names. Task 5 contains 20 training classes and 100 cumulative test
classes. The paired after-task-5 audit found identical ordered evaluation keys,
no duplicates, and no missing answers. The metric implementation uses the
paper's final-stage Last Accuracy and historical-best Forgetting definitions.

## Reduced-configuration findings

The task-5 compatibility changes - disabling cuDNN and lowering the initial
dynamic FP16 scale - were applied to both methods. They alter the 2080 Ti
execution path but not the RaPO equations. Successful logs contain the complete
20-value learning-rate sequence and no skipped update.

The audit did identify a boundary in the reduced launcher. Visual-RFT's pinned
trainer does not safely handle a multimodal prompt longer than
`max_prompt_length`: it slices the local prompt ID and mask variables but sends
the original inputs to generation. The first five task exports were measured
with the actual processor and training images:

| Task | Minimum tokens | Maximum tokens | Over 512 |
| ---: | ---: | ---: | ---: |
| 1 | 202 | 268 | 0/100 |
| 2 | 247 | 325 | 0/100 |
| 3 | 318 | 375 | 0/100 |
| 4 | 386 | 449 | 0/100 |
| 5 | 418 | 508 | 0/100 |

Therefore this upstream behavior did not affect tasks 1 through 5. The
task-6 text prompt adds 59 tokenizer tokens relative to task 5, so a 512-token
limit is no longer safe. Across all ten cumulative vocabularies, text-side
length grows from 143 tokens at task 1 to 703 at task 10. Using
`RAPO_SMOKE_MAX_PROMPT_LENGTH=1024` avoids the inconsistent truncation path for
the present protocol under the configured image-pixel cap.

## Remaining disclosed uncertainties

The audit cannot verify choices absent from the paper or unreleased official
code:

- the authors' three class orders and exact 5-shot samples;
- undisclosed optimizer, batch, precision, and generation settings;
- CTAN initialization from the first observed standard deviation;
- global rather than per-rank `sigma_batch`;
- whether CTAN should update on every gradient-accumulation microbatch or only
  once per optimizer step;
- the effect of the reduced 20-step budget versus the paper's two-epoch H100
  runs.

These remain explicit assumptions, not confirmed paper facts.

## Decision

Proceed to task 6 without changing the RaPO algorithm. Export and validate the
task-6 data, set the prompt limit to 1024 for both methods, retain the shared
task-5 2080 Ti compatibility controls, and first run a one-step RaPO load and
update gate on the eight idle GPUs. Run the full paired task-6 continuation only
if that gate completes without OOM, numerical failure, prompt truncation, or a
skipped optimizer update. A failure at this gate is the point to request a
higher-memory GPU rather than weakening the protocol further.
