# RTX 2080 Ti six-task comparison and stability boundary (2026-07-30--31)

## Scope and status

Status: **completed as an engineering run, paused at a numerical-stability
boundary**. The bounded order-0 comparison reached task 6 for both GRPO and
RaPO, including save, reload, and the same fixed evaluation. It is not a paper
reproduction result: it uses one class order, one run, 20 optimizer steps per
task, 20 test images per class, FP16, cuDNN disabled, and 8 x RTX 2080 Ti rather
than the paper's 8 x H100 setup.

Task-6 training used RaPO repository commit `b26b496` and the pinned Visual-RFT
commit `2ffad63b25ddd79bfe25d3e046645401201c89d6`. The task-6 prompt limit was
raised to 1,024 because the cumulative class-vocabulary prompt reached 567
tokens. All other bounded-run settings remained shared between the methods.

The equation-to-trainer audit performed after task 5 found no material RaPO
integration mismatch. Task 6 therefore tested the audited implementation, not
an unreviewed continuation.

## FP16 compatibility investigation

The first RaPO task-6 run completed all 20 optimizer steps with dynamic-FP16
initial scale 1,024. The corresponding GRPO run skipped an optimizer update at
step 8, visible as a repeated learning-rate value and stale gradient norm. That
run was stopped and was not used in the paired comparison.

Both methods were restarted from their respective task-5 models with the same
explicit scale-512 DeepSpeed configuration,
`configs/deepspeed_zero3_cpu_offload_scale9.json`. They then completed all 20
expected learning-rate values without a skipped update, OOM, NaN, or traceback.
This is the formal task-6 pair.

| Method | Initial FP16 scale | Runtime (s) | Mean loss | Final gradient norm |
| --- | ---: | ---: | ---: | ---: |
| GRPO | 512 | 755.41 | 0.000424 | 0.1823 |
| RaPO | 512 | 769.82 | 0.000547 | 0.7709 |

The final RaPO state records `task_index=6`, `normalizer.updates=228`, and
`normalizer.ema_std=0.577847948133668`. The update count continues from 190
after task 5 plus 38 task-6 forward passes.

Selected task-6 retention measurements were:

| Step | Retention reward | Drift |
| ---: | ---: | ---: |
| 1 | 1.000000 | 0.000000 |
| 2 | 0.983154 | 0.001017 |
| 4 | 0.951172 | 0.004717 |
| 6 | 0.914551 | 0.005785 |
| 8 | 0.881836 | 0.009483 |
| 9 | 0.805908 | 0.048096 |
| 10 | 0.929443 | 0.021283 |
| 12 | 0.868408 | 0.018093 |
| 14 | 0.757324 | 0.035706 |
| 16 | 0.841309 | 0.079315 |
| 17 | 0.932861 | 0.003925 |
| 18 | 0.831299 | 0.042652 |
| 19 | 0.853027 | 0.056965 |
| 20 | 0.791504 | 0.026833 |

## Formal scale-512 evaluation

Evaluation used greedy generation, at most 32 new tokens, and the first 20 test
images per class after sorting by manifest-relative path. Each cell contains
400 examples and the final row contains 2,400. Both methods padded the same
undersized image, `n07873807/deviantart_12.jpg`, from 27 x 30 to 28 x 30.

### GRPO

| After task | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 380/400 | - | - | - | - | - |
| 2 | 372/400 | 379/400 | - | - | - | - |
| 3 | 364/400 | 380/400 | 371/400 | - | - | - |
| 4 | 345/400 | 375/400 | 363/400 | 365/400 | - | - |
| 5 | 344/400 | 375/400 | 346/400 | 361/400 | 366/400 | - |
| 6 | 288/400 | 357/400 | 330/400 | 355/400 | 353/400 | 357/400 |

- Last Accuracy: `2040/2400 = 85.00%`
- Forgetting: `8.95%`

### RaPO

| After task | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 380/400 | - | - | - | - | - |
| 2 | 374/400 | 379/400 | - | - | - | - |
| 3 | 359/400 | 381/400 | 370/400 | - | - | - |
| 4 | 281/400 | 370/400 | 360/400 | 375/400 | - | - |
| 5 | 281/400 | 363/400 | 336/400 | 366/400 | 363/400 | - |
| 6 | 93/400 | 187/400 | 240/400 | 342/400 | 332/400 | 372/400 |

- Last Accuracy: `1566/2400 = 65.25%`
- Forgetting: `33.75%`

RaPO finishes with 474 fewer correct predictions than GRPO. Its final
per-task deficits are 195, 170, 90, 13, 21, and -15 predictions for tasks 1
through 6; the negative task-6 deficit denotes a 15-prediction RaPO advantage
on the newest task. This pattern is old-task forgetting rather than failure to
learn task 6.

## Paired result audit

The two task-6 files contain the same 2,400 ordered
`(after_task, eval_task, relative_path, target)` keys, 400 per task, with no
duplicates or missing parsed answers.

| Eval task | Both correct | GRPO only | RaPO only | Both wrong |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 92 | 196 | 1 | 111 |
| 2 | 186 | 171 | 1 | 42 |
| 3 | 239 | 91 | 1 | 69 |
| 4 | 333 | 22 | 9 | 36 |
| 5 | 323 | 30 | 9 | 38 |
| 6 | 356 | 1 | 16 | 27 |

Representative old-task errors include `axolotl` becoming `newt`, `stingray`
becoming `mantis`, and `pelican` becoming `bird`. The ordered-key audit, parsed
answers, and recent-task accuracy rule out an evaluator join error or missing
model as explanations for the gap.

## Loss-scale and inference-hardware diagnostics

The completed scale-1,024 RaPO model was retained as a sensitivity diagnostic.
Its task-6 final row was `[145, 260, 281, 355, 338, 366]`, giving Last Accuracy
`72.7083%` and Forgetting `24.50%`. It is better than the formal scale-512 RaPO
run but remains well below the scale-512 GRPO result. No scale-1,024 GRPO pair
exists because that run skipped an optimizer update, so this is not a fair
method comparison.

To isolate inference hardware, the formal scale-512 RaPO model was evaluated
again on one RTX 4090. The 4090 final row was
`[92, 187, 240, 343, 332, 372]`: the same total `1566/2400`, Last Accuracy
`65.25%`, and Forgetting `33.75%` as the 2080 Ti evaluation. Ordered sample keys
were identical. Across 2,400 samples, only four correctness decisions differed;
the task-1 and task-4 count changes cancelled. The large task-6 gap is therefore
not explained by 2080 Ti inference hardware.

Together, these diagnostics show that the reduced FP16 path is materially
sensitive to loss scaling while the observed forgetting is stable across the
two inference GPU architectures. A BF16 training check on newer GPUs is needed
before treating the scale-512 behavior as an algorithmic result.

## Remote artifacts

Retained task-6 models:

- `/home/zhenglifeng/outputs/rapo-smoke/2080ti-grpo-task06-step20-n8-scale9-nocudnn-prompt1024-nosave`
- `/home/zhenglifeng/outputs/rapo-smoke/2080ti-rapo-task06-step20-n8-scale9-nocudnn-prompt1024-nosave`
- `/home/zhenglifeng/outputs/rapo-smoke/2080ti-rapo-task06-step20-n8-scale10-nocudnn-prompt1024-nosave`

Predictions, metrics, and audits are under
`/home/zhenglifeng/results/rapo-smoke/eval-20-per-class-step20/`.

| Artifact | SHA-256 |
| --- | --- |
| `grpo-after6.jsonl` | `552588ce72e5f604f7821059e553bb1269d1dff67025607413e17d342782ac05` |
| `rapo-after6.jsonl` | `7d6b458daa4c1e38b6c28a7ba16f7a2319d681cdc598e18db700a23ee0a49a94` |
| `grpo-metrics-6task.json` | `6bdeb2e4016686a9a33e1a408986867469650fef72b3995c606a21b247e38665` |
| `rapo-metrics-6task.json` | `abecebce6fc9e557a04919efd63888101828a679d0300aa2f5081bae68689728` |
| `task06-paired-audit.json` | `2caa42323d040b6b386bc53ca6de25cbf8411c0f612ab8ac549b436111e1571f` |
| `rapo-scale10-after6.jsonl` | `8ff2a65983b08a4d60b6fe537900b38eb90597edf260c55e5e7c471968b0f895` |
| `rapo-scale10-metrics-6task.json` | `33013e5ce3660e8556bf082c3cf98fb5f5807e67f20ef59aeb5817dabaac7699` |
| `rapo-scale9-4090-after6.jsonl` | `d36586fffb4e81ee6d0b5f12a3f508b66ac51a184b3ff2dc4077ccc7be4bf281` |
| `rapo-scale9-4090-metrics-6task.json` | `dd7e5d19db5261c1d43846e0e6df534026778e063533eda4af52c94963756b2e` |
| `task06-scale9-inference-hardware-audit.json` | `878a5fe1b15c065cc25388d8aacb88c463ecb5ef2673cb4669228e38b8dd10eb` |

## Pause and resume decision

Task 7 has not been started. The present chain is paused after task 6 because:

1. both methods completed a technically fair scale-512 pair;
2. the RaPO result is strongly unfavorable and materially loss-scale-sensitive;
3. the inference-hardware control did not remove the gap; and
4. continuing the same reduced FP16 path would add cost without resolving the
   main validity question.

On resume, recheck `gpustat2` and request an idle, homogeneous 8-GPU group with
BF16 support. Preserve world size 8 and rollout grouping; silently reducing the
card count would change gradient accumulation and distributed CTAN semantics.
Run a paired task-6 BF16 stability gate from the same task-5 GRPO/RaPO models
before deciding whether to continue to task 7. At the last cluster survey no
complete high-end 8-GPU group was free, so availability must be checked again
rather than assumed.
