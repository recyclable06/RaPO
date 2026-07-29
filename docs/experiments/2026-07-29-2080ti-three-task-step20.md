# RTX 2080 Ti 20-step three-task comparison (2026-07-29)

## Scope and status

Status: **passed** for a three-task GRPO/RaPO engineering gate on 8 x RTX
2080 Ti. This extends the matching two-task experiment with the same order-0
ImageNet-R split, reduced generation budget, and fixed 20-image-per-class
evaluation subset. It remains a bounded reproduction check rather than a paper
result.

Task-3 training used RaPO repository commit `1818a98`, the pinned Visual-RFT
commit `2ffad63b25ddd79bfe25d3e046645401201c89d6`, and the same FP16, SDPA,
ZeRO-3 CPU optimizer offload, gradient accumulation 2, gradient checkpointing,
512 prompt tokens, 32 completion tokens, 100,352 image pixels, and eight
generations used by the two-task gate. Evaluation used commit `e4da23d`.

## Training results

Both methods started from their own task-2 final model and completed all 20
optimizer steps on task 3. No OOM, NaN, traceback, or skipped FP16 update was
observed.

| Method | Runtime (s) | Mean loss | Final gradient norm |
| --- | ---: | ---: | ---: |
| GRPO | 750.37 | 0.000231 | 0.0366 |
| RaPO | 890.54 | 0.000303 | 1.1808 |

The final RaPO state records `task_index=3`, `normalizer.updates=114`, and
`normalizer.ema_std=0.6144286799923909`. The update count is the expected
continuation from 76 after task 2 plus 38 task-3 forward passes.

Selected task-3 retention measurements were:

| Step | Retention reward | Drift |
| ---: | ---: | ---: |
| 1 | 1.00000 | 0.00000 |
| 5 | 0.89282 | 0.01201 |
| 10 | 0.89478 | 0.02103 |
| 15 | 0.93433 | 0.00483 |
| 20 | 0.88965 | 0.01240 |

DeepSpeed reported repeated allocator-cache flushes under high memory pressure.
They reduced task-3 throughput but did not interrupt either run or change the
acceptance result.

## Evaluation results

Evaluation used greedy generation, at most 32 new tokens, and the first 20 test
images per class after sorting by manifest-relative path. Each accuracy cell
therefore contains 400 examples.

### GRPO

| After task | Task 1 | Task 2 | Task 3 |
| ---: | ---: | ---: | ---: |
| 1 | 380/400 | - | - |
| 2 | 372/400 | 379/400 | - |
| 3 | 364/400 | 380/400 | 371/400 |

- Last Accuracy: `1115/1200 = 92.9167%`
- Forgetting: `2.0%`

### RaPO

| After task | Task 1 | Task 2 | Task 3 |
| ---: | ---: | ---: | ---: |
| 1 | 380/400 | - | - |
| 2 | 374/400 | 379/400 | - |
| 3 | 359/400 | 381/400 | 370/400 |

- Last Accuracy: `1110/1200 = 92.5%`
- Forgetting: `2.625%`

RaPO's small retention advantage after task 2 did not persist through task 3
on this subset. At the final stage it produced five fewer task-1 predictions,
one more task-2 prediction, and one fewer task-3 prediction than GRPO. The net
difference is five predictions out of 1,200 and covers one class order, so it
does not establish a method ranking or contradict the paper's broader result.

## Integration issues found

The first distributed task-3 launch failed before training because all eight
ranks concurrently wrote the same Hugging Face `Dataset.map` cache on the
shared filesystem. Commit `1818a98` wraps conversation preprocessing in
`training_args.main_process_first`; a cache-cold rerun showed rank 0 writing
once and the remaining ranks loading after the barrier.

The first paired evaluation reached 1,172/1,200 predictions before both methods
encountered `n07873807/deviantart_12.jpg`, whose 27-pixel width is below
Qwen2-VL's 28-pixel spatial factor. Commit `e4da23d` pads only undersized image
dimensions to the processor factor. A full subset scan found this single image,
which changed from 27 x 30 to 28 x 30. The evaluator's atomic output policy
prevented partial JSONL files from being mistaken for complete results.

## Remote artifacts

The retained task-3 models are:

- `/home/zhenglifeng/outputs/rapo-smoke/2080ti-grpo-task03-step20-n8-scale11-nosave`
- `/home/zhenglifeng/outputs/rapo-smoke/2080ti-rapo-task03-step20-n8-scale11-nosave`

Predictions and three-task metrics are under
`/home/zhenglifeng/results/rapo-smoke/eval-20-per-class-step20/`, using the
`*-after3.jsonl`, `*-predictions-3task.jsonl`, and `*-metrics-3task.json`
names. Full success and retained failure logs are under
`/home/zhenglifeng/logs/rapo-smoke/`.

## Decision

The three-task engineering gate passes, while the bounded accuracy comparison
does not favor RaPO. The next reproduction step is to continue the same paired
configuration through tasks 4-10, exporting only the next task stage each time
and preserving the one-order result before considering broader class orders or
larger budgets.
