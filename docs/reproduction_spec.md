# RaPO reproduction specification v0

This document is the executable contract for the first reproduction milestone.
It distinguishes paper-stated facts from implementation assumptions.

## Scope

- Benchmark: ImageNet-R class-incremental classification.
- Backbone: Qwen2-VL-2B.
- Protocol: 200 classes, 10 tasks, 20 disjoint classes per task, 5 training
  examples per class, no replay of previous-task training examples.
- Training: sequential tasks, 2 epochs per task.
- Initial comparison: GRPO versus RaPO under the same non-RaPO settings.
- Final reporting: three random class orders, Last Accuracy, Forgetting, and the
  full task-by-task accuracy matrix.
- Out of scope: OPD extensions, object detection, video classification, and
  domain-incremental learning.

## Paper-locked RaPO settings

- rollout group size: `n = 8`;
- retention sensitivity: `alpha = 20`;
- retention weight: `lambda = 0.5`;
- CTAN smoothing coefficient: `beta = 0.999`;
- activate Retention Reward starting from task 2;
- freeze the final policy of task `t-1` as the anchor for all of task `t`;
- persist CTAN state across task boundaries.

For generated trajectory `y`, the implementation computes:

```text
drift = max(mean_s(log pi_actor(y_s) - log pi_anchor(y_s)), 0)
R_ret = exp(-alpha * drift)
R_total = R_task + lambda * R_ret
advantage = (R_total - group_mean(R_total)) / (CTAN_std + epsilon)
```

The trajectory drift and retention reward are detached scalar feedback. They
must not introduce a direct gradient path to the actor logits.

## Recorded implementation assumptions

These choices are not fully specified in the paper and must remain explicit:

- CTAN is initialized from the first observed batch reward standard deviation.
- Standard deviation uses Bessel correction (`correction=1`) to match the
  PyTorch behavior in the pinned Visual-RFT trainer.
- CTAN's batch standard deviation must be computed from globally gathered
  rewards before distributed trainer integration.
- `epsilon = 1e-4`, matching the public GRPO baseline.
- The paper does not disclose its three class orders or selected 5-shot
  examples. Pending an official release, use class-order seeds `0`, `1`, and
  `2`, keep `sample_seed = 0`, select five deterministically ordered images per
  class for training, and use the remaining images for testing.
- Last Accuracy is computed as the micro-average over every final-stage test
  sample. Forgetting is the mean historical-best accuracy drop over tasks
  `1..T-1`.
- Undisclosed training hyperparameters will start from the closest Visual-RFT
  classification configuration and be explored only on a fixed development
  task order before the final configurations are frozen.

## Acceptance gates

1. Core component tests pass on CPU.
2. A reduced GRPO run completes 20-50 steps without OOM or NaN.
3. A two-task RaPO run verifies anchor loading, detached retention rewards,
   CTAN persistence, checkpoint resume, and evaluation output.
4. A full 10-task single-seed run shows a lower-forgetting trend than GRPO.
5. Three frozen class orders produce final mean and standard deviation.
