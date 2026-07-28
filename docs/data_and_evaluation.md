# ImageNet-R data and continual-learning evaluation

This stage turns the paper protocol into deterministic, inspectable artifacts.
It does not download ImageNet-R and does not commit dataset files.

## Protocol boundary

The paper specifies:

- 200 ImageNet-R classes;
- 10 disjoint tasks with 20 classes per task;
- five labeled training images per class and no replay;
- evaluation over every class observed so far;
- a prompt containing only the cumulative seen-class vocabulary;
- classification accuracy based on an exact match inside `<answer>` after
  lower-casing and mapping underscores, hyphens, and periods to spaces;
- averages over three random class orders.

The paper does not publish the three class orders, the selected five images per
class, or an explicit train/test split file. Until author code or metadata is
available, this repository makes the following provisional choices:

- class-order seeds are `0`, `1`, and `2`;
- `sample_seed=0` is fixed across all three class orders;
- for each class, five deterministically ordered images are training examples
  and every remaining image is a test example.

Every choice and relative image path is stored in `manifest.json`, so a later
official split can be compared or substituted without ambiguity.

## Input layout

Extract ImageNet-R into one WNID directory per class:

```text
/home/zhenglifeng/data/imagenet-r/
├── n01443537/
│   ├── image_001.jpg
│   └── ...
└── ...
```

The preferred class map is the `README.txt` included in the official
ImageNet-R archive. It ends with all 200 WNID-label pairs:

```text
n01443537 goldfish
n01484850 great_white_shark
```

Common ImageNet JSON layouts are also accepted:

```json
{
  "n01443537": "goldfish"
}
```

```json
{
  "1": ["n01443537", "goldfish"]
}
```

The builder scans the image root and requires exactly 200 non-empty class
directories. Using the included README avoids a separate metadata download. A
1,000-class ImageNet JSON map remains valid because only WNIDs present in
ImageNet-R are used.

The official archive used for this reproduction is:

```text
URL: https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
Bytes: 2191079936
SHA-256: 18c6bf493b39a0d975d48e587437f562caab9c52ae6327dcfa9dd8eb54aa1b52
```

After extraction it contains 200 class directories, 30,000 JPEG images, and
one top-level `README.txt`.

## Build manifests

The core manifest builder uses only the Python standard library. Run it once
per class order:

```bash
for seed in 0 1 2; do
  rapo-build-imagenet-r \
    /home/zhenglifeng/data/imagenet-r \
    /home/zhenglifeng/data/imagenet-r/README.txt \
    "/home/zhenglifeng/data/rapo-imagenet-r/order_${seed}" \
    --class-order-seed "${seed}" \
    --sample-seed 0
done
```

Each output contains one `manifest.json` with:

- the seeds and protocol dimensions;
- the exact global class order and task assignment;
- current-task training paths and held-out test paths for every class;
- current-task and cumulative test sizes;
- the cumulative class-name vocabulary used in each task prompt.

Generated data and manifests live below `data/` or another personal server
directory and are intentionally excluded from Git.

## Export for Visual-RFT

Visual-RFT consumes a Hugging Face `DatasetDict` with `image`, `problem`, and
`solution` fields. Install the lightweight export dependencies in the
environment that will prepare data:

```bash
python -m pip install --editable '.[data]'
```

Then add `--export-visual-rft` to the builder command. It writes:

```text
visual_rft/task_01/
visual_rft/task_02/
...
visual_rft/task_10/
```

Each task-stage dataset contains:

- `train`: only the current task's 100 training examples;
- `test`: all test examples from tasks observed so far;
- `test_task_01`, ..., `test_task_T`: per-task splits used to construct the
  accuracy matrix.

All rows at stage `T` use the same cumulative vocabulary through task `T`.
Image fields retain absolute paths to the source ImageNet-R tree, so the source
tree must remain at the same server path during training and evaluation.

## Prediction and metric formats

The evaluator accepts JSONL prediction rows:

```json
{"after_task": 1, "eval_task": 1, "completion": "<think>...</think><answer>goldfish</answer>", "target": "goldfish"}
```

After each trained task `t`, generate predictions separately on
`test_task_01` through `test_task_t` and record both task indices. Then run:

```bash
rapo-evaluate predictions.jsonl \
  --num-tasks 10 \
  --output continual_metrics.json
```

It also accepts already aggregated JSONL cells:

```json
{"after_task": 1, "eval_task": 1, "correct": 120, "total": 150}
```

The lower-triangular matrix must be complete. Last Accuracy is calculated as
the micro-average over all final-task test samples:

```text
A = sum_j correct[T, j] / sum_j total[T, j]
```

Forgetting is the macro-average historical-best drop over the previous tasks:

```text
F = mean over j < T of (max over t >= j accuracy[t, j] - accuracy[T, j])
```

The output includes decimal and percentage forms of both metrics, plus
accuracy, correct-count, and total-count matrices.
