# GPU compatibility and two-task smoke test

This runbook prepares the smallest GPU-backed checks needed before a full
ImageNet-R reproduction. It separates paper facts from compatibility settings.

## Frozen provenance

- Visual-RFT commit:
  `2ffad63b25ddd79bfe25d3e046645401201c89d6`;
- Qwen model: `Qwen/Qwen2-VL-2B-Instruct` at revision
  `895c3a49bc3fa70a340399125c650a463535e71c`;
- Transformers commit:
  `336dc69d63d56f232a183a3e7f52790429b871ef`
  (`4.49.0.dev0`);
- TRL: `0.14.0`;
- PyTorch: `2.5.1+cu124`;
- FlashAttention: `2.7.4.post1`.

The Visual-RFT trainer follows the GRPO implementation introduced in TRL
0.14.0. Its upstream installer is not used because it installs moving
`main` branches and two conflicting vLLM versions. The initial RaPO path does
not use or install vLLM.

## Paper facts and smoke-only settings

The paper states that all experiments use 8 NVIDIA H100 GPUs. It also fixes
eight rollouts and trains image-classification tasks for two epochs. It does
not disclose the learning rate, per-device batch size, gradient accumulation,
maximum sequence lengths, or image resolution.

For the smoke test, undisclosed settings follow Visual-RFT's closest public
classification script and TRL 0.14.0 defaults:

- learning rate `1e-6`;
- per-device batch size `1`;
- gradient accumulation `2`;
- GRPO KL coefficient `0.04`;
- prompt/completion limits `1024`/`256`;
- maximum image pixels `401408`;
- no gradient checkpointing;
- DeepSpeed ZeRO-3;
- no vLLM and no online experiment tracker.

These are engineering starting points, not claims about the paper's hidden
configuration. A smoke run uses `max_steps`; the final reproduction uses the
paper's two epochs only after the configuration is frozen.

## Prepare the isolated training environment

Run this on the target GPU node inside `tmux`. FlashAttention compilation is
limited to four build workers to comply with the server CPU-usage rule.

```bash
cd /home/zhenglifeng/projects/RaPO
INSTALL_FLASH_ATTN=1 MAX_JOBS=4 \
  bash scripts/bootstrap_train_env.sh \
  /home/zhenglifeng/projects/Visual-RFT-RaPO-2b5561a
```

Do not run Visual-RFT's `setup.sh`.

## Model download

The model repository is about 4.43 GB. Download the pinned revision into the
personal directory before requesting a GPU slot:

```bash
mkdir -p /home/zhenglifeng/models
hf download Qwen/Qwen2-VL-2B-Instruct \
  --revision 895c3a49bc3fa70a340399125c650a463535e71c \
  --local-dir /home/zhenglifeng/models/Qwen2-VL-2B-Instruct-895c3a4
```

Record file hashes after download. If direct Hugging Face access is unavailable,
use the Qwen-owned ModelScope mirror through the repository helper:

```bash
bash scripts/download_qwen_model.sh \
  /home/zhenglifeng/models/Qwen2-VL-2B-Instruct-895c3a4
```

The helper resumes interrupted downloads and verifies every downloaded file
against SHA-256 values obtained from the pinned Hugging Face revision. It also
writes `SHA256SUMS` and `PROVENANCE.txt` beside the model. Do not use an
unverified mirror or an unfrozen `main` snapshot.

## Resource gate

The full paper result used 8 H100 GPUs. The first engineering gate should use
four 24 GB or larger Ampere/Ada GPUs, preferably RTX 4090, for a one-step
ZeRO-3 launch. RTX 3090 is a slower fallback. Do not begin with RTX 5090:
the frozen PyTorch 2.5.1/CUDA 12.4 and FlashAttention 2.7.4 stack predates
Blackwell support, so using that card would first require a separate dependency
compatibility branch. This gate is a measurement, not a promise that four cards
can complete the final run. Record peak memory per GPU and then decide whether
the 20-step test needs more or larger cards.

Immediately before every launch, inspect `nvidia-smi`, choose idle devices, and
set `GPU_IDS` explicitly. Run long commands in `tmux`.

## Launch sequence

The following paths use the already exported order-0 datasets.

One-step GRPO gate:

```bash
GPU_IDS=0,1,2,3 \
VISUAL_RFT_ROOT=/home/zhenglifeng/projects/Visual-RFT-RaPO-2b5561a \
MODEL_PATH=/home/zhenglifeng/models/Qwen2-VL-2B-Instruct-895c3a4 \
DATASET_PATH=/home/zhenglifeng/data/rapo-imagenet-r/order_0/visual_rft/task_01 \
OUTPUT_DIR=/home/zhenglifeng/outputs/rapo-smoke/grpo-task01-step1 \
bash scripts/run_imagenet_r_smoke.sh grpo 1 1
```

If the one-step gate has no OOM, NaN, distributed hang, or FlashAttention
error, repeat it for 20 steps with a new output directory.

RaPO task 1:

```bash
GPU_IDS=0,1,2,3 \
VISUAL_RFT_ROOT=/home/zhenglifeng/projects/Visual-RFT-RaPO-2b5561a \
MODEL_PATH=/home/zhenglifeng/models/Qwen2-VL-2B-Instruct-895c3a4 \
DATASET_PATH=/home/zhenglifeng/data/rapo-imagenet-r/order_0/visual_rft/task_01 \
OUTPUT_DIR=/home/zhenglifeng/outputs/rapo-smoke/rapo-task01 \
bash scripts/run_imagenet_r_smoke.sh rapo 1 20
```

RaPO task 2 must start from task 1's final model and CTAN state:

```bash
GPU_IDS=0,1,2,3 \
VISUAL_RFT_ROOT=/home/zhenglifeng/projects/Visual-RFT-RaPO-2b5561a \
MODEL_PATH=/home/zhenglifeng/outputs/rapo-smoke/rapo-task01 \
DATASET_PATH=/home/zhenglifeng/data/rapo-imagenet-r/order_0/visual_rft/task_02 \
OUTPUT_DIR=/home/zhenglifeng/outputs/rapo-smoke/rapo-task02 \
RAPO_STATE_PATH=/home/zhenglifeng/outputs/rapo-smoke/rapo-task01/rapo_state.json \
bash scripts/run_imagenet_r_smoke.sh rapo 2 20
```

After each run, confirm that no GPU process remains. Preserve the command,
Conda package list, console log, peak GPU memory, exit code, and output hashes.
