#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: bash scripts/run_imagenet_r_smoke.sh <grpo|rapo> <task-index> [max-steps]" >&2
    exit 2
fi

method="$1"
task_index="$2"
max_steps="${3:-20}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-${HOME}/miniforge3/bin/conda}"
env_name="${RAPO_TRAIN_ENV:-rapo-train}"

required_variables=(
    GPU_IDS
    VISUAL_RFT_ROOT
    MODEL_PATH
    DATASET_PATH
    OUTPUT_DIR
)
for variable_name in "${required_variables[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "${variable_name} must be set." >&2
        exit 2
    fi
done

if [[ "${method}" != "grpo" && "${method}" != "rapo" ]]; then
    echo "Method must be either grpo or rapo." >&2
    exit 2
fi
if [[ ! "${task_index}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Task index must be a positive integer." >&2
    exit 2
fi
if [[ ! "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "max-steps must be a positive integer." >&2
    exit 2
fi
if [[ ! "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "GPU_IDS must be an explicit comma-separated list such as 0,1,2,3." >&2
    exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model directory not found: ${MODEL_PATH}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Dataset directory not found: ${DATASET_PATH}" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" && ! -d "${OUTPUT_DIR}" ]]; then
    echo "OUTPUT_DIR exists and is not a directory: ${OUTPUT_DIR}" >&2
    exit 1
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "OUTPUT_DIR already exists and is not empty: ${OUTPUT_DIR}" >&2
    exit 1
fi

entrypoint="${VISUAL_RFT_ROOT}/src/virft/src/open_r1/grpo_classification.py"
trainer_path="${VISUAL_RFT_ROOT}/src/virft/src/open_r1/trainer/grpo_trainer.py"
deepspeed_config="${DEEPSPEED_CONFIG:-${VISUAL_RFT_ROOT}/src/virft/local_scripts/zero3.json}"
if [[ ! -f "${entrypoint}" || ! -f "${trainer_path}" || ! -f "${deepspeed_config}" ]]; then
    echo "Visual-RFT entrypoint or DeepSpeed config is missing." >&2
    exit 1
fi
pinned_visual_rft="2ffad63b25ddd79bfe25d3e046645401201c89d6"
actual_visual_rft="$(git -C "${VISUAL_RFT_ROOT}" rev-parse HEAD)"
if [[ "${actual_visual_rft}" != "${pinned_visual_rft}" ]]; then
    echo "Visual-RFT must be checked out at ${pinned_visual_rft}; found ${actual_visual_rft}." >&2
    exit 1
fi
if ! grep -q "classification_answer_is_correct" "${entrypoint}" ||
    ! grep -q "rapo_config" "${trainer_path}"; then
    echo "The RaPO patch is not present in the pinned Visual-RFT checkout." >&2
    exit 1
fi

rapo_arguments=(--rapo_enabled false)
if [[ "${method}" == "rapo" ]]; then
    rapo_arguments=(
        --rapo_enabled true
        --rapo_task_index "${task_index}"
        --rapo_retention_alpha 20.0
        --rapo_retention_weight 0.5
        --rapo_ctan_beta 0.999
        --rapo_ctan_epsilon 0.0001
    )
    if (( task_index >= 2 )); then
        if [[ -z "${RAPO_STATE_PATH:-}" || ! -f "${RAPO_STATE_PATH}" ]]; then
            echo "Task ${task_index} RaPO requires RAPO_STATE_PATH from the previous task." >&2
            exit 2
        fi
        rapo_arguments+=(--rapo_state_path "${RAPO_STATE_PATH}")
    fi
fi

IFS=',' read -r -a gpu_id_array <<< "${GPU_IDS}"
nproc_per_node="${#gpu_id_array[@]}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${repo_root}/src:${VISUAL_RFT_ROOT}/src/virft/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

echo "Checking GPU state immediately before launch."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
echo "Launching ${method} task ${task_index} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."

"${conda_exe}" run --no-capture-output --name "${env_name}" \
    torchrun \
    --nproc_per_node "${nproc_per_node}" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    --master_port "${MASTER_PORT:-29501}" \
    "${entrypoint}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_PATH}" \
    --dataset_train_split train \
    --dataset_test_split test \
    --deepspeed "${deepspeed_config}" \
    --max_prompt_length 1024 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --logging_steps 1 \
    --bf16 true \
    --report_to none \
    --eval_strategy no \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --min_pixels 3136 \
    --max_steps "${max_steps}" \
    --save_steps "${max_steps}" \
    --num_generations 8 \
    --seed 0 \
    --data_seed 0 \
    "${rapo_arguments[@]}"
