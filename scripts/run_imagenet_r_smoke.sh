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
precision="${RAPO_SMOKE_PRECISION:-bf16}"
attn_implementation="${RAPO_SMOKE_ATTN_IMPLEMENTATION:-flash_attention_2}"
max_prompt_length="${RAPO_SMOKE_MAX_PROMPT_LENGTH:-1024}"
max_completion_length="${RAPO_SMOKE_MAX_COMPLETION_LENGTH:-256}"
gradient_accumulation_steps="${RAPO_SMOKE_GRADIENT_ACCUMULATION_STEPS:-2}"
gradient_checkpointing="${RAPO_SMOKE_GRADIENT_CHECKPOINTING:-false}"
max_pixels="${RAPO_SMOKE_MAX_PIXELS:-401408}"
min_pixels="${RAPO_SMOKE_MIN_PIXELS:-3136}"
num_generations="${RAPO_SMOKE_NUM_GENERATIONS:-8}"
save_strategy="${RAPO_SMOKE_SAVE_STRATEGY:-steps}"

required_variables=(
    GPU_IDS
    VISUAL_RFT_ROOT
    MODEL_PATH
    DATASET_PATH
    OUTPUT_DIR
    RUN_MANIFEST_PATH
    EXPERIMENT_ID
    RUN_ID
    DATA_MANIFEST_PATH
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
if (( task_index == 1 )); then
    if [[ -n "${PARENT_MANIFEST_PATH:-}" || -n "${RAPO_STATE_PATH:-}" ]]; then
        echo "Task 1 forbids PARENT_MANIFEST_PATH and RAPO_STATE_PATH." >&2
        exit 2
    fi
elif [[ -z "${PARENT_MANIFEST_PATH:-}" || ! -f "${PARENT_MANIFEST_PATH}" ]]; then
    echo "Task ${task_index} requires a finalized PARENT_MANIFEST_PATH." >&2
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
if [[ "${precision}" != "bf16" && "${precision}" != "fp16" ]]; then
    echo "RAPO_SMOKE_PRECISION must be either bf16 or fp16." >&2
    exit 2
fi
if [[ "${attn_implementation}" != "flash_attention_2" &&
      "${attn_implementation}" != "sdpa" &&
      "${attn_implementation}" != "eager" ]]; then
    echo "RAPO_SMOKE_ATTN_IMPLEMENTATION must be flash_attention_2, sdpa, or eager." >&2
    exit 2
fi
if [[ "${gradient_checkpointing}" != "true" && "${gradient_checkpointing}" != "false" ]]; then
    echo "RAPO_SMOKE_GRADIENT_CHECKPOINTING must be true or false." >&2
    exit 2
fi
if [[ "${save_strategy}" != "steps" && "${save_strategy}" != "no" ]]; then
    echo "RAPO_SMOKE_SAVE_STRATEGY must be either steps or no." >&2
    exit 2
fi
for value_name in \
    max_prompt_length \
    max_completion_length \
    gradient_accumulation_steps \
    max_pixels \
    min_pixels \
    num_generations; do
    value="${!value_name}"
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer; found ${value}." >&2
        exit 2
    fi
done
if (( min_pixels > max_pixels )); then
    echo "RAPO_SMOKE_MIN_PIXELS cannot exceed RAPO_SMOKE_MAX_PIXELS." >&2
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
if [[ ! -f "${DATA_MANIFEST_PATH}" ]]; then
    echo "Data manifest not found: ${DATA_MANIFEST_PATH}" >&2
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
patch_file="${repo_root}/patches/visual_rft_2ffad63_rapo.patch"
patched_files="$(git -C "${VISUAL_RFT_ROOT}" diff --name-only)"
expected_patched_files=$'src/virft/src/open_r1/grpo_classification.py\nsrc/virft/src/open_r1/trainer/grpo_trainer.py'
if [[ "${patched_files}" != "${expected_patched_files}" ]] ||
    ! git -C "${VISUAL_RFT_ROOT}" apply --reverse --check "${patch_file}"; then
    echo "Visual-RFT worktree does not exactly match the pinned RaPO patch." >&2
    exit 1
fi

reproduction_config="${RAPO_REPRODUCTION_CONFIG:-${repo_root}/configs/independent_reproduction.json}"
if [[ ! -f "${reproduction_config}" ]]; then
    echo "Independent reproduction config not found: ${reproduction_config}" >&2
    exit 1
fi

provenance_arguments=(
    --manifest "${RUN_MANIFEST_PATH}"
    --experiment-id "${EXPERIMENT_ID}"
    --run-id "${RUN_ID}"
    --method "${method}"
    --task-index "${task_index}"
    --repo-root "${repo_root}"
    --upstream-root "${VISUAL_RFT_ROOT}"
    --patch "${patch_file}"
    --input-model "${MODEL_PATH}"
    --output-model "${OUTPUT_DIR}"
    --data-manifest "${DATA_MANIFEST_PATH}"
    --stage-dataset "${DATASET_PATH}"
    --reproduction-config "${reproduction_config}"
)
if (( task_index >= 2 )); then
    provenance_arguments+=(--parent-manifest "${PARENT_MANIFEST_PATH}")
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
        provenance_arguments+=(--input-state "${RAPO_STATE_PATH}")
        rapo_arguments+=(--rapo_state_path "${RAPO_STATE_PATH}")
    fi
fi

IFS=',' read -r -a gpu_id_array <<< "${GPU_IDS}"
nproc_per_node="${#gpu_id_array[@]}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

contract_sha256="$(
    "${conda_exe}" run --no-capture-output --name "${env_name}" \
        python -m rapo.provenance prepare-run "${provenance_arguments[@]}"
)"
if [[ ! "${contract_sha256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Run-manifest preparation did not return a canonical contract SHA256." >&2
    exit 1
fi
if [[ "${method}" == "rapo" ]]; then
    rapo_arguments+=(
        --rapo_run_id "${RUN_ID}"
        --rapo_contract_sha256 "${contract_sha256}"
    )
fi

precision_arguments=(--bf16 false --fp16 true)
if [[ "${precision}" == "bf16" ]]; then
    precision_arguments=(--bf16 true --fp16 false)
fi
save_arguments=(--save_strategy "${save_strategy}")
if [[ "${save_strategy}" == "steps" ]]; then
    save_arguments+=(--save_steps "${max_steps}")
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${repo_root}/src:${VISUAL_RFT_ROOT}/src/virft/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true

echo "Checking GPU state immediately before launch."
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader
echo "Launching ${method} task ${task_index} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}."
echo "Smoke settings: precision=${precision}, attention=${attn_implementation}, prompt=${max_prompt_length}, completion=${max_completion_length}, generations=${num_generations}, max_pixels=${max_pixels}, gradient_checkpointing=${gradient_checkpointing}."

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
    --max_prompt_length "${max_prompt_length}" \
    --max_completion_length "${max_completion_length}" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps "${gradient_accumulation_steps}" \
    --learning_rate 1e-6 \
    --beta 0.04 \
    --logging_steps 1 \
    "${precision_arguments[@]}" \
    --report_to none \
    --eval_strategy no \
    --gradient_checkpointing "${gradient_checkpointing}" \
    --attn_implementation "${attn_implementation}" \
    --max_pixels "${max_pixels}" \
    --min_pixels "${min_pixels}" \
    --max_steps "${max_steps}" \
    "${save_arguments[@]}" \
    --num_generations "${num_generations}" \
    --seed 0 \
    --data_seed 0 \
    "${rapo_arguments[@]}"

finalize_arguments=(
    --manifest "${RUN_MANIFEST_PATH}"
    --output-model "${OUTPUT_DIR}"
)
if [[ "${method}" == "rapo" ]]; then
    finalize_arguments+=(--output-state "${OUTPUT_DIR}/rapo_state.json")
fi
"${conda_exe}" run --no-capture-output --name "${env_name}" \
    python -m rapo.provenance finalize-run "${finalize_arguments[@]}"
