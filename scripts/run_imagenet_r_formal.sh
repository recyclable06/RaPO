#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && "$3" != "--dry-run" ) ]]; then
    echo "Usage: bash scripts/run_imagenet_r_formal.sh <grpo|rapo> <task-index> [--dry-run]" >&2
    exit 2
fi

for variable_name in $(compgen -e); do
    if [[ "${variable_name}" == RAPO_SMOKE_* ||
          "${variable_name}" == "RAPO_CUDNN_ENABLED" ||
          "${variable_name}" == "RAPO_FP16_INITIAL_SCALE_POWER" ||
          "${variable_name}" == "DEEPSPEED_CONFIG" ]]; then
        echo "Formal runner rejects legacy environment variable ${variable_name}." >&2
        exit 2
    fi
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
profile_path="${repo_root}/configs/formal_profile.json"
export RAPO_RUN_MODE=formal
export RAPO_EXPERIMENT_PROFILE="${profile_path}"
export RAPO_FORMAL_NUM_TRAIN_EPOCHS=2
export RAPO_FORMAL_PRECISION=bf16
export RAPO_FORMAL_ATTN_IMPLEMENTATION=flash_attention_2
export RAPO_FORMAL_MAX_PROMPT_LENGTH=1024
export RAPO_FORMAL_MAX_COMPLETION_LENGTH=256
export RAPO_FORMAL_GRADIENT_ACCUMULATION_STEPS=2
export RAPO_FORMAL_GRADIENT_CHECKPOINTING=true
export RAPO_FORMAL_MAX_PIXELS=401408
export RAPO_FORMAL_MIN_PIXELS=3136
export RAPO_FORMAL_NUM_GENERATIONS=8
export RAPO_FORMAL_SAVE_STRATEGY=steps
export RAPO_FORMAL_SAVE_STEPS=5
export RAPO_FORMAL_DEEPSPEED_CONFIG="${repo_root}/configs/deepspeed_zero3_formal_bf16.json"

echo "formal profile status: pending_hardware_gate" >&2
if [[ "${3:-}" == "--dry-run" ]]; then
    if [[ ! "${RAPO_TRAIN_SAMPLE_COUNT:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "RAPO_TRAIN_SAMPLE_COUNT must be a positive integer for dry-run." >&2
        exit 2
    fi
    if [[ ! "${GPU_IDS:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "GPU_IDS must be an explicit comma-separated list for dry-run." >&2
        exit 2
    fi
    IFS=',' read -r -a gpu_ids <<< "${GPU_IDS}"
    PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${RAPO_CPU_PYTHON:-python}" -m rapo.formal_contract \
        --profile "${profile_path}" \
        --train-samples "${RAPO_TRAIN_SAMPLE_COUNT}" \
        --world-size "${#gpu_ids[@]}"
    exit 0
fi

exec bash "${script_dir}/run_imagenet_r_smoke.sh" "$1" "$2"
