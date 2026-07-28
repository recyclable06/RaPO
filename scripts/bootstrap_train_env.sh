#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/bootstrap_train_env.sh /path/to/Visual-RFT" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
visual_rft_root="$(cd "$1" && pwd)"
conda_exe="${CONDA_EXE:-${HOME}/miniforge3/bin/conda}"
env_name="${RAPO_TRAIN_ENV:-rapo-train}"
pinned_visual_rft="2ffad63b25ddd79bfe25d3e046645401201c89d6"
flash_attn_version="2.7.4.post1"

if [[ ! -x "${conda_exe}" ]]; then
    echo "Conda executable not found at ${conda_exe}." >&2
    exit 1
fi

if [[ ! -f "${visual_rft_root}/src/virft/src/open_r1/grpo_classification.py" ]]; then
    echo "Visual-RFT classification entrypoint not found under ${visual_rft_root}." >&2
    exit 1
fi

actual_visual_rft="$(git -C "${visual_rft_root}" rev-parse HEAD)"
if [[ "${actual_visual_rft}" != "${pinned_visual_rft}" ]]; then
    echo "Visual-RFT must be checked out at ${pinned_visual_rft}; found ${actual_visual_rft}." >&2
    exit 1
fi

if "${conda_exe}" env list | awk '{print $1}' | grep -qx "${env_name}"; then
    "${conda_exe}" env update --name "${env_name}" --file "${repo_root}/environment-train.yml"
else
    "${conda_exe}" env create --file "${repo_root}/environment-train.yml"
fi

"${conda_exe}" run --name "${env_name}" \
    python -m pip install --no-deps --editable "${repo_root}"

if [[ "${INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
    MAX_JOBS="${MAX_JOBS:-4}" "${conda_exe}" run --name "${env_name}" \
        python -m pip install "flash-attn==${flash_attn_version}" --no-build-isolation
fi

PYTHONPATH="${repo_root}/src:${visual_rft_root}/src/virft/src" \
    "${conda_exe}" run --name "${env_name}" python -c \
    'import torch, transformers, trl; import open_r1.grpo_classification; import rapo; print({"torch": torch.__version__, "transformers": transformers.__version__, "trl": trl.__version__})'

if [[ "${INSTALL_FLASH_ATTN:-0}" != "1" ]]; then
    echo "Base training environment is ready. Install FlashAttention on the target GPU node with:"
    echo "INSTALL_FLASH_ATTN=1 MAX_JOBS=4 bash scripts/bootstrap_train_env.sh ${visual_rft_root}"
fi
