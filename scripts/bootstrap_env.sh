#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_exe="${CONDA_EXE:-${HOME}/miniforge3/bin/conda}"
env_name="rapo"

if [[ ! -x "${conda_exe}" ]]; then
    echo "Conda executable not found at ${conda_exe}." >&2
    echo "Install Miniforge in ${HOME}/miniforge3 or set CONDA_EXE." >&2
    exit 1
fi

if "${conda_exe}" env list | awk '{print $1}' | grep -qx "${env_name}"; then
    "${conda_exe}" env update --name "${env_name}" --file "${repo_root}/environment.yml" --prune
else
    "${conda_exe}" env create --file "${repo_root}/environment.yml"
fi

"${conda_exe}" run --name "${env_name}" python -m pip install --editable "${repo_root}[test]"
"${conda_exe}" run --name "${env_name}" python -m pytest
