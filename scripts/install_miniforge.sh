#!/usr/bin/env bash
set -euo pipefail

version="26.3.2-2"
sha256="42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94"
prefix="${MINIFORGE_PREFIX:-${HOME}/miniforge3}"
url="https://github.com/conda-forge/miniforge/releases/download/${version}/Miniforge3-${version}-Linux-x86_64.sh"

if [[ -x "${prefix}/bin/conda" ]]; then
    "${prefix}/bin/conda" --version
    exit 0
fi

if [[ -e "${prefix}" ]]; then
    printf 'Refusing to overwrite incomplete path: %s\n' "${prefix}" >&2
    exit 1
fi

installer="$(mktemp "${TMPDIR:-/tmp}/miniforge-${USER}-XXXXXX.sh")"
trap 'rm -f -- "${installer}"' EXIT

curl --fail --location --retry 3 --output "${installer}" "${url}"
printf '%s  %s\n' "${sha256}" "${installer}" | sha256sum --check -
bash "${installer}" -b -p "${prefix}"
"${prefix}/bin/conda" --version
