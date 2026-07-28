#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
    echo "Usage: bash scripts/download_qwen_model.sh [target-directory]" >&2
    exit 2
fi

target="${1:-${HOME}/models/Qwen2-VL-2B-Instruct-895c3a4}"
base_url="https://modelscope.cn/models/Qwen/Qwen2-VL-2B-Instruct/resolve/master"
hf_revision="895c3a49bc3fa70a340399125c650a463535e71c"

declare -A expected_sha256=(
    [LICENSE]="832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
    [README.md]="74fbe869df708593087ae2ee66a6605d33fc2025937644721dc1360963d037f4"
    [chat_template.json]="ad60d90252ed0b0705ba14e2d0ad0fec0beac1ea955642b54059b36052d8bc96"
    [config.json]="422adefa19e62dd175961cec85bc0400344fe5bf9b22bd1182e05aaae78556e0"
    [generation_config.json]="d2864bf1edea5863d331edfff48106b586a366f5a2c41aa77731fadc53aa25d2"
    [merges.txt]="599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"
    [model-00001-of-00002.safetensors]="994ac2b03f97de8bc647d0fe5eba2e4b632b3e28dc03574c29bdfc36cf47e1b9"
    [model-00002-of-00002.safetensors]="92540d8353c8d226a589a3b179bdb33851c970ee2cc2ac7ba035f79425e7b833"
    [model.safetensors.index.json]="260ab9fa1418d6d6ab79daa1d9da2c47264f3b72edb4630fc799077ac67d27c6"
    [preprocessor_config.json]="b5eaad0c2815f07631535dcc58f3c462b0d73693638ad21d19f3c50820eae1cc"
    [tokenizer.json]="cb63a0a23eef3d5b01063a9880a1925a65aaf4d1591d519910ee3527852950a0"
    [tokenizer_config.json]="ff5c4fd898fe8c39591eb70e5d39d2782802d4204d6ae9ba1223252f354842a0"
    [vocab.json]="ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"
)
files=(
    LICENSE
    README.md
    chat_template.json
    config.json
    generation_config.json
    merges.txt
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    model.safetensors.index.json
    preprocessor_config.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
)

mkdir -p "${target}"
for file_name in "${files[@]}"; do
    file_path="${target}/${file_name}"
    expected_hash="${expected_sha256[$file_name]}"
    if [[ -f "${file_path}" ]] &&
        [[ "$(sha256sum "${file_path}" | awk '{print $1}')" == "${expected_hash}" ]]; then
        echo "Already verified: ${file_name}"
        continue
    fi

    curl --fail --location \
        --retry 20 --retry-delay 5 --retry-all-errors \
        --continue-at - \
        --output "${file_path}" \
        "${base_url}/${file_name}"

    actual_sha256="$(sha256sum "${file_path}" | awk '{print $1}')"
    if [[ "${actual_sha256}" != "${expected_hash}" ]]; then
        echo "SHA-256 mismatch for ${file_name}." >&2
        exit 1
    fi
done

checksum_lines=()
for file_name in "${files[@]}"; do
    checksum_lines+=("${expected_sha256[$file_name]}  ${file_name}")
done
printf '%s\n' "${checksum_lines[@]}" > "${target}/SHA256SUMS"
printf '%s\n' \
    "model=Qwen/Qwen2-VL-2B-Instruct" \
    "hugging_face_revision=${hf_revision}" \
    "download_source=${base_url}" \
    "verification=all files match the pinned Hugging Face revision by SHA-256" \
    > "${target}/PROVENANCE.txt"
echo "All model files match the pinned Hugging Face revision."
