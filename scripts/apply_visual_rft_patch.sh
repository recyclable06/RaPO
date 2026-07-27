#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/Visual-RFT" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
visual_rft_root="$(cd "$1" && pwd)"
patch_file="${repo_root}/patches/visual_rft_2ffad63_rapo.patch"
expected_commit="2ffad63b25ddd79bfe25d3e046645401201c89d6"

actual_commit="$(git -C "${visual_rft_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
    echo "Expected Visual-RFT ${expected_commit}, got ${actual_commit}." >&2
    exit 1
fi

if [[ -n "$(git -C "${visual_rft_root}" status --porcelain)" ]]; then
    echo "Refusing to patch a non-clean Visual-RFT checkout." >&2
    exit 1
fi

git -C "${visual_rft_root}" apply --check "${patch_file}"
git -C "${visual_rft_root}" apply "${patch_file}"
echo "Applied RaPO integration patch to ${visual_rft_root}."
