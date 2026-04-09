#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/kohya_ss/sd-scripts"

if [[ -d "${TARGET_DIR}" ]]; then
  echo "[setup_kohya] Found existing: ${TARGET_DIR}"
  exit 0
fi

mkdir -p "${ROOT_DIR}/kohya_ss"
echo "[setup_kohya] Cloning kohya-ss/sd-scripts into ${TARGET_DIR}"
git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git "${TARGET_DIR}"
echo "[setup_kohya] Done."
