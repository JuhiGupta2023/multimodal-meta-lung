#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Running smoke script"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "[INFO] Repo root: $REPO_ROOT"

PY=${PY:-python}

# 1) generate toy data (relative path to sample_data)
echo "[INFO] Generating toy dataset at: ${REPO_ROOT}/sample_data/toy_dataset"
$PY "${REPO_ROOT}/sample_data/generate_sample_data.py" || { echo "[ERR] toy data gen failed"; exit 1; }

# 2) locate alignment_train.py anywhere under src/
ALIGN_SCRIPT="$(find "${REPO_ROOT}" -type f -path "*/src/*" -iname "alignment_train.py" | head -n 1 || true)"
if [ -z "$ALIGN_SCRIPT" ]; then
  echo "[ERR] Could not find alignment_train.py under repo src/ - ensure the file is committed at src/alignment/alignment_train.py"
  echo "[INFO] Showing src tree for debugging:"
  ls -la "${REPO_ROOT}/src" || true
  exit 2
fi

echo "[INFO] Found alignment script at: $ALIGN_SCRIPT"

# 3) run alignment in smoke mode: pass data-root and epochs=1 to keep it small
echo "[INFO] Running alignment (1-epoch) using python: $PY"
$PY "$ALIGN_SCRIPT" --data-root "${REPO_ROOT}/sample_data/toy_dataset" --epochs 1 || { echo "[ERR] alignment failed"; exit 1; }

echo "[INFO] Smoke script finished successfully"
