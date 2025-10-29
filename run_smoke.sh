#!/usr/bin/env bash
set -euo pipefail
echo "[INFO] Running smoke script"

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "[INFO] Repo root: $REPO_ROOT"

# 1) generate toy data (your script path)
python "${REPO_ROOT}/sample_data/generate_sample_data.py" || { echo "toy data gen failed"; exit 1; }

# 2) run one-epoch alignment (use src path)
PY=${PY:-python}
echo "[INFO] Running alignment train (1 epoch smoke) at: ${REPO_ROOT}/src/alignment/alignment_train.py"
# pass a flag so alignment_train.py does 1 epoch and uses sample_data paths
$PY "${REPO_ROOT}/src/alignment/alignment_train.py" --data-root "${REPO_ROOT}/sample_data/toy_dataset" --epochs 1 || { echo "alignment failed"; exit 1; }

echo "[INFO] Smoke script finished"
