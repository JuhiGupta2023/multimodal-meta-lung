#!/usr/bin/env bash
set -euo pipefail
# create toy data
python sample_data/generate_sample_data.py --out ./sample_data/toy_dataset --n_patients 3 --slices_per_patient 2

# ensure alignment script points to toy dataset via env or edit ROOT at top of file
# We prefer env override inside script: alignment_train.py should read ROOT from env var if present.
export DATA_ROOT=./sample_data/toy_dataset
python src/alignment/alignment_train.py
