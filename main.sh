#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
infile="${1:-data/data_SCI_eng.xlsx}"
out_root="${2:-result}"
n_time="${N_TIME:-50}"
for preselect in both left right mean smaller larger; do
    python l1_lr_calibration_1.py \
        --input "$infile" --preselect "$preselect" \
        --out_dir "$out_root/$preselect" --n_time "$n_time" \
        --bootstrap_samples "${BOOTSTRAP_SAMPLES:-10000}" \
        --calibration_bins 10 --n_splits 3
done
