#!/usr/bin/env bash
# Full pipeline: OpenFOAM references -> PINN training -> amortised inference + figures.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests -q
python scripts/run_openfoam.py                                   # 7 reference designs
python scripts/run_openfoam.py --jobs 4 --designs 0.40 0.45 0.50 0.55 0.60 0.65 \
    0.70 0.75 0.80 0.85 0.90 0.95                                # 4-parallel throughput baseline
mkdir -p results/pinn
python scripts/train.py 2>&1 | tee results/pinn/train_log.txt
python scripts/infer_and_postprocess.py
