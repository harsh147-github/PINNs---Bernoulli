#!/usr/bin/env bash
# Full pipeline: OpenFOAM references -> PINN training -> amortised inference + figures.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests -q
python scripts/run_openfoam.py
mkdir -p results/pinn
python scripts/train.py 2>&1 | tee results/pinn/train_log.txt
python scripts/infer_and_postprocess.py
