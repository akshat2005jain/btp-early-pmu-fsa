# Phase 15 — Authoritative Clean-Environment Reproduction Instructions

## 1. Environment Setup (Fresh Checkout)

```bash
cd btp-early-pmu-fsa
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-build-isolation --no-deps
```

- **Supported Python**: `>= 3.9` (verified on Python `3.9.6`, NumPy `2.0.2`, SciPy `1.13.1`, scikit-learn `1.6.1`, Pandapower `2.14.11`).
- **Frozen package versions**: recorded in `reports/final_audit/environment_snapshot.txt`.
- **Canonical configuration**: `configs/default_50hz.yaml` (`seed: 42`, `f_nominal: 50.0`, `security_threshold_hz: 49.50`, `fps: 50`, `strict_nadir: true`).

---

## 2. Three Clearly Distinguished Execution Tiers

### Tier A — Quick Sanity Run (~15 seconds, Isolated Temp Output)
Verifies AC power flow, dynamic ODE solver, PMU channel recording, 14-feature extraction, training, and evaluation on an 8-scenario mini-batch without overwriting canonical artifacts:

```bash
.venv/bin/python scripts/run_simulations.py --n-scenarios 8 --seed 42 \
  --output-dir /tmp/pmu_sanity/raw --raw-sim-dir /tmp/pmu_sanity/raw_sim \
  --manifest-path /tmp/pmu_sanity/manifest.json --report-path /tmp/pmu_sanity/quality.md --validate
.venv/bin/python scripts/extract_features.py --input-dir /tmp/pmu_sanity/raw \
  --output-file /tmp/pmu_sanity/processed/features.npz --strict-nadir
```

### Tier B — Small Pilot Run (~45 seconds, Isolated Temp Output)
Runs a 36-scenario stratified pilot across simulation, multi-window extraction, model training, and benchmark evaluation:

```bash
.venv/bin/python scripts/run_simulations.py --n-scenarios 36 --seed 42 \
  --output-dir /tmp/pmu_pilot/raw --raw-sim-dir /tmp/pmu_pilot/raw_sim \
  --manifest-path /tmp/pmu_pilot/manifest.json --report-path /tmp/pmu_pilot/quality.md --validate
.venv/bin/python scripts/extract_features.py --input-dir /tmp/pmu_pilot/raw \
  --output-dir /tmp/pmu_pilot/processed --all-windows --strict-nadir --save-csv
.venv/bin/python scripts/train_models.py --features /tmp/pmu_pilot/processed/features_t1.0s.npz \
  --model-type all --output-dir /tmp/pmu_pilot/models --tables-dir /tmp/pmu_pilot/tables
.venv/bin/python scripts/evaluate_all.py --data-dir /tmp/pmu_pilot/raw \
  --features /tmp/pmu_pilot/processed/features_t1.0s.npz --models-dir /tmp/pmu_pilot/models \
  --tables-dir /tmp/pmu_pilot/tables --report-file /tmp/pmu_pilot/evaluation_report.md
```
*(Note: Tier A and Tier B are strictly sanity/pilot runs and must never be reported as full benchmark reproduction.)*

### Tier C — Full 750-Scenario Benchmark Reproduction

#### Option C1: Single Authoritative Master Command (Including Raw ODE Regeneration, ~3–4 min)
Because `data/raw/*.npz` and `data/raw_sim/*.npz` are git-ignored, run this command on a fresh clone to regenerate all 750 raw ODE + PMU `.npz` files, verify SHA-256 digests against `data/manifest.json`, extract all 4 causal windows, train all 14 models, run all benchmarks, and render all 14 PNG + PDF figures:

```bash
.venv/bin/python scripts/run_all.py --config configs/default_50hz.yaml --with-figures
```

#### Option C2: Fast Full Benchmark Reproduction (When `data/raw/*.npz` Already Exists on Disk, ~95 seconds)
```bash
.venv/bin/python scripts/run_all.py --config configs/default_50hz.yaml --skip-simulation --with-figures
```

#### Option C3: Explicit Ordered Stage-by-Stage Commands
```bash
# 1. Generate or validate 750 raw ODE & PMU scenarios + SHA-256 manifest
.venv/bin/python scripts/run_simulations.py --clean --n-scenarios 750 \
  --gen-buses 30 31 32 33 34 35 36 37 38 39 \
  --loss-fractions 1.0 0.70 0.40 \
  --penetrations 0.10 0.20 0.30 0.40 0.50 \
  --load-scales 0.80 0.90 1.00 1.10 1.20 \
  --fps 50 --f-nominal 50.0 --threshold 49.50 --seed 42 \
  --output-dir data/raw --raw-sim-dir data/raw_sim --validate

# 2. Extract strictly pre-nadir 14-feature datasets across {0.5, 1.0, 1.5, 2.0} s and re-validate manifest
.venv/bin/python scripts/extract_features.py \
  --input-dir data/raw --t-obs 1.0 --f-nominal 50.0 --threshold 49.50 \
  --all-windows --strict-nadir --save-csv --validate \
  --output-file data/processed/features.npz

# 3. Train all 7 regressors + 7 classifiers (including cross-fit Platt & PAVA Isotonic calibration)
.venv/bin/python scripts/train_models.py \
  --features data/processed/features.npz --model-type all \
  --calibration platt --threshold 49.50 --seed 42 \
  --output-dir models/ --tables-dir results/tables/

# 4. Run all evaluation, OOD, noise, latency, Holm statistical, threshold, and ablation benchmarks
.venv/bin/python scripts/evaluate_all.py \
  --data-dir data/raw --models-dir models/ \
  --features data/processed/features.npz --t-obs 1.0 --threshold 49.50 --seed 42 \
  --tables-dir results/tables/ --report-file results/evaluation_report.md

# 5. Regenerate all 14 quantitative & schematic figures (28 files: PNG + PDF)
.venv/bin/python scripts/generate_figures.py

# 6. Rebuild visual project guide PDF
.venv/bin/python scripts/generate_project_guide.py
```

---

## 3. Verifying Raw Trajectory Checksums Against `data/manifest.json`

```bash
.venv/bin/python -c '
import json
from pathlib import Path
from pmu_fsa.simulation.dataset_validator import compute_file_sha256

man = json.loads(Path("data/manifest.json").read_text())
assert man["all_passed"] is True
assert man["partition_scenario_hashes"]["zero_overlap_verified"] is True
for fname, expected in man["checksums_sha256"]["raw_pmu"].items():
    actual = compute_file_sha256(Path("data/raw") / fname)
    assert actual == expected, f"Mismatch on {fname}"
print(f"Verified {len(man[\"checksums_sha256\"][\"raw_pmu\"])}/750 raw PMU SHA-256 digests successfully.")
'
```

---

## 4. Running the Full Automated Test Suite

```bash
# Unit, integration, causality, statistical, and regression tests (237 tests)
.venv/bin/python -m pytest tests/ -q

# 5-tier end-to-end production pipeline runner (149 checks)
.venv/bin/python tests/e2e/run_e2e.py --tier all --report-path reports/final_audit/reproduction_e2e_report.md
```
