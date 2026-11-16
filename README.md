# Early PMU-Based Frequency-Security Assessment (`pmu_fsa`)

IEEE 39-bus (50.0 Hz) dynamic simulation, causal synchrophasor feature extraction, and probability-calibrated gradient boosting for early frequency nadir and security status prediction (`f_secure = 49.50 Hz`).

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

## Running the Pipeline

```bash
# 1. Generate 50 Hz disturbance scenarios
.venv/bin/python3 scripts/run_simulations.py --clean \
  --n-scenarios 450 \
  --gen-buses 30 31 32 33 34 35 36 37 38 39 \
  --loss-fractions 1.0 0.55 0.25 \
  --penetrations 0.10 0.20 0.30 0.40 0.50 \
  --load-scales 0.90 1.00 1.10 \
  --fps 50 --f-nominal 50.0 --threshold 49.50 --output-dir data/raw

# 2. Extract early post-disturbance PMU features
.venv/bin/python3 scripts/extract_features.py \
  --input-dir data/raw --t-obs 1.0 --f-nominal 50.0 \
  --all-windows --save-csv --output-file data/processed/features.npz

# 3. Train baseline and calibrated GBDT models
.venv/bin/python3 scripts/train_models.py \
  --features data/processed/features.npz --model-type all \
  --calibration platt --threshold 49.50 \
  --output-dir models/ --tables-dir results/tables/

# 4. Run evaluation benchmarks
.venv/bin/python3 scripts/evaluate_all.py \
  --data-dir data/raw --models-dir models/ \
  --tables-dir results/tables/ --report-file results/evaluation_report.md
```

## Tests

```bash
.venv/bin/pytest tests/ -v
```
