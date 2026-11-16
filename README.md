# Early PMU-Based Prediction of Frequency Nadir and Security Status in Renewable-Integrated Multi-Area Power Systems

This repository implements an end-to-end simulation, feature engineering, and machine learning pipeline (`pmu_fsa`) for early post-disturbance frequency-security assessment on the **IEEE 39-bus, 10-machine** multi-area power system standardized at **50.0 Hz** (`f_secure = 49.50 Hz`, `50 fps` IEEE C37.118 reporting).

Given only a short causal observation window `[t0, t0 + T_obs]` (`0.5 s` to `2.0 s` immediately after a generator outage—several seconds before the actual frequency nadir occurs), the pipeline extracts 14 physical and spatial synchrophasor features and predicts:
1. **Continuous Frequency Nadir (`f_nadir` in Hz)** bounded to `[45.0, 50.5] Hz`.
2. **Calibrated Security Probability (`P(f_nadir >= 49.50 Hz)`)** and binary security status (`1 = Secure`, `0 = Insecure`) with 95% epistemic confidence bounds.

---

## Why This Matters

When a large generator trips offline, active power demand exceeds mechanical supply and synchronous rotors decelerate. As converter-interfaced solar and wind plants (`10%–50%` penetration) displace conventional synchronous turbines, effective system inertia `H` drops—producing steeper initial Rate of Change of Frequency (ROCOF) and deeper frequency drops before turbine governors (`TGOV1`) can respond.

- **Numerical ODE simulation is too slow for large-scale online screening** (~`350 ms` per contingency).
- **Full-window ML classifiers leak post-nadir samples**, offering no operational lead time.
- **Uncalibrated classifiers distort borderline probabilities** near `49.50 Hz` and degrade sharply if raw two-point finite differences amplify PMU sensor noise.

By combining noise-resilient Ordinary Least Squares (OLS) ROCOF estimation, multi-area spatial features, and a two-stage **Calibrated Gradient Boosted Decision Tree (`CalibratedGBDTClassifier`)**, online assessment completes in **`0.485 ms`** per scenario (**721.6x speedup** over dynamic simulation) while preserving **`5.06 s–6.56 s`** of early-warning lead time.

---

## Repository Layout

```text
btp-early-pmu-fsa/
├── pyproject.toml                  # Package metadata & dependencies
├── requirements.txt                # Pip dependency list
├── setup.py                        # Editable install shim
├── BTP_Complete_Project_Guide.pdf  # 12-page visual explanation & viva guide (50.0 Hz)
├── src/pmu_fsa/
│   ├── simulation/
│   │   ├── ieee39_data.py          # 39 buses, 10 generators, 46 branches, 3-area map (50.0 Hz)
│   │   ├── renewables.py           # Converter-interfaced generation & inertia derating
│   │   ├── dynamic_engine.py       # Kron reduction + 30 coupled swing & TGOV1 governor ODEs
│   │   ├── pmu_recorder.py         # 50 fps IEEE C37.118.2 sampler, FIR ROCOF, binary CRC frames
│   │   └── scenario_generator.py   # Combinatorial outage generator & deterministic ID/OOD split
│   ├── features/
│   │   ├── window_slicer.py        # Strict [t0, t0 + T_obs] causal windowing
│   │   └── extractor.py            # 14 temporal, energy, and multi-area spatial PMU features
│   ├── models/
│   │   ├── regressors.py           # Ridge, Decision Tree, Random Forest, GBDT nadir regressors
│   │   ├── classifiers.py          # Logistic, Decision Tree, Random Forest, GBDT classifiers
│   │   ├── calibrated_gb.py        # Two-stage GBDT + L-BFGS Platt / PAVA Isotonic calibration
│   │   └── calibration_metrics.py  # Brier score, ECE, MCE, and reliability diagram binning
│   ├── evaluation/
│   │   ├── sensitivity.py          # Observation window sweep (0.5 s - 2.0 s) & lead-time tracking
│   │   ├── ood_generalization.py   # Domain shift: train 10-30% RES -> test 40-50% RES
│   │   ├── noise_stress.py         # IEEE C37.118 Gaussian PMU noise sweep (0 - 50 mHz)
│   │   └── latency.py              # Hardware perf_counter_ns timing vs. ODE simulation
│   └── pipeline/
│       └── run_experiments.py      # Single-command end-to-end experiment runner
├── scripts/
│   ├── run_simulations.py          # CLI 1: Simulate contingency trajectories -> data/raw/*.npz
│   ├── extract_features.py         # CLI 2: Extract windowed features -> data/processed/*.npz/.csv
│   ├── train_models.py             # CLI 3: Fit models & export regression/classification tables
│   └── evaluate_all.py             # CLI 4: Run robustness & latency benchmarks -> results/
├── data/
│   ├── raw/                        # Simulated .npz PMU trajectory archives (450 scenarios)
│   └── processed/                  # Tabular feature matrices (.npz and .csv for 0.5s - 2.0s)
├── models/                         # Saved .joblib estimators & training_metrics.json
├── results/
│   ├── evaluation_report.md        # Consolidated benchmark report
│   └── tables/                     # CSV & Markdown tables for all experiments
└── tests/                          # 454 unit, causality, boundary, and end-to-end tests
```

---

## Installation & Setup

Requires **Python 3.9+** on macOS or Linux.

```bash
cd /Users/akshat/Desktop/btp-early-pmu-fsa

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the package in editable mode with test dependencies
pip install -e ".[dev]"
```

*(If you prefer running commands without activating the shell environment, prefix commands with `.venv/bin/python3` or `.venv/bin/pytest` as shown below.)*

---

## How to Run the Pipeline

You can either run each of the 4 stages individually via `scripts/` (recommended for full experiments) or run the entire workflow in one command via `run_experiments.py`.

### 1. Generate 50.0 Hz Contingency Scenarios (`scripts/run_simulations.py`)

Simulates generator trips across buses `30–39`, renewable shares, outage loss fractions, and system load scales, saving compressed `.npz` synchrophasor archives into `data/raw/`:

```bash
# Full 450-scenario 50.0 Hz dataset (270 In-Distribution + 180 Out-of-Distribution)
.venv/bin/python3 scripts/run_simulations.py --clean \
  --n-scenarios 450 \
  --gen-buses 30 31 32 33 34 35 36 37 38 39 \
  --loss-fractions 1.0 0.55 0.25 \
  --penetrations 0.10 0.20 0.30 0.40 0.50 \
  --load-scales 0.90 1.00 1.10 \
  --fps 50 \
  --f-nominal 50.0 \
  --threshold 49.50 \
  --output-dir data/raw
```

Useful options for `run_simulations.py`:
- `--n-scenarios <int>`: Maximum number of contingencies to simulate (e.g., `20` for a quick test, `450` for the full combinatorial grid).
- `--no-macro-intertie`: Models Bus 39 as a local generator (`H = 42 s`) rather than an external grid equivalent (`H = 500 s`).
- `--inject-noise --noise-std 0.005`: Injects Gaussian PMU frequency noise during recording and recomputes 5-point FIR ROCOF.
- `--clean`: Deletes existing `.npz` files in `--output-dir` before generating a fresh dataset.
- `--fast`: Generates fast analytical trajectories for quick pipeline smoke-testing.

---

### 2. Extract Causal PMU Features (`scripts/extract_features.py`)

Slices each trajectory strictly on `[t0, t0 + T_obs]` (zero access to samples after `t0 + T_obs`) and computes the 14 physical and spatial PMU features:

```bash
# Extract primary 1.0s dataset (features.npz) plus all windows [0.5s, 1.0s, 1.5s, 2.0s] and CSVs
.venv/bin/python3 scripts/extract_features.py \
  --input-dir data/raw \
  --t-obs 1.0 \
  --f-nominal 50.0 \
  --all-windows \
  --save-csv \
  --output-file data/processed/features.npz
```

To extract a single custom window (for example `0.75 s`) without overwriting the multi-window sweep:

```bash
.venv/bin/python3 scripts/extract_features.py \
  --input-dir data/raw \
  --t-obs 0.75 \
  --output-file data/processed/features_t0.75s.npz
```

#### Extracted Synchrophasor Features (14 Core Physical Signals)

| # | Feature Column | Short Alias | Physical Meaning |
|---:|---|---|---|
| 1 | `rocof_init_coi` | `init_rocof` | OLS linear slope of COI frequency over `[t0, t0 + 0.1s]` (`-DeltaP / 2H`) |
| 2 | `rocof_peak_neg_coi` | `pk_rocof` | Minimum instantaneous COI ROCOF inside the observation window |
| 3 | `rocof_max_abs_coi` | `max_abs_rocof` | Peak absolute COI ROCOF magnitude |
| 4 | `rocof_mean_coi` | `mean_rocof` | Net average COI frequency decay rate `(f_end - f_start) / T_obs` |
| 5 | `delta_f_tobs_coi` | `inst_drop` | Instantaneous COI frequency drop at `t0 + T_obs` |
| 6 | `delta_f_max_drop_coi` | `max_drop` | Worst COI excursion below `50.0 Hz` observed within `[t0, t0 + T_obs]` |
| 7 | `freq_integral_dev_coi` | `integ_dev` | Trapezoidal integral of `(f_COI - 50.0)` in `Hz*s` (kinetic energy deficit) |
| 8 | `freq_integral_abs_coi` | `integ_abs` | Trapezoidal integral of `|f_COI - 50.0|` in `Hz*s` |
| 9 | `area_spread_max` | `spread` | Maximum pairwise mean frequency spread across Areas 1, 2, and 3 |
| 10 | `bus_freq_var_mean` | `spatial_var` | Cross-bus frequency variance across all 39 substations at `t0 + T_obs` |
| 11 | `angle_div_max` | `max_ang` | Modulo-wrapped maximum voltage phase-angle span across the network |
| 12 | `tie_line_angle_diff` | `tie_ang` | Wrapped phase-angle difference across primary inter-area tie-line Bus 4–5 |
| 13 | `coi_acc_init` | `init_acc` | Initial post-fault COI deceleration slope (`0.1 s`) |
| 14 | `coi_acc_mean` | `mean_acc` | Window-averaged COI deceleration (captures governor arrest curvature) |

---

### 3. Train Baselines & Calibrated GBDT (`scripts/train_models.py`)

Trains all four regression baselines (Ridge, Decision Tree, Random Forest, GBDT), all four classification baselines (Logistic Regression, Decision Tree, Random Forest, Uncalibrated GBDT), and the proposed `CalibratedGBDTClassifier` using disjoint `train`, `val` (calibration), and `test` partitions:

```bash
# Train every model family and export results/tables/nadir_regression.md & security_classification.md
.venv/bin/python3 scripts/train_models.py \
  --features data/processed/features.npz \
  --model-type all \
  --calibration platt \
  --threshold 49.50 \
  --output-dir models/ \
  --tables-dir results/tables/
```

Useful options for `train_models.py`:
- `--model-type {linear,dt,rf,gbdt,calibrated_gb,all}`: Train a single model or the full comparison suite.
- `--calibration {platt,isotonic}`: Select L-BFGS Platt scaling (enforcing positive slope `A > 0`) or PAVA isotonic regression (automatically falls back to Platt if `N_cal < 50`).
- Outputs serialized `.joblib` models (`linear.joblib`, `dt.joblib`, `rf.joblib`, `gbdt.joblib`, `calibrated_gb.joblib`, `calibrated_gb_platt.joblib`, `calibrated_gb_iso.joblib`) and `models/training_metrics.json`.

---

### 4. Run Evaluation & Stress Benchmarks (`scripts/evaluate_all.py`)

Runs the four robustness and timing benchmarks on `data/raw/` and writes CSV/Markdown tables to `results/tables/` plus the combined report at `results/evaluation_report.md`:

```bash
# Run all 4 benchmarks (window sensitivity, OOD shift, PMU noise stress, and latency)
.venv/bin/python3 scripts/evaluate_all.py \
  --data-dir data/raw \
  --models-dir models/ \
  --tables-dir results/tables/ \
  --report-file results/evaluation_report.md
```

To run only specific benchmarks (for example, just OOD shift and latency):

```bash
.venv/bin/python3 scripts/evaluate_all.py \
  --data-dir data/raw \
  --benchmarks ood latency
```

---

### Single-Command End-to-End Runner (`run_experiments.py`)

If you want to run simulation, multi-window extraction, training, benchmarking, and plot export (`results/figures/window_sensitivity_curve.png`) in one step:

```bash
# Run a compact 30-scenario end-to-end experiment from scratch
.venv/bin/python3 src/pmu_fsa/pipeline/run_experiments.py \
  --n-scenarios 30 \
  --t-obs 1.0 \
  --output-dir results/

# Or reuse existing scenarios in data/raw/ without re-simulating
.venv/bin/python3 src/pmu_fsa/pipeline/run_experiments.py --skip-simulation
```

---

## Using the Python API Directly

You can import `pmu_fsa` directly in Python scripts or notebooks for custom contingency experiments:

```python
import joblib
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
from pmu_fsa.simulation.renewables import RenewablePenetrationConfig
from pmu_fsa.features.extractor import FeatureExtractor

sim = DynamicSimulator(
    load_scaling=1.0,
    renewable_config=RenewablePenetrationConfig(p_ren=0.25),
    macro_intertie=True,
    f_nominal=50.0,
    security_threshold_hz=49.50,
)
traj = sim.simulate(t_span=(0.0, 10.0), t_disturb=1.0, tripped_bus=35, p_loss_mw=600.0, fps=50)

extractor = FeatureExtractor(t_obs=1.0, f_nominal=50.0)
X_row = extractor.extract_scenario(traj, t_obs=1.0)

gbdt_reg = joblib.load("models/gbdt.joblib")
cal_clf = joblib.load("models/calibrated_gb.joblib")

pred_nadir = gbdt_reg.predict(X_row)[0]
p_sec, ci_low, ci_high = cal_clf.predict_with_intervals(X_row, confidence_level=0.95)

print(f"True Nadir:      {traj['metadata']['ground_truth_nadir']:.4f} Hz")
print(f"Predicted Nadir: {pred_nadir:.4f} Hz")
print(f"P(Secure):       {p_sec[0]:.4f}  (95% CI: [{ci_low[0]:.4f}, {ci_high[0]:.4f}])")
```

---

## Running the Test Suite (454 Tests)

The test suite verifies Kron reduction physics, IEEE C37.118.2 binary frame serialization & `CRC-CCITT` checksums, zero future data leakage under post-window `NaN` poisoning, PAVA monotonicity, OOD partition disjointness, and sub-10 ms latency budgets:

```bash
# Run the complete 454-test suite (111 unit tests + 343 end-to-end tests)
.venv/bin/pytest tests/ -v

# Run only unit & causality tests
.venv/bin/pytest tests/test_*.py -v

# Run end-to-end suites by tier (1, 2, 3, 4, 5, or all)
.venv/bin/python3 tests/e2e/run_e2e.py --tier all
```

---

## 450-Scenario 50.0 Hz Pilot Results Summary

Dataset: **450 unique contingencies** (`10` generator buses × `3` loss fractions × `5` renewable levels `10%–50%` × `3` load scales `0.90–1.10`), partitioned into **270 In-Distribution (`10%–30%` RES: `186 train`, `48 val`, `36 test`)** and **180 Out-of-Distribution (`40%–50%` RES: `ood_test`)** scenarios.

### Continuous Nadir Regression (Held-Out ID Test Split, `T_obs = 1.0 s`)

| Model | MAE (Hz) | RMSE (Hz) | `R2` | 95% Bootstrap CI (Hz) | Inference (ms/scen) |
|---|---:|---:|---:|---|---:|
| Ridge Linear Regression | 0.0417 | 0.0515 | 0.9717 | `[0.0321, 0.0521]` | 0.0175 |
| Decision Tree (CART) | 0.0287 | 0.0403 | 0.9827 | `[0.0212, 0.0393]` | 0.0130 |
| Random Forest (50 Trees) | **0.0184** | **0.0232** | **0.9943** | `[0.0142, 0.0227]` | 0.2676 |
| Gradient Boosting (GBDT) | **0.0222** | **0.0289** | **0.9911** | `[0.0169, 0.0288]` | **0.0450** |

### Observation Window Sensitivity (All 450 Scenarios Combined)

| Window `T_obs` | Nadir MAE (Hz) | `R2` | Security F1 | ECE | Brier | `FNR_insecure` | Lead Time `t_lead` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.50 s` | 0.0531 | 0.9822 | 0.9293 | 0.0471 | 0.0412 | 0.0120 | **6.56 s** |
| `1.00 s` | 0.0531 | 0.9722 | 0.9307 | **0.0293** | 0.0300 | 0.0241 | **6.06 s** |
| `1.50 s` | 0.0531 | 0.9619 | **0.9400** | 0.0357 | **0.0272** | 0.0120 | **5.56 s** |
| `2.00 s` | 0.0531 | 0.9718 | **0.9400** | 0.0383 | 0.0295 | 0.0120 | **5.06 s** |

### OOD Generalization, Noise Robustness & Online Latency

- **In-Distribution (`10%–30%` RES, `N = 270`)**: `MAE = 0.0422 Hz`, `F1 = 0.9953`, `ECE = 0.0064`, `Brier = 0.0024`.
- **Out-of-Distribution (`40%–50%` RES, `N = 180`)**: `MAE = 0.2731 Hz`, `F1 = 0.8547`, `ECE = 0.0812`, `Brier = 0.0700`.
- **PMU Measurement Noise (`sigma_f` from `0` to `20 mHz`)**: Security `F1` degrades smoothly from `0.9848` (clean) -> `0.9279` (`5 mHz` Class M) -> `0.9003` (`10 mHz`) -> `0.8667` (`20 mHz`).
- **Online Execution Speed**: `0.279 ms` feature extraction + `0.206 ms` inference = **`0.485 ms` total** (`721.6x` faster than `350.0 ms` dynamic ODE integration).
