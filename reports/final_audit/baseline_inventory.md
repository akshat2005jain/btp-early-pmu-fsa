# Phase 0 — Baseline Inventory & Environment Snapshot

## 1. Repository & Git Baseline
- **Starting Branch**: `btp-scientific-validation`
- **Starting Commit Hash**: `e573837c249c7ef83a01f635db0d2e2ca2101a32`
- **Original Main Branch**: `main` at `774b5e6` (untouched)
- **Dedicated Audit & Remediation Branch**: `btp-publication-audit` (branched cleanly from `e573837` prior to any edits; no remote push performed)
- **Starting Working Tree**: Clean (`0` modified or untracked tracked paths)

## 2. Hardware & Software Environment
- **Operating System**: macOS (`Darwin 25.6.0 x86_64`)
- **CPU**: `Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz` (`12` logical cores)
- **RAM**: `16.0 GB` (`17,179,869,184` bytes)
- **Python Runtime**: Python `3.9.6` (`.venv/bin/python`)
- **Installed Scientific Packages**:
  - `numpy==2.0.2`
  - `scipy==1.13.1`
  - `pandas==2.2.3`
  - `scikit-learn==1.6.1`
  - `lightgbm==4.6.0` (note: OpenMP runtime `libomp.dylib` not present on macOS host, so `LightGBMNadirRegressor` / `LightGBMSecurityClassifier` fall back deterministically to `sklearn_hist_gbdt` as recorded in `models/training_metrics.json`)
  - `matplotlib==3.9.4`
  - `joblib==1.5.3`
  - `pytest==8.4.2`
  - Note: `PyYAML` is not installed in `.venv`; `ExperimentConfig.from_file()` parses flat YAML via `_parse_simple_yaml()`, while `setup.py` currently lacks `install_requires` and `extras_require` (logged as `ISS-008`).

## 3. Complete Repository Inventory

### 3.1 Source Modules (`src/pmu_fsa/`)
- **Simulation (`src/pmu_fsa/simulation/`)**:
  - `ieee39_data.py` — 50.0 Hz IEEE 39-bus, 10-machine, 46-branch parameters; `SECURITY_THRESHOLD_HZ = 49.50`; regional Bus 39 default (`H_39 = 42.0 s`, `H_tot = 324.7 s`) and opt-in `macro_intertie` (`H_39 = 500.0 s`).
  - `renewables.py` — Synchronous inertia, damping, and governor participation derating across 9 non-slack generator buses (`30, 32..39`) plus optional 1st-order fast inverter droop (`enable_droop`, `r_ren`, `t_inv`).
  - `dynamic_engine.py` — Vectorized Newton-Raphson AC power flow (`solve_power_flow`), Schur-complement Kron reduction (`compute_kron_reduction`), post-contingency complete/partial unit disconnection (`_build_post_disturbance_kron`), two-segment RK45 integration (`DynamicSimulator.simulate`), bus phasor/frequency reconstruction, and ground-truth COI nadir calculation.
  - `pmu_recorder.py` — Uniform `50 fps` / `25 fps` resampling with unwrapped-angle interpolation, seeded multi-channel Gaussian noise (`PMUNoiseConfig`), causal 5-point backward OLS ROCOF (`compute_causal_rocof`), area-weighted PMU frequency estimator (`estimate_pmu_system_frequency`), and simplified single-station C37.118.2 frame pack/parse.
  - `scenario_generator.py` — 750-scenario combinatorial grid (`10` gens × `3` loss fractions × `5` RES × `5` load scales), deterministic MD5 scenario-level partitioner (`assign_scenario_partition`), and separate `raw_sim` vs `raw` `.npz` writers.
  - `dataset_validator.py` — 14 automated integrity, unit, partition-disjointness, and pre-nadir causality checks plus SHA-256 `manifest.json` writer.
- **Features (`src/pmu_fsa/features/`)**:
  - `window_slicer.py` — `CausalWindow` and `extract_causal_window()` slicing `[t_0, t_0 + T_obs]` and stripping `FORBIDDEN_WINDOW_CHANNEL_KEYS`.
  - `extractor.py` — Canonical 14 spatiotemporal PMU features (`CANONICAL_FEATURE_NAMES`), missing-data/unobserved-bus masking, and `FeatureExtractor`.
- **Models (`src/pmu_fsa/models/`)**:
  - `regressors.py` — `MeanNadirRegressor`, `AnalyticalSwingNadirRegressor`, `LinearNadirRegressor` (Ridge), `DecisionTreeNadirRegressor`, `RandomForestNadirRegressor`, `GradientBoostingNadirRegressor`, `LightGBMNadirRegressor`.
  - `classifiers.py` — `BinarySecurityClassifier`, `LogisticSecurityClassifier`, `DecisionTreeSecurityClassifier`, `RandomForestSecurityClassifier`, `GradientBoostingSecurityClassifier`, `LightGBMSecurityClassifier`, `compute_binary_security_metrics`, `evaluate_probability_threshold_sweep`.
  - `calibrated_gb.py` — `CalibratedGBDTClassifier` (two-stage continuous GBDT nadir score + Platt sigmoid or PAVA isotonic calibration + quantile/conformal/OOD prediction intervals).
  - `calibration_metrics.py` — `brier_score`, `expected_calibration_error`, `maximum_calibration_error`, `reliability_diagram` (10 equal-width bins).
- **Evaluation & Pipeline (`src/pmu_fsa/evaluation/`, `src/pmu_fsa/pipeline/`)**:
  - `sensitivity.py` — `WindowSensitivityBenchmark` and `split_scenarios_by_partition`.
  - `ood_generalization.py` — `OODGeneralizationBenchmark` (`id_test`, `ood_res`, `ood_load`, `ood_combined`, `ood_all`).
  - `noise_stress.py` — `MeasurementNoiseBenchmark` (`0–50 mHz` Gaussian noise + OLS vs 2-pt ROCOF variance test).
  - `latency.py` — `LatencyBenchmark` (6-stage `perf_counter_ns` profiler + hardware-timed `DynamicSimulator` runs).
  - `interpretability.py` — `InterpretabilityAnalyzer` (permutation/impurity importance, Pearson/Spearman correlations, 7 feature/observability ablations, multi-seed rank stability, subgroup feature shifts).
  - `statistical_tests.py` — `paired_bootstrap_ci`, `paired_model_comparison_test`, `compare_all_models_statistical`, `LearningCurveEvaluator`, `MultiSeedStabilityEvaluator`.
  - `run_experiments.py` — `ExperimentConfig` and `ExperimentPipeline`.

### 3.2 CLI Scripts (`scripts/`) & Configurations (`configs/`)
- `configs/default_50hz.yaml` — Canonical 750-scenario 50.0 Hz configuration (`macro_intertie: false`, `security_threshold_hz: 49.50`, `fps: 50`, `strict_nadir: true`).
- `scripts/run_simulations.py`, `scripts/extract_features.py`, `scripts/train_models.py`, `scripts/evaluate_all.py`, `scripts/generate_figures.py`, `scripts/run_all.py`, `scripts/generate_project_guide.py`.

### 3.3 Datasets (`data/`), Models (`models/`), Results (`results/`), & Docs
- `data/raw/` (`750` `.npz` files, git-ignored) & `data/raw_sim/` (`750` `.npz` files, git-ignored) — both present on disk and matching `data/manifest.json` (`b543ffc51c757ddc...`). No stale 450-scenario artifacts remain.
- `data/processed/` — `features.npz`, `features_t{0.5,1.0,1.5,2.0}s.{npz,csv}`, `extraction_report.json`.
- `models/` — 18 `.joblib` artifacts + `training_metrics.json`.
- `results/` — `dataset_quality_report.md`, `evaluation_report.md`, `tables/` (`27` CSV + `25` MD files), `figures/` (`fig01..fig13.{png,pdf}`).
- Top-level & `docs/`: `README.md`, `BTP_Complete_Project_Guide.pdf`, `docs/IEEE_BTP_Research_Paper.{md,pdf}`, `docs/BTP_Final_Presentation.{html,pdf}` (note: per explicit user instruction, PPT and research paper will not be created/regenerated).

## 4. `.gitignore` & Reproducibility Assessment
- `.gitignore` excludes `.venv/`, `__pycache__/`, `data/raw/*.npz`, and `data/raw_sim/*.npz`, while tracking `!data/manifest.json`, `!configs/**`, `!results/**`, `data/processed/*`, and `models/*`.
- **Reproducibility from Clean Checkout**:
  1. `scripts/run_all.py --config configs/default_50hz.yaml` can regenerate `data/raw/*.npz` and `data/raw_sim/*.npz` deterministically from `seed=42`.
  2. **Identified Reproducibility Gaps at Baseline**:
     - `setup.py` omits `install_requires` and `extras_require`, so `pip install -e ".[dev]"` in a fresh virtual environment does not install runtime dependencies (`ISS-008`).
     - `data/manifest.json`, `data/processed/extraction_report.json`, and `results/dataset_quality_report.md` embed machine-specific absolute paths (`/Users/akshat/Desktop/btp-early-pmu-fsa/...`) (`ISS-007`).

## 5. Baseline Test Execution Summary
- `.venv/bin/python -m pytest tests/ -q` → **`232 passed, 9 warnings in 123.56s`**
- `.venv/bin/python tests/e2e/run_e2e.py --tier all --report-path reports/final_audit/baseline_e2e_report.md` → **`149/149 passed (0 failed, 0 skipped) in 54.83s`**
- Full output archived at `reports/final_audit/baseline_test_results.txt`.

## 6. Checkpoint 0 Evaluation
- **Checkpoint 0 Status**: **PASS**
  - Original commit (`e573837`) and branch states recorded in `starting_repository_state.json`.
  - Dedicated branch `btp-publication-audit` active; `main` and `btp-scientific-validation` untouched.
  - Complete baseline test suite executed and logged prior to any code modification.
