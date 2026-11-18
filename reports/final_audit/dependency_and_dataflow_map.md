# Phase 1 — Dependency and End-to-End Dataflow Map

## 1. End-to-End Pipeline Architecture

```mermaid
flowchart TD
    CFG["configs/default_50hz.yaml"] --> GEN["scripts/generate_dataset.py"]
    NET["src/pmu_fsa/simulation/ieee39_data.py<br/>IEEE 39-Bus, 10-Gen, 50 Hz AC PF"] --> GEN
    REN["src/pmu_fsa/simulation/renewables.py<br/>Approach B: Co-located GFL/GFM Derating"] --> GEN
    DYN["src/pmu_fsa/simulation/dynamic_engine.py<br/>Swing + IEEEG1 ODE + Kron Y_red"] --> GEN
    PMU["src/pmu_fsa/simulation/pmu_recorder.py<br/>50 fps Phasor Sampling + Causal lfilter"] --> GEN
    SCEN["src/pmu_fsa/simulation/scenario_generator.py<br/>270 ID + 480 OOD Scenarios, Split Assignment"] --> GEN

    GEN --> RAW_SIM["data/raw_sim/scenario_XXXX.npz<br/>Ground-Truth COI & Rotor States (Audit Only)"]
    GEN --> RAW_PMU["data/raw/scenario_XXXX.npz<br/>Observable PMU Channels + Scalar Labels"]
    GEN --> MAN["data/manifest.json<br/>SHA-256 Checksums + Partition Hashes + Config"]

    RAW_PMU --> VAL["src/pmu_fsa/simulation/dataset_validator.py<br/>results/dataset_quality_report.md"]
    MAN --> VAL

    RAW_PMU --> EXT["scripts/extract_features.py<br/>src/pmu_fsa/features/window_slicer.py + extractor.py"]
    EXT --> PROC["data/processed/<br/>X_{0.5,1.0,1.5,2.0}s.npy, y_nadir.npy, y_secure.npy, metadata.csv"]

    PROC --> TRN["scripts/train_models.py<br/>train=148 fit, val=70 select & calibrate"]
    TRN --> MOD["models/*.joblib + models/training_metrics.json<br/>artifact_metadata_ + Feature Schema v2.0"]

    MOD --> EVAL["scripts/evaluate_all.py"]
    PROC --> EVAL
    RAW_PMU --> EVAL

    EVAL --> SENS["evaluation/sensitivity.py<br/>Strict Pre-Nadir Window Sweep"]
    EVAL --> OOD["evaluation/ood_generalization.py<br/>3 OOD Axes + Combined"]
    EVAL --> NOISE["evaluation/noise_stress.py<br/>Clean, 45, 40, 34, 26 dB + TVE=1%"]
    EVAL --> INT["evaluation/interpretability.py<br/>MDI, Permutation, Single-Family & Group Ablations"]
    EVAL --> STAT["evaluation/statistical_tests.py<br/>Sign-Flip Perm + Paired Bootstrap CI + Holm"]
    EVAL --> LAT["evaluation/latency.py<br/>100+ Rep Production Artifact Timing"]
    EVAL --> CAL["evaluation/calibration_metrics.py<br/>ECE, MCE, Brier, Threshold Sweep"]

    SENS & OOD & NOISE & INT & STAT & LAT & CAL --> TABS["results/tables/*.csv & *.md<br/>results/evaluation_report.md"]
    TABS & MOD & PROC & RAW_PMU --> FIGS["scripts/generate_figures.py<br/>results/figures/fig01..fig14 (.png & .pdf)"]
```

---

## 2. Causal Firewall & Partition Isolation Boundaries

| Boundary | Allowed Inputs | Strictly Forbidden Inputs | Enforcement Mechanism |
| :--- | :--- | :--- | :--- |
| **PMU Feature Extraction (`extractor.py`)** | `t`, `bus_frequencies`, `bus_rocof`, `voltage_magnitudes`, `voltage_angles`, `pmu_sys_frequency` within $[t_0, t_0 + T_{\mathrm{obs}}]$ | `coi_frequency`, `rotor_angles`, `rotor_speeds`, `mech_powers`, `h_eff`, `p_lost_mw`, `ren_penetration`, samples at $t > t_0 + T_{\mathrm{obs}}$ | Hard check on `bus_frequencies` & `voltage_angles` presence; zero fallback to `coi_frequency`; `WindowSlicer` truncates at $t_0 + T_{\mathrm{obs}}$. |
| **Strict Pre-Nadir Gate (`strict_nadir=True`)** | Scenarios satisfying $t_0 + T_{\mathrm{obs}} < t_{\mathrm{nadir}}$ (`lead_time > 0`) | Any scenario where $t_{\mathrm{nadir}} \le t_0 + T_{\mathrm{obs}}$ (e.g., 8 OOD cases at $T_{\mathrm{obs}} = 2.0\text{ s}$) | Enforced identically in `extract_features.py`, `WindowSensitivityBenchmark`, `OODGeneralizationEvaluator`, `NoiseStressTest`, and `InterpretabilityAnalyzer`. |
| **Model Fitting (`train_models.py`)** | `split == "train"` (`148` ID scenarios) | `val` (`70`), `test` (`52`), `ood_test` (`480`) | Scenario-level partition mask verified by `DatasetValidator` and `partition_scenario_hashes`. |
| **Model Selection & Calibration (`train_models.py`, `calibrated_gb.py`)** | `split == "val"` (`70` ID scenarios) | `test` (`52`), `ood_test` (`480`) | Validation MAE selects `best_regressor` (`linear` Ridge); validation Brier selects `calibrated_gb` (`Platt`); calibrator sigmoid/isotonic map fit strictly on `val`. |
| **Held-Out & OOD Reporting (`evaluate_all.py`)** | `split == "test"` (`52` ID) and `split == "ood_test"` (`480` OOD; `472` at `2.0 s`) | Zero hyperparameter refitting, zero threshold tuning on test | Read-only `.joblib` estimators evaluated once per test protocol. |

---

## 3. Module Dependency Inventory

| Package Layer | Module | Upstream Dependencies | Downstream Consumers |
| :--- | :--- | :--- | :--- |
| **Simulation** | `simulation/ieee39_data.py` | `pandapower`, `numpy` | `renewables.py`, `dynamic_engine.py`, `scenario_generator.py`, `generate_figures.py` |
| **Simulation** | `simulation/renewables.py` | `numpy`, `ieee39_data.py` | `dynamic_engine.py`, `scenario_generator.py` |
| **Simulation** | `simulation/dynamic_engine.py` | `scipy.integrate`, `numpy`, `ieee39_data.py`, `renewables.py` | `scenario_generator.py` |
| **Simulation** | `simulation/pmu_recorder.py` | `scipy.signal` (`butter`, `lfilter`), `numpy` | `scenario_generator.py`, `noise_stress.py`, `latency.py` |
| **Simulation** | `simulation/scenario_generator.py` | `ieee39_data`, `renewables`, `dynamic_engine`, `pmu_recorder` | `scripts/generate_dataset.py` |
| **Simulation** | `simulation/dataset_validator.py` | `numpy`, `pandas`, `extractor.py` | `scripts/generate_dataset.py`, `scripts/extract_features.py` |
| **Features** | `features/window_slicer.py` | `numpy` | `features/extractor.py` |
| **Features** | `features/extractor.py` | `numpy`, `pandas`, `window_slicer.py` | `extract_features.py`, `sensitivity.py`, `noise_stress.py`, `interpretability.py`, `latency.py` |
| **Models** | `models/regressors.py` | `scikit-learn`, `numpy` | `train_models.py`, `evaluate_all.py`, `sensitivity.py`, `ood_generalization.py` |
| **Models** | `models/classifiers.py` | `scikit-learn`, `numpy` | `train_models.py`, `evaluate_all.py`, `calibrated_gb.py` |
| **Models** | `models/calibrated_gb.py` | `scikit-learn`, `scipy.optimize`, `classifiers.py` | `train_models.py`, `evaluate_all.py`, `sensitivity.py`, `noise_stress.py` |
| **Evaluation** | `evaluation/calibration_metrics.py` | `numpy` | `train_models.py`, `evaluate_all.py`, `sensitivity.py`, `ood_generalization.py` |
| **Evaluation** | `evaluation/sensitivity.py` | `extractor`, `regressors`, `calibrated_gb`, `calibration_metrics` | `evaluate_all.py` |
| **Evaluation** | `evaluation/ood_generalization.py` | `calibration_metrics`, `numpy`, `pandas` | `evaluate_all.py` |
| **Evaluation** | `evaluation/noise_stress.py` | `pmu_recorder`, `extractor`, `calibration_metrics` | `evaluate_all.py` |
| **Evaluation** | `evaluation/interpretability.py` | `scikit-learn`, `extractor`, `regressors`, `calibrated_gb` | `evaluate_all.py` |
| **Evaluation** | `evaluation/statistical_tests.py` | `scipy.stats`, `numpy`, `pandas` | `evaluate_all.py` |
| **Evaluation** | `evaluation/latency.py` | `time.perf_counter`, `pmu_recorder`, `extractor` | `evaluate_all.py` |
| **Pipeline / CLI** | `scripts/run_all.py` & `pipeline/run_experiments.py` | All `scripts/*.py` & `pmu_fsa.*` modules | Clean-environment end-to-end reproduction |
