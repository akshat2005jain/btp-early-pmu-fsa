# Phase-by-Phase Execution Log & Verification Evidence (`Phases 0–17`)

**Repository Branch:** `btp-publication-audit` (branched cleanly from `e573837c249c7ef83a01f635db0d2e2ca2101a32`; `main` and `btp-scientific-validation` untouched; zero remote pushes).

---

## Phase 0 — Establish a Safe Baseline
- **Files Inspected**: Git status/branches, `.venv`, `data/`, `models/`, `results/`, `tests/`.
- **Actions**: Created dedicated local audit branch `btp-publication-audit`; ran baseline `pytest tests/ -q` (`232 passed, 9 warnings`) and `tests/e2e/run_e2e.py --tier all` (`149/149 passed`); wrote `starting_repository_state.json`, `baseline_inventory.md`, `baseline_test_results.txt`, `baseline_e2e_report.md`, and `initial_issue_register.csv`.
- **Checkpoint 0**: **PASS**

## Phase 1 — Complete Repository and Scientific Audit
- **Files Inspected**: All modules in `src/pmu_fsa/`, all scripts in `scripts/`, `configs/default_50hz.yaml`, `README.md`, `data/manifest.json`, `models/training_metrics.json`, `results/evaluation_report.md`, and `tests/`.
- **Problems Found**: 10 active defects (`ISS-001`..`ISS-010`) + verified disposition of all 18 historical issues (`HIST-001`..`HIST-018`).
- **Evidence**: `reports/final_audit/repository_audit.md`, `reports/final_audit/dependency_and_dataflow_map.md`, `reports/final_audit/issue_register.csv`.
- **Checkpoint 1**: **PASS**

## Phase 2 — Lock the Scientific Definitions and Research Claims
- **Actions**: Locked 50.0 Hz IEEE 39-bus 10-machine specification (`H_39 = 42.0 s` regional default, 5% synchronous machine droop, `T_g = 4.20 s`), post-fault surviving-machine COI nadir target, `49.50 Hz` CERC IEGC 2023 contingency floor (100 mHz above `49.40 Hz` UFLS Stage-1), strict pre-nadir rule ($t_0 + T_{\mathrm{obs}} < t_{\mathrm{nadir}}$), falsifiable hypotheses `H1`–`H7`, Approach B scope limits, and 8 bibliographically verified primary citations with DOIs.
- **Evidence**: `reports/final_audit/scientific_claims_audit.md`.
- **Checkpoint 2**: **PASS**

## Phase 3 — Validate and Correct the Physical Simulation (`ISS-004`)
- **Changes Made**:
  - `src/pmu_fsa/simulation/dynamic_engine.py`: Fixed `DynamicSimulator.rhs()` to zero 4th converter-droop state (`state_eff[3 * m + trip_gen_idx] = 0.0`) upon complete unit disconnection when `has_ren_droop=True`.
  - `src/pmu_fsa/simulation/renewables.py`, `README.md`, `scripts/generate_project_guide.py`, `scripts/evaluate_all.py`: Explicitly documented **Approach B** (co-located constant-power GFL effective inertia, damping, and governor participation derating at unchanged terminal bus active-power dispatch) and stated omitted network/PLL effects.
- **Verification**: `tests/test_simulation.py::test_rhs_complete_trip_with_renewable_droop_zeros_fourth_state` PASS + flat-run zero-drift and power-flow convergence tests PASS.
- **Checkpoint 3**: **PASS**

## Phase 4 — PMU Measurement and Feature Pipeline Audit (`ISS-003`)
- **Changes Made**:
  - `src/pmu_fsa/features/window_slicer.py` & `src/pmu_fsa/features/extractor.py`: Removed legacy `coi_frequency` fallback; required non-empty observable PMU `bus_frequencies` and `voltage_angles`, raising `KeyError`/`ValueError` immediately when missing.
- **Verification**: `tests/test_feature_causality.py::TestCausalityAndZeroLeakage::test_missing_pmu_channels_fail_without_coi_fallback` PASS.
- **Checkpoint 4**: **PASS**

## Phase 5 — Define and Enforce Strict Causal Observation Windows (`ISS-001`)
- **Changes Made**:
  - `src/pmu_fsa/evaluation/sensitivity.py`, `interpretability.py`, `noise_stress.py`, `pipeline/run_experiments.py`, and `scripts/extract_features.py`: Enforced strict pre-nadir filtering ($t_0 + T_{\mathrm{obs}} < t_{\mathrm{nadir}}$) by default; recorded `n_eligible`, `n_excluded`, `excluded_scenario_ids`, and exported both `window_specific_causal` (`480, 480, 480, 472`) and `common_causal_intersection` (`472` across all 4 windows).
- **Verification**: `results/tables/window_sensitivity_ood.csv` (`n_excluded=8` at `2.0 s`, `t_lead_min=+0.0200 s`), `window_sensitivity_ood_common.csv`, and `TestAuditRegressionGuardrails.test_strict_pre_nadir_exclusion_and_common_intersection_in_sensitivity` PASS.
- **Checkpoint 5**: **PASS**

## Phase 6 & Phase 7 — Validate Complete Dataset & Guarantee Scenario-Level Partitioning (`ISS-007`, `ISS-008`)
- **Changes Made**:
  - `src/pmu_fsa/simulation/dataset_validator.py` & `scripts/extract_features.py`: Normalized paths to portable POSIX repo-relative strings; validated `data/raw` (750) + `data/raw_sim` (750) + `data/processed` (`750` for `0.5–1.5 s`, `742` for `2.0 s`); enriched `data/manifest.json` with `dataset_version`, `feature_schema_version`, `git_commit`, `simulation_config`, `partition_scenario_hashes` (`train=148`, `val=70`, `test=52`, `ood_test=480`, `zero_overlap_verified=True`), and all 750 `scenario_records`.
- **Verification**: All 14 automated dataset quality checks PASS (`data/manifest.json`, `results/dataset_quality_report.md`).
- **Checkpoint 6**: **PASS**
- **Checkpoint 7**: **PASS**

## Phase 8 & Phase 9 — Rebuild Regression, Classification & Calibrated Models (`ISS-005`, `ISS-006`)
- **Changes Made**:
  - `scripts/train_models.py`: Attached `artifact_metadata_` to all 17 saved `.joblib` models; validated schema on load in `evaluate_all.py`; removed duplicate `calibrated_gb_isotonic.joblib`; selected `calibrated_gb.joblib` via validation Brier (`Platt 0.0056` vs `Isotonic 0.0063`); exported `model_configurations.{csv,md}` and `calibration_comparison.{csv,md}`.
- **Verification**: `models/training_metrics.json`, `results/tables/nadir_regression.csv`, `security_classification.csv`, `calibration_comparison.csv`.
- **Checkpoint 8**: **PASS**
- **Checkpoint 9**: **PASS**

## Phase 10 — Standardise All Evaluation Experiments (`ISS-005`, `ISS-009`)
- **Changes Made**:
  - `scripts/evaluate_all.py` & `src/pmu_fsa/evaluation/interpretability.py`: Evaluated both validation-selected `linear` (Ridge) and proposed `gbdt` across Window Sensitivity, OOD Generalization, Measurement Noise, Latency, and Statistical Tests; added physical security-threshold sensitivity (`49.20–49.80 Hz`) and single-family drop ablations (`no_rocof`, `no_freq_level`, `no_curvature`).
- **Verification**: `results/evaluation_report.md` and `results/tables/*.csv`.
- **Checkpoint 10**: **PASS**

## Phase 11 — Repair the Statistical Analysis (`ISS-002`)
- **Changes Made**:
  - `src/pmu_fsa/evaluation/statistical_tests.py`: Replaced uncentered bootstrap `p_boot` and unadjusted `OR` rule with null-centered paired sign-flip permutation test (`p_value_permutation`), null-centered bootstrap `p_value_bootstrap`, paired bootstrap 95% CI, step-down **Holm (1979) FWER adjustment** (`p_holm_adjusted`, `significant_holm_005`), separate Wilcoxon diagnostic (`test_disagreement_note`), and `practical_significance`.
- **Verification**: `TestAuditRegressionGuardrails.test_sign_flip_permutation_holm_and_wilcoxon_disagreement_handling` PASS; `statistical_tests_classification.csv` confirms zero-crossing CIs (`Platt` vs `Logistic` and `DT`) have `p_holm = 1.0000`, `significant_holm_005 = False`.
- **Checkpoint 11**: **PASS**

## Phase 12 — Correct Latency and Computational-Performance Benchmarking (`ISS-009`)
- **Changes Made**:
  - `src/pmu_fsa/evaluation/latency.py` & `scripts/evaluate_all.py`: Retained all 100 per-repetition timings in `results/tables/latency_trials.csv`; benchmarked both `linear` (`2.281 ms` median) and `gbdt` (`2.531 ms` median) against measured 10 s ODE simulation (`161.17 ms` median, `63.7x` desktop speedup) and `20.0 ms` 50-fps frame interval; removed any `"hard real-time"` claims.
- **Verification**: `results/tables/latency_profile.{csv,md}`, `latency_stages.{csv,md}`, `latency_trials.csv`, `latency_model_comparison.{csv,md}`.
- **Checkpoint 12**: **PASS**

## Phase 13 — Rebuild Figures and Tables From Verified Data (`ISS-009`)
- **Changes Made**:
  - `scripts/generate_figures.py`: Updated `fig01` rank note (`Rank 13/14 Clean, 14/14 Noisy`), updated `fig13` sub-frame budget caption, added `fig14_statistical_and_latency_distributions.{png,pdf}` directly from `statistical_tests_*.csv` and `latency_trials.csv`, and regenerated all 14 PNG + 14 PDF figures (`28` files) plus all reference/experimental CSV & Markdown tables.
- **Checkpoint 13**: **PASS**

## Phase 14 & Phase 15 — Automated Testing, Packaging & Reproducibility (`ISS-007`, `ISS-008`)
- **Changes Made**:
  - Added `pyproject.toml`, `requirements.txt`, updated `setup.py`, wrote `reproduction_instructions.md`, `environment_snapshot.txt`, `reproduction_run_manifest.json`, `reproduction_test_results.md`, and executed full `pytest` (`237` tests) + 5-tier E2E (`149` checks).
- **Checkpoint 14**: **PASS**
- **Checkpoint 15**: **PASS**

## Phase 16 & Phase 17 — Documentation, Team Attribution & Final Acceptance Gate (`ISS-010`)
- **Changes Made**:
  - Updated `README.md`, `pyproject.toml`, and regenerated `BTP_Complete_Project_Guide.pdf` with verified results and all 4 team members (**Aditya Jha, Akshat Jain, Arshdeep Singh, Raj Bathla**) + Mentor (**Dr. Deepak Kumar Gupta**).
  - **Explicit User Override Recorded**: Per explicit user instruction at the end of the prompt (`"dont make ppt or any research paper"`), `docs/IEEE_BTP_Research_Paper.*` and `docs/BTP_Final_Presentation.*` were intentionally left untouched and not regenerated/created.
- **Checkpoint 16**: **BLOCKED (User Override: `"dont make ppt or any research paper"`)** — `README.md`, `BTP_Complete_Project_Guide.pdf`, and verified literature matrix (`scientific_claims_audit.md`) are 100% updated and verified; research paper and PPT updates were explicitly skipped at user request.
- **Checkpoint 17**: **PASS (Technical, Scientific, Dataset, ML, Statistical & Reproducibility Gates 100% PASS; Paper/PPT deferred per explicit user instruction)**
