# Phase 17 — Final Acceptance Audit Checklist

**Audit Date:** 2026-11-18  
**Branch:** `btp-publication-audit`

---

## A. Scientific Validity

- [x] Research question is precise (`reports/final_audit/scientific_claims_audit.md`).
- [x] Project scope remains generation-loss under-frequency prediction on 50.0 Hz IEEE 39-bus benchmark.
- [x] System model is fully documented (`H_39 = 42.0 s` regional default, 5% synchronous droop, `T_g = 4.20 s`, `S_base = 100 MVA`).
- [x] Generator trips are physically represented in the model (`Y_red_post` Norton disconnection, `rhs()` 4th-state droop zeroing fixed, surviving-machine COI).
- [x] Renewable integration claims match the equations and implementation (**Approach B**: co-located constant-power GFL effective inertia/damping/droop derating explicitly stated across code, `README.md`, manifest, and guide).
- [x] Power-flow initialisation and balance are validated (`< 1e-11 pu` mismatch, `0.0000 Hz` pre-disturbance flat-run drift).
- [x] Frequency-nadir target is precisely defined (post-disturbance minimum of surviving-machine COI frequency).
- [x] Security threshold is justified and documented (`49.50 Hz` per CERC IEGC 2023 contingency floor above `49.40 Hz` UFLS Stage-1, plus sensitivity sweep across `49.20–49.80 Hz`).
- [x] PMU-only feature path is enforced (`bus_frequencies` and `voltage_angles` required; zero fallback to `coi_frequency`).
- [x] No hidden simulator-state leakage exists (`data/raw_sim` isolated from `data/raw`; negative tests verified).
- [x] ROCOF estimation is causal (closed-form 5-point backward OLS slope).
- [x] Noise assumptions are consistent (`0, 5, 10, 20, 50 mHz` Gaussian noise propagated into all 14 canonical features).
- [x] Every evaluated observation window is strictly pre-nadir (`750/750` at `0.5–1.5 s`; `472/480` OOD at `2.0 s` with `8` non-causal scenarios excluded and `t_lead_min = +0.0200 s > 0`).
- [x] Limitations are accurately stated (`results/tables/limitations_and_assumptions.md`, `README.md`).

## B. Dataset Validity

- [x] Scenario manifest is accurate (`data/manifest.json` v2.0.0 with 750 `scenario_records` and `simulation_config`).
- [x] Scenario IDs are unique (`750` unique deterministic IDs, Check 1 PASS).
- [x] Dataset checksums are valid (`750` `raw_pmu` + `750` `raw_sim` + `10` `processed` SHA-256 digests verified).
- [x] Data generation is reproducible (`seed = 42`, verified bitwise reproducibility in `test_simulation.py`).
- [x] Feature schema is canonical (`14` `CANONICAL_FEATURE_NAMES`, v2.0.0).
- [x] Targets and labels are verified (`750/750` match `raw_sim` COI minimum and `49.50 Hz` threshold, Checks 8 & 9 PASS).
- [x] No invalid rows are silently retained (`8` non-causal `2.0 s` scenarios explicitly logged and excluded).
- [x] ID/OOD definitions are documented (`270` ID vs `180 ood_res` + `180 ood_load` + `120 ood_combined`).
- [x] Scenario-level partitioning is verified (`148 train` / `70 val` / `52 test` / `480 ood_test`, `partition_scenario_hashes`).
- [x] No train/validation/test/OOD leakage exists (`zero_overlap_verified = True`, Check 11 PASS).

## C. ML Validity

- [x] Analytical and simple ML baselines are evaluated (`Mean`, `Analytical Swing`, `Ridge`, `Logistic`, `DT`).
- [x] Regression models use valid training/validation procedures (fit on `train=148` for validation selection on `val=70`, equalized `train+val=218` refit, zero `test` exposure).
- [x] Classification models use valid training/validation procedures.
- [x] Model selection is traceable (`models/training_metrics.json`, `results/tables/model_configurations.md`).
- [x] Validation-selected (`linear` Ridge) and proposed (`gbdt`) models are clearly distinguished and evaluated side by side across ID, OOD, window, noise, latency, and statistical benchmarks.
- [x] Calibration does not use test data (`5-fold` stratified OOF across `train+val`, `N_cal = 218`, `fallback_used = False`).
- [x] Test data remains untouched during model selection.
- [x] Regression and classification metrics are complete (`MAE, RMSE, Bias, Median AE, P95 AE, R2, Precision, Recall, F1, PR-AUC, ROC-AUC, Brier, ECE, MCE, FNR_insecure, FPR_secure`).
- [x] False-negative analysis is present (`threshold_sweep.md`: `FNR_insecure = 0.0000` at `tau = 0.70`).
- [x] OOD and robustness evaluations are valid (`ood_model_comparison.md`, `measurement_noise.md`, `measurement_noise_ridge.md`).

## D. Statistical Validity

- [x] Paired comparisons are correctly implemented at the scenario level (`N = 52` ID, `N = 480` OOD).
- [x] Bootstrap confidence intervals are verified (`n_boot = 2000` scenario-paired resampling).
- [x] P-value procedure matches its stated null hypothesis (two-sided paired sign-flip permutation test under $H_0: \mathbb{E}[d_i] = 0$, plus null-centered bootstrap; Wilcoxon reported separately without `OR` rule).
- [x] Multiple comparisons are handled (step-down **Holm 1979 FWER adjustment** across all baseline families).
- [x] Effect sizes are reported (`delta_ref_minus_base` with 95% CIs).
- [x] Significance claims are consistent with the results (zero-crossing Brier CIs have `p_holm = 1.0000`, `significant_holm_005 = False`).
- [x] Practical and statistical significance are distinguished (`practical_significance` column & `< 5 mHz` noise-floor threshold).

## E. Software and Reproducibility

- [x] Full unit test suite passes (`237/237` pytest tests PASS).
- [x] Full integration test suite passes (included in `237/237` pytest suite).
- [x] Full E2E test suite passes (`149/149` E2E checks PASS).
- [x] New regression tests pass (`5/5` new audit regression tests PASS).
- [x] Full benchmark execution is recorded (`results/evaluation_report.md`).
- [x] Actual production models are used in evaluation and latency tests (`17` `.joblib` models with verified `artifact_metadata_`).
- [x] No metric manipulation or placeholder evaluation remains.
- [x] No quantitative figure uses unsupported hardcoded data (`fig01`–`fig14` regenerated from final artifacts).
- [x] Paths are portable (repo-relative POSIX paths across `README.md`, `manifest.json`, `extraction_report.json`, and `dataset_quality_report.md`).
- [x] Dependencies are documented (`pyproject.toml`, `requirements.txt`, `setup.py`, `environment_snapshot.txt`).
- [x] Raw-data regeneration is documented (`reports/final_audit/reproduction_instructions.md`).
- [x] Environment, seeds, configurations and code version are recorded (`reproduction_run_manifest.json`, `data/manifest.json`).

## F. Academic Deliverables

- [ ] Final paper matches the verified experiments — **INTENTIONALLY SKIPPED PER EXPLICIT USER INSTRUCTION (`"dont make ppt or any research paper"`)**.
- [x] All citations are real and verified (`reports/final_audit/scientific_claims_audit.md` — 8 verified primary DOIs/standards).
- [x] Figures and tables are regenerated (`14` PNG + `14` PDF figures in `results/figures/`; `31` CSV + `28` MD tables in `results/tables/`).
- [x] Limitations are disclosed (`README.md`, `results/tables/limitations_and_assumptions.md`).
- [x] README and reproduction instructions are current (including all 4 team members: **Aditya Jha, Akshat Jain, Arshdeep Singh, Raj Bathla**).
- [ ] Final presentation matches the paper and code — **INTENTIONALLY SKIPPED PER EXPLICIT USER INSTRUCTION (`"dont make ppt or any research paper"`)**.
- [x] Target-venue formatting requirements & suitability considerations documented (`scientific_claims_audit.md`).
- [x] Supervisor (**Dr. Deepak Kumar Gupta**) and authorship approval requirements are identified.
