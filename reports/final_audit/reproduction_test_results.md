# Phase 14 & Phase 15 — Automated Test Suite & Clean Reproduction Verification Results

## 1. Automated Test Suite Summary

| Suite | Command | Total | Passed | Failed | Skipped | Warnings | Duration | Log / Artifact |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Baseline Pytest (Phase 0)** | `.venv/bin/python -m pytest tests/ -q` | 232 | 232 | 0 | 0 | 9 | 123.56 s | `reports/final_audit/baseline_test_results.txt` |
| **Baseline E2E (Phase 0)** | `.venv/bin/python tests/e2e/run_e2e.py --tier all` | 149 | 149 | 0 | 0 | 6 | 54.83 s | `reports/final_audit/baseline_e2e_report.md` |
| **Final Pytest (With 5 New Audit Regression Tests)** | `.venv/bin/python -m pytest tests/ -q` | **237** | **237** | **0** | **0** | **9** | 172.23 s | `reports/final_audit/final_pytest_results.txt` |
| **Final 5-Tier E2E Runner** | `.venv/bin/python tests/e2e/run_e2e.py --tier all` | **149** | **149** | **0** | **0** | **6** | 63.42 s | `reports/final_audit/reproduction_e2e_report.md` |

---

## 2. New Regression Tests Added During Audit

1. `tests/test_simulation.py::test_rhs_complete_trip_with_renewable_droop_zeros_fourth_state`: Verifies `DynamicSimulator.rhs()` zeros the 4th converter-droop state upon complete generator disconnection when `enable_droop=True` (`ISS-004`).
2. `tests/test_feature_causality.py::TestCausalityAndZeroLeakage::test_missing_pmu_channels_fail_without_coi_fallback`: Verifies removing `bus_frequencies` or `voltage_angles` while leaving `coi_frequency` present raises `KeyError`/`ValueError` rather than falling back to simulator COI (`ISS-003`).
3. `tests/test_benchmarks.py::TestAuditRegressionGuardrails::test_strict_pre_nadir_exclusion_and_common_intersection_in_sensitivity`: Verifies `WindowSensitivityBenchmark` excludes non-causal `2.0 s` scenarios (`t_lead_min > 0`) and constructs the exact `common_causal_intersection` (`ISS-001`).
4. `tests/test_benchmarks.py::TestAuditRegressionGuardrails::test_sign_flip_permutation_holm_and_wilcoxon_disagreement_handling`: Verifies null-centered sign-flip permutation p-value, step-down Holm FWER adjustment, and rejection of false positives where Wilcoxon ranks tiny offsets against one outlier (`ISS-002`).
5. `tests/test_benchmarks.py::TestAuditRegressionGuardrails::test_single_family_drop_ablations_ten_rows` + CLI artifact checks: Verifies 10-row single-family drop ablations (`no_rocof`, `no_freq_level`, `no_curvature`), embedded `artifact_metadata_` on `.joblib` models, and removal of duplicate `calibrated_gb_isotonic.joblib` (`ISS-005`, `ISS-006`, `ISS-009`).

---

## 3. Explanation of All 9 Expected `UserWarning`s (Zero Suppressed Warnings)

Per Phase 14 rule 18, no legitimate production guardrails were silenced. Every `UserWarning` emitted during `pytest` arises from intentional negative/boundary tests exercising safety fallbacks:
1. `test_t2_isotonic_regression_constant_target` (2 warnings): Passes `N_cal = 4 < 30` with constant target `[49.6, 49.6, 49.6, 49.6]`, intentionally triggering the `N_cal < 30` isotonic->Platt fallback warning and single-class physics-default sigmoid warning.
2. `test_t2_window_sensitivity_exceeding_trajectory_length` (1 warning): Calls `extract_causal_window(..., t_obs = max_t + 10.0, strict_nadir=False)` to test truncation + pre-nadir causality warning emission.
3. `test_t5_cal_pava_sigmoid_degenerate_fallback_and_interval_ordering` (1 warning): Passes adversarial inverted/flat labels to verify non-positive Platt slope (`A <= 0`) guardrail fallback.
4. `test_t5_benchmarks_edge_cases` (2 warnings): Runs `WindowSensitivityBenchmark` on a single dummy scenario (`N = 1`), triggering single-sample calibration guardrail warnings.
5. `test_probe_window_slicer_extremes` (1 warning): Probes `t_obs = 6.0 s > t_nadir = 2.5 s` with `strict_nadir=False` to verify pre-nadir warning generation.
6. `test_pipeline_end_to_end_and_deterministic_seed_isolated` (2 warnings): Runs an isolated tiny 18-scenario mini-pipeline (`N_val = 9 < 30`), triggering the `N_cal < 30` isotonic->Platt guardrail warning across two identical-seed determinism runs.

*(Note: On the actual 750-scenario benchmark dataset [`N_cal = 218 >= 30`], `fallback_used = False` for both Platt and PAVA Isotonic calibration with zero calibration warnings.)*
