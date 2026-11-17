# IEEE 39-Bus PMU-FSA Dataset Quality & Causality Validation Report

- **Generated (UTC)**: `2026-11-17T20:01:49.171010+00:00`
- **Overall Validation Status**: **PASS**
- **Total Scenarios**: `750`
- **Nominal Frequency / Threshold**: `50.0 Hz` / `49.5 Hz`
- **Raw PMU Directory**: `data/raw`
- **Raw Simulation Directory**: `data/raw_sim`
- **Processed Feature Directory**: `data/processed`
- **Aggregate Dataset SHA-256**: `b543ffc51c757ddc36437391eced8df375de0ac793141e80a6e3b9be2aeaab11`

## 1. Scenario-Level Partition & Class Balance Summary

| Partition | Total Scenarios | Secure (`>= 49.50 Hz`) | Insecure (`< 49.50 Hz`) | Secure Fraction |
| :--- | ---: | ---: | ---: | ---: |
| `train` | 148 | 71 | 77 | 0.480 |
| `val` | 70 | 34 | 36 | 0.486 |
| `test` | 52 | 21 | 31 | 0.404 |
| `ood_test` | 480 | 177 | 303 | 0.369 |

### OOD Regime Breakdown

| Regime (`ood_type`) | Total Scenarios | Secure (`1`) | Insecure (`0`) |
| :--- | ---: | ---: | ---: |
| `id` | 270 | 126 | 144 |
| `ood_res` | 180 | 49 | 131 |
| `ood_load` | 180 | 95 | 85 |
| `ood_combined` | 120 | 33 | 87 |

## 2. Phase 5 Quality & Integrity Checklist (14 Automated Checks)

| Check | Status | Summary |
| :--- | :---: | :--- |
| `1_duplicate_scenario_ids` | **PASS** | 0 violations |
| `2_missing_scenarios` | **PASS** | 0 violations |
| `3_duplicate_feature_rows` | **PASS** | 0 violations |
| `4_nan_inf_and_channel_separation` | **PASS** | 0 violations |
| `5_frequency_angle_rocof_units` | **PASS** | 0 violations |
| `6_uniform_sampling_intervals` | **PASS** | 0 violations |
| `7_expected_window_sample_counts` | **PASS** | 0 violations |
| `8_valid_coi_nadir_labels` | **PASS** | 0 violations |
| `9_security_label_consistency` | **PASS** | 0 violations |
| `10_renewable_penetration_consistency` | **PASS** | 0 violations |
| `11_zero_partition_overlap_and_window_consistency` | **PASS** | 0 violations |
| `12_canonical_14_feature_ordering` | **PASS** | 0 violations |
| `13_physical_target_bounds` | **PASS** | 0 violations |
| `14_pre_nadir_causality_audit` | **PASS** | 0 violations |

## 3. Pre-Nadir Causality & Lead-Time Audit (`t_0 + T_obs < t_nadir`)

| Window `T_obs` (s) | Total | Strictly Causal | Violating | Causal Rate | Mean Lead (s) | Median Lead (s) | Min Lead (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 750 | 750 | 0 | 100.00% | 1.772 | 1.780 | 1.280 |
| 1.0 | 750 | 750 | 0 | 100.00% | 1.272 | 1.280 | 0.780 |
| 1.5 | 750 | 750 | 0 | 100.00% | 0.772 | 0.780 | 0.280 |
| 2.0 | 750 | 742 | 8 | 98.93% | 0.273 | 0.280 | -0.220 |

### Scenarios Violating Pre-Nadir Causality
- **`T_obs = 2.0s`** (8 non-causal): `scen_gen31_loss573_ren10_ld120_000, scen_gen32_loss650_ren10_ld120_000, scen_gen32_loss650_ren20_ld120_000, scen_gen35_loss650_ren30_ld120_000, scen_gen35_loss650_ren40_ld120_000, scen_gen38_loss332_ren40_ld120_000, scen_gen39_loss1000_ren40_ld110_000, scen_gen39_loss700_ren40_ld120_000`

## 4. COI Frequency Nadir Distribution

- **Nadir Range (Hz)**: `[48.2556, 49.9109]` (Mean: `49.3694 Hz`, Median: `49.4223 Hz`)
- **Time-to-Nadir Range (s)**: `[2.78, 3.7]` (Median: `3.28 s`)
- **Manifest File**: `data/manifest.json`
