# PMU-Based Early Frequency-Security Assessment: Verified Benchmark Evaluation Report

- **System Standard**: `50.0 Hz` nominal, `49.50 Hz` contingency security threshold
- **Loaded Scenarios**: `750` (`synthetic_fallback=False`)
- **Validation-Selected Best Regressor**: `linear` (`best_regressor.joblib`)
- **Proposed Nonlinear Regressor**: `gbdt` (`gbdt.joblib`)
- **Validation-Selected Best Classifier**: `calibrated_gb_platt` (`calibrated_gb.joblib`)
- **Loaded Model Artifacts**: `['analytical', 'best_classifier', 'best_regressor', 'calibrated_gb', 'calibrated_gb_iso', 'calibrated_gb_platt', 'dt', 'dt_clf', 'gbdt', 'gbdt_clf', 'lgbm', 'lgbm_clf', 'linear', 'logistic_clf', 'mean', 'rf', 'rf_clf']`

## 1. Observation Window Sensitivity ($T_{\text{obs}} \in [0.5, 2.0]$ s — Proposed GBDT + Calibrated GBDT)

### 1A. Held-Out ID Test (`N = 52`, 100% Strictly Pre-Nadir)
| window_obs | mae | rmse | r2 | f1 | ece | evaluation_population | n_causal | n_eligible | n_excluded | n_total | causal_valid_ratio | excluded_scenario_ids | t_lead | t_lead_median | t_lead_min | bias | median_ae | p95_ae | precision | recall | pr_auc | roc_auc | brier_score | fnr | fpr_secure | fnr_insecure | missed_insecure_count | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.0124 | 0.0167 | 0.9956 | 1.0000 | 0.0457 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 1.7942 | 1.7800 | 1.5800 | -0.0031 | 0.0086 | 0.0366 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0115 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0124 | 0.0167 | 1.0000 |
| 1.0000 | 0.0077 | 0.0113 | 0.9980 | 0.9767 | 0.0267 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 1.2942 | 1.2800 | 1.0800 | 0.0014 | 0.0059 | 0.0179 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 0.0141 | 0.0000 | 0.0000 | 0.0323 | 1 | 0.0077 | 0.0113 | 0.9767 |
| 1.5000 | 0.0069 | 0.0118 | 0.9978 | 1.0000 | 0.0441 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 0.7942 | 0.7800 | 0.5800 | -0.0001 | 0.0046 | 0.0188 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0112 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0069 | 0.0118 | 1.0000 |
| 2.0000 | 0.0074 | 0.0136 | 0.9971 | 0.9767 | 0.0258 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 0.2942 | 0.2800 | 0.0800 | 0.0023 | 0.0041 | 0.0232 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 0.0172 | 0.0000 | 0.0000 | 0.0323 | 1 | 0.0074 | 0.0136 | 0.9767 |

### 1B. Held-Out OOD Test (`window_specific_causal`: non-causal `2.0 s` cases excluded)
| window_obs | mae | rmse | r2 | f1 | ece | evaluation_population | n_causal | n_eligible | n_excluded | n_total | causal_valid_ratio | excluded_scenario_ids | t_lead | t_lead_median | t_lead_min | bias | median_ae | p95_ae | precision | recall | pr_auc | roc_auc | brier_score | fnr | fpr_secure | fnr_insecure | missed_insecure_count | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.0279 | 0.0621 | 0.9664 | 0.9831 | 0.0319 | window_specific_causal | 480 | 480 | 0 | 480 | 1.0000 |  | 1.7749 | 1.7800 | 1.2800 | 0.0094 | 0.0107 | 0.1180 | 0.9831 | 0.9831 | 0.9988 | 0.9993 | 0.0161 | 0.0169 | 0.0169 | 0.0099 | 3 | 0.0279 | 0.0621 | 0.9831 |
| 1.0000 | 0.0184 | 0.0533 | 0.9753 | 0.9860 | 0.0224 | window_specific_causal | 480 | 480 | 0 | 480 | 1.0000 |  | 1.2749 | 1.2800 | 0.7800 | 0.0106 | 0.0062 | 0.0559 | 0.9778 | 0.9944 | 0.9998 | 0.9999 | 0.0086 | 0.0056 | 0.0056 | 0.0132 | 4 | 0.0184 | 0.0533 | 0.9860 |
| 1.5000 | 0.0218 | 0.0580 | 0.9707 | 0.9943 | 0.0304 | window_specific_causal | 480 | 480 | 0 | 480 | 1.0000 |  | 0.7749 | 0.7800 | 0.2800 | 0.0083 | 0.0067 | 0.0874 | 1.0000 | 0.9887 | 0.9999 | 0.9999 | 0.0100 | 0.0113 | 0.0113 | 0.0000 | 0 | 0.0218 | 0.0580 | 0.9943 |
| 2.0000 | 0.0229 | 0.0610 | 0.9679 | 0.9861 | 0.0346 | window_specific_causal | 472 | 472 | 8 | 480 | 0.9833 | scen_gen31_loss573_ren10_ld120_000;scen_gen32_loss650_ren10_ld120_000;scen_gen32_loss650_ren20_ld120_000;scen_gen35_loss650_ren30_ld120_000;scen_gen35_loss650_ren40_ld120_000;scen_gen38_loss332_ren40_ld120_000;scen_gen39_loss1000_ren40_ld110_000;scen_gen39_loss700_ren40_ld120_000 | 0.2812 | 0.2800 | 0.0200 | 0.0100 | 0.0073 | 0.0909 | 0.9725 | 1.0000 | 0.9999 | 0.9999 | 0.0122 | 0.0000 | 0.0000 | 0.0169 | 5 | 0.0229 | 0.0610 | 0.9861 |

### 1C. Held-Out OOD Test (`common_causal_intersection` across all 4 windows)
| window_obs | mae | rmse | r2 | f1 | ece | evaluation_population | n_causal | n_eligible | n_excluded | n_total | causal_valid_ratio | excluded_scenario_ids | t_lead | t_lead_median | t_lead_min | bias | median_ae | p95_ae | precision | recall | pr_auc | roc_auc | brier_score | fnr | fpr_secure | fnr_insecure | missed_insecure_count | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.0272 | 0.0618 | 0.9670 | 0.9831 | 0.0331 | common_causal_intersection | 472 | 472 | 8 | 480 | 0.9833 | scen_gen31_loss573_ren10_ld120_000;scen_gen32_loss650_ren10_ld120_000;scen_gen32_loss650_ren20_ld120_000;scen_gen35_loss650_ren30_ld120_000;scen_gen35_loss650_ren40_ld120_000;scen_gen38_loss332_ren40_ld120_000;scen_gen39_loss1000_ren40_ld110_000;scen_gen39_loss700_ren40_ld120_000 | 1.7812 | 1.7800 | 1.5200 | 0.0103 | 0.0102 | 0.1170 | 0.9831 | 0.9831 | 0.9988 | 0.9993 | 0.0162 | 0.0169 | 0.0169 | 0.0102 | 3 | 0.0272 | 0.0618 | 0.9831 |
| 1.0000 | 0.0184 | 0.0537 | 0.9752 | 0.9860 | 0.0226 | common_causal_intersection | 472 | 472 | 8 | 480 | 0.9833 | scen_gen31_loss573_ren10_ld120_000;scen_gen32_loss650_ren10_ld120_000;scen_gen32_loss650_ren20_ld120_000;scen_gen35_loss650_ren30_ld120_000;scen_gen35_loss650_ren40_ld120_000;scen_gen38_loss332_ren40_ld120_000;scen_gen39_loss1000_ren40_ld110_000;scen_gen39_loss700_ren40_ld120_000 | 1.2812 | 1.2800 | 1.0200 | 0.0110 | 0.0061 | 0.0588 | 0.9778 | 0.9944 | 0.9998 | 0.9999 | 0.0087 | 0.0056 | 0.0056 | 0.0136 | 4 | 0.0184 | 0.0537 | 0.9860 |
| 1.5000 | 0.0218 | 0.0584 | 0.9706 | 0.9943 | 0.0307 | common_causal_intersection | 472 | 472 | 8 | 480 | 0.9833 | scen_gen31_loss573_ren10_ld120_000;scen_gen32_loss650_ren10_ld120_000;scen_gen32_loss650_ren20_ld120_000;scen_gen35_loss650_ren30_ld120_000;scen_gen35_loss650_ren40_ld120_000;scen_gen38_loss332_ren40_ld120_000;scen_gen39_loss1000_ren40_ld110_000;scen_gen39_loss700_ren40_ld120_000 | 0.7812 | 0.7800 | 0.5200 | 0.0087 | 0.0065 | 0.0877 | 1.0000 | 0.9887 | 0.9999 | 0.9999 | 0.0102 | 0.0113 | 0.0113 | 0.0000 | 0 | 0.0218 | 0.0584 | 0.9943 |
| 2.0000 | 0.0229 | 0.0610 | 0.9679 | 0.9861 | 0.0346 | common_causal_intersection | 472 | 472 | 8 | 480 | 0.9833 | scen_gen31_loss573_ren10_ld120_000;scen_gen32_loss650_ren10_ld120_000;scen_gen32_loss650_ren20_ld120_000;scen_gen35_loss650_ren30_ld120_000;scen_gen35_loss650_ren40_ld120_000;scen_gen38_loss332_ren40_ld120_000;scen_gen39_loss1000_ren40_ld110_000;scen_gen39_loss700_ren40_ld120_000 | 0.2812 | 0.2800 | 0.0200 | 0.0100 | 0.0073 | 0.0909 | 0.9725 | 1.0000 | 0.9999 | 0.9999 | 0.0122 | 0.0000 | 0.0000 | 0.0169 | 5 | 0.0229 | 0.0610 | 0.9861 |

### 1D. Validation-Selected Regressor (`linear`) Window Sensitivity (ID Test)
| window_obs | mae | rmse | r2 | f1 | ece | evaluation_population | n_causal | n_eligible | n_excluded | n_total | causal_valid_ratio | excluded_scenario_ids | t_lead | t_lead_median | t_lead_min | bias | median_ae | p95_ae | precision | recall | pr_auc | roc_auc | brier_score | fnr | fpr_secure | fnr_insecure | missed_insecure_count | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.0063 | 0.0093 | 0.9986 | 1.0000 | 0.0457 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 1.7942 | 1.7800 | 1.5800 | -0.0009 | 0.0043 | 0.0206 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0115 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0063 | 0.0093 | 1.0000 |
| 1.0000 | 0.0034 | 0.0046 | 0.9997 | 0.9767 | 0.0267 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 1.2942 | 1.2800 | 1.0800 | -0.0004 | 0.0023 | 0.0095 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 0.0141 | 0.0000 | 0.0000 | 0.0323 | 1 | 0.0034 | 0.0046 | 0.9767 |
| 1.5000 | 0.0030 | 0.0044 | 0.9997 | 1.0000 | 0.0441 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 0.7942 | 0.7800 | 0.5800 | -0.0004 | 0.0023 | 0.0086 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0112 | 0.0000 | 0.0000 | 0.0000 | 0 | 0.0030 | 0.0044 | 1.0000 |
| 2.0000 | 0.0026 | 0.0037 | 0.9998 | 0.9767 | 0.0258 | window_specific_causal | 52 | 52 | 0 | 52 | 1.0000 |  | 0.2942 | 0.2800 | 0.0800 | -0.0002 | 0.0019 | 0.0068 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 0.0172 | 0.0000 | 0.0000 | 0.0323 | 1 | 0.0026 | 0.0037 | 0.9767 |

## 2. Out-of-Distribution Generalization (ID Held-Out Test vs OOD-RES / Load / Combined)

### 2A. Proposed GBDT + Calibrated GBDT
| Regime | N | MAE (Hz) | RMSE (Hz) | Bias (Hz) | P95 AE (Hz) | R2 | F1 | PR-AUC | Brier | ECE | FNR_insecure | Missed_Insecure | Interval_Hz |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID Held-Out Test | 52 | 0.0077 | 0.0113 | 0.0014 | 0.0179 | 0.9980 | 0.9767 | 1.0000 | 0.0141 | 0.0267 | 0.0323 | 1 | 0.6616 |
| OOD-RES (40-50%) | 180 | 0.0267 | 0.0720 | 0.0191 | 0.1271 | 0.9593 | 1.0000 | 1.0000 | 0.0055 | 0.0216 | 0.0000 | 0 | 0.7428 |
| OOD-Load (0.80, 1.20) | 180 | 0.0078 | 0.0105 | 0.0006 | 0.0201 | 0.9981 | 0.9792 | 0.9996 | 0.0118 | 0.0263 | 0.0353 | 3 | 0.6671 |
| OOD-Combined (RES + Load) | 120 | 0.0217 | 0.0584 | 0.0128 | 0.0659 | 0.9707 | 0.9851 | 1.0000 | 0.0084 | 0.0250 | 0.0115 | 1 | 0.7272 |
| OOD-All | 480 | 0.0184 | 0.0533 | 0.0106 | 0.0559 | 0.9753 | 0.9860 | 0.9998 | 0.0086 | 0.0224 | 0.0132 | 4 | 0.7105 |

| Metric | ID Held-Out Test (10-30%) | OOD Test (40-50% / Load Shift) | Generalization Gap |
| --- | --- | --- | --- |
| MAE | 0.0077 | 0.0184 | 0.0107 |
| RMSE | 0.0113 | 0.0533 | 0.0420 |
| BIAS | 0.0014 | 0.0106 | 0.0092 |
| P95_AE | 0.0179 | 0.0559 | 0.0380 |
| R2 | 0.9980 | 0.9753 | -0.0227 |
| F1 | 0.9767 | 0.9860 | -0.0093 |
| PR_AUC | 1.0000 | 0.9998 | -0.0002 |
| ECE | 0.0267 | 0.0224 | -0.0043 |
| BRIER_SCORE | 0.0141 | 0.0086 | -0.0055 |
| FNR_INSECURE | 0.0323 | 0.0132 | -0.0191 |

- **Partition Protocol**: `metadata_partitions` (train=148, val=70, held-out test=52)
- **ID Scenarios**: 270 (evaluated strictly on 52 held-out test)
- **OOD Scenarios**: 480 (RES=180, Load=180, Combined=120)
- **Distribution Shift Detected**: True
- **Generalization Gap (Delta MAE)**: 0.0107 Hz

### 2B. Validation-Selected `linear` vs Proposed `gbdt` Across OOD Regimes
| Regime | N | Validation_Selected_Ridge_MAE_Hz | Validation_Selected_Ridge_RMSE_Hz | Validation_Selected_Ridge_R2 | Proposed_GBDT_MAE_Hz | Proposed_GBDT_RMSE_Hz | Proposed_GBDT_R2 | Calibrated_GBDT_F1 | Calibrated_GBDT_FNR_Insecure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID Held-Out Test | 52 | 0.0034 | 0.0046 | 0.9997 | 0.0077 | 0.0113 | 0.9980 | 0.9767 | 0.0323 |
| OOD-RES (40-50%) | 180 | 0.0054 | 0.0076 | 0.9995 | 0.0267 | 0.0720 | 0.9593 | 1.0000 | 0.0000 |
| OOD-Load (0.80, 1.20) | 180 | 0.0046 | 0.0070 | 0.9992 | 0.0078 | 0.0105 | 0.9981 | 0.9792 | 0.0353 |
| OOD-Combined (RES + Load) | 120 | 0.0069 | 0.0096 | 0.9992 | 0.0217 | 0.0584 | 0.9707 | 0.9851 | 0.0115 |
| OOD-All | 480 | 0.0054 | 0.0080 | 0.9994 | 0.0184 | 0.0533 | 0.9753 | 0.9860 | 0.0132 |

## 3. PMU Measurement Noise Stress Benchmark

### 3A. Proposed GBDT + Calibrated GBDT
| noise_std_hz | mae | rmse | ece | bias | median_ae | p95_ae | r2 | precision | recall | f1 | fnr_insecure | missed_insecure_count | fpr_secure | brier_score | prob_stability_mad | n_features_used | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0077 | 0.0113 | 0.0267 | 0.0014 | 0.0059 | 0.0179 | 0.9980 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0141 | 0.0000 | 14.0000 | 0.0077 | 0.0113 | 0.9767 |
| 0.0050 | 0.0082 | 0.0113 | 0.0380 | 0.0002 | 0.0062 | 0.0228 | 0.9980 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0140 | 0.0007 | 14.0000 | 0.0082 | 0.0113 | 0.9767 |
| 0.0100 | 0.0085 | 0.0119 | 0.0272 | -0.0006 | 0.0060 | 0.0205 | 0.9977 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0139 | 0.0018 | 14.0000 | 0.0085 | 0.0119 | 0.9767 |
| 0.0200 | 0.0101 | 0.0144 | 0.0245 | -0.0035 | 0.0074 | 0.0270 | 0.9967 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0127 | 0.0065 | 14.0000 | 0.0101 | 0.0144 | 0.9767 |
| 0.0500 | 0.0114 | 0.0164 | 0.0418 | -0.0051 | 0.0071 | 0.0390 | 0.9957 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0210 | 0.0131 | 14.0000 | 0.0114 | 0.0164 | 0.9767 |

### 3B. Validation-Selected `linear` Under Identical PMU Noise
| noise_std_hz | mae | rmse | ece | bias | median_ae | p95_ae | r2 | precision | recall | f1 | fnr_insecure | missed_insecure_count | fpr_secure | brier_score | prob_stability_mad | n_features_used | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0034 | 0.0046 | 0.0267 | -0.0004 | 0.0023 | 0.0095 | 0.9997 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0141 | 0.0000 | 14.0000 | 0.0034 | 0.0046 | 0.9767 |
| 0.0050 | 0.0050 | 0.0065 | 0.0380 | -0.0034 | 0.0040 | 0.0136 | 0.9993 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0140 | 0.0007 | 14.0000 | 0.0050 | 0.0065 | 0.9767 |
| 0.0100 | 0.0087 | 0.0107 | 0.0272 | -0.0075 | 0.0069 | 0.0196 | 0.9982 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0139 | 0.0018 | 14.0000 | 0.0087 | 0.0107 | 0.9767 |
| 0.0200 | 0.0183 | 0.0208 | 0.0245 | -0.0180 | 0.0185 | 0.0340 | 0.9931 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0127 | 0.0065 | 14.0000 | 0.0183 | 0.0208 | 0.9767 |
| 0.0500 | 0.0571 | 0.0610 | 0.0418 | -0.0569 | 0.0562 | 0.0880 | 0.9410 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0210 | 0.0131 | 14.0000 | 0.0571 | 0.0610 | 0.9767 |

- **Causal Windowed OLS vs 2-Point ROCOF Variance Ratio**: `0.0003` (`var_ols=0.001616`, `var_2pt=4.642054`)

## 4. Computational Latency & Measured ODE Simulation Speedup (Desktop Benchmark)

| Stage | Median (ms) | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1. Causal Window Slicing | 0.1444 | 0.1650 | 0.0732 | 0.2642 | 0.3963 |
| 2. 14-Feature Extraction | 0.8056 | 0.8721 | 0.1726 | 1.1835 | 1.5245 |
| 3. Nadir Regression Inference | 0.7672 | 0.7992 | 0.1686 | 1.1760 | 1.2869 |
| 4. Security Classification Inference | 0.6516 | 0.7077 | 0.1686 | 1.0592 | 1.1599 |
| 5. Probability Calibration & Decision | 0.0270 | 0.0289 | 0.0076 | 0.0477 | 0.0503 |
| 6. Total End-to-End Online Prediction | 2.5306 | 2.5768 | 0.4348 | 3.2345 | 3.9396 |
| Reference: 10s Time-Domain ODE Simulation | 161.1669 | 167.1225 | 40.6379 | 221.2675 | 231.0011 |

| Metric | Value | Target |
| --- | --- | --- |
| End-to-End Median Online Latency | 2.531 ms | < 10.0 ms |
| End-to-End P95 / P99 Tail Latency | 3.235 / 3.940 ms | < 10.0 ms |
| Measured 10s Dynamic ODE Simulation (Median) | 161.17 ms (hardware_measured=True) | Hardware Reference |
| Computational Speedup Ratio (Median Sim / Median Online) | 63.7x | Sub-Frame Screening (20 ms 50-fps frame; desktop benchmark) |
| Repetitions / Warm-up Runs | 100 / 10 | >= 100 / 10 |
| Latency Compliant (< 10 ms) | True | PASS |

- **Hardware / OS**: `Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz` (12 cores), `macOS-26.6.1-x86_64-i386-64bit`
- **Software Stack**: Python `3.9.6`, NumPy `2.0.2`, SciPy `1.13.1`, scikit-learn `1.6.1`

### 4B. Model Variant Latency Comparison (`N = 100` repetitions)
| Pipeline_Variant | Median_Total_ms | Mean_Total_ms | P95_Total_ms | P99_Total_ms | Regressor_Stage_Median_ms | Nominal_50fps_Frame_ms |
| --- | --- | --- | --- | --- | --- | --- |
| Validation-Selected (linear) + Calibrated GBDT | 2.2810 | 2.3808 | 3.0800 | 3.5657 | 0.5426 | 20.0000 |
| Proposed GBDT + Calibrated GBDT | 2.5310 | 2.5768 | 3.2345 | 3.9396 | 0.7672 | 20.0000 |

## 5. Paired Statistical Significance Tests (Sign-Flip Permutation + Holm FWER Adjustment)

### 5A. Nadir Regression vs Proposed `gbdt` (ID Test `N = 52`)
| reference_model | baseline_model | n_paired | metric | direction | reference_score | baseline_score | delta_ref_minus_base | ci95_low | ci95_high | test_statistic | p_value_permutation | p_holm_adjusted | p_value_bootstrap | p_value_wilcoxon | significant_unadjusted_005 | significant_holm_005 | significant_p005 | practical_significance | test_disagreement_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gbdt | mean | 52 | mae | lower_is_better | 0.0077 | 0.2104 | -0.2027 | -0.2413 | -0.1672 | -0.2027 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | material (>=20 mHz MAE difference) | consistent |
| gbdt | analytical | 52 | mae | lower_is_better | 0.0077 | 0.0042 | 0.0035 | 0.0017 | 0.0056 | 0.0035 | 0.0004 | 0.0016 | 0.0010 | 0.0001 | True | True | True | negligible (<5 mHz; below typical PMU frequency noise floor) | consistent |
| gbdt | linear | 52 | mae | lower_is_better | 0.0077 | 0.0034 | 0.0043 | 0.0025 | 0.0065 | 0.0043 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | negligible (<5 mHz; below typical PMU frequency noise floor) | consistent |
| gbdt | dt | 52 | mae | lower_is_better | 0.0077 | 0.0111 | -0.0034 | -0.0053 | -0.0015 | -0.0034 | 0.0014 | 0.0042 | 0.0005 | 0.0042 | True | True | True | negligible (<5 mHz; below typical PMU frequency noise floor) | consistent |
| gbdt | rf | 52 | mae | lower_is_better | 0.0077 | 0.0091 | -0.0013 | -0.0028 | 0.0000 | -0.0013 | 0.0710 | 0.1420 | 0.0645 | 0.0956 | False | False | False | negligible (<5 mHz; below typical PMU frequency noise floor) | consistent |
| gbdt | gbdt | 52 | mae | lower_is_better | 0.0077 | 0.0077 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False | False | False | self_reference | self_reference |
| gbdt | lgbm | 52 | mae | lower_is_better | 0.0077 | 0.0086 | -0.0009 | -0.0028 | 0.0010 | -0.0009 | 0.3769 | 0.3769 | 0.3648 | 0.0594 | False | False | False | negligible (<5 mHz; below typical PMU frequency noise floor) | consistent |

### 5B. Nadir Regression vs Validation-Selected `linear` (ID Test `N = 52`)
| reference_model | baseline_model | n_paired | metric | direction | reference_score | baseline_score | delta_ref_minus_base | ci95_low | ci95_high | test_statistic | p_value_permutation | p_holm_adjusted | p_value_bootstrap | p_value_wilcoxon | significant_unadjusted_005 | significant_holm_005 | significant_p005 | practical_significance | test_disagreement_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | mean | 52 | mae | lower_is_better | 0.0034 | 0.2104 | -0.2070 | -0.2461 | -0.1715 | -0.2070 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | material (>=20 mHz MAE difference) | consistent |
| linear | analytical | 52 | mae | lower_is_better | 0.0034 | 0.0042 | -0.0008 | -0.0016 | 0.0001 | -0.0008 | 0.0878 | 0.0878 | 0.0710 | 0.0316 | False | False | False | negligible (<5 mHz; below typical PMU frequency noise floor) | Wilcoxon tests rank symmetry of non-zero differences (sensitive to many tiny probability/residual offsets vs rare large errors), whereas primary sign-flip permutation tests mean loss difference E[d_i]=0. |
| linear | linear | 52 | mae | lower_is_better | 0.0034 | 0.0034 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False | False | False | self_reference | self_reference |
| linear | dt | 52 | mae | lower_is_better | 0.0034 | 0.0111 | -0.0077 | -0.0104 | -0.0053 | -0.0077 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | moderate (5-20 mHz MAE difference) | consistent |
| linear | rf | 52 | mae | lower_is_better | 0.0034 | 0.0091 | -0.0056 | -0.0086 | -0.0032 | -0.0056 | 0.0002 | 0.0012 | 0.0020 | 0.0000 | True | True | True | moderate (5-20 mHz MAE difference) | consistent |
| linear | gbdt | 52 | mae | lower_is_better | 0.0034 | 0.0077 | -0.0043 | -0.0066 | -0.0024 | -0.0043 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | negligible (<5 mHz; below typical PMU frequency noise floor) | consistent |
| linear | lgbm | 52 | mae | lower_is_better | 0.0034 | 0.0086 | -0.0052 | -0.0076 | -0.0032 | -0.0052 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | moderate (5-20 mHz MAE difference) | consistent |

### 5C. Nadir Regression vs Validation-Selected `linear` (OOD Test `N = 480`)
| reference_model | baseline_model | n_paired | metric | direction | reference_score | baseline_score | delta_ref_minus_base | ci95_low | ci95_high | test_statistic | p_value_permutation | p_holm_adjusted | p_value_bootstrap | p_value_wilcoxon | significant_unadjusted_005 | significant_holm_005 | significant_p005 | practical_significance | test_disagreement_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| linear | mean | 480 | mae | lower_is_better | 0.0054 | 0.2799 | -0.2745 | -0.2953 | -0.2532 | -0.2745 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | material (>=20 mHz MAE difference) | consistent |
| linear | analytical | 480 | mae | lower_is_better | 0.0054 | 0.0058 | -0.0004 | -0.0009 | 0.0001 | -0.0004 | 0.1568 | 0.1568 | 0.1594 | 0.9368 | False | False | False | negligible (<5 mHz; below typical PMU frequency noise floor) | consistent |
| linear | linear | 480 | mae | lower_is_better | 0.0054 | 0.0054 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False | False | False | self_reference | self_reference |
| linear | dt | 480 | mae | lower_is_better | 0.0054 | 0.0244 | -0.0190 | -0.0236 | -0.0147 | -0.0190 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | moderate (5-20 mHz MAE difference) | consistent |
| linear | rf | 480 | mae | lower_is_better | 0.0054 | 0.0227 | -0.0173 | -0.0226 | -0.0125 | -0.0173 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | moderate (5-20 mHz MAE difference) | consistent |
| linear | gbdt | 480 | mae | lower_is_better | 0.0054 | 0.0184 | -0.0129 | -0.0176 | -0.0089 | -0.0129 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | moderate (5-20 mHz MAE difference) | consistent |
| linear | lgbm | 480 | mae | lower_is_better | 0.0054 | 0.0274 | -0.0220 | -0.0276 | -0.0167 | -0.0220 | 0.0002 | 0.0012 | 0.0005 | 0.0000 | True | True | True | material (>=20 mHz MAE difference) | consistent |

### 5D. Calibrated Probability Brier Loss vs Baselines (Holm-Adjusted)
| reference_model | baseline_model | n_paired | metric | direction | reference_score | baseline_score | delta_ref_minus_base | ci95_low | ci95_high | test_statistic | p_value_permutation | p_holm_adjusted | p_value_bootstrap | p_value_wilcoxon | significant_unadjusted_005 | significant_holm_005 | significant_p005 | practical_significance | test_disagreement_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated_gb_platt | logistic_clf | 52 | brier_score | lower_is_better | 0.0141 | 0.0132 | 0.0009 | -0.0127 | 0.0203 | 0.0009 | 1.0000 | 1.0000 | 0.9220 | 0.0000 | False | False | False | negligible (<0.002 Brier difference) | Wilcoxon tests rank symmetry of non-zero differences (sensitive to many tiny probability/residual offsets vs rare large errors), whereas primary sign-flip permutation tests mean loss difference E[d_i]=0. |
| calibrated_gb_platt | dt_clf | 52 | brier_score | lower_is_better | 0.0141 | 0.0192 | -0.0051 | -0.0292 | 0.0102 | -0.0051 | 1.0000 | 1.0000 | 0.6357 | 0.0000 | False | False | False | small (0.002-0.010 Brier difference) | Wilcoxon tests rank symmetry of non-zero differences (sensitive to many tiny probability/residual offsets vs rare large errors), whereas primary sign-flip permutation tests mean loss difference E[d_i]=0. |
| calibrated_gb_platt | rf_clf | 52 | brier_score | lower_is_better | 0.0141 | 0.0093 | 0.0048 | 0.0008 | 0.0097 | 0.0048 | 0.0410 | 0.2050 | 0.0365 | 0.0000 | True | False | False | small (0.002-0.010 Brier difference) | consistent |
| calibrated_gb_platt | gbdt_clf | 52 | brier_score | lower_is_better | 0.0141 | 0.0192 | -0.0051 | -0.0305 | 0.0097 | -0.0051 | 0.9998 | 1.0000 | 0.6462 | 0.7916 | False | False | False | small (0.002-0.010 Brier difference) | consistent |
| calibrated_gb_platt | lgbm_clf | 52 | brier_score | lower_is_better | 0.0141 | 0.0181 | -0.0040 | -0.0260 | 0.0101 | -0.0040 | 0.9992 | 1.0000 | 0.6642 | 0.5178 | False | False | False | small (0.002-0.010 Brier difference) | consistent |
| calibrated_gb_platt | calibrated_gb_platt | 52 | brier_score | lower_is_better | 0.0141 | 0.0141 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | False | False | False | self_reference | self_reference |
| calibrated_gb_platt | calibrated_gb_iso | 52 | brier_score | lower_is_better | 0.0141 | 0.0080 | 0.0061 | 0.0018 | 0.0124 | 0.0061 | 0.0002 | 0.0012 | 0.0295 | 0.0000 | True | True | True | small (0.002-0.010 Brier difference) | consistent |

### 5E. Probability Decision-Threshold Sweep (`tau_prob` on Held-Out ID Test)
| proba_threshold | tp | tn | fp | fn | missed_insecure_count | false_alarm_count | fnr_insecure | fpr_secure | precision | recall | f1 | accuracy | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | 21.0000 | 30.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0323 | 0.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9808 | 1.0000 | 1.0000 |
| 0.5000 | 21.0000 | 30.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0323 | 0.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9808 | 1.0000 | 1.0000 |
| 0.6000 | 21.0000 | 30.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0323 | 0.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9808 | 1.0000 | 1.0000 |
| 0.7000 | 20.0000 | 31.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0476 | 1.0000 | 0.9524 | 0.9756 | 0.9808 | 1.0000 | 1.0000 |
| 0.8000 | 19.0000 | 31.0000 | 0.0000 | 2.0000 | 0.0000 | 2.0000 | 0.0000 | 0.0952 | 1.0000 | 0.9048 | 0.9500 | 0.9615 | 1.0000 | 1.0000 |
| 0.9000 | 19.0000 | 31.0000 | 0.0000 | 2.0000 | 0.0000 | 2.0000 | 0.0000 | 0.0952 | 1.0000 | 0.9048 | 0.9500 | 0.9615 | 1.0000 | 1.0000 |

### 5F. Physical Security-Threshold Sensitivity (`f_thresh` in 49.20–49.80 Hz)
| f_threshold_hz | id_test_n | id_secure_frac | id_f1 | id_recall | id_fnr_insecure | id_missed_insecure | id_brier | id_ece | ood_n | ood_secure_frac | ood_f1 | ood_fnr_insecure | ood_brier | ood_ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49.2000 | 52.0000 | 0.8077 | 0.9882 | 1.0000 | 0.1000 | 1.0000 | 0.0091 | 0.0243 | 480.0000 | 0.6750 | 0.9969 | 0.0064 | 0.0053 | 0.0136 |
| 49.4000 | 52.0000 | 0.5385 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0051 | 480.0000 | 0.4813 | 1.0000 | 0.0000 | 0.0050 | 0.0190 |
| 49.5000 | 52.0000 | 0.4038 | 0.9767 | 1.0000 | 0.0323 | 1.0000 | 0.0141 | 0.0267 | 480.0000 | 0.3688 | 0.9860 | 0.0132 | 0.0086 | 0.0224 |
| 49.6000 | 52.0000 | 0.2692 | 0.9286 | 0.9286 | 0.0263 | 1.0000 | 0.0182 | 0.0428 | 480.0000 | 0.2417 | 0.9957 | 0.0000 | 0.0055 | 0.0179 |
| 49.7000 | 52.0000 | 0.2115 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0017 | 0.0091 | 480.0000 | 0.1167 | 0.9912 | 0.0024 | 0.0046 | 0.0118 |
| 49.8000 | 52.0000 | 0.0577 | 0.5000 | 0.3333 | 0.0000 | 0.0000 | 0.0262 | 0.0362 | 480.0000 | 0.0437 | 0.7647 | 0.0000 | 0.0132 | 0.0195 |

### 5G. Sample-Complexity Learning Curve (20%–100% Train)
| train_fraction | n_train_samples | mae_mean | mae_std | rmse_mean | r2_mean | f1_mean | f1_std | brier_mean | ece_mean | fnr_insecure_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 30.0000 | 0.0258 | 0.0009 | 0.0369 | 0.9782 | 0.9425 | 0.0139 | 0.0226 | 0.0398 | 0.0215 |
| 0.4000 | 59.0000 | 0.0204 | 0.0033 | 0.0344 | 0.9807 | 0.9845 | 0.0110 | 0.0167 | 0.0328 | 0.0215 |
| 0.6000 | 89.0000 | 0.0129 | 0.0012 | 0.0235 | 0.9911 | 0.9767 | 0.0000 | 0.0170 | 0.0369 | 0.0323 |
| 0.8000 | 118.0000 | 0.0084 | 0.0006 | 0.0112 | 0.9980 | 0.9845 | 0.0110 | 0.0162 | 0.0441 | 0.0215 |
| 1.0000 | 148.0000 | 0.0093 | 0.0000 | 0.0142 | 0.9968 | 0.9767 | 0.0000 | 0.0164 | 0.0354 | 0.0323 |

### 5H. Multi-Seed Stability (Seeds 42–46)
| model | n_seeds | mae_mean | mae_std | rmse_mean | rmse_std | r2_mean | f1_mean | f1_std | ece_mean | ece_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reg_rf | 5 | 0.0112 | 0.0006 | 0.0176 | 0.0009 | 0.9951 | - | - | - | - |
| reg_gbdt | 5 | 0.0085 | 0.0005 | 0.0122 | 0.0016 | 0.9976 | - | - | - | - |
| reg_lgbm | 5 | 0.0121 | 0.0000 | 0.0188 | 0.0000 | 0.9944 | - | - | - | - |
| calibrated_gb_platt | 5 | - | - | - | - | - | 0.9860 | 0.0114 | 0.0392 | 0.0090 |

## 6. Physical Interpretability & Observability Ablations

- **Clean 1.0s Feature Matrix Rank**: `13 / 14` (`freq_drop_end == freq_drop_max` on monotonic pre-nadir ramps; `14 / 14` under PMU noise; max off-diagonal |Spearman| = `1.0000`, multi-seed rank Spearman = `0.8409`)

### 6A. Permutation & Impurity Feature Importance (Held-Out ID Test)
| feature | importance_mean | importance_std | rank | impurity_importance | impurity_rank |
| --- | --- | --- | --- | --- | --- |
| freq_dev_integral | 0.1695 | 0.0142 | 1 | 0.7654 | 1 |
| freq_drop_max | 0.0312 | 0.0030 | 2 | 0.1004 | 2 |
| freq_drop_end | 0.0300 | 0.0029 | 3 | 0.0974 | 3 |
| rocof_mean | 0.0164 | 0.0019 | 4 | 0.0296 | 4 |
| rocof_end | 0.0010 | 0.0004 | 5 | 0.0006 | 8 |
| bus_drop_worst | 0.0006 | 0.0001 | 6 | 0.0023 | 5 |
| bus_freq_std_mean | 0.0005 | 0.0001 | 7 | 0.0003 | 9 |
| freq_curvature_mean | 0.0005 | 0.0003 | 8 | 0.0001 | 10 |
| rocof_peak_neg | 0.0003 | 0.0001 | 9 | 0.0014 | 7 |
| area_freq_spread_max | 0.0003 | 0.0002 | 10 | 0.0000 | 11 |
| freq_curvature_init | 0.0001 | 0.0001 | 11 | 0.0000 | 12 |
| tie_line_angle_excursion | 0.0001 | 0.0000 | 12 | 0.0000 | 14 |
| angle_divergence_max | 0.0000 | 0.0000 | 13 | 0.0000 | 13 |
| rocof_init | -0.0000 | 0.0000 | 14 | 0.0023 | 6 |

### 6B. Feature-Group, Single-Family Drop, Partial 13-Bus PMU & 25-fps Ablations
| ablation | description | n_features | id_mae | id_rmse | id_r2 | id_f1 | id_brier | id_ece | id_fnr_insecure | ood_mae | ood_f1 | ood_ece | ood_fnr_insecure | delta_id_mae_vs_50fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_14 | All 14 Canonical PMU Features (50 fps, 39-bus) | 14 | 0.0093 | 0.0142 | 0.9968 | 0.9767 | 0.0164 | 0.0354 | 0.0323 | 0.0242 | 0.9887 | 0.0294 | 0.0066 | - |
| freq_only | Aggregate Frequency Deviation Only (3 features) | 3 | 0.0092 | 0.0126 | 0.9975 | 0.9767 | 0.0172 | 0.0312 | 0.0323 | 0.0191 | 0.9887 | 0.0301 | 0.0066 | - |
| freq_plus_rocof | Frequency + First-Order ROCOF (7 features) | 7 | 0.0089 | 0.0128 | 0.9974 | 0.9767 | 0.0166 | 0.0318 | 0.0323 | 0.0226 | 0.9887 | 0.0299 | 0.0066 | - |
| no_spatial_angle | Temporal Frequency + ROCOF + Curvature, No Spatial/Angle (9 features) | 9 | 0.0094 | 0.0138 | 0.9970 | 1.0000 | 0.0152 | 0.0501 | 0.0000 | 0.0215 | 0.9915 | 0.0315 | 0.0033 | - |
| spatial_angle_only | Multi-Area Spatial Dispersion & Phase Angles Only (5 features) | 5 | 0.0310 | 0.0444 | 0.9688 | 0.9302 | 0.0433 | 0.0541 | 0.0645 | 0.0505 | 0.9459 | 0.0212 | 0.0264 | - |
| no_rocof | Drop First-Order ROCOF Family (10 features) | 10 | 0.0084 | 0.0136 | 0.9971 | 1.0000 | 0.0147 | 0.0488 | 0.0000 | 0.0205 | 0.9887 | 0.0297 | 0.0066 | - |
| no_freq_level | Drop Aggregate Frequency Level & Integral Family (10 features) | 11 | 0.0144 | 0.0183 | 0.9947 | 0.9767 | 0.0186 | 0.0347 | 0.0323 | 0.0304 | 0.9659 | 0.0221 | 0.0165 | - |
| no_curvature | Drop Second-Order Curvature Family (12 features) | 12 | 0.0100 | 0.0147 | 0.9966 | 0.9767 | 0.0178 | 0.0316 | 0.0323 | 0.0236 | 0.9887 | 0.0290 | 0.0066 | - |
| partial_13bus_pmu | Partial Spatial Observability (13/39 PMU Buses, 33.3%) | 14 | 0.0108 | 0.0174 | 0.9952 | 1.0000 | 0.0160 | 0.0513 | 0.0000 | 0.0238 | 0.9858 | 0.0275 | 0.0066 | - |
| fps_25_vs_50 | Reduced PMU Reporting Rate (25 frames/s vs 50 frames/s baseline) | 14 | 0.0094 | 0.0142 | 0.9968 | 0.9767 | 0.0174 | 0.0366 | 0.0323 | 0.0239 | 0.9915 | 0.0309 | 0.0066 | 0.0001 |
