# PMU-Based Early Frequency-Security Assessment: Honest Benchmark Evaluation Report

- **System Standard**: `50.0 Hz` nominal, `49.50 Hz` security threshold
- **Loaded Scenarios**: `750` (`synthetic_fallback=False`)
- **Loaded Model Artifacts**: `['analytical', 'best_classifier', 'best_regressor', 'calibrated_gb', 'calibrated_gb_iso', 'calibrated_gb_isotonic', 'calibrated_gb_platt', 'dt', 'dt_clf', 'gbdt', 'gbdt_clf', 'lgbm', 'lgbm_clf', 'linear', 'logistic_clf', 'mean', 'rf', 'rf_clf']`

## 1. Observation Window Sensitivity ($T_{\text{obs}} \in [0.5, 2.0]$ s — Held-Out ID Test)

| window_obs | mae | rmse | r2 | f1 | ece | n_causal | n_total | causal_valid_ratio | t_lead | t_lead_median | t_lead_min | bias | median_ae | p95_ae | precision | recall | pr_auc | roc_auc | brier_score | fnr | fpr_secure | fnr_insecure | missed_insecure_count | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.0124 | 0.0167 | 0.9956 | 1.0000 | 0.0457 | 52.0000 | 52.0000 | 1.0000 | 1.7942 | 1.7800 | 1.5800 | -0.0031 | 0.0086 | 0.0366 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0124 | 0.0167 | 1.0000 |
| 1.0000 | 0.0077 | 0.0113 | 0.9980 | 0.9767 | 0.0267 | 52.0000 | 52.0000 | 1.0000 | 1.2942 | 1.2800 | 1.0800 | 0.0014 | 0.0059 | 0.0179 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 0.0141 | 0.0000 | 0.0000 | 0.0323 | 1.0000 | 0.0077 | 0.0113 | 0.9767 |
| 1.5000 | 0.0069 | 0.0118 | 0.9978 | 1.0000 | 0.0441 | 52.0000 | 52.0000 | 1.0000 | 0.7942 | 0.7800 | 0.5800 | -0.0001 | 0.0046 | 0.0188 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0112 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0069 | 0.0118 | 1.0000 |
| 2.0000 | 0.0074 | 0.0136 | 0.9971 | 0.9767 | 0.0258 | 52.0000 | 52.0000 | 1.0000 | 0.2942 | 0.2800 | 0.0800 | 0.0023 | 0.0041 | 0.0232 | 0.9545 | 1.0000 | 1.0000 | 1.0000 | 0.0172 | 0.0000 | 0.0000 | 0.0323 | 1.0000 | 0.0074 | 0.0136 | 0.9767 |

### 1B. Observation Window Sensitivity on Held-Out OOD Test

| window_obs | mae | rmse | r2 | f1 | ece | n_causal | n_total | causal_valid_ratio | t_lead | t_lead_median | t_lead_min | bias | median_ae | p95_ae | precision | recall | pr_auc | roc_auc | brier_score | fnr | fpr_secure | fnr_insecure | missed_insecure_count | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.0279 | 0.0621 | 0.9664 | 0.9831 | 0.0319 | 480.0000 | 480.0000 | 1.0000 | 1.7749 | 1.7800 | 1.2800 | 0.0094 | 0.0107 | 0.1180 | 0.9831 | 0.9831 | 0.9988 | 0.9993 | 0.0161 | 0.0169 | 0.0169 | 0.0099 | 3.0000 | 0.0279 | 0.0621 | 0.9831 |
| 1.0000 | 0.0184 | 0.0533 | 0.9753 | 0.9860 | 0.0224 | 480.0000 | 480.0000 | 1.0000 | 1.2749 | 1.2800 | 0.7800 | 0.0106 | 0.0062 | 0.0559 | 0.9778 | 0.9944 | 0.9998 | 0.9999 | 0.0086 | 0.0056 | 0.0056 | 0.0132 | 4.0000 | 0.0184 | 0.0533 | 0.9860 |
| 1.5000 | 0.0218 | 0.0580 | 0.9707 | 0.9943 | 0.0304 | 480.0000 | 480.0000 | 1.0000 | 0.7749 | 0.7800 | 0.2800 | 0.0083 | 0.0067 | 0.0874 | 1.0000 | 0.9887 | 0.9999 | 0.9999 | 0.0100 | 0.0113 | 0.0113 | 0.0000 | 0.0000 | 0.0218 | 0.0580 | 0.9943 |
| 2.0000 | 0.0229 | 0.0607 | 0.9679 | 0.9861 | 0.0342 | 472.0000 | 480.0000 | 0.9833 | 0.2749 | 0.2800 | -0.2200 | 0.0094 | 0.0073 | 0.0886 | 0.9725 | 1.0000 | 0.9999 | 0.9999 | 0.0120 | 0.0000 | 0.0000 | 0.0165 | 5.0000 | 0.0229 | 0.0607 | 0.9861 |

## 2. Out-of-Distribution Generalization (ID Held-Out Test vs OOD-RES / Load / Combined)

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

## 3. PMU Measurement Noise Stress Benchmark

| noise_std_hz | mae | rmse | ece | bias | median_ae | p95_ae | r2 | precision | recall | f1 | fnr_insecure | missed_insecure_count | fpr_secure | brier_score | prob_stability_mad | n_features_used | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.0077 | 0.0113 | 0.0267 | 0.0014 | 0.0059 | 0.0179 | 0.9980 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0141 | 0.0000 | 14.0000 | 0.0077 | 0.0113 | 0.9767 |
| 0.0050 | 0.0082 | 0.0113 | 0.0380 | 0.0002 | 0.0062 | 0.0228 | 0.9980 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0140 | 0.0007 | 14.0000 | 0.0082 | 0.0113 | 0.9767 |
| 0.0100 | 0.0085 | 0.0119 | 0.0272 | -0.0006 | 0.0060 | 0.0205 | 0.9977 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0139 | 0.0018 | 14.0000 | 0.0085 | 0.0119 | 0.9767 |
| 0.0200 | 0.0101 | 0.0144 | 0.0245 | -0.0035 | 0.0074 | 0.0270 | 0.9967 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0127 | 0.0065 | 14.0000 | 0.0101 | 0.0144 | 0.9767 |
| 0.0500 | 0.0114 | 0.0164 | 0.0418 | -0.0051 | 0.0071 | 0.0390 | 0.9957 | 0.9545 | 1.0000 | 0.9767 | 0.0323 | 1.0000 | 0.0000 | 0.0210 | 0.0131 | 14.0000 | 0.0114 | 0.0164 | 0.9767 |

- **Causal Windowed OLS vs 2-Point ROCOF Variance Ratio**: `0.0003` (`var_ols=0.001616`, `var_2pt=4.642054`)

## 4. Online Computational Latency & Measured ODE Simulation Speedup

| Stage | Median (ms) | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1. Causal Window Slicing | 0.1420 | 0.1476 | 0.0259 | 0.2046 | 0.2198 |
| 2. 14-Feature Extraction | 0.8208 | 0.8515 | 0.1556 | 1.0204 | 1.6300 |
| 3. Nadir Regression Inference | 0.7493 | 0.7603 | 0.1185 | 0.9492 | 1.0514 |
| 4. Security Classification Inference | 0.6353 | 0.6612 | 0.1240 | 0.8147 | 0.9522 |
| 5. Probability Calibration & Decision | 0.0293 | 0.0303 | 0.0093 | 0.0470 | 0.0633 |
| 6. Total End-to-End Online Prediction | 2.3805 | 2.4341 | 0.3409 | 2.9045 | 3.3641 |
| Reference: 10s Time-Domain ODE Simulation | 125.3759 | 127.5130 | 26.3144 | 161.1115 | 169.6625 |

| Metric | Value | Target |
| --- | --- | --- |
| End-to-End Median Online Latency | 2.380 ms | < 10.0 ms |
| End-to-End P95 / P99 Tail Latency | 2.905 / 3.364 ms | < 10.0 ms |
| Measured 10s Dynamic ODE Simulation (Median) | 125.38 ms (hardware_measured=True) | Hardware Reference |
| Computational Speedup Ratio (Median Sim / Median Online) | 52.7x | Real-Time Screening |
| Repetitions / Warm-up Runs | 100 / 10 | >= 100 / 10 |
| Latency Compliant (< 10 ms) | True | PASS |

- **Hardware / OS**: `Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz` (12 cores), `macOS-26.6.1-x86_64-i386-64bit`
- **Software Stack**: Python `3.9.6`, NumPy `2.0.2`, SciPy `1.13.1`, scikit-learn `1.6.1`

## 5. Paired Statistical Significance Tests (Bootstrap n=1000 & Wilcoxon)

### 5A. Nadir Regression vs GBDT
| reference_model | baseline_model | metric | reference_score | baseline_score | delta_ref_minus_base | ci95_low | ci95_high | p_value_wilcoxon | p_value_bootstrap | significant_p005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gbdt | mean | mae | 0.0077 | 0.2104 | -0.2027 | -0.2420 | -0.1645 | 0.0000 | 0.0000 | True |
| gbdt | analytical | mae | 0.0077 | 0.0042 | 0.0035 | 0.0017 | 0.0056 | 0.0001 | 0.0000 | True |
| gbdt | linear | mae | 0.0077 | 0.0034 | 0.0043 | 0.0026 | 0.0065 | 0.0000 | 0.0000 | True |
| gbdt | dt | mae | 0.0077 | 0.0111 | -0.0034 | -0.0054 | -0.0015 | 0.0042 | 0.0000 | True |
| gbdt | rf | mae | 0.0077 | 0.0091 | -0.0013 | -0.0027 | -0.0000 | 0.0956 | 0.0440 | True |
| gbdt | gbdt | mae | 0.0077 | 0.0077 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | False |
| gbdt | lgbm | mae | 0.0077 | 0.0086 | -0.0009 | -0.0028 | 0.0010 | 0.0594 | 0.3080 | False |

### 5B. Calibrated Probability Brier Loss vs Baselines
| reference_model | baseline_model | metric | reference_score | baseline_score | delta_ref_minus_base | ci95_low | ci95_high | p_value_wilcoxon | p_value_bootstrap | significant_p005 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated_gb_platt | logistic_clf | brier_score | 0.0141 | 0.0132 | 0.0009 | -0.0131 | 0.0206 | 0.0000 | 0.9640 | True |
| calibrated_gb_platt | dt_clf | brier_score | 0.0141 | 0.0192 | -0.0051 | -0.0298 | 0.0100 | 0.0000 | 0.7440 | True |
| calibrated_gb_platt | rf_clf | brier_score | 0.0141 | 0.0093 | 0.0048 | 0.0007 | 0.0094 | 0.0000 | 0.0200 | True |
| calibrated_gb_platt | gbdt_clf | brier_score | 0.0141 | 0.0192 | -0.0051 | -0.0305 | 0.0095 | 0.7916 | 0.7420 | False |
| calibrated_gb_platt | lgbm_clf | brier_score | 0.0141 | 0.0181 | -0.0040 | -0.0253 | 0.0102 | 0.5178 | 0.8380 | False |
| calibrated_gb_platt | calibrated_gb_platt | brier_score | 0.0141 | 0.0141 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | False |
| calibrated_gb_platt | calibrated_gb_iso | brier_score | 0.0141 | 0.0080 | 0.0061 | 0.0020 | 0.0125 | 0.0000 | 0.0000 | True |

### 5C. Safety Decision-Threshold Sweep (Held-Out Test)
| proba_threshold | tp | tn | fp | fn | missed_insecure_count | false_alarm_count | fnr_insecure | fpr_secure | precision | recall | f1 | accuracy | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3000 | 21.0000 | 30.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0323 | 0.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9808 | 1.0000 | 1.0000 |
| 0.5000 | 21.0000 | 30.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0323 | 0.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9808 | 1.0000 | 1.0000 |
| 0.6000 | 21.0000 | 30.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0323 | 0.0000 | 0.9545 | 1.0000 | 0.9767 | 0.9808 | 1.0000 | 1.0000 |
| 0.7000 | 20.0000 | 31.0000 | 0.0000 | 1.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0476 | 1.0000 | 0.9524 | 0.9756 | 0.9808 | 1.0000 | 1.0000 |
| 0.8000 | 19.0000 | 31.0000 | 0.0000 | 2.0000 | 0.0000 | 2.0000 | 0.0000 | 0.0952 | 1.0000 | 0.9048 | 0.9500 | 0.9615 | 1.0000 | 1.0000 |
| 0.9000 | 19.0000 | 31.0000 | 0.0000 | 2.0000 | 0.0000 | 2.0000 | 0.0000 | 0.0952 | 1.0000 | 0.9048 | 0.9500 | 0.9615 | 1.0000 | 1.0000 |

### 5D. Sample-Complexity Learning Curve (20%–100% Train)
| train_fraction | n_train_samples | mae_mean | mae_std | rmse_mean | r2_mean | f1_mean | f1_std | brier_mean | ece_mean | fnr_insecure_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.2000 | 30.0000 | 0.0258 | 0.0009 | 0.0369 | 0.9782 | 0.9425 | 0.0139 | 0.0226 | 0.0398 | 0.0215 |
| 0.4000 | 59.0000 | 0.0204 | 0.0033 | 0.0344 | 0.9807 | 0.9845 | 0.0110 | 0.0167 | 0.0328 | 0.0215 |
| 0.6000 | 89.0000 | 0.0129 | 0.0012 | 0.0235 | 0.9911 | 0.9767 | 0.0000 | 0.0170 | 0.0369 | 0.0323 |
| 0.8000 | 118.0000 | 0.0084 | 0.0006 | 0.0112 | 0.9980 | 0.9845 | 0.0110 | 0.0162 | 0.0441 | 0.0215 |
| 1.0000 | 148.0000 | 0.0093 | 0.0000 | 0.0142 | 0.9968 | 0.9767 | 0.0000 | 0.0164 | 0.0354 | 0.0323 |

### 5E. Multi-Seed Stability (Seeds 42–46)
| model | n_seeds | mae_mean | mae_std | rmse_mean | rmse_std | r2_mean | f1_mean | f1_std | ece_mean | ece_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reg_rf | 5 | 0.0112 | 0.0006 | 0.0176 | 0.0009 | 0.9951 | - | - | - | - |
| reg_gbdt | 5 | 0.0085 | 0.0005 | 0.0122 | 0.0016 | 0.9976 | - | - | - | - |
| reg_lgbm | 5 | 0.0121 | 0.0000 | 0.0188 | 0.0000 | 0.9944 | - | - | - | - |
| calibrated_gb_platt | 5 | - | - | - | - | - | 0.9860 | 0.0114 | 0.0392 | 0.0090 |

## 6. Physical Interpretability & Observability Ablations

- **Feature Matrix Rank**: `13 / 14` (max off-diagonal |Spearman| = `1.0000`, multi-seed rank Spearman = `0.8409`)

### 6A. Permutation & Impurity Feature Importance (Held-Out Test)
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

### 6B. Feature-Group, Partial 13-Bus PMU & 25-fps Reporting Ablations
| ablation | description | n_features | id_mae | id_rmse | id_r2 | id_f1 | id_brier | id_ece | id_fnr_insecure | ood_mae | ood_f1 | ood_ece | ood_fnr_insecure | delta_id_mae_vs_50fps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_14 | All 14 Canonical PMU Features (50 fps, 39-bus) | 14 | 0.0093 | 0.0142 | 0.9968 | 0.9767 | 0.0164 | 0.0354 | 0.0323 | 0.0242 | 0.9887 | 0.0294 | 0.0066 | - |
| freq_only | Aggregate Frequency Deviation Only (3 features) | 3 | 0.0092 | 0.0126 | 0.9975 | 0.9767 | 0.0172 | 0.0312 | 0.0323 | 0.0191 | 0.9887 | 0.0301 | 0.0066 | - |
| freq_plus_rocof | Frequency + First-Order ROCOF (7 features) | 7 | 0.0089 | 0.0128 | 0.9974 | 0.9767 | 0.0166 | 0.0318 | 0.0323 | 0.0226 | 0.9887 | 0.0299 | 0.0066 | - |
| no_spatial_angle | Temporal Frequency + ROCOF + Curvature, No Spatial/Angle (9 features) | 9 | 0.0094 | 0.0138 | 0.9970 | 1.0000 | 0.0152 | 0.0501 | 0.0000 | 0.0215 | 0.9915 | 0.0315 | 0.0033 | - |
| spatial_angle_only | Multi-Area Spatial Dispersion & Phase Angles Only (5 features) | 5 | 0.0310 | 0.0444 | 0.9688 | 0.9302 | 0.0433 | 0.0541 | 0.0645 | 0.0505 | 0.9459 | 0.0212 | 0.0264 | - |
| partial_13bus_pmu | Partial Spatial Observability (13/39 PMU Buses, 33.3%) | 14 | 0.0108 | 0.0174 | 0.9952 | 1.0000 | 0.0160 | 0.0513 | 0.0000 | 0.0238 | 0.9858 | 0.0275 | 0.0066 | - |
| fps_25_vs_50 | Reduced PMU Reporting Rate (25 frames/s vs 50 frames/s baseline) | 14 | 0.0094 | 0.0142 | 0.9968 | 0.9767 | 0.0174 | 0.0366 | 0.0323 | 0.0239 | 0.9915 | 0.0309 | 0.0066 | 0.0001 |
