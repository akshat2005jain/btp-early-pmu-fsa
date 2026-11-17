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