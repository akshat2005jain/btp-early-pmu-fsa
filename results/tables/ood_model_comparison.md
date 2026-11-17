| Regime | N | Validation_Selected_Ridge_MAE_Hz | Validation_Selected_Ridge_RMSE_Hz | Validation_Selected_Ridge_R2 | Proposed_GBDT_MAE_Hz | Proposed_GBDT_RMSE_Hz | Proposed_GBDT_R2 | Calibrated_GBDT_F1 | Calibrated_GBDT_FNR_Insecure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ID Held-Out Test | 52 | 0.0034 | 0.0046 | 0.9997 | 0.0077 | 0.0113 | 0.9980 | 0.9767 | 0.0323 |
| OOD-RES (40-50%) | 180 | 0.0054 | 0.0076 | 0.9995 | 0.0267 | 0.0720 | 0.9593 | 1.0000 | 0.0000 |
| OOD-Load (0.80, 1.20) | 180 | 0.0046 | 0.0070 | 0.9992 | 0.0078 | 0.0105 | 0.9981 | 0.9792 | 0.0353 |
| OOD-Combined (RES + Load) | 120 | 0.0069 | 0.0096 | 0.9992 | 0.0217 | 0.0584 | 0.9707 | 0.9851 | 0.0115 |
| OOD-All | 480 | 0.0054 | 0.0080 | 0.9994 | 0.0184 | 0.0533 | 0.9753 | 0.9860 | 0.0132 |