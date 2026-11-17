| Partition | Regime | Operating Regime Description | N_Scenarios | Share_Pct | Secure_Count | Insecure_Count | Secure_Rate | Mean_Nadir_Hz | Min_Nadir_Hz | Mean_Lead_Time_1s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | id | ID Train (RES <= 30%, Load 0.90-1.10x) | 148 | 19.7300 | 71 | 77 | 0.4797 | 49.4612 | 48.7188 | 1.2659 |
| val | id | ID Validation (RES <= 30%, Load 0.90-1.10x) | 70 | 9.3300 | 34 | 36 | 0.4857 | 49.4395 | 48.7358 | 1.2540 |
| test | id | ID Held-Out Test (RES <= 30%, Load 0.90-1.10x) | 52 | 6.9300 | 21 | 31 | 0.4038 | 49.4271 | 48.8165 | 1.2942 |
| ood_test | ood_res | OOD-RES (RES 40-50%, Load 0.90-1.10x) | 180 | 24.0000 | 49 | 131 | 0.2722 | 49.2054 | 48.2689 | 1.2904 |
| ood_test | ood_load | OOD-Load (RES <= 30%, Load {0.80, 1.20}x) | 180 | 24.0000 | 95 | 85 | 0.5278 | 49.4874 | 48.7342 | 1.2557 |
| ood_test | ood_combined | OOD-Combined (RES 40-50%, Load {0.80, 1.20}x) | 120 | 16.0000 | 33 | 87 | 0.2750 | 49.2595 | 48.2556 | 1.2803 |
| ood_test (all) | all_ood | All OOD Test Combined | 480 | 64.0000 | 177 | 303 | 0.3688 | 49.3247 | 48.2556 | 1.2749 |
| total | all | Full Dataset Total | 750 | 100.0000 | 303 | 447 | 0.4040 | 49.3694 | 48.2556 | 1.2725 |