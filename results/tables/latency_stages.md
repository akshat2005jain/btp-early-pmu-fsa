| Stage | Median (ms) | Mean (ms) | Std (ms) | P95 (ms) | P99 (ms) |
| --- | --- | --- | --- | --- | --- |
| 1. Causal Window Slicing | 0.1420 | 0.1476 | 0.0259 | 0.2046 | 0.2198 |
| 2. 14-Feature Extraction | 0.8208 | 0.8515 | 0.1556 | 1.0204 | 1.6300 |
| 3. Nadir Regression Inference | 0.7493 | 0.7603 | 0.1185 | 0.9492 | 1.0514 |
| 4. Security Classification Inference | 0.6353 | 0.6612 | 0.1240 | 0.8147 | 0.9522 |
| 5. Probability Calibration & Decision | 0.0293 | 0.0303 | 0.0093 | 0.0470 | 0.0633 |
| 6. Total End-to-End Online Prediction | 2.3805 | 2.4341 | 0.3409 | 2.9045 | 3.3641 |
| Reference: 10s Time-Domain ODE Simulation | 125.3759 | 127.5130 | 26.3144 | 161.1115 | 169.6625 |