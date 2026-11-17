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