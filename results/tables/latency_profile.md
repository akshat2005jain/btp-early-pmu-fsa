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