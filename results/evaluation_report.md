# PMU-FSA Evaluation & Robustness Benchmark Report

## 1. Window Sensitivity Benchmark ($T_{obs} \in [0.5, 2.0]$ s)

| window_obs | mae | rmse | r2 | f1 | ece | mae_raw | rmse_raw | f1_raw | brier_score | t_lead | fnr | fnr_insecure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5000 | 0.0531 | 0.0802 | 0.9822 | 0.9293 | 0.0471 | 0.0531 | 0.0802 | 0.9293 | 0.0412 | 6.5620 | 0.1154 | 0.0120 |
| 1.0000 | 0.0531 | 0.0664 | 0.9722 | 0.9307 | 0.0293 | 0.0647 | 0.1001 | 0.9307 | 0.0300 | 6.0620 | 0.0962 | 0.0241 |
| 1.5000 | 0.0531 | 0.0664 | 0.9619 | 0.9400 | 0.0357 | 0.0648 | 0.1172 | 0.9400 | 0.0272 | 5.5620 | 0.0962 | 0.0120 |
| 2.0000 | 0.0531 | 0.0664 | 0.9718 | 0.9400 | 0.0383 | 0.0642 | 0.1008 | 0.9400 | 0.0295 | 5.0620 | 0.0962 | 0.0120 |


## 2. Out-of-Distribution Generalization Benchmark

| Metric | ID (10-30%) | OOD (40-50%) | Generalization Gap |
| --- | --- | --- | --- |
| MAE | 0.0422 | 0.2731 | 0.2308 |
| RMSE | 0.0579 | 0.6201 | 0.5622 |
| R2 | 0.9712 | 0.3036 | -0.6676 |
| F1 | 0.9953 | 0.8547 | 0.1406 |
| ECE | 0.0064 | 0.0812 | 0.0748 |
| BRIER_SCORE | 0.0024 | 0.0700 | 0.0676 |

- **ID Scenarios**: 270
- **OOD Scenarios**: 180
- **Distribution Shift Detected**: True
- **Generalization Gap (Delta MAE)**: 0.2308 Hz


## 3. PMU Measurement Noise Stress Benchmark

| noise_std_hz | mae | rmse | ece | f1 | brier_score | mae_raw | rmse_raw | f1_raw |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0000 | 0.1224 | 0.1955 | 0.0107 | 0.9848 | 0.0114 | 0.1224 | 0.1955 | 0.9848 |
| 0.0050 | 0.2159 | 0.2822 | 0.0337 | 0.9279 | 0.0379 | 0.2159 | 0.2822 | 0.9279 |
| 0.0100 | 0.3349 | 0.4229 | 0.0599 | 0.9003 | 0.0617 | 0.3349 | 0.4229 | 0.9003 |
| 0.0200 | 0.5902 | 0.7539 | 0.0732 | 0.8667 | 0.0762 | 0.5902 | 0.7539 | 0.8667 |
| 0.0500 | 0.9402 | 1.6265 | 0.1994 | 0.6160 | 0.1898 | 1.3052 | 1.6265 | 0.6160 |


## 4. Latency & Inference Profiling Benchmark

| Metric | Value | Budget |
| --- | --- | --- |
| Feature Extraction Latency (t_feat) | 0.279 ms | < 5.0 ms |
| Model Inference Latency (t_infer) | 0.206 ms | < 2.0 ms |
| Total Online Execution (t_total) | 0.485 ms | < 10.0 ms |
| Numerical Dynamic ODE Simulation | 350.0 ms | Baseline Reference |
| Computational Speedup Ratio | 721.6x | > 200x |
| Latency Compliant (< 10 ms) | True | PASS |

