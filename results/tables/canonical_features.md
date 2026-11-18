| Index | Feature | Group | Unit | Definition |
| --- | --- | --- | --- | --- |
| 1 | rocof_init | pmu_causal | - | OLS_slope(t, f_sys_pmu) over [t0, min(t0 + 0.10, t_end)] |
| 2 | rocof_peak_neg | pmu_causal | - | min_{t in W} rocof_sys_pmu(t) |
| 3 | rocof_end | pmu_causal | - | rocof_sys_pmu(t0 + T_obs) |
| 4 | rocof_mean | pmu_causal | - | OLS_slope(t, f_sys_pmu) over [t0, t0 + T_obs] |
| 5 | freq_drop_end | pmu_causal | - | f_sys_pmu(t0 + T_obs) - f_sys_pmu(t0) |
| 6 | freq_drop_max | pmu_causal | - | min_{t in W} f_sys_pmu(t) - f_nominal |
| 7 | freq_dev_integral | pmu_causal | - | int_{t0}^{t0 + T_obs} (f_sys_pmu(t) - f_nominal) dt |
| 8 | bus_drop_worst | pmu_causal | - | nanmin_{t in W, b in B_obs} (f_bus(t, b) - f_nominal) |
| 9 | area_freq_spread_max | pmu_causal | - | max_{t in W} max_{a, b in {1,2,3}} |mean_Area_a(t) - mean_Area_b(t)| |
| 10 | bus_freq_std_mean | pmu_causal | - | mean_{t in W} nanstd_{b in B_obs} f_bus(t, b) |
| 11 | angle_divergence_max | pmu_causal | - | max_{t in W} max_{a, b in B_obs} |wrap_180((theta_a(t)-theta_a(0)) - (theta_b(t)-theta_b(0)))| |
| 12 | tie_line_angle_excursion | pmu_causal | - | max_{t in W} max_{(a,b) in Ties} |wrap_180((theta_a(t)-theta_b(t)) - (theta_a(0)-theta_b(0)))| |
| 13 | freq_curvature_init | pmu_causal | - | OLS_slope(tau, rocof_sys_pmu) over [t0, min(t0 + 0.20, t_end)] |
| 14 | freq_curvature_mean | pmu_causal | - | OLS_slope(tau, rocof_sys_pmu) over [t0, t0 + T_obs] |