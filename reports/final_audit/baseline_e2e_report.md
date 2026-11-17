# E2E Test Suite Execution Report (50.0 Hz Canonical Standard)

- **Timestamp**: `2026-11-17T16:08:40.817605+00:00`
- **Tier**: `Tier all`
- **Total Tests**: `149`
- **Passed**: `149`
- **Failed**: `0`
- **Skipped**: `0`
- **Pass Rate**: `100.0%`
- **Execution Time**: `54.831s`

## Summary Results Table

| Module | Test Case | Status | Duration (s) | Notes |
|---|---|:---:|:---:|---|
| `test_tier1_r1_simulation.py` | `test_f1_branch_connectivity_and_reactances` | PASSED | 0.0004 | - |
| `test_tier1_r1_simulation.py` | `test_f1_bus_topology_and_50hz_nominal_standard` | PASSED | 0.0004 | - |
| `test_tier1_r1_simulation.py` | `test_f1_generator_parameters_and_intertie_modes` | PASSED | 0.0007 | - |
| `test_tier1_r1_simulation.py` | `test_f1_nr_power_flow_equilibrium_and_power_balance` | PASSED | 0.0044 | - |
| `test_tier1_r1_simulation.py` | `test_f1_three_area_partitioning` | PASSED | 0.0003 | - |
| `test_tier1_r1_simulation.py` | `test_f2_complete_generator_disconnection_and_partial_loss_scaling` | PASSED | 0.1630 | - |
| `test_tier1_r1_simulation.py` | `test_f2_kron_admittance_and_50hz_steady_state_equilibrium` | PASSED | 0.0166 | - |
| `test_tier1_r1_simulation.py` | `test_f2_tgov1_governor_and_initial_swing_rocof` | PASSED | 0.1717 | - |
| `test_tier1_r1_simulation.py` | `test_f3_linear_renewable_displacement_without_saturation` | PASSED | 0.0025 | - |
| `test_tier1_r1_simulation.py` | `test_f3_renewable_penetration_deepens_50hz_nadir_and_virtual_inertia_mitigates` | PASSED | 0.2589 | - |
| `test_tier1_r1_simulation.py` | `test_f4_pmu_recorder_sampling_rates_soc_fracsec_noise_and_c37_118` | PASSED | 0.1269 | - |
| `test_tier1_r1_simulation.py` | `test_f5_scenario_batch_metadata_49_5hz_labeling_and_dataset_validator` | PASSED | 1.0807 | - |
| `test_tier1_r2_features.py` | `test_f10_trapezoidal_50hz_frequency_integral` | PASSED | 0.0030 | - |
| `test_tier1_r2_features.py` | `test_f11_interarea_dispersion_and_bus_std` | PASSED | 0.0007 | - |
| `test_tier1_r2_features.py` | `test_f12_wrap_safe_angle_divergence_in_degrees` | PASSED | 0.0007 | - |
| `test_tier1_r2_features.py` | `test_f13_frequency_curvature_hz_per_s2_and_noise_propagation` | PASSED | 0.0003 | - |
| `test_tier1_r2_features.py` | `test_f6_canonical_14_feature_schema_and_window_duration` | PASSED | 0.0018 | - |
| `test_tier1_r2_features.py` | `test_f6_time_range_sample_counts_and_invalid_windows` | PASSED | 0.0023 | - |
| `test_tier1_r2_features.py` | `test_f6_zero_future_and_zero_coi_leakage` | PASSED | 0.0138 | - |
| `test_tier1_r2_features.py` | `test_f7_epicenter_bus_and_disturbance_scaling` | PASSED | 0.0040 | - |
| `test_tier1_r2_features.py` | `test_f7_ols_initial_rocof_50hz_ramp_and_noise_resilience` | PASSED | 0.0111 | - |
| `test_tier1_r2_features.py` | `test_f8_causal_rocof_metrics_ordering` | PASSED | 0.0011 | - |
| `test_tier1_r2_features.py` | `test_f9_frequency_drop_metrics_at_50hz` | PASSED | 0.0027 | - |
| `test_tier1_r3_models.py` | `test_f14_all_seven_production_regressors_fit_predict_save_load` | PASSED | 0.2465 | - |
| `test_tier1_r3_models.py` | `test_f14_prediction_monotonicity_with_50hz_disturbance_size` | PASSED | 0.0057 | - |
| `test_tier1_r3_models.py` | `test_f15_all_security_classifiers_49_5hz_threshold_and_zero_tp_f1` | PASSED | 0.2672 | - |
| `test_tier1_r3_models.py` | `test_f16_calibrated_gbdt_platt_vs_isotonic_and_ood_intervals` | PASSED | 0.8455 | - |
| `test_tier1_r3_models.py` | `test_f17_calibration_metrics_brier_ece_mce_and_10bin_reliability` | PASSED | 0.0045 | - |
| `test_tier1_r4_benchmarks.py` | `test_f18_custom_window_grid_and_causality` | PASSED | 0.8187 | - |
| `test_tier1_r4_benchmarks.py` | `test_f18_window_sensitivity_held_out_unsmoothed_contract` | PASSED | 1.3541 | - |
| `test_tier1_r4_benchmarks.py` | `test_f19_ood_generalization_unclamped_gaps_and_breakdown` | PASSED | 0.3146 | - |
| `test_tier1_r4_benchmarks.py` | `test_f20_measurement_noise_benchmark_unsmoothed_and_ols_resilience` | PASSED | 0.2888 | - |
| `test_tier1_r4_benchmarks.py` | `test_f21_six_stage_hardware_latency_profiler` | PASSED | 1.4815 | - |
| `test_tier1_r4_benchmarks.py` | `test_f21b_interpretability_and_paired_statistical_tests` | PASSED | 1.8026 | - |
| `test_tier1_r4_benchmarks.py` | `test_f22_cli_entrypoints_help_and_isolated_execution` | PASSED | 18.2972 | - |
| `test_tier2_boundaries.py` | `test_t2_active_power_loss_negative_rejection` | PASSED | 0.0007 | - |
| `test_tier2_boundaries.py` | `test_t2_all_negative_class_training_set` | PASSED | 0.0039 | - |
| `test_tier2_boundaries.py` | `test_t2_all_positive_class_training_set` | PASSED | 0.0023 | - |
| `test_tier2_boundaries.py` | `test_t2_asymmetric_interarea_phase_swing` | PASSED | 0.0014 | - |
| `test_tier2_boundaries.py` | `test_t2_brier_score_identical_predictions` | PASSED | 0.0003 | - |
| `test_tier2_boundaries.py` | `test_t2_cli_missing_required_argument` | PASSED | 0.0024 | - |
| `test_tier2_boundaries.py` | `test_t2_cli_nonexistent_input_file` | PASSED | 0.0001 | - |
| `test_tier2_boundaries.py` | `test_t2_cli_nonexistent_output_directory_creation` | PASSED | 0.0014 | - |
| `test_tier2_boundaries.py` | `test_t2_collinear_features_linear_regressor` | PASSED | 0.0034 | - |
| `test_tier2_boundaries.py` | `test_t2_contingency_unknown_bus_trip` | PASSED | 0.0003 | - |
| `test_tier2_boundaries.py` | `test_t2_dropped_pmu_frames_10pct` | PASSED | 0.0011 | - |
| `test_tier2_boundaries.py` | `test_t2_duplicate_timestamps_handling` | PASSED | 0.0013 | - |
| `test_tier2_boundaries.py` | `test_t2_ece_empty_bins_handling` | PASSED | 0.0002 | - |
| `test_tier2_boundaries.py` | `test_t2_ece_single_bin_boundary` | PASSED | 0.0002 | - |
| `test_tier2_boundaries.py` | `test_t2_empty_test_matrix_rejection` | PASSED | 0.0018 | - |
| `test_tier2_boundaries.py` | `test_t2_extreme_class_imbalance_99_to_1` | PASSED | 0.0030 | - |
| `test_tier2_boundaries.py` | `test_t2_extreme_disturbance_deep_plunge` | PASSED | 0.0014 | - |
| `test_tier2_boundaries.py` | `test_t2_extreme_pmu_noise_0_5hz` | PASSED | 0.0018 | - |
| `test_tier2_boundaries.py` | `test_t2_f18_to_f22_benchmark_and_cli_boundaries` | PASSED | 1.5476 | - |
| `test_tier2_boundaries.py` | `test_t2_f1_disjoint_area_partitioning_and_gen_bounds` | PASSED | 0.0021 | - |
| `test_tier2_boundaries.py` | `test_t2_f2_governor_stiff_limit_and_damping_rejection` | PASSED | 0.0027 | - |
| `test_tier2_boundaries.py` | `test_t2_f3_zero_penetration_and_inertia_floor` | PASSED | 0.0027 | - |
| `test_tier2_boundaries.py` | `test_t2_f4_high_reporting_rate_120fps_and_subcycle_uniformity` | PASSED | 0.0015 | - |
| `test_tier2_boundaries.py` | `test_t2_f5_out_of_range_disturbance_time_and_zero_loss` | PASSED | 0.0019 | - |
| `test_tier2_boundaries.py` | `test_t2_f6_window_boundary_and_post_window_invariance` | PASSED | 0.0019 | - |
| `test_tier2_boundaries.py` | `test_t2_f7_to_f13_steady_state_50hz_identities_and_invariants` | PASSED | 0.0026 | - |
| `test_tier2_boundaries.py` | `test_t2_feature_dimension_mismatch_rejection` | PASSED | 0.0032 | - |
| `test_tier2_boundaries.py` | `test_t2_frequency_rise_overfrequency_event` | PASSED | 0.0010 | - |
| `test_tier2_boundaries.py` | `test_t2_generator_trip_at_high_load_vs_light_load` | PASSED | 0.0017 | - |
| `test_tier2_boundaries.py` | `test_t2_high_reporting_rate_200fps` | PASSED | 0.0019 | - |
| `test_tier2_boundaries.py` | `test_t2_identical_bus_frequencies_zero_dispersion` | PASSED | 0.0001 | - |
| `test_tier2_boundaries.py` | `test_t2_incoherent_area_frequency_oscillation_boundary` | PASSED | 0.0010 | - |
| `test_tier2_boundaries.py` | `test_t2_inf_input_rejection` | PASSED | 0.0010 | - |
| `test_tier2_boundaries.py` | `test_t2_isotonic_regression_constant_target` | PASSED | 0.1081 | - |
| `test_tier2_boundaries.py` | `test_t2_isotonic_regression_strictly_decreasing_inputs` | PASSED | 0.2541 | - |
| `test_tier2_boundaries.py` | `test_t2_latency_benchmark_batch_size_1` | PASSED | 1.5826 | - |
| `test_tier2_boundaries.py` | `test_t2_latency_benchmark_large_batch_500` | PASSED | 0.0045 | - |
| `test_tier2_boundaries.py` | `test_t2_latency_timer_precision_positive` | PASSED | 1.5138 | - |
| `test_tier2_boundaries.py` | `test_t2_malformed_scenario_id_handling` | PASSED | 0.0032 | - |
| `test_tier2_boundaries.py` | `test_t2_marginal_nadir_exactly_on_threshold` | PASSED | 0.0038 | - |
| `test_tier2_boundaries.py` | `test_t2_marginal_nadir_infinitesimally_below_threshold` | PASSED | 0.0023 | - |
| `test_tier2_boundaries.py` | `test_t2_massive_generator_trip_2500mw` | PASSED | 0.0014 | - |
| `test_tier2_boundaries.py` | `test_t2_mce_upper_bound_constraint` | PASSED | 0.0005 | - |
| `test_tier2_boundaries.py` | `test_t2_missing_metadata_key_rejection` | PASSED | 0.0015 | - |
| `test_tier2_boundaries.py` | `test_t2_nadir_prediction_clipping_above_50_5` | PASSED | 0.0028 | - |
| `test_tier2_boundaries.py` | `test_t2_nadir_prediction_clipping_below_45` | PASSED | 0.0028 | - |
| `test_tier2_boundaries.py` | `test_t2_nan_input_rejection` | PASSED | 0.0012 | - |
| `test_tier2_boundaries.py` | `test_t2_negative_brier_score_impossible` | PASSED | 0.0002 | - |
| `test_tier2_boundaries.py` | `test_t2_negative_reporting_rate_rejection` | PASSED | 0.0003 | - |
| `test_tier2_boundaries.py` | `test_t2_noise_benchmark_negative_sigma` | PASSED | 0.0002 | - |
| `test_tier2_boundaries.py` | `test_t2_noise_benchmark_zero_noise_exact_match` | PASSED | 0.0022 | - |
| `test_tier2_boundaries.py` | `test_t2_non_integer_sample_rate_handling` | PASSED | 0.0007 | - |
| `test_tier2_boundaries.py` | `test_t2_non_monotonic_timestamps_rejection` | PASSED | 0.0012 | - |
| `test_tier2_boundaries.py` | `test_t2_ood_empty_regime_handling` | PASSED | 0.0001 | - |
| `test_tier2_boundaries.py` | `test_t2_phase_angle_boundary_wrapping_pi` | PASSED | 0.0000 | - |
| `test_tier2_boundaries.py` | `test_t2_platt_scaling_extreme_negative_log_odds` | PASSED | 0.0001 | - |
| `test_tier2_boundaries.py` | `test_t2_platt_scaling_extreme_positive_log_odds` | PASSED | 0.0000 | - |
| `test_tier2_boundaries.py` | `test_t2_rapid_governor_time_constant_boundary` | PASSED | 0.0002 | - |
| `test_tier2_boundaries.py` | `test_t2_reliability_diagram_all_zero_and_all_one_confidence` | PASSED | 0.0003 | - |
| `test_tier2_boundaries.py` | `test_t2_renewable_penetration_out_of_bounds` | PASSED | 0.0002 | - |
| `test_tier2_boundaries.py` | `test_t2_saturated_turbine_governor_valve_position` | PASSED | 0.0014 | - |
| `test_tier2_boundaries.py` | `test_t2_single_sample_window_boundary` | PASSED | 0.0012 | - |
| `test_tier2_boundaries.py` | `test_t2_single_training_scenario_boundary` | PASSED | 0.0029 | - |
| `test_tier2_boundaries.py` | `test_t2_ultra_high_renewable_penetration_80pct` | PASSED | 0.0013 | - |
| `test_tier2_boundaries.py` | `test_t2_ultra_low_inertia_h1` | PASSED | 0.0014 | - |
| `test_tier2_boundaries.py` | `test_t2_ultra_short_observation_window_0_1s` | PASSED | 0.0015 | - |
| `test_tier2_boundaries.py` | `test_t2_voltage_magnitude_zero_or_negative_rejection` | PASSED | 0.0015 | - |
| `test_tier2_boundaries.py` | `test_t2_window_sensitivity_exceeding_trajectory_length` | PASSED | 0.0016 | - |
| `test_tier2_boundaries.py` | `test_t2_window_sensitivity_negative_t_obs` | PASSED | 0.0011 | - |
| `test_tier2_boundaries.py` | `test_t2_window_sensitivity_zero_t_obs` | PASSED | 0.0009 | - |
| `test_tier2_boundaries.py` | `test_t2_zero_frequency_deviation_steady_state` | PASSED | 0.0012 | - |
| `test_tier2_boundaries.py` | `test_t2_zero_generator_damping_boundary` | PASSED | 0.0002 | - |
| `test_tier2_boundaries.py` | `test_t2_zero_length_trajectory_array_rejection` | PASSED | 0.0000 | - |
| `test_tier2_boundaries.py` | `test_t2_zero_variance_features_constant_column` | PASSED | 0.0025 | - |
| `test_tier3_combinations.py` | `test_flow1_multibus_spatial_ordering_in_sliced_window` | PASSED | 0.0013 | - |
| `test_tier3_combinations.py` | `test_flow1_network_to_sliced_50hz_pmu_tensor` | PASSED | 0.0019 | - |
| `test_tier3_combinations.py` | `test_flow1_pmu_sampling_rate_25_50_60_slicing` | PASSED | 0.0040 | - |
| `test_tier3_combinations.py` | `test_flow2_contingency_size_vs_renewable_penetration_coupling` | PASSED | 0.0032 | - |
| `test_tier3_combinations.py` | `test_flow2_renewable_inertia_steepens_50hz_rocof` | PASSED | 0.0029 | - |
| `test_tier3_combinations.py` | `test_flow3_causal_window_to_14_feature_matrix_and_7_regressors` | PASSED | 0.1134 | - |
| `test_tier3_combinations.py` | `test_flow4_dynamic_features_to_calibrated_gbdt_and_intervals` | PASSED | 0.2587 | - |
| `test_tier3_combinations.py` | `test_flow5_platt_vs_isotonic_and_window_sensitivity_calibration` | PASSED | 0.9612 | - |
| `test_tier3_combinations.py` | `test_flow6_pmu_noise_propagation_ols_advantage_and_unsmoothed_benchmark` | PASSED | 0.2539 | - |
| `test_tier3_combinations.py` | `test_flow7_ood_generalization_held_out_partitions` | PASSED | 0.2649 | - |
| `test_tier3_combinations.py` | `test_flow8_streaming_latency_budget_and_speedup` | PASSED | 1.7733 | - |
| `test_tier3_combinations.py` | `test_flow9_full_chain_50hz_pipeline_integration` | PASSED | 0.1869 | - |
| `test_tier3_combinations.py` | `test_pairwise_invariants` | PASSED | 0.0542 | - |
| `test_tier4_scenarios.py` | `test_scenario_10_multi_area_boundary_islanding_prevention_alert` | PASSED | 0.0012 | - |
| `test_tier4_scenarios.py` | `test_scenario_1_severe_generator_outage_high_load` | PASSED | 0.2078 | - |
| `test_tier4_scenarios.py` | `test_scenario_2_high_renewable_low_inertia_shift` | PASSED | 0.1983 | - |
| `test_tier4_scenarios.py` | `test_scenario_3_multi_area_boundary_tie_line_oscillation` | PASSED | 0.0390 | - |
| `test_tier4_scenarios.py` | `test_scenario_4_marginal_disturbance_near_49_50hz_threshold` | PASSED | 0.0338 | - |
| `test_tier4_scenarios.py` | `test_scenario_5_noisy_pmu_stream_with_jitter_and_dropped_frames` | PASSED | 0.1987 | - |
| `test_tier4_scenarios.py` | `test_scenario_6_ultra_early_0_5s_vs_standard_1_5s_comparison` | PASSED | 0.3952 | - |
| `test_tier4_scenarios.py` | `test_scenario_7_online_inference_latency_compliance_50fps` | PASSED | 0.0476 | - |
| `test_tier4_scenarios.py` | `test_scenario_8_full_end_to_end_pipeline_execution` | PASSED | 2.7820 | - |
| `test_tier4_scenarios.py` | `test_scenario_9_under_frequency_load_shedding_readiness` | PASSED | 0.0467 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_feat_50hz_identities_wrap_around_nan_fallback_and_latency` | PASSED | 0.1904 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_pmu_c37_118_rates_truncated_frames_causal_rocof_and_tve` | PASSED | 0.0017 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_ren_linear_non_saturating_displacement_and_alpha_cap` | PASSED | 0.0025 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_ren_virtual_inertia_and_config_validation` | PASSED | 0.0008 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_sim_kron_reduction_zero_and_heavy_load` | PASSED | 0.0102 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_sim_non_generator_bus_trip_warning_and_flat_identity` | PASSED | 0.0157 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_sim_overfrequency_post_end_disturbance_zero_voltage_and_solvers` | PASSED | 0.1144 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_sim_zero_and_extreme_damping_stability` | PASSED | 0.0554 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_sim_zero_and_ultra_low_inertia` | PASSED | 0.1035 | - |
| `test_tier5_adversarial_sim_features.py` | `test_t5_slice_edge_cases_and_memory_isolation` | PASSED | 0.0020 | - |
| `test_tier5_adversarial_models_eval.py` | `test_t5_benchmarks_edge_cases` | PASSED | 1.9417 | - |
| `test_tier5_adversarial_models_eval.py` | `test_t5_cal_pava_sigmoid_degenerate_fallback_and_interval_ordering` | PASSED | 1.0235 | - |
| `test_tier5_adversarial_models_eval.py` | `test_t5_cls_49_5hz_threshold_degenerate_classes_and_all_backends` | PASSED | 0.4479 | - |
| `test_tier5_adversarial_models_eval.py` | `test_t5_met_calibration_edge_cases` | PASSED | 0.0015 | - |
| `test_tier5_adversarial_models_eval.py` | `test_t5_pipe_corrupted_config_and_isolated_cli_exit_codes` | PASSED | 6.9391 | - |
| `test_tier5_adversarial_models_eval.py` | `test_t5_reg_50hz_extreme_nadir_clipping_bounds` | PASSED | 0.0477 | - |
| `test_tier5_adversarial_models_eval.py` | `test_t5_reg_collinear_constant_single_sample_and_validation` | PASSED | 0.0289 | - |

## Status: ALL TESTS PASSED
