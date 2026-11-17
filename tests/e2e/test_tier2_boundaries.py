"""Boundary and edge-case E2E tests across 50 Hz simulation, features, models, benchmarks, and CLI."""

from __future__ import annotations

import argparse
import copy
import io
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import pytest

from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.features.extractor import (
    calc_angle_divergence,
    calc_coi_acceleration,
    calc_delta_f,
    calc_freq_integral,
    calc_initial_rocof,
    calc_interarea_dispersion,
    calc_rocof_metrics,
)
from pmu_fsa.features.window_slicer import extract_causal_window
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
)
from pmu_fsa.models.classifiers import BinarySecurityClassifier
from pmu_fsa.models.regressors import LinearNadirRegressor
from pmu_fsa.simulation.scenario_generator import load_scenario_from_npz
from tests.e2e.conftest import (
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
    IEEE39_GEN_PARAMS,
    _has_inf,
    _has_nan,
    _to_float_list_1d,
    _to_float_list_2d,
    assert_valid_pmu_trajectory,
    generate_synthetic_pmu_stream,
)


def test_t2_ultra_low_inertia_h1():
    traj = generate_synthetic_pmu_stream(f_nadir=47.5, renewable_penetration=0.85, active_loss_mw=900.0, seed=101)
    assert_valid_pmu_trajectory(traj)
    assert traj["metadata"]["ground_truth_nadir"] < 49.0
    assert traj["metadata"]["is_secure"] == 0


def test_t2_ultra_high_renewable_penetration_80pct():
    traj = generate_synthetic_pmu_stream(renewable_penetration=0.80, active_loss_mw=600.0, f_nadir=48.5, seed=102)
    assert_valid_pmu_trajectory(traj)
    base_H = sum(g["H"] for g in IEEE39_GEN_PARAMS.values())
    assert traj["metadata"]["effective_H_sum"] < base_H * 0.5


def test_t2_ultra_short_observation_window_0_1s(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    t_0 = traj["metadata"]["t_disturb"]
    idx = int(np.searchsorted(traj["timestamps"], t_0 + 0.1, side="right"))
    assert idx >= 5
    mean_val = float(np.mean(traj["coi_frequency"][:idx]))
    assert 45.0 <= mean_val <= 50.5


def test_t2_single_sample_window_boundary(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    idx_start = int(np.searchsorted(traj["timestamps"], traj["metadata"]["t_disturb"], side="left"))
    single_val = float(traj["coi_frequency"][idx_start])
    assert 45.0 <= single_val <= 50.5


def test_t2_zero_frequency_deviation_steady_state():
    traj = generate_synthetic_pmu_stream(active_loss_mw=0.0, f_nadir=50.0, t_disturb=1.0, noise_std=0.0, seed=103)
    assert_valid_pmu_trajectory(traj)
    assert float(np.max(traj["bus_frequencies"])) == pytest.approx(50.0, abs=1e-6)
    assert float(np.min(traj["bus_frequencies"])) == pytest.approx(50.0, abs=1e-6)
    assert traj["metadata"]["is_secure"] == 1


def test_t2_extreme_disturbance_deep_plunge():
    traj = generate_synthetic_pmu_stream(active_loss_mw=2500.0, f_nadir=44.5, seed=104)
    assert traj["metadata"]["is_secure"] == 0
    assert traj["metadata"]["ground_truth_nadir"] < 45.0
    with pytest.raises(AssertionError):
        assert_valid_pmu_trajectory(traj)


def test_t2_frequency_rise_overfrequency_event():
    traj = generate_synthetic_pmu_stream(f_nadir=50.0, active_loss_mw=0.0, seed=105)
    assert traj["metadata"]["is_secure"] == 1
    assert traj["metadata"]["ground_truth_nadir"] >= 49.50


def test_t2_zero_generator_damping_boundary():
    params = copy.deepcopy(IEEE39_GEN_PARAMS)
    for g in params.values():
        g["D"] = 0.0
        assert g["D"] == 0.0 and g["H"] > 0.0


def test_t2_rapid_governor_time_constant_boundary():
    params = copy.deepcopy(IEEE39_GEN_PARAMS)
    for g in params.values():
        g["Tg"] = 0.05
        assert g["Tg"] > 0.0


def test_t2_identical_bus_frequencies_zero_dispersion():
    f_uniform = [50.0] * 39
    m1 = sum(f_uniform[b - 1] for b in AREA_1_BUSES) / len(AREA_1_BUSES)
    m2 = sum(f_uniform[b - 1] for b in AREA_2_BUSES) / len(AREA_2_BUSES)
    m3 = sum(f_uniform[b - 1] for b in AREA_3_BUSES) / len(AREA_3_BUSES)
    assert max(abs(m1 - m2), abs(m1 - m3), abs(m2 - m3)) == 0.0


def test_t2_massive_generator_trip_2500mw():
    traj = generate_synthetic_pmu_stream(disturbed_gen=39, active_loss_mw=2500.0, f_nadir=45.6, seed=106)
    assert_valid_pmu_trajectory(traj)
    assert traj["metadata"]["active_loss_mw"] == 2500.0


def test_t2_generator_trip_at_high_load_vs_light_load():
    traj_light = generate_synthetic_pmu_stream(f_nadir=49.65, active_loss_mw=500.0, renewable_penetration=0.1)
    traj_heavy = generate_synthetic_pmu_stream(f_nadir=49.10, active_loss_mw=500.0, renewable_penetration=0.4)
    assert traj_heavy["metadata"]["ground_truth_nadir"] < traj_light["metadata"]["ground_truth_nadir"]


def test_t2_saturated_turbine_governor_valve_position():
    for g in IEEE39_GEN_PARAMS.values():
        p_max = g["P0_mw"] * 1.2
        p_gov = g["P0_mw"]
        for _ in range(300):
            p_gov = min(p_gov + ((3000.0 - p_gov) / g["Tg"]) * 0.05, p_max)
        assert p_gov == pytest.approx(p_max)


def test_t2_incoherent_area_frequency_oscillation_boundary():
    traj = generate_synthetic_pmu_stream(disturbed_gen=30, active_loss_mw=500.0, seed=107)
    assert not _has_nan(traj["coi_frequency"])
    assert not _has_inf(traj["coi_frequency"])


def test_t2_extreme_pmu_noise_0_5hz():
    traj = generate_synthetic_pmu_stream(noise_std=0.50, seed=201)
    assert not _has_nan(traj["bus_frequencies"])
    assert not _has_inf(traj["bus_frequencies"])


def test_t2_dropped_pmu_frames_10pct():
    rng = random.Random(202)
    traj = generate_synthetic_pmu_stream(n_samples=200, seed=202)
    ts = _to_float_list_1d(traj["timestamps"])
    ts_dropped = [ts[i] for i in range(len(ts)) if rng.random() > 0.10]
    assert len(ts_dropped) >= 150
    for i in range(len(ts_dropped) - 1):
        assert ts_dropped[i + 1] > ts_dropped[i]


def test_t2_nan_input_rejection():
    traj_corrupt = generate_synthetic_pmu_stream(seed=203)
    traj_corrupt["bus_frequencies"][10, 5] = float("nan")
    assert _has_nan(traj_corrupt["bus_frequencies"]) is True
    with pytest.raises((AssertionError, ValueError)):
        assert_valid_pmu_trajectory(traj_corrupt)


def test_t2_inf_input_rejection():
    traj_corrupt = generate_synthetic_pmu_stream(seed=204)
    traj_corrupt["bus_frequencies"][10, 5] = float("inf")
    assert _has_inf(traj_corrupt["bus_frequencies"]) is True
    with pytest.raises((AssertionError, ValueError)):
        assert_valid_pmu_trajectory(traj_corrupt)


def test_t2_phase_angle_boundary_wrapping_pi():
    a1, a2 = 3.10, -3.10
    circ_diff = math.atan2(math.sin(a1 - a2), math.cos(a1 - a2))
    assert abs(circ_diff) < 0.15


def test_t2_non_monotonic_timestamps_rejection():
    traj = generate_synthetic_pmu_stream(seed=205)
    traj["timestamps"][5] = traj["timestamps"][4] - 0.02
    with pytest.raises(AssertionError, match="monotonic"):
        assert_valid_pmu_trajectory(traj)


def test_t2_negative_reporting_rate_rejection():
    with pytest.raises(ValueError, match="positive|Reporting rate"):
        generate_synthetic_pmu_stream(fps=-50)


def test_t2_voltage_magnitude_zero_or_negative_rejection():
    traj = generate_synthetic_pmu_stream(seed=205)
    traj["voltage_magnitudes"][5, 0] = -0.5
    with pytest.raises(AssertionError):
        assert_valid_pmu_trajectory(traj)


def test_t2_duplicate_timestamps_handling():
    traj = generate_synthetic_pmu_stream(seed=206)
    traj["timestamps"][4] = traj["timestamps"][3]
    with pytest.raises(AssertionError, match="monotonic|duplicate"):
        assert_valid_pmu_trajectory(traj)


def test_t2_asymmetric_interarea_phase_swing():
    traj = generate_synthetic_pmu_stream(seed=206)
    row = traj["voltage_angles"][-1]
    div = float(np.max(row) - np.min(row))
    assert math.isfinite(div)


def test_t2_missing_metadata_key_rejection():
    traj = generate_synthetic_pmu_stream(seed=207)
    del traj["metadata"]["is_secure"]
    with pytest.raises((AssertionError, KeyError)):
        assert_valid_pmu_trajectory(traj)


def test_t2_zero_length_trajectory_array_rejection():
    empty_traj = {"timestamps": np.array([]), "bus_frequencies": np.array([])}
    with pytest.raises((AssertionError, KeyError)):
        assert_valid_pmu_trajectory(empty_traj)


def test_t2_high_reporting_rate_200fps():
    traj = generate_synthetic_pmu_stream(fps=200, n_samples=400, seed=208)
    assert_valid_pmu_trajectory(traj)
    assert abs((traj["timestamps"][1] - traj["timestamps"][0]) - 0.005) < 1e-6


def test_t2_non_integer_sample_rate_handling():
    traj = generate_synthetic_pmu_stream(fps=50, n_samples=100, seed=209)
    assert len(traj["timestamps"]) == 100


def test_t2_single_training_scenario_boundary():
    X_single = [[0.5, 0.2, -0.3, -0.4]]
    y_single = [49.4]
    model = LinearNadirRegressor()
    model.fit(X_single, y_single)
    pred = _to_float_list_1d(model.predict(X_single))[0]
    assert 45.0 <= pred <= 50.5


def test_t2_zero_variance_features_constant_column():
    X = [[1.0, 0.1, -0.2], [1.0, 0.2, -0.3], [1.0, 0.3, -0.4], [1.0, 0.4, -0.5]]
    y = [49.6, 49.4, 49.2, 49.0]
    model = LinearNadirRegressor(alpha=1.0)
    model.fit(X, y)
    preds = _to_float_list_1d(model.predict(X))
    assert all(45.0 <= p <= 50.5 for p in preds)


def test_t2_extreme_class_imbalance_99_to_1():
    X = [[float(i), float(i * 2)] for i in range(100)]
    y_nadir = [49.7] * 99 + [48.8]
    clf = BinarySecurityClassifier(threshold=49.50)
    clf.fit(X, y_nadir)
    preds = _to_float_list_1d(clf.predict(X))
    assert len(preds) == 100 and all(p in (0, 1) for p in preds)


def test_t2_all_positive_class_training_set():
    X = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    y_nadir = [49.8, 49.7, 49.9]
    clf = BinarySecurityClassifier(threshold=49.50)
    clf.fit(X, y_nadir)
    probs = _to_float_list_2d(clf.predict_proba(X))
    assert all(0.5 <= p_sec <= 1.0 for _p_insec, p_sec in probs)


def test_t2_all_negative_class_training_set():
    X = [[0.8, 0.9], [0.7, 0.8], [0.9, 1.0]]
    y_nadir = [48.5, 48.7, 48.2]
    clf = BinarySecurityClassifier(threshold=49.50)
    clf.fit(X, y_nadir)
    probs = _to_float_list_2d(clf.predict_proba(X))
    assert all(0.0 <= p_sec <= 0.5 for _p_insec, p_sec in probs)


def test_t2_marginal_nadir_exactly_on_threshold():
    clf = BinarySecurityClassifier(threshold=49.50, alpha=1e-9)
    clf.fit([[49.0], [50.0]], [49.0, 50.0])
    pred = clf.predict([[49.500000]])
    assert int(pred[0]) == 1


def test_t2_marginal_nadir_infinitesimally_below_threshold():
    clf = BinarySecurityClassifier(threshold=49.50, alpha=1e-9)
    clf.fit([[49.0], [50.0]], [49.0, 50.0])
    pred = clf.predict([[49.499000]])
    assert int(pred[0]) == 0


def test_t2_platt_scaling_extreme_positive_log_odds():
    clf = CalibratedGBDTClassifier(method="platt", threshold=49.50)
    p = clf._calibrate_single(120.0)
    assert p == 1.0 and not math.isinf(p)


def test_t2_platt_scaling_extreme_negative_log_odds():
    clf = CalibratedGBDTClassifier(method="platt", threshold=49.50)
    p = clf._calibrate_single(0.0)
    assert p == 0.0 and not math.isnan(p)


def test_t2_isotonic_regression_constant_target():
    clf = CalibratedGBDTClassifier(method="isotonic", threshold=49.50)
    with warnings_suppress_small_cal():
        clf.fit([[0.1], [0.2], [0.3], [0.4]], [49.6, 49.6, 49.6, 49.6])
    assert 0.0 <= clf._calibrate_single(49.6) <= 1.0


def warnings_suppress_small_cal():
    import warnings
    return warnings.catch_warnings()


def test_t2_isotonic_regression_strictly_decreasing_inputs():
    clf = CalibratedGBDTClassifier(method="isotonic", threshold=49.50)
    X = [[float(i)] for i in range(40)]
    y_nadir = np.linspace(49.9, 49.1, 40)
    clf.fit(X, y_nadir, X_cal=X, y_cal=y_nadir)
    assert clf._calibrate_single(49.3) <= clf._calibrate_single(49.7) + 1e-6


def test_t2_brier_score_identical_predictions():
    assert brier_score([1.0, 0.0, 1.0, 0.0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.25)


def test_t2_ece_single_bin_boundary():
    assert expected_calibration_error([1.0, 1.0, 0.0, 0.0], [0.7, 0.7, 0.7, 0.7], n_bins=1) == pytest.approx(0.20)


def test_t2_ece_empty_bins_handling():
    assert 0.0 <= expected_calibration_error([1.0, 0.0], [0.55, 0.55], n_bins=10) <= 1.0


def test_t2_mce_upper_bound_constraint():
    ece = expected_calibration_error([1.0, 0.0, 1.0, 0.0, 1.0], [0.9, 0.1, 0.6, 0.4, 0.8], n_bins=5)
    mce = maximum_calibration_error([1.0, 0.0, 1.0, 0.0, 1.0], [0.9, 0.1, 0.6, 0.4, 0.8], n_bins=5)
    assert 0.0 <= ece <= mce <= 1.0 + 1e-9


def test_t2_empty_test_matrix_rejection():
    model = LinearNadirRegressor()
    model.fit([[1.0, 2.0], [2.0, 3.0]], [49.5, 49.2])
    assert len(model.predict([])) == 0


def test_t2_feature_dimension_mismatch_rejection():
    model = LinearNadirRegressor()
    model.fit([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], [49.5, 49.2])
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        model.predict([[1.0, 2.0]])


def test_t2_nadir_prediction_clipping_below_45():
    model = LinearNadirRegressor(clip_bounds=(45.0, 50.5))
    model.fit([[1.0], [2.0]], [46.0, 45.0])
    pred = _to_float_list_1d(model.predict([[100.0]]))[0]
    assert pred >= 45.0


def test_t2_nadir_prediction_clipping_above_50_5():
    model = LinearNadirRegressor(clip_bounds=(45.0, 50.5))
    model.fit([[1.0], [2.0]], [49.5, 50.0])
    pred = _to_float_list_1d(model.predict([[-100.0]]))[0]
    assert pred <= 50.5


def test_t2_collinear_features_linear_regressor():
    X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
    y = [49.8, 49.6, 49.4, 49.2]
    model = LinearNadirRegressor(alpha=1e-2)
    model.fit(X, y)
    preds = _to_float_list_1d(model.predict(X))
    assert len(preds) == 4 and all(math.isfinite(p) for p in preds)


def test_t2_window_sensitivity_negative_t_obs(sample_pmu_trajectory):
    with pytest.raises(ValueError, match="positive"):
        extract_causal_window(sample_pmu_trajectory, t_obs=-0.5)


def test_t2_window_sensitivity_zero_t_obs(sample_pmu_trajectory):
    with pytest.raises(ValueError, match="positive"):
        extract_causal_window(sample_pmu_trajectory, t_obs=0.0)


def test_t2_window_sensitivity_exceeding_trajectory_length(sample_pmu_trajectory):
    max_t = float(sample_pmu_trajectory["timestamps"][-1])
    with pytest.warns(UserWarning, match="exceeds trajectory duration"):
        w = extract_causal_window(sample_pmu_trajectory, t_obs=max_t + 10.0)
    assert w.metadata["is_truncated"] is True


def test_t2_ood_empty_regime_handling():
    benchmark = OODGeneralizationBenchmark(id_range=(0.80, 0.90), ood_range=(0.95, 1.0))
    report = benchmark.run([])
    assert report["id_scenario_count"] == 0
    assert report["ood_scenario_count"] == 0


def test_t2_noise_benchmark_negative_sigma():
    with pytest.raises(ValueError, match="negative|noise"):
        generate_synthetic_pmu_stream(noise_std=-0.05)


def test_t2_noise_benchmark_zero_noise_exact_match():
    traj1 = generate_synthetic_pmu_stream(noise_std=0.0, seed=42)
    traj2 = generate_synthetic_pmu_stream(noise_std=0.0, seed=42)
    np.testing.assert_array_equal(traj1["bus_frequencies"], traj2["bus_frequencies"])


def test_t2_latency_benchmark_batch_size_1():
    res = LatencyBenchmark().profile(n_iter=5)
    assert res["t_feat_ms"] > 0.0 and res["t_infer_ms"] > 0.0


def test_t2_latency_benchmark_large_batch_500():
    model = LinearNadirRegressor()
    model.fit([[1.0, 2.0], [2.0, 3.0]], [49.5, 49.2])
    X_large = [[1.5, 2.5] for _ in range(500)]
    _ = model.predict(X_large[:2])
    t0 = time.perf_counter()
    preds = model.predict(X_large)
    elap = (time.perf_counter() - t0) * 1000.0
    assert len(preds) == 500 and elap < 100.0


def test_t2_latency_timer_precision_positive():
    assert LatencyBenchmark().profile(n_iter=2)["t_total_ms"] > 0.0


def test_t2_cli_missing_required_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-arg", required=True)
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        with pytest.raises(SystemExit):
            parser.parse_args([])
    finally:
        sys.stderr = old_stderr


def test_t2_cli_nonexistent_input_file():
    with pytest.raises((FileNotFoundError, OSError)):
        load_scenario_from_npz("/nonexistent_dir_abc123/bad_file.npz")


def test_t2_cli_nonexistent_output_directory_creation(tmp_path: Path):
    nested_report = tmp_path / "sub1" / "sub2" / "report.md"
    nested_report.parent.mkdir(parents=True, exist_ok=True)
    nested_report.write_text("# Isolated report\n", encoding="utf-8")
    assert nested_report.exists() and nested_report.stat().st_size > 0


def test_t2_malformed_scenario_id_handling():
    traj = generate_synthetic_pmu_stream(seed=301)
    traj["metadata"]["scenario_id"] = ""
    with pytest.raises(AssertionError, match="scenario_id"):
        assert_valid_pmu_trajectory(traj)


def test_t2_contingency_unknown_bus_trip():
    with pytest.raises(ValueError, match="generator|disturbed_gen"):
        generate_synthetic_pmu_stream(disturbed_gen=15)


def test_t2_active_power_loss_negative_rejection():
    with pytest.raises(ValueError, match="non-negative|active_loss"):
        generate_synthetic_pmu_stream(active_loss_mw=-500.0)


def test_t2_renewable_penetration_out_of_bounds():
    for bad_ren in (1.5, -0.2):
        with pytest.raises(ValueError, match="renewable_penetration"):
            generate_synthetic_pmu_stream(renewable_penetration=bad_ren)


def test_t2_reliability_diagram_all_zero_and_all_one_confidence():
    diag0 = reliability_diagram([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], n_bins=5)
    assert diag0["bin_counts"][0] == 3
    diag1 = reliability_diagram([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], n_bins=5)
    assert diag1["bin_counts"][-1] == 3


def test_t2_negative_brier_score_impossible():
    assert brier_score([1.0, 0.0, 1.0], [0.8, 0.2, 0.9]) >= 0.0


def test_t2_f1_disjoint_area_partitioning_and_gen_bounds(ieee39_topology):
    area_map = ieee39_topology["area_partition"]
    all_partitioned = [b for buses in area_map.values() for b in buses]
    assert len(all_partitioned) == 39 and set(all_partitioned) == set(range(1, 40))
    assert set(ieee39_topology["generator_buses"]) == set(range(30, 40))


def test_t2_f2_governor_stiff_limit_and_damping_rejection():
    traj = generate_synthetic_pmu_stream(governor_Tg=0.02, seed=500)
    assert_valid_pmu_trajectory(traj)
    with pytest.raises(ValueError, match="Damping factor|non-negative"):
        generate_synthetic_pmu_stream(damping=-0.5)


def test_t2_f3_zero_penetration_and_inertia_floor():
    traj0 = generate_synthetic_pmu_stream(renewable_penetration=0.0, seed=501)
    base_H = sum(p["H"] for p in IEEE39_GEN_PARAMS.values())
    assert traj0["metadata"]["effective_H_sum"] == pytest.approx(base_H, abs=1e-4)

    traj80 = generate_synthetic_pmu_stream(renewable_penetration=0.80, seed=502)
    assert traj80["metadata"]["effective_H_sum"] > 50.0


def test_t2_f4_high_reporting_rate_120fps_and_subcycle_uniformity():
    traj = generate_synthetic_pmu_stream(n_samples=240, fps=120, seed=503)
    assert_valid_pmu_trajectory(traj)
    dts = np.diff(traj["timestamps"])
    assert float(np.max(dts) - np.min(dts)) < 1e-4


def test_t2_f5_out_of_range_disturbance_time_and_zero_loss():
    traj = generate_synthetic_pmu_stream(seed=505)
    with pytest.raises(ValueError, match="outside trajectory bounds"):
        extract_causal_window(traj, t_obs=1.0, t_0=50.0)

    traj0 = generate_synthetic_pmu_stream(active_loss_mw=0.0, f_nadir=50.0, seed=506)
    assert traj0["metadata"]["ground_truth_nadir"] == 50.0
    assert traj0["metadata"]["is_secure"] == 1


def test_t2_f6_window_boundary_and_post_window_invariance(sample_pmu_trajectory):
    w_clean = extract_causal_window(sample_pmu_trajectory, t_obs=0.5)
    assert len(w_clean["timestamps"]) in (25, 26)
    traj_corrupt = copy.deepcopy(sample_pmu_trajectory)
    traj_corrupt["bus_frequencies"][120:, :] = 40.0
    w_corrupt = extract_causal_window(traj_corrupt, t_obs=0.5)
    np.testing.assert_allclose(w_clean["bus_frequencies"], w_corrupt["bus_frequencies"])


def test_t2_f7_to_f13_steady_state_50hz_identities_and_invariants(sample_pmu_trajectory):
    traj_flat = copy.deepcopy(sample_pmu_trajectory)
    traj_flat["bus_frequencies"][:, :] = 50.0
    traj_flat["voltage_angles"][:, :] = 0.5
    traj_flat["pmu_sys_frequency"][:] = 50.0
    traj_flat["coi_frequency"][:] = 50.0
    w_flat = extract_causal_window(traj_flat, t_obs=1.0)

    assert abs(calc_initial_rocof(w_flat)) < 1e-10
    pk_neg, r_end, mean_r = calc_rocof_metrics(w_flat)
    assert abs(pk_neg) < 1e-10 and abs(r_end) < 1e-10 and abs(mean_r) < 1e-10
    df_end, df_max = calc_delta_f(w_flat, f_nominal=50.0)
    assert abs(df_end) < 1e-10 and abs(df_max) < 1e-10
    idv, iab = calc_freq_integral(w_flat, f_nominal=50.0)
    assert abs(idv) < 1e-10 and abs(iab) < 1e-10
    sp, std = calc_interarea_dispersion(w_flat)
    assert abs(sp) < 1e-10 and abs(std) < 1e-10
    adiv, atie = calc_angle_divergence(w_flat)
    assert abs(adiv) < 1e-10 and abs(atie) < 1e-10
    ci, cm = calc_coi_acceleration(w_flat)
    assert abs(ci) < 1e-10 and abs(cm) < 1e-10

    # Active trajectory invariants
    w_act = extract_causal_window(sample_pmu_trajectory, t_obs=1.0)
    pk_a, _re_a, mr_a = calc_rocof_metrics(w_act)
    assert pk_a <= mr_a + 1e-6
    de_a, dm_a = calc_delta_f(w_act, f_nominal=50.0)
    assert abs(dm_a) >= abs(de_a) - 1e-6


def test_t2_f18_to_f22_benchmark_and_cli_boundaries(synthetic_scenario_batch):
    res_dense = WindowSensitivityBenchmark(window_lengths=[0.4, 0.8, 1.2, 1.6, 2.0]).run(synthetic_scenario_batch)
    assert len(res_dense["results"]) == 5

    with pytest.raises(ValueError, match="overlap"):
        OODGeneralizationBenchmark(id_range=(0.10, 0.35), ood_range=(0.30, 0.50))

    res_same = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.10, 0.30)).run(synthetic_scenario_batch)
    assert res_same["delta_mae"] == pytest.approx(0.0, abs=1e-5)

    res_noise = MeasurementNoiseBenchmark().run(synthetic_scenario_batch)
    for row in res_noise["results"]:
        assert row["mae"] >= 0.0 and row["rmse"] >= row["mae"] - 1e-6

    res_help = subprocess.run([sys.executable, "tests/e2e/run_e2e.py", "--help"], capture_output=True, text=True)
    assert res_help.returncode == 0 and "--tier" in res_help.stdout
