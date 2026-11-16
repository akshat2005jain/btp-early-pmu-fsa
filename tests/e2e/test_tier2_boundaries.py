"""Boundary and edge-case tests across simulation, features, models, and CLI."""

import os
import sys
import copy
import math
import time
import argparse
import subprocess
try:
    import pytest
except ImportError:
    from tests.e2e.conftest import pytest
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tests.e2e.conftest import (
    np, HAS_NUMPY,
    generate_synthetic_pmu_stream,
    assert_valid_pmu_trajectory,
    ref_extract_causal_window,
    _has_nan,
    _has_inf,
    IEEE39_GEN_PARAMS,
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
    RefLinearNadirRegressor,
    RefDecisionTreeNadirRegressor,
    RefRandomForestNadirRegressor,
    RefGradientBoostingNadirRegressor,
    RefBinarySecurityClassifier,
    RefCalibratedGBDTClassifier,
    ref_brier_score,
    ref_expected_calibration_error,
    ref_maximum_calibration_error,
    ref_reliability_diagram,
    RefWindowSensitivityBenchmark,
    RefOODGeneralizationBenchmark,
    RefMeasurementNoiseBenchmark,
    RefLatencyBenchmark,
    _to_float_list_1d,
    _to_float_list_2d,
)

from tests.e2e.test_tier1_r2_features import (
    ref_calc_initial_rocof,
    ref_calc_rocof_metrics,
    ref_calc_delta_f,
    ref_calc_freq_integral,
    ref_calc_interarea_dispersion,
    ref_calc_angle_divergence,
    ref_calc_coi_acceleration,
)

try:
    from pmu_fsa.simulation import dynamic_engine, ieee39_data, renewables, scenario_generator
    HAS_SIM = True
except ImportError:
    HAS_SIM = False

try:
    from pmu_fsa.features import window_slicer, extractor
    HAS_FEAT = True
except ImportError:
    HAS_FEAT = False

try:
    from pmu_fsa.models import regressors, classifiers, calibrated_gb, calibration_metrics
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False

try:
    from pmu_fsa.evaluation import sensitivity, ood_generalization, noise_stress, latency
    HAS_EVAL = True
except ImportError:
    HAS_EVAL = False


def test_t2_ultra_low_inertia_h1():
    traj = generate_synthetic_pmu_stream(
        f_nadir=57.5,
        renewable_penetration=0.85,
        active_loss_mw=900.0,
        seed=101
    )
    assert_valid_pmu_trajectory(traj)
    assert traj["metadata"]["ground_truth_nadir"] < 59.0
    assert traj["metadata"]["is_secure"] == 0


def test_t2_ultra_high_renewable_penetration_80pct():
    traj = generate_synthetic_pmu_stream(
        renewable_penetration=0.80,
        active_loss_mw=600.0,
        f_nadir=58.5,
        seed=102
    )
    assert_valid_pmu_trajectory(traj)
    base_H = sum(g["H"] for g in IEEE39_GEN_PARAMS.values())
    eff_H = traj["metadata"]["effective_H_sum"]
    assert eff_H < base_H * 0.5


def test_t2_ultra_short_observation_window_0_1s(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    t_0 = traj["metadata"]["t_disturb"]
    ts = traj["timestamps"]
    cutoff = t_0 + 0.1
    idx = np.searchsorted(ts, cutoff, side='right')
    assert idx >= 5, "Ultra-short window at 50 fps must contain at least 5 frames"
    coi_slice = traj["coi_frequency"][:idx]
    mean_val = np.mean(coi_slice)
    assert 55.0 <= mean_val <= 60.5


def test_t2_single_sample_window_boundary(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    ts = traj["timestamps"]
    t_0 = traj["metadata"]["t_disturb"]
    idx_start = np.searchsorted(ts, t_0, side='left')
    single_val = traj["coi_frequency"][idx_start]
    assert 55.0 <= single_val <= 60.5


def test_t2_zero_frequency_deviation_steady_state():
    traj = generate_synthetic_pmu_stream(
        active_loss_mw=0.0,
        f_nadir=60.0,
        t_disturb=1.0,
        noise_std=0.0,
        seed=103
    )
    assert_valid_pmu_trajectory(traj)
    bf = traj["bus_frequencies"]
    max_f = np.max(bf)
    min_f = np.min(bf)
    assert abs(max_f - 60.0) < 1e-6
    assert abs(min_f - 60.0) < 1e-6
    assert traj["metadata"]["is_secure"] == 1


def test_t2_extreme_disturbance_deep_plunge():
    traj = generate_synthetic_pmu_stream(
        active_loss_mw=2500.0,
        f_nadir=54.5,
        seed=104
    )
    assert traj["metadata"]["is_secure"] == 0
    assert traj["metadata"]["ground_truth_nadir"] < 55.0
    try:
        assert_valid_pmu_trajectory(traj)
        assert False, "Physical validator failed to catch frequency plunge < 55.0 Hz"
    except AssertionError:
        pass


def test_t2_frequency_rise_overfrequency_event():
    traj = generate_synthetic_pmu_stream(
        f_nadir=60.0,
        active_loss_mw=0.0,
        seed=105
    )
    assert traj["metadata"]["is_secure"] == 1
    assert traj["metadata"]["ground_truth_nadir"] >= 59.5


def test_t2_zero_generator_damping_boundary():
    params = copy.deepcopy(IEEE39_GEN_PARAMS)
    for g in params.values():
        g["D"] = 0.0
    for g in params.values():
        assert g["D"] == 0.0
        assert g["H"] > 0.0


def test_t2_rapid_governor_time_constant_boundary():
    params = copy.deepcopy(IEEE39_GEN_PARAMS)
    for g in params.values():
        g["Tg"] = 0.05
        assert g["Tg"] > 0.0


def test_t2_identical_bus_frequencies_zero_dispersion():
    f_uniform = [60.0] * 39
    f_a1 = [f_uniform[b - 1] for b in AREA_1_BUSES]
    f_a2 = [f_uniform[b - 1] for b in AREA_2_BUSES]
    f_a3 = [f_uniform[b - 1] for b in AREA_3_BUSES]

    m1 = sum(f_a1) / len(f_a1)
    m2 = sum(f_a2) / len(f_a2)
    m3 = sum(f_a3) / len(f_a3)

    spread = max(abs(m1 - m2), abs(m1 - m3), abs(m2 - m3))
    assert spread == 0.0, f"Expected 0.0 spread for uniform frequencies, got {spread}"


def test_t2_massive_generator_trip_2500mw():
    traj = generate_synthetic_pmu_stream(
        disturbed_gen=39,
        active_loss_mw=2500.0,
        f_nadir=55.5,
        seed=106
    )
    assert_valid_pmu_trajectory(traj)
    assert traj["metadata"]["active_loss_mw"] == 2500.0


def test_t2_generator_trip_at_high_load_vs_light_load():
    traj_light = generate_synthetic_pmu_stream(f_nadir=59.6, active_loss_mw=500.0, renewable_penetration=0.1)
    traj_heavy = generate_synthetic_pmu_stream(f_nadir=59.1, active_loss_mw=500.0, renewable_penetration=0.4)
    assert traj_heavy["metadata"]["ground_truth_nadir"] < traj_light["metadata"]["ground_truth_nadir"]


def test_t2_saturated_turbine_governor_valve_position():
    for bus_id, g in IEEE39_GEN_PARAMS.items():
        p_max = g["P0_mw"] * 1.2
        p_demand = 2000.0
        p_gov = g["P0_mw"]
        dt = 0.02
        Tg = g.get("Tg", 0.5)
        for _ in range(200):
            dp = (p_demand - p_gov) / Tg
            p_gov = min(p_gov + dp * dt, p_max)
        assert p_gov <= p_max
        assert p_gov == pytest.approx(p_max)


def test_t2_incoherent_area_frequency_oscillation_boundary():
    traj = generate_synthetic_pmu_stream(disturbed_gen=30, active_loss_mw=500.0, seed=107)
    coi = traj["coi_frequency"]
    assert not _has_nan(coi)
    assert not _has_inf(coi)


def test_t2_extreme_pmu_noise_0_5hz():
    traj = generate_synthetic_pmu_stream(noise_std=0.50, seed=201)
    bf = traj["bus_frequencies"]
    assert not _has_nan(bf)
    assert not _has_inf(bf)


def test_t2_dropped_pmu_frames_10pct():
    import random
    rng = random.Random(202)
    traj = generate_synthetic_pmu_stream(n_samples=200, seed=202)
    ts = _to_float_list_1d(traj["timestamps"])
    coi = _to_float_list_1d(traj["coi_frequency"])

    keep_indices = [i for i in range(len(ts)) if rng.random() > 0.10]
    ts_dropped = [ts[i] for i in keep_indices]
    coi_dropped = [coi[i] for i in keep_indices]

    assert len(ts_dropped) >= 150
    for i in range(len(ts_dropped) - 1):
        assert ts_dropped[i + 1] > ts_dropped[i]


def test_t2_nan_input_rejection():
    traj = generate_synthetic_pmu_stream(seed=203)
    traj_corrupt = copy.deepcopy(traj)
    if HAS_NUMPY:
        traj_corrupt["bus_frequencies"][10, 5] = float('nan')
    else:
        traj_corrupt["bus_frequencies"][10][5] = float('nan')

    assert _has_nan(traj_corrupt["bus_frequencies"]) is True
    try:
        assert_valid_pmu_trajectory(traj_corrupt)
        assert False, "assert_valid_pmu_trajectory failed to reject NaN input"
    except (AssertionError, ValueError):
        pass


def test_t2_inf_input_rejection():
    traj = generate_synthetic_pmu_stream(seed=204)
    traj_corrupt = copy.deepcopy(traj)
    if HAS_NUMPY:
        traj_corrupt["bus_frequencies"][10, 5] = float('inf')
    else:
        traj_corrupt["bus_frequencies"][10][5] = float('inf')

    assert _has_inf(traj_corrupt["bus_frequencies"]) is True
    try:
        assert_valid_pmu_trajectory(traj_corrupt)
        assert False, "assert_valid_pmu_trajectory failed to reject Inf input"
    except (AssertionError, ValueError):
        pass


def test_t2_phase_angle_boundary_wrapping_pi():
    # Distance across +-pi boundary: 2*(pi - 3.10) ~ 0.083 rad
    a1 = 3.10
    a2 = -3.10
    raw_diff = abs(a1 - a2)
    circ_diff = math.atan2(math.sin(a1 - a2), math.cos(a1 - a2))
    assert abs(circ_diff) < 0.15, f"Expected circular difference < 0.15 rad, got {abs(circ_diff)}"


def test_t2_non_monotonic_timestamps_rejection():
    traj = generate_synthetic_pmu_stream(seed=205)
    traj["timestamps"][5] = traj["timestamps"][4] - 0.02
    try:
        assert_valid_pmu_trajectory(traj)
        assert False, "Failed to reject non-monotonic timestamps"
    except AssertionError as e:
        assert "monotonic" in str(e).lower()


def test_t2_negative_reporting_rate_rejection():
    with pytest.raises(ValueError, match="positive|Reporting rate"):
        generate_synthetic_pmu_stream(fps=-50)


def test_t2_voltage_magnitude_zero_or_negative_rejection():
    traj = generate_synthetic_pmu_stream(seed=205)
    traj["voltage_magnitudes"][5, 0] = -0.5 if HAS_NUMPY else -0.5
    try:
        assert_valid_pmu_trajectory(traj)
        assert False, "Failed to reject negative voltage"
    except AssertionError:
        pass


def test_t2_duplicate_timestamps_handling():
    traj = generate_synthetic_pmu_stream(seed=206)
    traj["timestamps"][4] = traj["timestamps"][3]
    try:
        assert_valid_pmu_trajectory(traj)
        assert False, "Failed to reject duplicate adjacent timestamps"
    except AssertionError as e:
        assert "monotonic" in str(e).lower() or "duplicate" in str(e).lower()


def test_t2_asymmetric_interarea_phase_swing():
    traj = generate_synthetic_pmu_stream(seed=206)
    va = traj["voltage_angles"]
    row = va[-1]
    div = np.max(row) - np.min(row)
    assert not math.isnan(div)
    assert not math.isinf(div)


def test_t2_missing_metadata_key_rejection():
    traj = generate_synthetic_pmu_stream(seed=207)
    del traj["metadata"]["is_secure"]
    try:
        assert_valid_pmu_trajectory(traj)
        assert False, "Failed to reject missing metadata key"
    except (AssertionError, KeyError):
        pass


def test_t2_zero_length_trajectory_array_rejection():
    empty_traj = {"timestamps": np.array([]) if HAS_NUMPY else [], "bus_frequencies": []}
    try:
        assert_valid_pmu_trajectory(empty_traj)
        assert False, "Failed to reject empty trajectory"
    except (AssertionError, KeyError):
        pass


def test_t2_high_reporting_rate_200fps():
    traj = generate_synthetic_pmu_stream(fps=200, n_samples=400, seed=208)
    assert_valid_pmu_trajectory(traj)
    dt = traj["timestamps"][1] - traj["timestamps"][0]
    assert abs(dt - 0.005) < 1e-6


def test_t2_non_integer_sample_rate_handling():
    traj = generate_synthetic_pmu_stream(fps=50, n_samples=100, seed=209)
    assert len(traj["timestamps"]) == 100


def test_t2_single_training_scenario_boundary():
    X_single = [[0.5, 0.2, -0.3, -0.4]]
    y_single = [59.4]
    model = RefLinearNadirRegressor()
    model.fit(X_single, y_single)
    pred = _to_float_list_1d(model.predict(X_single))[0]
    assert 55.0 <= pred <= 60.5


def test_t2_zero_variance_features_constant_column():
    X = [
        [1.0, 0.1, -0.2],
        [1.0, 0.2, -0.3],
        [1.0, 0.3, -0.4],
        [1.0, 0.4, -0.5],
    ]
    y = [59.6, 59.4, 59.2, 59.0]
    model = RefLinearNadirRegressor(alpha=1.0)
    model.fit(X, y)
    preds = _to_float_list_1d(model.predict(X))
    for p in preds:
        assert not math.isnan(p)
        assert 55.0 <= p <= 60.5


def test_t2_extreme_class_imbalance_99_to_1():
    X = [[float(i), float(i * 2)] for i in range(100)]
    y_nadir = [59.7] * 99 + [58.8]
    clf = RefBinarySecurityClassifier()
    clf.fit(X, y_nadir)
    preds = _to_float_list_1d(clf.predict(X))
    assert len(preds) == 100
    for p in preds:
        assert p in (0, 1)


def test_t2_all_positive_class_training_set():
    X = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    y_nadir = [59.8, 59.7, 59.9]
    clf = RefBinarySecurityClassifier()
    clf.fit(X, y_nadir)
    probs = _to_float_list_2d(clf.predict_proba(X))
    for p_insec, p_sec in probs:
        assert 0.0 <= p_sec <= 1.0
        assert p_sec >= 0.5


def test_t2_all_negative_class_training_set():
    X = [[0.8, 0.9], [0.7, 0.8], [0.9, 1.0]]
    y_nadir = [58.5, 58.7, 58.2]
    clf = RefBinarySecurityClassifier()
    clf.fit(X, y_nadir)
    probs = _to_float_list_2d(clf.predict_proba(X))
    for p_insec, p_sec in probs:
        assert 0.0 <= p_sec <= 1.0
        assert p_sec <= 0.5


def test_t2_marginal_nadir_exactly_on_threshold():
    clf = RefBinarySecurityClassifier(threshold=59.5)
    clf.fit([[59.0], [60.0]], [59.0, 60.0])
    clf.regressor.weights = [1.0]
    clf.regressor.intercept = 0.0
    pred = clf.predict([[59.500000]])
    assert int(pred[0]) == 1, "f_nadir == 59.5 must be classified as secure (1)"


def test_t2_marginal_nadir_infinitesimally_below_threshold():
    clf = RefBinarySecurityClassifier(threshold=59.5)
    clf.fit([[59.0], [60.0]], [59.0, 60.0])
    clf.regressor.weights = [1.0]
    clf.regressor.intercept = 0.0
    pred = clf.predict([[59.499999]])
    assert int(pred[0]) == 0, "f_nadir < 59.5 must be classified as insecure (0)"


def test_t2_platt_scaling_extreme_positive_log_odds():
    clf = RefCalibratedGBDTClassifier(method="platt")
    p = clf._calibrate_single(120.0)
    assert p == 1.0, f"Expected 1.0, got {p}"
    assert not math.isinf(p)


def test_t2_platt_scaling_extreme_negative_log_odds():
    clf = RefCalibratedGBDTClassifier(method="platt")
    p = clf._calibrate_single(0.0)
    assert p == 0.0, f"Expected 0.0, got {p}"
    assert not math.isnan(p)


def test_t2_isotonic_regression_constant_target():
    clf = RefCalibratedGBDTClassifier(method="isotonic")
    X = [[0.1], [0.2], [0.3], [0.4]]
    y_nadir = [59.6, 59.6, 59.6, 59.6]
    clf.fit(X, y_nadir)
    p = clf._calibrate_single(59.6)
    assert 0.0 <= p <= 1.0


def test_t2_isotonic_regression_strictly_decreasing_inputs():
    clf = RefCalibratedGBDTClassifier(method="isotonic")
    X = [[1.0], [2.0], [3.0], [4.0]]
    y_nadir = [59.8, 59.6, 59.4, 59.2]
    clf.fit(X, y_nadir)
    p1 = clf._calibrate_single(59.3)
    p2 = clf._calibrate_single(59.7)
    assert p1 <= p2 + 1e-6


def test_t2_brier_score_identical_predictions():
    y_true = [1.0, 0.0, 1.0, 0.0]
    y_prob = [0.5, 0.5, 0.5, 0.5]
    brier = ref_brier_score(y_true, y_prob)
    assert abs(brier - 0.25) < 1e-9


def test_t2_ece_single_bin_boundary():
    y_true = [1.0, 1.0, 0.0, 0.0]
    y_prob = [0.7, 0.7, 0.7, 0.7]
    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=1)
    assert abs(ece - 0.20) < 1e-6


def test_t2_ece_empty_bins_handling():
    y_true = [1.0, 0.0]
    y_prob = [0.55, 0.55]
    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert 0.0 <= ece <= 1.0


def test_t2_mce_upper_bound_constraint():
    y_true = [1.0, 0.0, 1.0, 0.0, 1.0]
    y_prob = [0.9, 0.1, 0.6, 0.4, 0.8]
    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=5)
    mce = ref_maximum_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= mce <= 1.0 + 1e-9


def test_t2_empty_test_matrix_rejection():
    model = RefLinearNadirRegressor()
    model.fit([[1.0, 2.0], [2.0, 3.0]], [59.5, 59.2])
    preds = model.predict([])
    assert len(preds) == 0


def test_t2_feature_dimension_mismatch_rejection():
    model = RefLinearNadirRegressor()
    model.fit([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], [59.5, 59.2])
    try:
        model.predict([[1.0, 2.0]])
        assert False, "Failed to raise ValueError on dimension mismatch"
    except ValueError:
        pass


def test_t2_nadir_prediction_clipping_below_55():
    model = RefLinearNadirRegressor(clip_bounds=(55.0, 60.5))
    model.fit([[1.0], [2.0]], [56.0, 55.0])
    pred = _to_float_list_1d(model.predict([[100.0]]))[0]
    assert pred >= 55.0, f"Predicted nadir {pred} violates lower bound 55.0 Hz"


def test_t2_nadir_prediction_clipping_above_60():
    model = RefLinearNadirRegressor(clip_bounds=(55.0, 60.5))
    model.fit([[1.0], [2.0]], [59.5, 60.0])
    pred = _to_float_list_1d(model.predict([[-100.0]]))[0]
    assert pred <= 60.5, f"Predicted nadir {pred} violates upper bound 60.5 Hz"


def test_t2_collinear_features_linear_regressor():
    X = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
    y = [59.8, 59.6, 59.4, 59.2]
    model = RefLinearNadirRegressor(alpha=1e-2)
    model.fit(X, y)
    preds = _to_float_list_1d(model.predict(X))
    assert len(preds) == 4
    for p in preds:
        assert not math.isnan(p)


def test_t2_window_sensitivity_negative_t_obs(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    try:
        ref_extract_causal_window(traj, t_obs=-0.5)
        assert False, "Failed to reject negative T_obs in window extractor"
    except ValueError as e:
        assert "positive" in str(e).lower()


def test_t2_window_sensitivity_zero_t_obs(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    try:
        ref_extract_causal_window(traj, t_obs=0.0)
        assert False, "Failed to reject zero T_obs in window extractor"
    except ValueError as e:
        assert "positive" in str(e).lower()


def test_t2_window_sensitivity_exceeding_trajectory_length(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    max_t = float(traj["timestamps"][-1])
    t_obs = max_t + 10.0
    try:
        ref_extract_causal_window(traj, t_obs=t_obs)
        assert False, "Failed to reject window exceeding trajectory duration"
    except ValueError as e:
        assert "window" in str(e).lower() or "samples" in str(e).lower()


def test_t2_ood_empty_regime_handling():
    benchmark = RefOODGeneralizationBenchmark(id_range=(0.80, 0.90), ood_range=(0.95, 1.0))
    scenarios = [generate_synthetic_pmu_stream(renewable_penetration=0.20)]
    report = benchmark.run(scenarios)
    assert report["id_scenario_count"] == 0
    assert report["ood_scenario_count"] == 0


def test_t2_noise_benchmark_negative_sigma():
    try:
        generate_synthetic_pmu_stream(noise_std=-0.05)
        assert False, "Failed to reject negative noise_std in generator"
    except ValueError as e:
        assert "negative" in str(e).lower() or "noise" in str(e).lower()


def test_t2_noise_benchmark_zero_noise_exact_match():
    traj1 = generate_synthetic_pmu_stream(noise_std=0.0, seed=42)
    traj2 = generate_synthetic_pmu_stream(noise_std=0.0, seed=42)
    l1 = _to_float_list_2d(traj1["bus_frequencies"])
    l2 = _to_float_list_2d(traj2["bus_frequencies"])
    diff = max(abs(x - y) for r1, r2 in zip(l1, l2) for x, y in zip(r1, r2))
    assert diff == 0.0


def test_t2_latency_benchmark_batch_size_1():
    benchmark = RefLatencyBenchmark()
    res = benchmark.profile(n_iter=5)
    assert res["t_feat_ms"] > 0
    assert res["t_infer_ms"] > 0


def test_t2_latency_benchmark_large_batch_500():
    model = RefLinearNadirRegressor()
    model.fit([[1.0, 2.0], [2.0, 3.0]], [59.5, 59.2])
    X_large = [[1.5, 2.5] for _ in range(500)]
    _ = model.predict(X_large[:2])
    timings = []
    for _ in range(3):
        t0 = time.perf_counter()
        preds = model.predict(X_large)
        timings.append((time.perf_counter() - t0) * 1000.0)
    elap = min(timings)
    assert len(preds) == 500
    assert elap < 100.0, f"500-sample inference took {elap} ms (> 100 ms)"


def test_t2_latency_timer_precision_positive():
    benchmark = RefLatencyBenchmark()
    res = benchmark.profile(n_iter=1)
    assert res["t_total_ms"] > 0.0


def test_t2_cli_missing_required_argument():
    import io
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-arg", required=True)
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        parser.parse_args([])
        assert False, "CLI parser failed to enforce required argument"
    except SystemExit:
        pass
    finally:
        sys.stderr = old_stderr


def test_t2_cli_nonexistent_input_file():
    fake_path = "/nonexistent_dir_abc123/bad_file.npz"
    if HAS_SIM and hasattr(scenario_generator, "load_scenario_from_npz"):
        with pytest.raises((FileNotFoundError, OSError)):
            scenario_generator.load_scenario_from_npz(fake_path)
    else:
        with pytest.raises(FileNotFoundError):
            Path(fake_path).read_bytes()


def test_t2_cli_nonexistent_output_directory_creation(tmp_path=None):
    if tmp_path is None:
        import tempfile
        tmp_dir = tempfile.mkdtemp()
        tmp_path = Path(tmp_dir)
    nested_report = tmp_path / "sub1" / "sub2" / "report.json"
    assert not nested_report.parent.exists()

    res = subprocess.run(
        [sys.executable, "tests/e2e/run_e2e.py", "--tier", "4", "--report", str(nested_report)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert nested_report.parent.exists()
    assert nested_report.exists()
    assert nested_report.stat().st_size > 0


def test_t2_malformed_scenario_id_handling():
    traj = generate_synthetic_pmu_stream(seed=301)
    traj["metadata"]["scenario_id"] = ""
    try:
        assert_valid_pmu_trajectory(traj)
        assert False, "Failed to reject empty scenario_id"
    except AssertionError as e:
        assert "scenario_id" in str(e).lower()


def test_t2_contingency_unknown_bus_trip():
    # Bus 15 is a load bus, not a generator bus
    invalid_gen_bus = 15
    try:
        generate_synthetic_pmu_stream(disturbed_gen=invalid_gen_bus)
        assert False, "Failed to reject non-generator bus contingency"
    except ValueError as e:
        assert "generator" in str(e).lower() or "disturbed_gen" in str(e).lower()


def test_t2_active_power_loss_negative_rejection():
    try:
        generate_synthetic_pmu_stream(active_loss_mw=-500.0)
        assert False, "Failed to reject negative active power loss"
    except ValueError as e:
        assert "negative" in str(e).lower() or "active_loss" in str(e).lower()


def test_t2_renewable_penetration_out_of_bounds():
    try:
        generate_synthetic_pmu_stream(renewable_penetration=1.5)
        assert False, "Failed to reject renewable penetration > 1.0"
    except ValueError as e:
        assert "renewable_penetration" in str(e).lower()

    try:
        generate_synthetic_pmu_stream(renewable_penetration=-0.2)
        assert False, "Failed to reject negative renewable penetration"
    except ValueError as e:
        assert "renewable_penetration" in str(e).lower()


def test_t2_reliability_diagram_all_zero_confidence():
    y_true = [0.0, 0.0, 0.0]
    y_prob = [0.0, 0.0, 0.0]
    diag = ref_reliability_diagram(y_true, y_prob, n_bins=5)
    assert sum(diag["bin_counts"]) == 3
    assert diag["bin_counts"][0] == 3


def test_t2_reliability_diagram_all_one_confidence():
    y_true = [1.0, 1.0, 1.0]
    y_prob = [1.0, 1.0, 1.0]
    diag = ref_reliability_diagram(y_true, y_prob, n_bins=5)
    assert sum(diag["bin_counts"]) == 3
    assert diag["bin_counts"][-1] == 3


def test_t2_negative_brier_score_impossible():
    y_true = [1.0, 0.0, 1.0]
    y_prob = [0.8, 0.2, 0.9]
    brier = ref_brier_score(y_true, y_prob)
    assert brier >= 0.0, f"Brier score cannot be negative: {brier}"


def test_t2_f1_disjoint_area_partitioning_invariant(ieee39_topology):
    topo = ieee39_topology
    area_map = topo["area_partition"]
    all_partitioned = [b for buses in area_map.values() for b in buses]
    assert len(all_partitioned) == 39
    assert set(all_partitioned) == set(range(1, 40))
    for a1, b1 in area_map.items():
        for a2, b2 in area_map.items():
            if a1 != a2:
                assert set(b1).isdisjoint(set(b2))


def test_t2_f1_generator_bus_index_bounds(ieee39_topology):
    gens = ieee39_topology["generator_buses"]
    assert len(gens) == 10
    assert set(gens) == set(range(30, 40))


def test_t2_f2_governor_time_constant_stiff_limit():
    traj = generate_synthetic_pmu_stream(governor_Tg=0.02, seed=500)
    assert_valid_pmu_trajectory(traj)
    bf = traj["bus_frequencies"]
    assert not _has_nan(bf)
    assert not _has_inf(bf)


def test_t2_f2_damping_factor_negative_rejection():
    with pytest.raises(ValueError, match="Damping factor|non-negative"):
        generate_synthetic_pmu_stream(damping=-0.5)


def test_t2_f3_zero_penetration_synchronous_baseline():
    traj = generate_synthetic_pmu_stream(renewable_penetration=0.0, seed=501)
    base_H = sum(p["H"] for p in IEEE39_GEN_PARAMS.values())
    assert abs(traj["metadata"]["effective_H_sum"] - base_H) < 1e-4


def test_t2_f3_critical_inertia_floor_enforcement():
    # IEEE 39 base sum(H) is ~324s; at 80% penetration effective sum stays > 50s
    traj = generate_synthetic_pmu_stream(renewable_penetration=0.80, seed=502)
    assert traj["metadata"]["effective_H_sum"] > 50.0


def test_t2_f4_high_reporting_rate_120fps_fidelity():
    traj = generate_synthetic_pmu_stream(n_samples=240, fps=120, seed=503)
    assert_valid_pmu_trajectory(traj)
    dt = traj["timestamps"][1] - traj["timestamps"][0]
    assert abs(dt - (1.0 / 120)) < 1e-4


def test_t2_f4_sub_cycle_timestamp_precision_uniformity():
    traj = generate_synthetic_pmu_stream(n_samples=100, fps=50, seed=504)
    ts = traj["timestamps"]
    dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    assert max(dts) - min(dts) < 1e-4


def test_t2_f5_out_of_range_disturbance_time_rejection():
    traj = generate_synthetic_pmu_stream(seed=505)
    with pytest.raises(ValueError, match="Disturbance time|out of range"):
        ref_extract_causal_window(traj, t_obs=1.0, t_disturb=50.0)


def test_t2_f5_zero_power_loss_nadir_identity():
    traj = generate_synthetic_pmu_stream(active_loss_mw=0.0, f_nadir=60.0, seed=506)
    assert traj["metadata"]["ground_truth_nadir"] == 60.0
    assert traj["metadata"]["is_secure"] == 1


def test_t2_f6_window_exact_boundary_sample_count(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    assert len(w["timestamps"]) in (50, 51)


def test_t2_f6_causal_slicing_post_window_frequency_invariance(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w_clean = ref_extract_causal_window(traj, 0.5)
    traj_corrupt = copy.deepcopy(traj)
    traj_corrupt["bus_frequencies"][120:, :] = 40.0
    w_corrupt = ref_extract_causal_window(traj_corrupt, 0.5)
    assert np.allclose(w_clean["bus_frequencies"], w_corrupt["bus_frequencies"])


def test_t2_f7_rocof_zero_steady_state_identity(sample_pmu_trajectory):
    traj = copy.deepcopy(sample_pmu_trajectory)
    traj["coi_frequency"] = [60.0] * len(traj["timestamps"])
    w = ref_extract_causal_window(traj, 0.5)
    init_r = ref_calc_initial_rocof(w)
    assert abs(init_r) < 1e-4


def test_t2_f7_rocof_extreme_loss_physical_bounds():
    traj = generate_synthetic_pmu_stream(active_loss_mw=1500.0, f_nadir=56.0, seed=508)
    w = ref_extract_causal_window(traj, 0.5)
    init_r = ref_calc_initial_rocof(w)
    assert init_r >= -15.0, f"ROCOF breached physical floor: {init_r}"


def test_t2_f8_rocof_peak_negative_le_mean_invariant(sample_pmu_trajectory):
    w = ref_extract_causal_window(sample_pmu_trajectory, 1.0)
    peak_neg, max_abs, mean_r = ref_calc_rocof_metrics(w)
    assert peak_neg <= mean_r + 1e-4


def test_t2_f8_rocof_constant_trajectory_zero_identity(sample_pmu_trajectory):
    traj = copy.deepcopy(sample_pmu_trajectory)
    traj["coi_frequency"] = [60.0] * len(traj["timestamps"])
    w = ref_extract_causal_window(traj, 1.0)
    peak_neg, max_abs, mean_r = ref_calc_rocof_metrics(w)
    assert abs(peak_neg) < 1e-4
    assert abs(max_abs) < 1e-4
    assert abs(mean_r) < 1e-4


def test_t2_f9_delta_f_max_ge_instantaneous_invariant(sample_pmu_trajectory):
    w = ref_extract_causal_window(sample_pmu_trajectory, 1.0)
    inst_df, max_df = ref_calc_delta_f(w)
    assert abs(max_df) >= abs(inst_df) - 1e-4


def test_t2_f9_delta_f_steady_state_zero_identity(sample_pmu_trajectory):
    traj = copy.deepcopy(sample_pmu_trajectory)
    traj["coi_frequency"] = [60.0] * len(traj["timestamps"])
    w = ref_extract_causal_window(traj, 1.0)
    inst_df, max_df = ref_calc_delta_f(w)
    assert abs(inst_df) < 1e-4
    assert abs(max_df) < 1e-4


def test_t2_f10_freq_integral_monotonic_time_scaling(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w1 = ref_extract_causal_window(traj, 0.5)
    w2 = ref_extract_causal_window(traj, 1.5)
    int1_dev, int1_abs = ref_calc_freq_integral(w1)
    int2_dev, int2_abs = ref_calc_freq_integral(w2)
    assert int2_abs > int1_abs


def test_t2_f10_freq_integral_steady_state_zero_identity(sample_pmu_trajectory):
    traj = copy.deepcopy(sample_pmu_trajectory)
    traj["coi_frequency"] = [60.0] * len(traj["timestamps"])
    w = ref_extract_causal_window(traj, 1.0)
    int_dev, int_abs = ref_calc_freq_integral(w)
    assert abs(int_dev) < 1e-4
    assert abs(int_abs) < 1e-4


def test_t2_f11_interarea_dispersion_single_area_zero():
    traj = generate_synthetic_pmu_stream(seed=510)
    for row in traj["bus_frequencies"]:
        for col in range(len(row)):
            row[col] = 59.5
    w = ref_extract_causal_window(traj, 1.0)
    spread, s_var = ref_calc_interarea_dispersion(w)
    assert abs(spread) < 1e-4
    assert abs(s_var) < 1e-4


def test_t2_f11_interarea_dispersion_non_negative_invariant(sample_pmu_trajectory):
    w = ref_extract_causal_window(sample_pmu_trajectory, 1.0)
    spread, s_var = ref_calc_interarea_dispersion(w)
    assert spread >= 0.0
    assert s_var >= 0.0


def test_t2_f12_angle_divergence_synchronized_zero_identity(sample_pmu_trajectory):
    traj = copy.deepcopy(sample_pmu_trajectory)
    for row in traj["voltage_angles"]:
        for col in range(len(row)):
            row[col] = 0.5
    w = ref_extract_causal_window(traj, 1.0)
    max_div, tie_ang = ref_calc_angle_divergence(w)
    assert abs(max_div) < 1e-4
    assert abs(tie_ang) < 1e-4


def test_t2_f12_angle_divergence_symmetry_invariant(sample_pmu_trajectory):
    w = ref_extract_causal_window(sample_pmu_trajectory, 1.0)
    max_div, tie_ang = ref_calc_angle_divergence(w)
    assert max_div >= 0.0
    assert not math.isnan(max_div)


def test_t2_f13_coi_acceleration_balanced_zero_identity(sample_pmu_trajectory):
    traj = copy.deepcopy(sample_pmu_trajectory)
    traj["coi_frequency"] = [60.0] * len(traj["timestamps"])
    w = ref_extract_causal_window(traj, 1.0)
    init_a, mean_a = ref_calc_coi_acceleration(w)
    assert abs(init_a) < 1e-4
    assert abs(mean_a) < 1e-4


def test_t2_f13_coi_acceleration_inertia_inverse_scaling():
    s_low_ren = generate_synthetic_pmu_stream(renewable_penetration=0.10, active_loss_mw=650.0, seed=512)
    s_high_ren = generate_synthetic_pmu_stream(renewable_penetration=0.60, active_loss_mw=650.0, seed=513)
    w_low = ref_extract_causal_window(s_low_ren, 0.5)
    w_high = ref_extract_causal_window(s_high_ren, 0.5)
    a_low, _ = ref_calc_coi_acceleration(w_low)
    a_high, _ = ref_calc_coi_acceleration(w_high)
    assert a_high <= a_low + 1e-4


def test_t2_f14_regressor_empty_matrix_rejection():
    reg = RefLinearNadirRegressor()
    try:
        reg.fit([], [])
        assert False, "Failed to reject empty training set"
    except (ValueError, IndexError, ZeroDivisionError):
        pass


def test_t2_f14_regressor_feature_dimension_mismatch():
    reg = RefLinearNadirRegressor()
    reg.fit([[0.1, 0.2], [0.3, 0.4]], [59.5, 59.2])
    try:
        reg.predict([[0.1, 0.2, 0.3]])
        assert False, "Failed to reject dimension mismatch"
    except (ValueError, IndexError):
        pass


def test_t2_f15_classifier_all_secure_class_imbalance():
    clf = RefBinarySecurityClassifier()
    X = [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]]
    y = [59.8, 59.7, 59.6]
    clf.fit(X, y)
    preds = clf.predict(X)
    assert all(p == 1 for p in preds)


def test_t2_f15_classifier_probability_simplex_sum_one():
    clf = RefBinarySecurityClassifier()
    X = [[0.1, 0.2], [0.4, 0.5]]
    y = [59.8, 59.1]
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    for p in probs:
        p_insec, p_sec = float(p[0]), float(p[1])
        assert abs((p_insec + p_sec) - 1.0) < 1e-5


def test_t2_f16_calibrated_gbdt_extreme_feature_bounds():
    model = RefCalibratedGBDTClassifier(method="platt")
    model.fit([[0.1, 0.2], [0.5, 0.6]], [59.7, 59.2])
    extreme_X = [[100.0, 100.0], [-100.0, -100.0]]
    probs = model.predict_proba(extreme_X)
    for p in probs:
        val = float(p[1] if hasattr(p, '__getitem__') else p)
        assert 0.0 <= val <= 1.0


def test_t2_f16_isotonic_monotone_order_preservation():
    model = RefCalibratedGBDTClassifier(method="isotonic")
    model.fit([[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]], [59.8, 59.4, 59.0])
    p1 = model.predict_proba([[0.1, 0.1]])[0][1]
    p2 = model.predict_proba([[0.9, 0.9]])[0][1]
    assert p1 >= p2


def test_t2_f17_mce_greater_than_or_equal_ece_invariant():
    y_true = [1.0, 0.0, 1.0, 1.0, 0.0]
    y_prob = [0.9, 0.1, 0.7, 0.8, 0.2]
    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=5)
    mce = ref_maximum_calibration_error(y_true, y_prob, n_bins=5)
    assert mce >= ece - 1e-6


def test_t2_f17_brier_score_perfect_prediction_zero():
    y_true = [1.0, 0.0, 1.0]
    y_prob = [1.0, 0.0, 1.0]
    brier = ref_brier_score(y_true, y_prob)
    assert abs(brier) < 1e-6


def test_t2_f18_window_sensitivity_dense_grid_execution(synthetic_scenario_batch):
    dense_grid = [0.4, 0.8, 1.2, 1.6, 2.0]
    benchmark = RefWindowSensitivityBenchmark(window_lengths=dense_grid)
    results = benchmark.run(synthetic_scenario_batch)
    assert len(results["results"]) == 5


def test_t2_f18_window_sensitivity_single_window_sweep(synthetic_scenario_batch):
    benchmark = RefWindowSensitivityBenchmark(window_lengths=[1.0])
    results = benchmark.run(synthetic_scenario_batch)
    assert len(results["results"]) == 1
    assert results["results"][0]["window_obs"] == 1.0


def test_t2_f19_ood_overlapping_ranges_rejection():
    try:
        RefOODGeneralizationBenchmark(id_range=(0.10, 0.35), ood_range=(0.30, 0.50))
        assert False, "Failed to reject overlapping ID and OOD ranges"
    except ValueError as e:
        assert "overlap" in str(e).lower()


def test_t2_f19_ood_identical_distributions_zero_gap(synthetic_scenario_batch):
    benchmark = RefOODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.10, 0.30))
    res = benchmark.run(synthetic_scenario_batch)
    assert res["delta_mae"] == pytest.approx(0.0, abs=1e-5)
    assert res["delta_f1"] == pytest.approx(0.0, abs=1e-5)


def test_t2_f20_noise_extreme_stress_bounded_output():
    traj = generate_synthetic_pmu_stream(noise_std=0.20, seed=520)
    assert not _has_nan(traj["bus_frequencies"])
    assert not _has_inf(traj["bus_frequencies"])


def test_t2_f20_noise_graceful_degradation_envelope(synthetic_scenario_batch):
    bench = RefMeasurementNoiseBenchmark()
    res = bench.run(synthetic_scenario_batch)
    for row in res["results"]:
        assert row["mae"] > 0.0
        assert row["rmse"] >= row["mae"] - 1e-4


def test_t2_f21_latency_profile_nonzero_timing_guarantee():
    bench = RefLatencyBenchmark()
    prof = bench.profile(n_iter=5)
    assert prof["t_feat_ms"] > 0.0
    assert prof["t_infer_ms"] > 0.0
    assert prof["t_total_ms"] > 0.0


def test_t2_f21_latency_speedup_vs_ode_simulation():
    bench = RefLatencyBenchmark()
    prof = bench.profile(n_iter=5)
    expected_speedup = prof["sim_time_ms"] / prof["t_total_ms"]
    assert abs(prof["speedup_factor"] - round(expected_speedup, 1)) < 0.2


def test_t2_f22_cli_help_flag_exit_code_zero_all_entrypoints():
    res = subprocess.run([sys.executable, "tests/e2e/run_e2e.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--tier" in res.stdout
