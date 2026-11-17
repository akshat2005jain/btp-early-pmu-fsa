"""End-to-end 50 Hz disturbance and grid security assessment scenario tests (Tier 4)."""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import pandas as pd
import pytest

from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.features.extractor import (
    calc_angle_divergence,
    calc_coi_acceleration,
    calc_delta_f,
    calc_initial_rocof,
    calc_interarea_dispersion,
)
from pmu_fsa.features.window_slicer import extract_causal_window
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import brier_score, expected_calibration_error
from pmu_fsa.models.classifiers import BinarySecurityClassifier
from pmu_fsa.models.regressors import (
    AnalyticalSwingNadirRegressor,
    DecisionTreeNadirRegressor,
    GradientBoostingNadirRegressor,
    LightGBMNadirRegressor,
    LinearNadirRegressor,
    MeanNadirRegressor,
    RandomForestNadirRegressor,
)
from tests.e2e.conftest import (
    IEEE39_GEN_PARAMS,
    _to_float_list_1d,
    assert_valid_pmu_trajectory,
    generate_synthetic_pmu_stream,
)
from tests.e2e.test_tier3_combinations import FEATURE_NAMES, _extract_tabular_features


def test_scenario_1_severe_generator_outage_high_load(synthetic_scenario_batch):
    stream = generate_synthetic_pmu_stream(
        n_samples=301,
        fps=50,
        f_nominal=50.0,
        f_nadir=49.18,
        t_disturb=1.0,
        disturbed_gen=35,
        renewable_penetration=0.20,
        active_loss_mw=650.0,
        seed=101,
    )
    assert_valid_pmu_trajectory(stream)
    assert stream["metadata"]["ground_truth_nadir"] < 49.20
    assert stream["metadata"]["is_secure"] == 0

    w = extract_causal_window(stream, t_obs=1.0)
    assert calc_initial_rocof(w) < -0.35
    drop_end, _ = calc_delta_f(w, f_nominal=50.0)
    assert drop_end < -0.25

    X_tr = np.array([_extract_tabular_features(s, 1.0) for s in synthetic_scenario_batch])
    y_tr = np.array([s["metadata"]["ground_truth_nadir"] for s in synthetic_scenario_batch])

    reg = LinearNadirRegressor().fit(X_tr, y_tr)
    cal_gb = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X_tr, y_tr)

    fv = _extract_tabular_features(stream, 1.0)
    assert len(fv) == len(FEATURE_NAMES) == 14

    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        v = _extract_tabular_features(stream, 1.0)
        _ = reg.predict([v])
        _ = cal_gb.predict_proba([v])
        timings.append((time.perf_counter() - t0) * 1000.0)
    assert min(timings) < 10.0

    pred_nadir = float(_to_float_list_1d(reg.predict([fv]))[0])
    p_sec = float(cal_gb.predict_proba([fv])[0, 1])
    assert pred_nadir < 49.40
    assert p_sec < 0.25


def test_scenario_2_high_renewable_low_inertia_shift(synthetic_scenario_batch):
    stream = generate_synthetic_pmu_stream(
        n_samples=251,
        fps=50,
        f_nominal=50.0,
        f_nadir=49.12,
        t_disturb=1.0,
        disturbed_gen=31,
        renewable_penetration=0.45,
        active_loss_mw=572.0,
        seed=202,
    )
    base_H_sum = sum(p["H"] for p in IEEE39_GEN_PARAMS.values())
    assert stream["metadata"]["effective_H_sum"] < base_H_sum * 0.70

    w = extract_causal_window(stream, t_obs=1.0)
    assert calc_initial_rocof(w) < -0.35
    curv_init, _ = calc_coi_acceleration(w)
    assert math.isfinite(curv_init)

    X_cal = np.array([_extract_tabular_features(s, 1.0) for s in synthetic_scenario_batch])
    y_cal = np.array([s["metadata"]["ground_truth_nadir"] for s in synthetic_scenario_batch])
    cal_gb = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X_cal, y_cal)

    feat_vec = _extract_tabular_features(stream, 1.0)
    p_sec, low_b, high_b = cal_gb.predict_with_intervals([feat_vec], confidence_level=0.95)
    assert float(p_sec[0]) < 0.20
    assert 0.0 <= float(low_b[0]) <= float(p_sec[0]) + 1e-9
    assert float(p_sec[0]) - 1e-9 <= float(high_b[0]) <= 1.0


def test_scenario_3_multi_area_boundary_tie_line_oscillation(real_pmu_trajectory, synthetic_scenario_batch):
    w = extract_causal_window(real_pmu_trajectory, t_obs=1.0)
    area_spread, bus_std = calc_interarea_dispersion(w)
    max_ang_div, tie_ang = calc_angle_divergence(w)

    assert area_spread > 0.01
    assert bus_std > 0.001
    assert max_ang_div > 1.5  # degrees
    assert tie_ang > 0.2      # degrees

    X_tr = np.array([_extract_tabular_features(s, 1.0) for s in synthetic_scenario_batch])
    y_tr = np.array([s["metadata"]["ground_truth_nadir"] for s in synthetic_scenario_batch])
    clf = BinarySecurityClassifier(threshold=49.50, alpha=0.5).fit(X_tr, y_tr)
    real_vec = _extract_tabular_features(real_pmu_trajectory, 1.0)
    assert real_pmu_trajectory["metadata"]["is_secure"] == 0
    assert int(clf.predict([real_vec])[0]) == 0


def test_scenario_4_marginal_disturbance_near_49_50hz_threshold():
    stream = generate_synthetic_pmu_stream(
        n_samples=251,
        fps=50,
        f_nominal=50.0,
        f_nadir=49.49,
        t_disturb=1.0,
        disturbed_gen=30,
        renewable_penetration=0.15,
        active_loss_mw=180.0,
        seed=404,
    )
    assert stream["metadata"]["ground_truth_nadir"] == pytest.approx(49.49)
    assert stream["metadata"]["is_secure"] == 0

    nadirs = [49.70, 49.60, 49.50, 49.40, 49.30]
    X_cal = [
        _extract_tabular_features(
            generate_synthetic_pmu_stream(n_samples=201, f_nadir=n_val, t_disturb=1.0, disturbed_gen=30, renewable_penetration=0.15),
            1.0,
        )
        for n_val in nadirs
    ]
    cal_gb = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X_cal, nadirs)

    feat_vec = _extract_tabular_features(stream, 1.0)
    p_sec = float(cal_gb.predict_proba([feat_vec])[0, 1])
    assert 0.15 <= p_sec <= 0.80


def test_scenario_5_noisy_pmu_stream_with_jitter_and_dropped_frames(synthetic_scenario_batch):
    s_clean = generate_synthetic_pmu_stream(
        n_samples=251, fps=50, t_disturb=1.0, disturbed_gen=35, active_loss_mw=650.0, f_nadir=49.25, noise_std=0.0, seed=505
    )
    s_noisy = generate_synthetic_pmu_stream(
        n_samples=251, fps=50, t_disturb=1.0, disturbed_gen=35, active_loss_mw=650.0, f_nadir=49.25, noise_std=0.02, seed=505
    )

    drop_indices = {35, 36, 75, 110, 180}
    kept = [i for i in range(len(s_noisy["timestamps"])) if i not in drop_indices]
    s_jittered = copy.deepcopy(s_noisy)
    for k in ("timestamps", "bus_frequencies", "bus_rocof", "voltage_magnitudes", "voltage_angles", "pmu_sys_frequency", "coi_frequency"):
        s_jittered[k] = s_jittered[k][kept]

    rocof_clean = calc_initial_rocof(extract_causal_window(s_clean, t_obs=1.0))
    rocof_noisy = calc_initial_rocof(extract_causal_window(s_jittered, t_obs=1.0))
    assert abs(rocof_clean - rocof_noisy) < 0.25

    feat_clean = _extract_tabular_features(s_clean, 1.0)
    feat_noisy = _extract_tabular_features(s_jittered, 1.0)
    assert all(math.isfinite(v) for v in feat_noisy)

    X_tr = [_extract_tabular_features(s, 1.0) for s in synthetic_scenario_batch]
    y_tr = [s["metadata"]["ground_truth_nadir"] for s in synthetic_scenario_batch]
    cal_gb = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X_tr, y_tr)
    p_clean = float(cal_gb.predict_proba([feat_clean])[0, 1])
    p_noisy = float(cal_gb.predict_proba([feat_noisy])[0, 1])
    assert (p_clean >= 0.5) == (p_noisy >= 0.5)


def test_scenario_6_ultra_early_0_5s_vs_standard_1_5s_comparison(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    y_true = np.array([s["metadata"]["ground_truth_nadir"] for s in batch])
    y_sec = np.array([s["metadata"]["is_secure"] for s in batch])

    X_05s = np.array([_extract_tabular_features(s, 0.5) for s in batch])
    X_15s = np.array([_extract_tabular_features(s, 1.5) for s in batch])

    reg_05 = LinearNadirRegressor().fit(X_05s, y_true)
    reg_15 = LinearNadirRegressor().fit(X_15s, y_true)
    mae_05 = float(np.mean(np.abs(reg_05.predict(X_05s) - y_true)))
    mae_15 = float(np.mean(np.abs(reg_15.predict(X_15s) - y_true)))
    assert mae_05 < 0.20 and mae_15 < 0.20

    cal_05 = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X_05s, y_true)
    cal_15 = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X_15s, y_true)
    assert brier_score(y_sec, cal_05.predict_proba(X_05s)[:, 1], threshold=49.50) < 0.20
    assert brier_score(y_sec, cal_15.predict_proba(X_15s)[:, 1], threshold=49.50) < 0.20


def test_scenario_7_online_inference_latency_compliance_50fps(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    reg = LinearNadirRegressor().fit([[0.1 * i] * 14 for i in range(10)], [50.0 - 0.1 * i for i in range(10)])

    _ = _extract_tabular_features(traj, 0.5)
    latencies_ms = []
    for _ in range(30):
        t0 = time.perf_counter()
        feat = _extract_tabular_features(traj, 0.5)
        _ = reg.predict([feat])
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    assert float(np.mean(latencies_ms)) < 6.0
    assert min(latencies_ms) < 5.0


def test_scenario_8_full_end_to_end_pipeline_execution(synthetic_scenario_batch):
    scenarios = synthetic_scenario_batch
    dataset_10 = pd.DataFrame([_extract_tabular_features(s, 1.0) for s in scenarios], columns=FEATURE_NAMES)
    y_nadir = np.array([s["metadata"]["ground_truth_nadir"] for s in scenarios])
    y_sec = np.array([s["metadata"]["is_secure"] for s in scenarios])

    models = {
        "Mean": MeanNadirRegressor(),
        "AnalyticalSwing": AnalyticalSwingNadirRegressor(),
        "Linear": LinearNadirRegressor(),
        "DecisionTree": DecisionTreeNadirRegressor(max_depth=3),
        "RandomForest": RandomForestNadirRegressor(n_estimators=10, max_depth=3),
        "GradientBoosting": GradientBoostingNadirRegressor(n_estimators=15, max_depth=2),
        "LightGBM": LightGBMNadirRegressor(n_estimators=15, max_depth=2),
    }
    for name, m in models.items():
        m.fit(dataset_10, y_nadir)
        mae = m.evaluate(dataset_10, y_nadir)["mae"]
        if name != "Mean":
            assert mae < 0.25

    cal_platt = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(dataset_10, y_nadir, y_sec)
    assert brier_score(y_sec, cal_platt.predict_proba(dataset_10)[:, 1], threshold=49.50) < 0.25
    assert expected_calibration_error(y_sec, cal_platt.predict_proba(dataset_10)[:, 1], n_bins=5, threshold=49.50) < 0.25

    assert len(WindowSensitivityBenchmark(threshold=49.50).run(scenarios)["results"]) == 4
    assert OODGeneralizationBenchmark(threshold=49.50).run(scenarios)["distribution_shift_detected"] is True
    assert len(MeasurementNoiseBenchmark(threshold=49.50).run(scenarios)["results"]) == 5
    assert LatencyBenchmark().profile(n_iter=15, warmup_reps=2, measure_simulation=False)["t_total_ms"] < 10.0


def test_scenario_9_under_frequency_load_shedding_readiness(synthetic_scenario_batch):
    stream = generate_synthetic_pmu_stream(
        n_samples=251,
        fps=50,
        f_nominal=50.0,
        f_nadir=49.02,
        t_disturb=1.0,
        disturbed_gen=38,
        renewable_penetration=0.40,
        active_loss_mw=830.0,
        seed=909,
    )
    feat_vec = _extract_tabular_features(stream, 0.8)

    X_tr = np.array([_extract_tabular_features(s, 0.8) for s in synthetic_scenario_batch])
    y_tr = np.array([s["metadata"]["ground_truth_nadir"] for s in synthetic_scenario_batch])

    ufls_thresh = 49.20
    reg = LinearNadirRegressor(alpha=0.1).fit(X_tr, y_tr)
    pred_nadir = float(reg.predict([feat_vec])[0])
    assert pred_nadir < ufls_thresh


def test_scenario_10_multi_area_boundary_islanding_prevention_alert(real_pmu_trajectory):
    w = extract_causal_window(real_pmu_trajectory, t_obs=1.5)
    max_ang_div, tie_ang = calc_angle_divergence(w)
    assert max_ang_div > 2.0  # degrees
    assert tie_ang > 0.3      # degrees
