"""Cross-module E2E integration tests across 50 Hz simulation, 14-column features, models, and benchmarks."""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.features.extractor import (
    CANONICAL_FEATURE_NAMES,
    calc_initial_rocof,
    extract_feature_vector,
)
from pmu_fsa.features.window_slicer import extract_causal_window
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import brier_score, reliability_diagram
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
    assert_feature_matrix_integrity,
    assert_valid_pmu_trajectory,
    generate_synthetic_pmu_stream,
)

FEATURE_NAMES: List[str] = list(CANONICAL_FEATURE_NAMES)


def _extract_tabular_features(trajectory: Dict[str, Any], t_obs: float = 1.0) -> List[float]:
    w = extract_causal_window(trajectory, t_obs=t_obs)
    vec = extract_feature_vector(w, f_nominal=50.0)
    return [float(v) for v in vec]


def test_flow1_network_to_sliced_50hz_pmu_tensor(ieee39_topology):
    assert ieee39_topology["n_buses"] == 39
    stream = generate_synthetic_pmu_stream(
        n_samples=251,
        fps=50,
        f_nominal=50.0,
        f_nadir=49.35,
        t_disturb=1.0,
        disturbed_gen=35,
        renewable_penetration=0.20,
        active_loss_mw=650.0,
        seed=101,
    )
    assert_valid_pmu_trajectory(stream)

    w = extract_causal_window(stream, t_obs=1.0)
    assert w["bus_frequencies"].shape == (51, 39)
    assert w["timestamps"][0] == pytest.approx(1.0, abs=1e-6)
    assert w["timestamps"][-1] == pytest.approx(2.0, abs=1e-6)


def test_flow1_pmu_sampling_rate_25_50_60_slicing():
    s_25 = generate_synthetic_pmu_stream(n_samples=126, fps=25, t_disturb=1.0, seed=10)
    s_50 = generate_synthetic_pmu_stream(n_samples=251, fps=50, t_disturb=1.0, seed=10)
    s_60 = generate_synthetic_pmu_stream(n_samples=301, fps=60, t_disturb=1.0, seed=10)

    assert len(extract_causal_window(s_25, t_obs=1.0)["timestamps"]) == 26
    assert len(extract_causal_window(s_50, t_obs=1.0)["timestamps"]) == 51
    assert len(extract_causal_window(s_60, t_obs=1.0)["timestamps"]) == 61


def test_flow1_multibus_spatial_ordering_in_sliced_window():
    stream = generate_synthetic_pmu_stream(n_samples=201, fps=50, t_disturb=1.0, disturbed_gen=35, seed=42)
    w = extract_causal_window(stream, t_obs=0.8)
    assert w["bus_frequencies"][-1, 34] <= w["bus_frequencies"][-1, 0]
    assert w["voltage_angles"].shape[1] == 39


def test_flow2_renewable_inertia_steepens_50hz_rocof():
    configs = [(0.10, 49.58), (0.25, 49.40), (0.45, 49.15)]
    rocofs = []
    for ren, nadir in configs:
        stream = generate_synthetic_pmu_stream(
            n_samples=251,
            fps=50,
            t_disturb=1.0,
            disturbed_gen=35,
            renewable_penetration=ren,
            active_loss_mw=650.0,
            f_nadir=nadir,
            seed=55,
        )
        rocofs.append(calc_initial_rocof(extract_causal_window(stream, t_obs=0.5)))
    assert rocofs[0] > rocofs[1] > rocofs[2]


def test_flow2_contingency_size_vs_renewable_penetration_coupling():
    s_small = generate_synthetic_pmu_stream(active_loss_mw=300.0, renewable_penetration=0.40, f_nadir=49.62, seed=1)
    s_large_low = generate_synthetic_pmu_stream(active_loss_mw=650.0, renewable_penetration=0.10, f_nadir=49.45, seed=2)
    s_compound = generate_synthetic_pmu_stream(active_loss_mw=650.0, renewable_penetration=0.40, f_nadir=49.12, seed=3)

    r1 = calc_initial_rocof(extract_causal_window(s_small, t_obs=0.5))
    r2 = calc_initial_rocof(extract_causal_window(s_large_low, t_obs=0.5))
    r3 = calc_initial_rocof(extract_causal_window(s_compound, t_obs=0.5))
    assert r3 < r1 and r3 < r2


def test_flow3_causal_window_to_14_feature_matrix_and_7_regressors(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    X = np.array([_extract_tabular_features(s, 1.0) for s in batch])
    X_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    y_nadir = np.array([s["metadata"]["ground_truth_nadir"] for s in batch])
    y_secure = (y_nadir >= 49.50).astype(int)

    feat_dict = {
        "X": X,
        "y_nadir": y_nadir,
        "y_secure": y_secure,
        "scenario_ids": [s["metadata"]["scenario_id"] for s in batch],
        "feature_names": FEATURE_NAMES,
    }
    assert_feature_matrix_integrity(feat_dict)

    models = [
        MeanNadirRegressor(),
        AnalyticalSwingNadirRegressor(),
        LinearNadirRegressor(),
        DecisionTreeNadirRegressor(max_depth=3),
        RandomForestNadirRegressor(n_estimators=10, max_depth=3),
        GradientBoostingNadirRegressor(n_estimators=15, max_depth=2),
        LightGBMNadirRegressor(n_estimators=15, max_depth=2),
    ]
    for m in models:
        m.fit(X_df, y_nadir)
        preds = m.predict(X_df)
        assert preds.shape == (len(batch),)
        assert np.all((preds >= 45.0) & (preds <= 50.5))


def test_flow4_dynamic_features_to_calibrated_gbdt_and_intervals(sample_causal_features):
    X = sample_causal_features["X"]
    y_nadir = sample_causal_features["y_nadir"]
    y_sec = sample_causal_features["y_secure"]

    clf = BinarySecurityClassifier(threshold=49.50).fit(X, y_nadir)
    cal_gb = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X, y_nadir, y_sec)

    assert len(clf.predict(X)) == len(cal_gb.predict(X))
    probs = cal_gb.predict_proba(X)
    np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)

    p_sec, low_b, high_b = cal_gb.predict_with_intervals(X, confidence_level=0.95)
    assert np.all((low_b <= p_sec + 1e-9) & (p_sec - 1e-9 <= high_b))


def test_flow5_platt_vs_isotonic_and_window_sensitivity_calibration(
    synthetic_scenario_batch,
    sample_classification_dataset,
):
    bench = WindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0], threshold=49.50)
    res = bench.run(synthetic_scenario_batch)
    assert len(res["results"]) == 4
    for row in res["results"]:
        assert 0.0 <= row["ece"] <= 1.0

    X = sample_classification_dataset["X"]
    y_n = sample_classification_dataset["y_nadir"]
    y_s = sample_classification_dataset["y_secure"]
    tr = np.arange(0, 60, 3)
    ca = np.array([i for i in range(60) if i % 3 != 0])

    platt = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20, fit_full_after_cal=False)
    iso = CalibratedGBDTClassifier(method="isotonic", threshold=49.50, n_estimators=20, fit_full_after_cal=False)
    platt.fit(X[tr], y_n[tr], y_s[tr], X_cal=X[ca], y_cal=y_n[ca])
    iso.fit(X[tr], y_n[tr], y_s[tr], X_cal=X[ca], y_cal=y_n[ca])
    assert len(iso.isotonic_x) > 0

    assert brier_score(y_s, platt.predict_proba(X)[:, 1], threshold=49.50) < 0.25
    assert brier_score(y_s, iso.predict_proba(X)[:, 1], threshold=49.50) < 0.25

    diag = reliability_diagram([0, 0, 0, 0, 1, 0, 1, 1, 1, 1], [0.1, 0.15, 0.2, 0.35, 0.45, 0.55, 0.7, 0.8, 0.85, 0.95], n_bins=5)
    assert sum(diag["bin_counts"]) == 10


def test_flow6_pmu_noise_propagation_ols_advantage_and_unsmoothed_benchmark(synthetic_scenario_batch):
    s_clean = generate_synthetic_pmu_stream(noise_std=0.0, seed=123)
    s_noisy = generate_synthetic_pmu_stream(noise_std=0.02, seed=123)

    f_clean = _extract_tabular_features(s_clean, 1.0)
    f_noisy = _extract_tabular_features(s_noisy, 1.0)
    assert abs(f_clean[0] - f_noisy[0]) < 0.20

    res = MeasurementNoiseBenchmark(noise_levels=[0.0, 0.01, 0.03, 0.05], threshold=49.50).run(synthetic_scenario_batch)
    assert len(res["results"]) == 4
    for row in res["results"]:
        assert row["mae"] == row["mae_raw"]
        assert 0.0 <= row["ece"] <= 1.0


def test_flow7_ood_generalization_held_out_partitions(synthetic_scenario_batch):
    res = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.40, 0.50), threshold=49.50).run(synthetic_scenario_batch)
    assert res["id_test_count"] == 4
    assert res["ood_scenario_count"] == 6
    assert res["distribution_shift_detected"] is True
    assert 0.0 <= res["ood_metrics"]["mae"] < 1.5
    assert 0.0 <= res["ood_metrics"]["ece"] <= 1.0


def test_flow8_streaming_latency_budget_and_speedup(real_pmu_trajectory):
    bench = LatencyBenchmark()
    prof = bench.profile(
        scenarios=[real_pmu_trajectory],
        n_iter=12,
        warmup_reps=2,
        measure_simulation=True,
        n_sim_runs=2,
    )
    assert prof["t_total_ms"] < 10.0
    assert prof["speedup_factor"] > 5.0


def test_flow9_full_chain_50hz_pipeline_integration(ieee39_topology, synthetic_scenario_batch):
    assert ieee39_topology["nominal_freq"] == 50.0
    X = np.array([_extract_tabular_features(s, t_obs=1.0) for s in synthetic_scenario_batch])
    y_nadir = np.array([s["metadata"]["ground_truth_nadir"] for s in synthetic_scenario_batch])
    y_sec = np.array([s["metadata"]["is_secure"] for s in synthetic_scenario_batch])
    assert X.shape == (20, 14)

    cal_gb = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=20).fit(X[:10], y_nadir[:10], y_sec[:10])
    probs = cal_gb.predict_proba(X[10:])
    assert probs.shape == (10, 2)
    assert brier_score(y_sec[10:], probs[:, 1], threshold=49.50) < 0.30


def test_pairwise_invariants(ieee39_topology, sample_pmu_trajectory, sample_causal_features):
    areas = ieee39_topology["area_partition"]
    all_area_buses = set(areas["Area 1"]) | set(areas["Area 2"]) | set(areas["Area 3"])
    assert set(ieee39_topology["generator_buses"]).issubset(all_area_buses)

    stream = generate_synthetic_pmu_stream(
        disturbed_gen=39, renewable_penetration=0.45, active_loss_mw=1000.0, f_nadir=48.85, seed=999
    )
    assert_valid_pmu_trajectory(stream)
    assert float(np.min(stream["bus_frequencies"])) >= 45.0

    reg = LinearNadirRegressor(alpha=1e-3).fit(sample_causal_features["X"], sample_causal_features["y_nadir"])
    assert len(reg.weights) == 14
    assert float(np.max(np.abs(reg.weights))) > 0.01

    _ = _extract_tabular_features(sample_pmu_trajectory, 0.5)
    for t_w in (0.5, 1.0, 1.5, 2.0):
        trials = []
        for _ in range(3):
            t0 = time.perf_counter()
            _ = _extract_tabular_features(sample_pmu_trajectory, t_w)
            trials.append((time.perf_counter() - t0) * 1000.0)
        assert min(trials) < 12.0
