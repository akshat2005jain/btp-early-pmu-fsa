"""Stress and numerical edge-case tests for models, calibration, benchmarks, and pipeline."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from pmu_fsa.models.regressors import (
    LinearNadirRegressor,
    DecisionTreeNadirRegressor,
    RandomForestNadirRegressor,
    GradientBoostingNadirRegressor,
    get_nadir_regressor,
)
from pmu_fsa.models.classifiers import (
    BinarySecurityClassifier,
    LogisticSecurityClassifier,
    DecisionTreeSecurityClassifier,
    RandomForestSecurityClassifier,
    GradientBoostingSecurityClassifier,
    get_security_classifier,
)
from pmu_fsa.models.calibrated_gb import (
    CalibratedGBDTClassifier,
    _pava_isotonic,
)
from pmu_fsa.models.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
    compute_calibration_metrics,
)
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.pipeline.run_experiments import (
    ExperimentConfig,
    ExperimentPipeline,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_py_exe() -> str:
    cand = PROJECT_ROOT / ".venv" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def test_t5_reg_extreme_nadir_clipping_bounds():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
    y_extreme = np.array([38.0, 42.0, 68.0, 74.0])

    models = [
        LinearNadirRegressor(clip_bounds=(55.0, 60.5)),
        DecisionTreeNadirRegressor(clip_bounds=(55.0, 60.5)),
        RandomForestNadirRegressor(n_estimators=5, clip_bounds=(55.0, 60.5)),
        GradientBoostingNadirRegressor(n_estimators=5, clip_bounds=(55.0, 60.5)),
    ]

    for model in models:
        model.fit(X, y_extreme)
        preds = model.predict(X)
        for p in preds:
            assert 55.0 <= p <= 60.5, f"{model.model_name} failed to clip extreme prediction {p}"

    reg_unclipped = LinearNadirRegressor(clip_bounds=None)
    reg_unclipped.fit(X, y_extreme)
    p_unclipped = reg_unclipped.predict(X)
    assert np.any(p_unclipped < 55.0) or np.any(p_unclipped > 60.5)


def test_t5_reg_collinear_feature_matrix_ridge():
    col0 = np.linspace(0.1, 1.0, 20)
    col1 = 2.0 * col0
    X = np.column_stack([col0, col1])
    y = 59.5 - 0.5 * col0

    reg = LinearNadirRegressor(alpha=1e-3, scale_features=True)
    reg.fit(X, y)
    preds = reg.predict(X)

    mae = float(np.mean(np.abs(y - preds)))
    assert mae < 0.05, f"Ridge failed on collinear features, MAE={mae}"
    assert np.all(np.isfinite(reg.weights))
    assert np.isfinite(reg.intercept)


def test_t5_reg_zero_variance_constant_features():
    X = np.column_stack([np.ones(15), np.linspace(0.1, 0.5, 15)])
    y = 59.4 - 0.2 * X[:, 1]

    reg = LinearNadirRegressor(scale_features=True)
    reg.fit(X, y)
    preds = reg.predict(X)

    assert not np.any(np.isnan(preds))
    assert not np.any(np.isinf(preds))
    assert np.all((preds >= 55.0) & (preds <= 60.5))


def test_t5_reg_single_sample_fit_identity():
    X_single = np.array([[1.5, 2.5]])
    y_single = np.array([59.25])

    models = [
        LinearNadirRegressor(),
        DecisionTreeNadirRegressor(),
        RandomForestNadirRegressor(n_estimators=5),
        GradientBoostingNadirRegressor(n_estimators=5),
    ]

    for model in models:
        model.fit(X_single, y_single)
        pred = model.predict(X_single)
        assert len(pred) == 1
        assert abs(pred[0] - 59.25) < 0.1, f"{model.model_name} failed single-sample identity"


def test_t5_reg_unfitted_model_rejection():
    reg = LinearNadirRegressor()
    with pytest.raises(RuntimeError, match="not fitted"):
        reg.predict([[1.0, 2.0]])


def test_t5_reg_empty_and_mismatched_dimensions():
    reg = LinearNadirRegressor()
    with pytest.raises(ValueError, match="empty dataset"):
        reg.fit([], [])

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = np.array([59.5, 59.0])
    reg.fit(X, y)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        reg.predict([[1.0, 2.0, 3.0]])


def test_t5_reg_zero_variance_target_r2_metric():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([59.5, 59.5, 59.5])

    reg = LinearNadirRegressor()
    reg.fit(X, y)
    metrics = reg.evaluate(X, y)

    assert "r2" in metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert not math.isnan(metrics["r2"])
    assert metrics["mae"] < 1e-4


def test_t5_cls_threshold_boundary_precision():
    clf = BinarySecurityClassifier(threshold=59.5)
    y_prep = clf._prepare_y(np.array([59.500000, 59.499999, 59.500001]))

    assert y_prep[0] == 1, "59.500000 Hz must be Secure (1)"
    assert y_prep[1] == 0, "59.499999 Hz must be Insecure (0)"
    assert y_prep[2] == 1, "59.500001 Hz must be Secure (1)"


def test_t5_cls_degenerate_all_secure_training():
    # BaseSecurityClassifier subclasses set constant_class_=1; regressor backend fits continuous regression
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_all_sec = np.array([59.8, 59.7, 59.9])

    log_clf = LogisticSecurityClassifier(threshold=59.5)
    log_clf.fit(X, y_all_sec)
    assert log_clf.constant_class_ == 1
    probs_log = log_clf.predict_proba(X)
    for row in probs_log:
        assert row[0] == 0.0 and row[1] == 1.0

    bin_clf = BinarySecurityClassifier(threshold=59.5, backend="regressor")
    bin_clf.fit(X, y_all_sec)
    probs_bin = bin_clf.predict_proba(X)
    for row in probs_bin:
        assert row[1] > 0.50, f"Expected secure probability > 0.5, got {row[1]}"
        assert abs(row[0] + row[1] - 1.0) < 1e-6


def test_t5_cls_degenerate_all_insecure_training():
    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    y_all_insec = np.array([58.5, 58.7, 58.2])

    clf = LogisticSecurityClassifier(threshold=59.5)
    clf.fit(X, y_all_insec)
    assert clf.constant_class_ == 0

    probs = clf.predict_proba(X)
    assert len(probs) == 3
    for row in probs:
        assert row[0] == 1.0 and row[1] == 0.0


def test_t5_cls_extreme_imbalance_safety_critical_fnr():
    X = np.random.randn(100, 4)
    y_nadir = np.array([59.8] * 99 + [58.5] * 1)

    clf = BinarySecurityClassifier(threshold=59.5)
    clf.fit(X, y_nadir)
    metrics = clf.evaluate(X, y_nadir)

    assert "fnr_insecure" in metrics
    assert not math.isnan(metrics["fnr_insecure"])
    assert 0.0 <= metrics["fnr_insecure"] <= 1.0


def test_t5_cls_all_backends_execution():
    X = np.array([
        [0.1, 0.2], [0.2, 0.3], [0.4, 0.5],
        [0.6, 0.7], [0.8, 0.9], [0.9, 1.0]
    ])
    y_sec = np.array([1, 1, 1, 0, 0, 0])

    backends = ["regressor", "logistic", "decision_tree", "random_forest", "gradient_boosting"]
    for b in backends:
        clf = BinarySecurityClassifier(backend=b, threshold=59.5)
        clf.fit(X, y_sec)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        m = clf.evaluate(X, y_sec)

        assert len(preds) == 6
        assert len(probs) == 6
        assert 0.0 <= m["accuracy"] <= 1.0
        assert 0.0 <= m["f1"] <= 1.0


def test_t5_cls_threshold_monotonicity():
    X = np.random.randn(30, 4)
    y = np.array([59.8] * 15 + [58.8] * 15)

    clf = BinarySecurityClassifier(threshold=59.5)
    clf.fit(X, y)

    counts_secure = []
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    for t in thresholds:
        preds = clf.predict(X, proba_thresh=t)
        counts_secure.append(int(np.sum(preds == 1)))

    for i in range(len(counts_secure) - 1):
        assert counts_secure[i] >= counts_secure[i + 1], (
            f"Threshold {thresholds[i]} produced fewer secure predictions than {thresholds[i+1]}"
        )


def test_t5_cls_probability_simplex_normalization():
    X = np.random.randn(20, 4) * 100.0
    clf = LogisticSecurityClassifier()
    clf.fit(X, [1] * 10 + [0] * 10)
    probs = clf.predict_proba(X)

    for row in probs:
        assert 0.0 <= row[0] <= 1.0
        assert 0.0 <= row[1] <= 1.0
        assert abs(row[0] + row[1] - 1.0) < 1e-6


def test_t5_cal_pava_isotonic_tied_scores():
    scores = [59.0, 59.0, 59.0, 59.0]
    targets = [0.0, 1.0, 0.0, 1.0]
    x_pts, y_pts = _pava_isotonic(scores, targets)

    assert len(x_pts) >= 1
    assert len(y_pts) == len(x_pts)
    for i in range(len(y_pts) - 1):
        assert y_pts[i] <= y_pts[i + 1] + 1e-7


def test_t5_cal_pava_isotonic_reverse_order():
    scores = [60.0, 59.5, 59.0, 58.5]
    targets = [1.0, 1.0, 0.0, 0.0]
    x_pts, y_pts = _pava_isotonic(scores, targets)

    assert len(x_pts) > 0
    assert x_pts == sorted(x_pts)
    for i in range(len(y_pts) - 1):
        assert y_pts[i] <= y_pts[i + 1] + 1e-7


def test_t5_cal_pava_isotonic_single_sample():
    x_pts, y_pts = _pava_isotonic([59.2], [1.0])
    assert x_pts == [59.2]
    assert y_pts == [1.0]


def test_t5_cal_sigmoid_extreme_saturation():
    clf = CalibratedGBDTClassifier(method="platt")
    p_pos = clf._calibrate_single(1000.0)
    p_neg = clf._calibrate_single(-1000.0)
    p_inf = clf._calibrate_single(float("inf"))
    p_ninf = clf._calibrate_single(float("-inf"))

    assert p_pos == 1.0
    assert p_neg == 0.0
    assert p_inf == 1.0
    assert p_ninf == 0.0


def test_t5_cal_degenerate_calibration_set_fallback():
    X_tr = np.random.randn(20, 2)
    y_tr = np.array([59.8] * 10 + [58.8] * 10)
    X_cal = np.random.randn(10, 2)
    y_cal_single = np.array([59.8] * 10)

    clf = CalibratedGBDTClassifier(method="platt")
    with pytest.warns(UserWarning, match="falling back to physics default"):
        clf.fit(X_tr, y_tr, X_cal=X_cal, y_cal=y_cal_single)

    assert clf.platt_a == 8.0
    assert clf.calibrator_ is None


def test_t5_cal_validation_fraction_minority_class_defect():
    # Single-member minority class either triggers stratified split ValueError or falls back cleanly
    X = np.random.randn(25, 4)
    y_nadir = np.array([59.8] * 24 + [58.5])
    clf = CalibratedGBDTClassifier(validation_fraction=0.2, random_state=42)

    try:
        clf.fit(X, y_nadir)
        assert clf.calibrator_ is not None or clf.platt_a == 8.0
    except ValueError as exc:
        assert "least populated class" in str(exc) or "too few" in str(exc)


def test_t5_cal_epistemic_confidence_intervals_ordering():
    X = np.random.randn(20, 4)
    y = np.array([59.8] * 10 + [58.8] * 10)

    clf = CalibratedGBDTClassifier(method="platt")
    clf.fit(X, y)

    p_sec, low95, high95 = clf.predict_with_intervals(X, confidence_level=0.95)
    _, low99, high99 = clf.predict_with_intervals(X, confidence_level=0.99)

    for p, l95, h95, l99, h99 in zip(p_sec, low95, high95, low99, high99):
        assert 0.0 <= l95 <= p + 1e-6
        assert p - 1e-6 <= h95 <= 1.0
        assert (h99 - l99) >= (h95 - l95) - 1e-6


def test_t5_met_single_class_ground_truth():
    y_true_all1 = [1, 1, 1, 1]
    y_prob = [0.9, 0.8, 0.85, 0.95]
    comp1 = compute_calibration_metrics(y_true_all1, y_prob)

    assert 0.0 <= comp1["brier_score"] <= 1.0
    assert 0.0 <= comp1["ece"] <= 1.0
    assert 0.0 <= comp1["mce"] <= 1.0

    y_true_all0 = [0, 0, 0, 0]
    comp0 = compute_calibration_metrics(y_true_all0, [0.1, 0.2, 0.05, 0.15])
    assert 0.0 <= comp0["brier_score"] <= 1.0
    assert 0.0 <= comp0["ece"] <= 1.0


def test_t5_met_single_bin_limit():
    yt = [1.0, 0.0, 1.0]
    yp = [0.8, 0.2, 0.9]
    ece1 = expected_calibration_error(yt, yp, n_bins=1)
    mce1 = maximum_calibration_error(yt, yp, n_bins=1)

    expected_diff = abs(np.mean(yt) - np.mean(yp))
    assert abs(ece1 - expected_diff) < 1e-6
    assert abs(mce1 - expected_diff) < 1e-6


def test_t5_met_nan_inf_prob_handling():
    yt = [1.0, 0.0]
    yp_nan = [0.8, np.nan]

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        brier_score(yt, yp_nan)

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        expected_calibration_error(yt, yp_nan)


def test_t5_met_perfect_vs_inverted_extremes():
    yt = [0.0, 1.0, 0.0, 1.0]
    assert abs(brier_score(yt, [0.0, 1.0, 0.0, 1.0])) < 1e-9
    assert abs(brier_score(yt, [1.0, 0.0, 1.0, 0.0]) - 1.0) < 1e-9


def test_t5_met_mce_ge_ece_mathematical_invariant():
    rng = np.random.RandomState(42)
    for _ in range(20):
        n = rng.randint(5, 50)
        yt = rng.randint(0, 2, size=n)
        yp = rng.uniform(0.0, 1.0, size=n)
        n_bins = rng.randint(2, 10)

        ece = expected_calibration_error(yt, yp, n_bins=n_bins)
        mce = maximum_calibration_error(yt, yp, n_bins=n_bins)

        assert 0.0 <= ece <= 1.0
        assert 0.0 <= mce <= 1.0
        assert mce >= ece - 1e-9, f"Invariant violated: MCE ({mce}) < ECE ({ece})"


def _make_dummy_scenario(scen_id: str = "dummy", ren_pen: float = 0.20, nadir: float = 59.4) -> Dict[str, Any]:
    n_samples = 250
    dt = 0.02
    ts = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    coi = np.full(n_samples, 60.0)
    coi[50:] = nadir
    bf = np.tile(coi[:, np.newaxis], (1, 39))
    rocof = np.gradient(bf, dt, axis=0)
    return {
        "timestamps": ts,
        "bus_frequencies": bf,
        "bus_rocof": rocof,
        "voltage_magnitudes": np.ones((n_samples, 39)),
        "voltage_angles": np.zeros((n_samples, 39)),
        "coi_frequency": coi,
        "metadata": {
            "scenario_id": scen_id,
            "t_disturb": 1.0,
            "renewable_penetration": ren_pen,
            "ground_truth_nadir": nadir,
            "ground_truth_t_nadir": 3.5,
            "is_secure": 1 if nadir >= 59.5 else 0,
        },
    }


def test_t5_sens_empty_scenario_list_schema():
    bench = WindowSensitivityBenchmark()
    res = bench.run([])
    assert "columns" in res
    assert "results" in res
    assert len(res["results"]) == 0


def test_t5_sens_single_scenario_graceful_run():
    scen = _make_dummy_scenario()
    bench = WindowSensitivityBenchmark(window_lengths=[0.5, 1.0])
    res = bench.run([scen])
    assert len(res["results"]) == 2


def test_t5_sens_invalid_and_duplicate_window_lengths():
    with pytest.raises(ValueError, match="strictly positive"):
        WindowSensitivityBenchmark(window_lengths=[0.5, -1.0])

    with pytest.raises(ValueError, match="strictly positive"):
        WindowSensitivityBenchmark(window_lengths=[0.0, 1.0])

    bench = WindowSensitivityBenchmark(window_lengths=[1.5, 0.5, 1.5, 0.5])
    assert bench.window_lengths == [0.5, 1.5]


def test_t5_ood_overlapping_intervals_strict_rejection():
    with pytest.raises(ValueError, match="overlaps with OOD range"):
        OODGeneralizationBenchmark(id_range=(0.10, 0.35), ood_range=(0.30, 0.50))

    with pytest.raises(ValueError, match="overlaps with OOD range"):
        OODGeneralizationBenchmark(id_range=(0.10, 0.40), ood_range=(0.40, 0.50))


def test_t5_ood_identical_intervals_zero_gap_contract():
    scens = [_make_dummy_scenario(f"s_{i}", ren_pen=0.20) for i in range(5)]
    bench = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.10, 0.30))
    res = bench.run(scens)

    assert res["delta_mae"] == 0.0
    assert res["delta_f1"] == 0.0
    assert res["distribution_shift_detected"] is False


def test_t5_noise_negative_std_rejection():
    with pytest.raises(ValueError, match="non-negative"):
        MeasurementNoiseBenchmark(noise_levels=[-0.05, 0.01])


def test_t5_noise_extreme_noise_physical_clipping():
    scens = [_make_dummy_scenario(f"s_{i}", ren_pen=0.20) for i in range(5)]
    bench = MeasurementNoiseBenchmark(noise_levels=[0.0, 2.0])
    res = bench.run(scens)

    for row in res["results"]:
        assert row["mae"] > 0
        assert row["rmse"] >= row["mae"]


def test_t5_noise_ols_zero_trials_defect():
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        MeasurementNoiseBenchmark.compare_ols_vs_endpoint_rocof(n_trials=0)


def test_t5_lat_batch_scaling_and_1d_input():
    lb = LatencyBenchmark()
    prof = lb.profile(sample_data=np.random.randn(14))
    assert prof["t_total_ms"] > 0.0
    assert prof["speedup_factor"] > 0.0

    batch_res = lb.profile_batch_scaling(batch_sizes=[1, 5, 20])
    assert "batch_scaling" in batch_res
    for b_size in [1, 5, 20]:
        assert batch_res["batch_scaling"][b_size]["per_sample_ms"] < 2.0


def test_t5_pipe_config_corrupted_file_fallback():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("invalid: [broken yaml")
        f_path = f.name

    try:
        cfg = ExperimentConfig.from_file(f_path)
        assert cfg.seed == 42
        assert cfg.n_scenarios == 30
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_t5_cli_missing_data_exit_codes():
    py_exe = _resolve_py_exe()

    res_train = subprocess.run(
        [py_exe, str(PROJECT_ROOT / "scripts" / "train_models.py"), "--features", "/tmp/nonexistent_feat_999.npz"],
        capture_output=True,
        text=True,
    )
    assert res_train.returncode == 1

    res_eval = subprocess.run(
        [
            py_exe, str(PROJECT_ROOT / "scripts" / "evaluate_all.py"),
            "--data-dir", "/tmp/empty_raw_scens_999",
            "--tables-dir", "/tmp/t5_test_tables",
            "--report-file", "/tmp/t5_test_tables/report.md",
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    assert res_eval.returncode == 0
