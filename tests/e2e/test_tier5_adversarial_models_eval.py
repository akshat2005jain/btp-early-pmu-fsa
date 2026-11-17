"""Stress and numerical edge-case E2E tests for 50 Hz models, calibration, benchmarks, and isolated CLIs."""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict

import numpy as np
import pytest

from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier, _pava_isotonic
from pmu_fsa.models.calibration_metrics import (
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
)
from pmu_fsa.models.classifiers import BinarySecurityClassifier, LogisticSecurityClassifier
from pmu_fsa.models.regressors import (
    AnalyticalSwingNadirRegressor,
    DecisionTreeNadirRegressor,
    GradientBoostingNadirRegressor,
    LightGBMNadirRegressor,
    LinearNadirRegressor,
    MeanNadirRegressor,
    RandomForestNadirRegressor,
)
from pmu_fsa.pipeline.run_experiments import ExperimentConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_py_exe() -> str:
    for name in ("python3", "python"):
        cand = PROJECT_ROOT / ".venv" / "bin" / name
        if cand.exists():
            return str(cand)
    return sys.executable


def _make_dummy_50hz_scenario(
    scen_id: str = "dummy",
    ren_pen: float = 0.20,
    nadir: float = 49.35,
    partition: str = "train",
) -> Dict[str, Any]:
    n_samples = 251
    dt = 0.02
    ts = np.round(np.arange(n_samples, dtype=float) * dt, 6)
    coi = np.full(n_samples, 50.0)
    coi[50:] = nadir
    bf = np.tile(coi[:, None], (1, 39))
    return {
        "timestamps": ts,
        "bus_frequencies": bf,
        "bus_rocof": np.zeros((n_samples, 39)),
        "voltage_magnitudes": np.ones((n_samples, 39)),
        "voltage_angles": np.zeros((n_samples, 39)),
        "pmu_sys_frequency": coi.copy(),
        "coi_frequency": coi.copy(),
        "metadata": {
            "scenario_id": scen_id,
            "partition": partition,
            "t_disturb": 1.0,
            "fps": 50,
            "f_nominal": 50.0,
            "renewable_penetration": ren_pen,
            "load_scale": 1.0,
            "ground_truth_nadir": nadir,
            "ground_truth_t_nadir": 3.5,
            "is_secure": 1 if nadir >= 49.50 else 0,
        },
    }


def test_t5_reg_50hz_extreme_nadir_clipping_bounds():
    X = np.array([[1.0, 2.0, 0.5], [2.0, 3.0, 0.6], [3.0, 4.0, 0.7], [4.0, 5.0, 0.8]])
    y_extreme = np.array([32.0, 38.0, 62.0, 68.0])

    models = [
        MeanNadirRegressor(clip_bounds=(45.0, 50.5)),
        AnalyticalSwingNadirRegressor(clip_bounds=(45.0, 50.5)),
        LinearNadirRegressor(clip_bounds=(45.0, 50.5)),
        DecisionTreeNadirRegressor(clip_bounds=(45.0, 50.5)),
        RandomForestNadirRegressor(n_estimators=5, clip_bounds=(45.0, 50.5)),
        GradientBoostingNadirRegressor(n_estimators=5, clip_bounds=(45.0, 50.5)),
        LightGBMNadirRegressor(n_estimators=5, clip_bounds=(45.0, 50.5)),
    ]
    for model in models:
        model.fit(X, y_extreme)
        preds = model.predict(X)
        assert np.all((preds >= 45.0) & (preds <= 50.5)), f"{model.model_name} violated [45.0, 50.5]"

    reg_unclipped = LinearNadirRegressor(clip_bounds=None)
    reg_unclipped.fit(X, y_extreme)
    p_unclipped = reg_unclipped.predict(X)
    assert np.any(p_unclipped < 45.0) or np.any(p_unclipped > 50.5)


def test_t5_reg_collinear_constant_single_sample_and_validation():
    col0 = np.linspace(0.1, 1.0, 20)
    X_col = np.column_stack([col0, 2.0 * col0])
    y_col = 49.8 - 0.5 * col0
    reg = LinearNadirRegressor(alpha=1e-3, scale_features=True).fit(X_col, y_col)
    assert float(np.mean(np.abs(y_col - reg.predict(X_col)))) < 0.05

    X_const = np.column_stack([np.ones(15), np.linspace(0.1, 0.5, 15)])
    y_const = 49.4 - 0.2 * X_const[:, 1]
    preds_const = LinearNadirRegressor(scale_features=True).fit(X_const, y_const).predict(X_const)
    assert np.all(np.isfinite(preds_const) & (preds_const >= 45.0) & (preds_const <= 50.5))

    X_single = np.array([[1.5, 2.5, 0.2]])
    y_single = np.array([49.25])
    for m in (
        MeanNadirRegressor(),
        AnalyticalSwingNadirRegressor(),
        LinearNadirRegressor(),
        DecisionTreeNadirRegressor(),
        RandomForestNadirRegressor(n_estimators=5),
        GradientBoostingNadirRegressor(n_estimators=5),
        LightGBMNadirRegressor(n_estimators=5),
    ):
        m.fit(X_single, y_single)
        assert abs(float(m.predict(X_single)[0]) - 49.25) < 0.1

    with pytest.raises(RuntimeError, match="not fitted"):
        LinearNadirRegressor().predict([[1.0, 2.0]])
    with pytest.raises(ValueError, match="empty dataset"):
        LinearNadirRegressor().fit([], [])
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        LinearNadirRegressor().fit([[1.0, 2.0], [3.0, 4.0]], [49.5, 49.0]).predict([[1.0, 2.0, 3.0]])

    m_zero_var = LinearNadirRegressor().fit([[1.0], [2.0], [3.0]], [49.5, 49.5, 49.5]).evaluate(
        [[1.0], [2.0], [3.0]], [49.5, 49.5, 49.5]
    )
    assert math.isfinite(m_zero_var["r2"]) and m_zero_var["mae"] < 1e-4


def test_t5_cls_49_5hz_threshold_degenerate_classes_and_all_backends():
    clf = BinarySecurityClassifier(threshold=49.50)
    y_prep = clf._prepare_y(np.array([49.500000, 49.499999, 49.500001]))
    np.testing.assert_array_equal(y_prep, [1, 0, 1])

    X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    log_sec = LogisticSecurityClassifier(threshold=49.50).fit(X, np.array([49.8, 49.7, 49.9]))
    assert log_sec.constant_class_ == 1
    np.testing.assert_array_equal(log_sec.predict_proba(X), [[0.0, 1.0]] * 3)

    log_insec = LogisticSecurityClassifier(threshold=49.50).fit(X, np.array([48.5, 48.7, 48.2]))
    assert log_insec.constant_class_ == 0
    np.testing.assert_array_equal(log_insec.predict_proba(X), [[1.0, 0.0]] * 3)

    X6 = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5], [0.6, 0.7], [0.8, 0.9], [0.9, 1.0]])
    y6 = np.array([1, 1, 1, 0, 0, 0])
    for b in ("regressor", "logistic", "decision_tree", "random_forest", "gradient_boosting", "lightgbm"):
        c = BinarySecurityClassifier(backend=b, threshold=49.50).fit(X6, y6)
        m = c.evaluate(X6, y6)
        assert 0.0 <= m["accuracy"] <= 1.0 and 0.0 <= m["f1"] <= 1.0

    rng = np.random.default_rng(42)
    X_imb = rng.normal(size=(50, 4))
    y_imb = np.array([49.8] * 49 + [48.5])
    m_imb = BinarySecurityClassifier(threshold=49.50).fit(X_imb, y_imb).evaluate(X_imb, y_imb)
    assert 0.0 <= m_imb["fnr_insecure"] <= 1.0


def test_t5_cal_pava_sigmoid_degenerate_fallback_and_interval_ordering():
    x_pts, y_pts = _pava_isotonic([49.0, 49.0, 49.0, 49.0], [0.0, 1.0, 0.0, 1.0])
    assert len(x_pts) >= 1 and all(y_pts[i] <= y_pts[i + 1] + 1e-7 for i in range(len(y_pts) - 1))

    x_rev, y_rev = _pava_isotonic([50.0, 49.5, 49.0, 48.5], [1.0, 1.0, 0.0, 0.0])
    assert x_rev == sorted(x_rev)

    assert _pava_isotonic([49.2], [1.0]) == ([49.2], [1.0])

    clf = CalibratedGBDTClassifier(method="platt", threshold=49.50)
    assert clf._calibrate_single(1000.0) == 1.0
    assert clf._calibrate_single(-1000.0) == 0.0
    assert clf._calibrate_single(float("inf")) == 1.0
    assert clf._calibrate_single(float("-inf")) == 0.0

    rng = np.random.default_rng(11)
    X_tr = rng.normal(size=(20, 2))
    y_tr = np.array([49.8] * 10 + [48.8] * 10)
    X_cal = rng.normal(size=(10, 2))
    y_cal_single = np.array([49.8] * 10)

    clf_deg = CalibratedGBDTClassifier(method="platt", threshold=49.50)
    with pytest.warns(UserWarning, match="falling back to physics default"):
        clf_deg.fit(X_tr, y_tr, X_cal=X_cal, y_cal=y_cal_single)
    assert clf_deg.platt_a == 8.0 and clf_deg.calibrator_ is None

    clf_ci = CalibratedGBDTClassifier(method="platt", threshold=49.50).fit(X_tr, y_tr)
    p_sec, l95, h95 = clf_ci.predict_with_intervals(X_tr, confidence_level=0.95)
    _, l99, h99 = clf_ci.predict_with_intervals(X_tr, confidence_level=0.99)
    assert np.all((h99 - l99) >= (h95 - l95) - 1e-9)
    assert np.all((l95 <= p_sec + 1e-9) & (p_sec - 1e-9 <= h95))


def test_t5_met_calibration_edge_cases():
    comp1 = compute_calibration_metrics([1, 1, 1, 1], [0.9, 0.8, 0.85, 0.95], threshold=49.50)
    assert 0.0 <= comp1["brier_score"] <= 1.0 and 0.0 <= comp1["ece"] <= 1.0

    ece1 = expected_calibration_error([1.0, 0.0, 1.0], [0.8, 0.2, 0.9], n_bins=1)
    mce1 = maximum_calibration_error([1.0, 0.0, 1.0], [0.8, 0.2, 0.9], n_bins=1)
    assert ece1 == pytest.approx(mce1, abs=1e-9)

    with pytest.raises(ValueError, match="contains NaN or Inf"):
        brier_score([1.0, 0.0], [0.8, np.nan])
    with pytest.raises(ValueError, match="contains NaN or Inf"):
        expected_calibration_error([1.0, 0.0], [0.8, np.nan])


def test_t5_benchmarks_edge_cases():
    assert len(WindowSensitivityBenchmark().run([])["results"]) == 0
    assert len(WindowSensitivityBenchmark(window_lengths=[0.5, 1.0]).run([_make_dummy_50hz_scenario()])["results"]) == 2

    for bad_w in ([0.5, -1.0], [0.0, 1.0]):
        with pytest.raises(ValueError, match="strictly positive"):
            WindowSensitivityBenchmark(window_lengths=bad_w)
    assert WindowSensitivityBenchmark(window_lengths=[1.5, 0.5, 1.5, 0.5]).window_lengths == [0.5, 1.5]

    with pytest.raises(ValueError, match="overlaps with OOD range"):
        OODGeneralizationBenchmark(id_range=(0.10, 0.35), ood_range=(0.30, 0.50))

    scens = [
        _make_dummy_50hz_scenario("s_tr_0", ren_pen=0.20, nadir=49.70, partition="train"),
        _make_dummy_50hz_scenario("s_tr_1", ren_pen=0.20, nadir=49.20, partition="train"),
        _make_dummy_50hz_scenario("s_te_0", ren_pen=0.20, nadir=49.65, partition="test"),
        _make_dummy_50hz_scenario("s_te_1", ren_pen=0.20, nadir=49.25, partition="test"),
    ]
    res_ident = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.10, 0.30)).run(scens)
    assert res_ident["delta_mae"] == 0.0 and res_ident["distribution_shift_detected"] is False

    with pytest.raises(ValueError, match="non-negative"):
        MeasurementNoiseBenchmark(noise_levels=[-0.05, 0.01])
    with pytest.raises(ValueError, match="n_trials must be >= 1"):
        MeasurementNoiseBenchmark.compare_ols_vs_endpoint_rocof(n_trials=0)

    res_noise = MeasurementNoiseBenchmark(noise_levels=[0.0, 1.5]).run(scens)
    for row in res_noise["results"]:
        assert row["mae"] >= 0.0 and row["rmse"] >= row["mae"] - 1e-9

    lb = LatencyBenchmark()
    prof = lb.profile(sample_data=np.zeros(14))
    assert prof["t_total_ms"] > 0.0 and prof["speedup_factor"] > 0.0


def test_t5_pipe_corrupted_config_and_isolated_cli_exit_codes(tmp_path: Path):
    bad_yaml = tmp_path / "broken.yaml"
    bad_yaml.write_text("invalid: [broken yaml", encoding="utf-8")
    cfg = ExperimentConfig.from_file(bad_yaml)
    assert cfg.seed == 42

    py_exe = _resolve_py_exe()
    res_train = subprocess.run(
        [
            py_exe,
            str(PROJECT_ROOT / "scripts" / "train_models.py"),
            "--features",
            str(tmp_path / "nonexistent_feat.npz"),
            "--output-dir",
            str(tmp_path / "models"),
            "--tables-dir",
            str(tmp_path / "tables"),
        ],
        capture_output=True,
        text=True,
    )
    assert res_train.returncode == 1

    res_eval = subprocess.run(
        [
            py_exe,
            str(PROJECT_ROOT / "scripts" / "evaluate_all.py"),
            "--data-dir",
            str(tmp_path / "empty_raw"),
            "--models-dir",
            str(tmp_path / "models"),
            "--tables-dir",
            str(tmp_path / "tables"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--report-file",
            str(tmp_path / "report.md"),
            "--benchmarks",
            "sensitivity",
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    assert res_eval.returncode == 0
