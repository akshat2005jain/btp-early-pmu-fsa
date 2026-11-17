"""End-to-end Tier 1 tests for 50 Hz nadir regressors, security classifiers, and calibration."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score,
    compute_calibration_metrics,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_diagram,
)
from pmu_fsa.models.classifiers import (
    BinarySecurityClassifier,
    DecisionTreeSecurityClassifier,
    GradientBoostingSecurityClassifier,
    LightGBMSecurityClassifier,
    LogisticSecurityClassifier,
    RandomForestSecurityClassifier,
    compute_binary_security_metrics,
    evaluate_probability_threshold_sweep,
)
from pmu_fsa.models.regressors import (
    AnalyticalSwingNadirRegressor,
    DecisionTreeNadirRegressor,
    GradientBoostingNadirRegressor,
    LightGBMNadirRegressor,
    LinearNadirRegressor,
    MeanNadirRegressor,
    RandomForestNadirRegressor,
)
from pmu_fsa.simulation.ieee39_data import PHYSICAL_FREQ_BOUNDS, SECURITY_THRESHOLD_HZ


def test_f14_all_seven_production_regressors_fit_predict_save_load(
    tmp_path: Path,
    sample_regression_dataset,
):
    X = sample_regression_dataset["X"]
    y = sample_regression_dataset["y_nadir"]
    feature_names = sample_regression_dataset["feature_names"]
    X_df = pd.DataFrame(X, columns=feature_names)
    assert X_df.shape[1] == 14

    all_regressors = [
        ("mean", MeanNadirRegressor()),
        ("analytical_swing", AnalyticalSwingNadirRegressor()),
        ("linear", LinearNadirRegressor()),
        ("decision_tree", DecisionTreeNadirRegressor(max_depth=4)),
        ("random_forest", RandomForestNadirRegressor(n_estimators=15, max_depth=4)),
        ("gradient_boosting", GradientBoostingNadirRegressor(n_estimators=20, max_depth=3)),
        ("lightgbm", LightGBMNadirRegressor(n_estimators=20, max_depth=3)),
    ]

    low_b, high_b = PHYSICAL_FREQ_BOUNDS
    for name, reg in all_regressors:
        reg.fit(X_df, y)
        preds = reg.predict(X_df)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))
        assert np.all((preds >= low_b) & (preds <= high_b))

        metrics = reg.evaluate(X_df, y)
        assert metrics["mae"] >= 0.0
        assert metrics["rmse"] >= metrics["mae"] - 1e-9
        assert metrics["r2"] <= 1.0 + 1e-9
        if name not in ("mean",):
            assert metrics["mae"] < 0.30

        save_path = tmp_path / f"reg_{name}.joblib"
        joblib.dump(reg, save_path)
        loaded = joblib.load(save_path)
        np.testing.assert_allclose(loaded.predict(X_df), preds, atol=1e-10)


def test_f14_prediction_monotonicity_with_50hz_disturbance_size():
    X_train = np.array([
        [-0.15, -0.20, -0.08, -0.11, -0.10, -0.12, -0.05, -0.14, 0.02, 0.005, 1.2, 0.5, 0.05, 0.08],
        [-0.28, -0.35, -0.14, -0.20, -0.20, -0.22, -0.10, -0.25, 0.04, 0.008, 2.1, 0.9, 0.10, 0.14],
        [-0.42, -0.52, -0.22, -0.32, -0.34, -0.37, -0.17, -0.41, 0.06, 0.012, 3.2, 1.4, 0.15, 0.21],
        [-0.58, -0.70, -0.30, -0.44, -0.48, -0.51, -0.24, -0.57, 0.08, 0.016, 4.5, 2.0, 0.21, 0.29],
        [-0.75, -0.92, -0.40, -0.58, -0.62, -0.66, -0.31, -0.73, 0.10, 0.020, 5.8, 2.6, 0.27, 0.38],
    ])
    y_train = np.array([49.82, 49.64, 49.44, 49.24, 49.02])

    reg = LinearNadirRegressor(alpha=1e-2)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_train)
    for i in range(len(preds) - 1):
        assert preds[i] >= preds[i + 1] - 1e-4


def test_f15_all_security_classifiers_49_5hz_threshold_and_zero_tp_f1(
    tmp_path: Path,
    sample_classification_dataset,
):
    X = sample_classification_dataset["X"]
    y_nadir = sample_classification_dataset["y_nadir"]
    y_sec = sample_classification_dataset["y_secure"]
    assert SECURITY_THRESHOLD_HZ == 49.50
    np.testing.assert_array_equal(y_sec, (y_nadir >= 49.50).astype(int))

    classifiers_list = [
        ("bin_reg", BinarySecurityClassifier(backend="regressor", threshold=49.50)),
        ("logistic", LogisticSecurityClassifier(threshold=49.50)),
        ("dt", DecisionTreeSecurityClassifier(max_depth=4, threshold=49.50)),
        ("rf", RandomForestSecurityClassifier(n_estimators=15, max_depth=4, threshold=49.50)),
        ("gbdt", GradientBoostingSecurityClassifier(n_estimators=20, max_depth=3, threshold=49.50)),
        ("lgbm", LightGBMSecurityClassifier(n_estimators=20, max_depth=3, threshold=49.50)),
    ]

    for name, clf in classifiers_list:
        clf.fit(X, y_nadir)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        assert preds.shape == (len(y_sec),)
        assert probs.shape == (len(y_sec), 2)
        assert set(np.unique(preds)).issubset({0, 1})
        assert np.all((probs >= 0.0) & (probs <= 1.0))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)

        m = clf.evaluate(X, y_sec)
        assert 0.0 <= m["f1"] <= 1.0
        assert 0.0 <= m["fnr_insecure"] <= 1.0
        assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == len(y_sec)

        p = tmp_path / f"clf_{name}.joblib"
        joblib.dump(clf, p)
        reloaded = joblib.load(p)
        np.testing.assert_allclose(reloaded.predict_proba(X), probs, atol=1e-10)

    m_zero = compute_binary_security_metrics(np.array([1, 1, 0, 0]), np.array([0, 0, 0, 0]))
    assert m_zero["tp"] == 0
    assert m_zero["f1"] == 0.0

    probs_pos = classifiers_list[1][1].predict_proba(X)[:, 1]
    sweep = evaluate_probability_threshold_sweep(y_sec, probs_pos, thresholds=[0.2, 0.4, 0.6, 0.8])
    recalls = [row["recall"] for row in sweep]
    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1] - 1e-9


def test_f16_calibrated_gbdt_platt_vs_isotonic_and_ood_intervals(
    tmp_path: Path,
    sample_classification_dataset,
):
    X = sample_classification_dataset["X"]
    y_nadir = sample_classification_dataset["y_nadir"]
    y_sec = sample_classification_dataset["y_secure"]

    tr_idx = np.arange(0, 60, 3)  # 20 train
    cal_idx = np.array([i for i in range(60) if i % 3 != 0])  # 40 calibration (N_cal >= 30)

    platt = CalibratedGBDTClassifier(
        method="platt", threshold=49.50, n_estimators=35, max_depth=3, random_state=42, fit_full_after_cal=True
    )
    iso = CalibratedGBDTClassifier(
        method="isotonic", threshold=49.50, n_estimators=35, max_depth=3, random_state=42, fit_full_after_cal=True
    )
    platt.fit(X[tr_idx], y_nadir[tr_idx], y_sec[tr_idx], X_cal=X[cal_idx], y_cal=y_nadir[cal_idx])
    iso.fit(X[tr_idx], y_nadir[tr_idx], y_sec[tr_idx], X_cal=X[cal_idx], y_cal=y_nadir[cal_idx])

    assert platt.method == "platt" and platt.effective_method_ == "platt"
    assert iso.method == "isotonic" and iso.effective_method_ == "isotonic"
    assert iso.fallback_used_ is False
    assert len(iso.isotonic_x) > 0

    probs_platt = platt.predict_proba(X)[:, 1]
    probs_iso = iso.predict_proba(X)[:, 1]
    assert not np.allclose(probs_platt, probs_iso, atol=1e-4)

    score_grid = np.linspace(48.6, 50.0, 15)
    p_grid_iso = [iso._calibrate_single(float(s)) for s in score_grid]
    for i in range(len(p_grid_iso) - 1):
        assert p_grid_iso[i] <= p_grid_iso[i + 1] + 1e-7

    feat_std = np.std(X, axis=0)
    X_id = X[15:30]
    X_ood = X_id - 4.5 * feat_std
    _, lo_id, hi_id = platt.predict_nadir_intervals(X_id, confidence_level=0.95)
    _, lo_ood, hi_ood = platt.predict_nadir_intervals(X_ood, confidence_level=0.95)
    assert float(np.mean(hi_ood - lo_ood)) > float(np.mean(hi_id - lo_id))

    p_sec, p_lo, p_hi = platt.predict_with_intervals(X_id, confidence_level=0.95)
    assert np.all((p_lo >= 0.0) & (p_lo <= p_sec + 1e-9) & (p_sec - 1e-9 <= p_hi) & (p_hi <= 1.0))

    iso_path = tmp_path / "cal_iso.joblib"
    joblib.dump(iso, iso_path)
    iso_loaded = joblib.load(iso_path)
    np.testing.assert_allclose(iso_loaded.predict_proba(X), iso.predict_proba(X), atol=1e-10)


def test_f17_calibration_metrics_brier_ece_mce_and_10bin_reliability():
    y_true = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert brier_score(y_true, [1.0, 0.0, 1.0, 1.0, 0.0], threshold=49.50) == pytest.approx(0.0)
    assert brier_score(y_true, [0.0, 1.0, 0.0, 0.0, 1.0], threshold=49.50) == pytest.approx(1.0)
    assert brier_score(y_true, [0.5] * 5, threshold=49.50) == pytest.approx(0.25)

    ece = expected_calibration_error([1, 1, 0, 0], [0.8, 0.8, 0.2, 0.2], n_bins=10, threshold=49.50)
    mce = maximum_calibration_error([1, 1, 0, 0], [0.8, 0.8, 0.2, 0.2], n_bins=10, threshold=49.50)
    assert ece == pytest.approx(0.20, abs=1e-6)
    assert 0.0 <= ece <= mce <= 1.0

    diag = reliability_diagram([1, 0, 1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2, 0.7, 0.3], n_bins=10, threshold=49.50)
    assert len(diag["bin_accuracies"]) == 10
    assert len(diag["bin_edges"]) == 11
    assert sum(diag["bin_counts"]) == 6

    comp = compute_calibration_metrics([49.8, 49.1, 49.6, 49.3], [0.85, 0.15, 0.75, 0.25], threshold=49.50)
    assert comp["n_bins"] == 10
    assert 0.0 <= comp["brier_score"] < 0.10
