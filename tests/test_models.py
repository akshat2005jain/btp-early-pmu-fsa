"""Tests for 50 Hz nadir regressors, security classifiers, and probability calibration."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import joblib
import numpy as np
import pandas as pd
import pytest

from pmu_fsa.features.extractor import CANONICAL_FEATURE_NAMES, FeatureExtractor
from pmu_fsa.models import calibrated_gb, calibration_metrics, classifiers, regressors
from pmu_fsa.simulation.scenario_generator import ScenarioGenerator


def generate_synthetic_50hz_regression_dataset(
    n_samples: int = 80,
    random_state: int = 42,
    noise_level: float = 0.01,
) -> Dict[str, Any]:
    rng = np.random.RandomState(random_state)
    loss_mw = rng.uniform(220.0, 950.0, n_samples)
    ren_pen = rng.uniform(0.10, 0.45, n_samples)

    f0 = 50.0
    h_tot = 3.25 * (1.0 - 0.55 * ren_pen)
    p_loss_pu = loss_mw / 100.0

    rocof_init = -f0 * p_loss_pu / (2.0 * h_tot * 100.0) + rng.normal(0, noise_level, n_samples)
    rocof_peak_neg = rocof_init * 1.18
    rocof_end = rocof_init * 0.45
    rocof_mean = rocof_init * 0.72
    freq_drop_end = rocof_mean * 1.0
    freq_drop_max = rocof_peak_neg * 0.85
    freq_dev_integral = 0.5 * freq_drop_end * 1.0
    bus_drop_worst = freq_drop_max * 1.12
    area_freq_spread_max = 0.03 + 0.02 * p_loss_pu + np.abs(rng.normal(0, 0.003, n_samples))
    bus_freq_std_mean = 0.01 + 0.006 * p_loss_pu
    angle_divergence_max = 2.0 + 1.8 * p_loss_pu
    tie_line_angle_excursion = 0.8 + 0.9 * p_loss_pu
    freq_curvature_init = -0.35 * rocof_init
    freq_curvature_mean = -0.55 * rocof_init

    X = np.column_stack([
        rocof_init,
        rocof_peak_neg,
        rocof_end,
        rocof_mean,
        freq_drop_end,
        freq_drop_max,
        freq_dev_integral,
        bus_drop_worst,
        area_freq_spread_max,
        bus_freq_std_mean,
        angle_divergence_max,
        tie_line_angle_excursion,
        freq_curvature_init,
        freq_curvature_mean,
    ])

    y_nadir = 50.0 - 0.095 * (p_loss_pu / (h_tot / 3.25)) + rng.normal(0, noise_level, n_samples)
    y_nadir = np.clip(y_nadir, 47.8, 49.95)

    return {
        "X": X,
        "X_df": pd.DataFrame(X, columns=CANONICAL_FEATURE_NAMES),
        "y_nadir": y_nadir,
        "n_samples": n_samples,
        "n_features": 14,
    }


def generate_synthetic_50hz_classification_dataset(
    n_samples: int = 80,
    threshold: float = 49.50,
    random_state: int = 42,
) -> Dict[str, Any]:
    data = generate_synthetic_50hz_regression_dataset(n_samples=n_samples, random_state=random_state)
    data["y_secure"] = (data["y_nadir"] >= threshold).astype(int)
    data["threshold"] = threshold
    return data


def generate_synthetic_calibration_splits(
    n_train: int = 100,
    n_cal: int = 50,
    n_test: int = 50,
    threshold: float = 49.50,
    random_state: int = 42,
) -> Dict[str, Any]:
    total_n = n_train + n_cal + n_test
    data = generate_synthetic_50hz_classification_dataset(
        n_samples=total_n,
        threshold=threshold,
        random_state=random_state,
    )
    X, y_n, y_s = data["X"], data["y_nadir"], data["y_secure"]
    return {
        "train": {"X": X[:n_train], "y_nadir": y_n[:n_train], "y_secure": y_s[:n_train]},
        "cal": {
            "X": X[n_train : n_train + n_cal],
            "y_nadir": y_n[n_train : n_train + n_cal],
            "y_secure": y_s[n_train : n_train + n_cal],
        },
        "test": {
            "X": X[n_train + n_cal :],
            "y_nadir": y_n[n_train + n_cal :],
            "y_secure": y_s[n_train + n_cal :],
        },
        "threshold": threshold,
    }


class TestAllSevenNadirRegressors:

    @pytest.fixture
    def synth_data(self):
        return generate_synthetic_50hz_regression_dataset(n_samples=70, random_state=42)

    def get_all_7_regressors(self):
        return [
            ("mean", regressors.MeanNadirRegressor()),
            ("analytical", regressors.AnalyticalSwingNadirRegressor(alpha=1e-2)),
            ("linear", regressors.LinearNadirRegressor(alpha=1e-2)),
            ("dt", regressors.DecisionTreeNadirRegressor(max_depth=4, random_state=42)),
            ("rf", regressors.RandomForestNadirRegressor(n_estimators=15, max_depth=5, random_state=42)),
            ("gbdt", regressors.GradientBoostingNadirRegressor(n_estimators=25, max_depth=3, random_state=42)),
            ("lgbm", regressors.LightGBMNadirRegressor(n_estimators=25, num_leaves=10, max_depth=4, random_state=42)),
        ]

    def test_all_7_regressors_fit_predict_evaluate_and_save_load(self, tmp_path: Path, synth_data):
        X_df, y = synth_data["X_df"], synth_data["y_nadir"]
        X_tr, y_tr = X_df.iloc[:50], y[:50]
        X_te, y_te = X_df.iloc[50:], y[50:]

        for key, model in self.get_all_7_regressors():
            reg_from_registry = regressors.get_nadir_regressor(key)
            assert type(reg_from_registry) is type(model)

            model.fit(X_tr, y_tr)
            preds_single = model.predict(X_te.iloc[:1])
            assert preds_single.shape == (1,)

            preds = model.predict(X_te)
            assert preds.shape == (len(y_te),)
            assert np.all(np.isfinite(preds))
            assert np.all((preds >= 45.0) & (preds <= 50.5))

            metrics = model.evaluate(X_te, y_te)
            for req_key in ("mae", "rmse", "r2", "bias", "median_ae", "p95_ae"):
                assert req_key in metrics
                assert math.isfinite(metrics[req_key])
            assert metrics["rmse"] >= metrics["mae"] - 1e-9
            assert metrics["p95_ae"] >= metrics["median_ae"] - 1e-9

            if key in ("analytical", "linear", "rf", "gbdt", "lgbm"):
                assert metrics["r2"] > 0.70, f"{key} expected R2 > 0.70, got {metrics['r2']}"

            save_path = tmp_path / f"{key}.joblib"
            joblib.dump(model, save_path)
            loaded = joblib.load(save_path)
            np.testing.assert_allclose(loaded.predict(X_te), preds, atol=1e-12)

    def test_unclamped_negative_r2_and_physical_50hz_clipping(self, synth_data):
        X, y = synth_data["X"], synth_data["y_nadir"]
        mean_reg = regressors.MeanNadirRegressor()
        mean_reg.fit(X[:10], np.full(10, 48.0))
        eval_bad = mean_reg.evaluate(X[10:30], np.linspace(49.5, 49.9, 20))
        assert eval_bad["r2"] < 0.0, f"Expected authentic negative R2 without clamping, got {eval_bad['r2']}"

        X_extreme = X * 25.0
        for name, model in self.get_all_7_regressors():
            model.fit(X, y)
            preds = model.predict(X_extreme)
            assert np.all((preds >= 45.0) & (preds <= 50.5)), f"{name} violated [45.0, 50.5] Hz bounds"

    def test_empty_input_zero_variance_and_dimension_mismatch(self, synth_data):
        X, y = synth_data["X"], synth_data["y_nadir"]
        model = regressors.LinearNadirRegressor()
        model.fit(X, y)
        assert len(model.predict([])) == 0
        with pytest.raises(ValueError, match="Feature dimension mismatch"):
            model.predict(X[:, :3])

        X_const = np.column_stack([np.ones(10), np.linspace(0.1, 1.0, 10)])
        y_const = 49.8 - 0.4 * X_const[:, 1]
        model.fit(X_const, y_const)
        assert np.all(np.isfinite(model.predict(X_const)))


class TestAllSevenSecurityClassifiers:

    @pytest.fixture
    def synth_cls_data(self):
        return generate_synthetic_50hz_classification_dataset(n_samples=80, threshold=49.50, random_state=42)

    def get_all_classifiers(self):
        return [
            ("binary", classifiers.BinarySecurityClassifier(threshold=49.50)),
            ("logistic", classifiers.LogisticSecurityClassifier(threshold=49.50, random_state=42)),
            ("dt", classifiers.DecisionTreeSecurityClassifier(threshold=49.50, max_depth=4, random_state=42)),
            ("rf", classifiers.RandomForestSecurityClassifier(threshold=49.50, n_estimators=15, max_depth=5, random_state=42)),
            ("gbdt", classifiers.GradientBoostingSecurityClassifier(threshold=49.50, n_estimators=20, max_depth=3, random_state=42)),
            ("lgbm", classifiers.LightGBMSecurityClassifier(threshold=49.50, n_estimators=20, num_leaves=10, random_state=42)),
            ("calibrated_gb_platt", calibrated_gb.CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=25, random_state=42)),
            ("calibrated_gb_iso", calibrated_gb.CalibratedGBDTClassifier(method="isotonic", threshold=49.50, n_estimators=25, random_state=42)),
        ]

    def test_all_classifiers_fit_predict_proba_evaluate_and_save_load(self, tmp_path: Path, synth_cls_data):
        X, y_nadir, y_sec = synth_cls_data["X"], synth_cls_data["y_nadir"], synth_cls_data["y_secure"]
        X_tr, yn_tr, ys_tr = X[:55], y_nadir[:55], y_sec[:55]
        X_te, yn_te, ys_te = X[55:], y_nadir[55:], y_sec[55:]

        for name, clf in self.get_all_classifiers():
            if isinstance(clf, calibrated_gb.CalibratedGBDTClassifier):
                clf.fit(X_tr, yn_tr, ys_tr)
            else:
                clf.fit(X_tr, yn_tr)

            preds = clf.predict(X_te)
            probs = clf.predict_proba(X_te)

            assert preds.shape == (len(ys_te),)
            assert probs.shape == (len(ys_te), 2)
            assert set(np.unique(preds)).issubset({0, 1})
            assert np.all((probs >= 0.0) & (probs <= 1.0))
            np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-6)

            eval_m = clf.evaluate(X_te, ys_te)
            for k in ("tp", "tn", "fp", "fn", "precision", "recall", "f1", "fnr_insecure", "fpr_secure", "pr_auc", "roc_auc"):
                assert k in eval_m
            assert eval_m["tp"] + eval_m["tn"] + eval_m["fp"] + eval_m["fn"] == len(ys_te)

            save_path = tmp_path / f"{name}.joblib"
            joblib.dump(clf, save_path)
            loaded = joblib.load(save_path)
            np.testing.assert_allclose(loaded.predict_proba(X_te), probs, atol=1e-12)

    def test_tp_zero_yields_f1_zero_never_one(self):
        # Case 1: tp=0 with fp>0 and fn>0
        m1 = classifiers.compute_binary_security_metrics(
            y_true=np.array([1, 1, 0, 0]),
            y_pred=np.array([0, 0, 1, 0]),
        )
        assert m1["tp"] == 0
        assert m1["precision"] == 0.0
        assert m1["recall"] == 0.0
        assert m1["f1"] == 0.0
        assert m1["missed_insecure_count"] == 1
        assert m1["fnr_insecure"] == pytest.approx(0.5)
        assert m1["fpr_secure"] == pytest.approx(1.0)

        # Case 2: tp=0, fp=0, fn=0 (all insecure, all predicted insecure)
        m2 = classifiers.compute_binary_security_metrics(
            y_true=np.array([0, 0, 0]),
            y_pred=np.array([0, 0, 0]),
        )
        assert m2["tp"] == 0
        assert m2["precision"] == 0.0
        assert m2["recall"] == 0.0
        assert m2["f1"] == 0.0
        assert m2["fnr_insecure"] == 0.0

    def test_threshold_boundary_49_50hz_and_threshold_sweep(self, synth_cls_data):
        clf = classifiers.BinarySecurityClassifier(threshold=49.50)
        y_bin = clf._prepare_y(np.array([49.500000, 49.499999]))
        assert int(y_bin[0]) == 1
        assert int(y_bin[1]) == 0

        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([49.500000, 49.499999])
        clf.regressor_ = mock_reg
        clf.model_ = mock_reg
        preds = clf.predict(np.array([[1.0], [2.0]]))
        probs = clf.predict_proba(np.array([[1.0], [2.0]]))
        assert int(preds[0]) == 1 and int(preds[1]) == 0
        assert probs[0, 1] == pytest.approx(0.5, abs=1e-6)
        assert probs[1, 1] < 0.5

        X, yn, ys = synth_cls_data["X"], synth_cls_data["y_nadir"], synth_cls_data["y_secure"]
        cal = calibrated_gb.CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=25, random_state=42)
        cal.fit(X[:55], yn[:55], ys[:55])
        sweep = cal.evaluate_threshold_sweep(X[55:], ys[55:], thresholds=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
        assert len(sweep) == 6
        recalls = [r["recall"] for r in sweep]
        fnrs_insec = [r["fnr_insecure"] for r in sweep]
        for i in range(len(sweep) - 1):
            assert recalls[i] >= recalls[i + 1] - 1e-9
            assert fnrs_insec[i] >= fnrs_insec[i + 1] - 1e-9


class TestCalibratedGBDTAndUncertainty:

    @pytest.fixture
    def disjoint_data(self):
        return generate_synthetic_calibration_splits(
            n_train=90,
            n_cal=50,
            n_test=40,
            threshold=49.50,
            random_state=42,
        )

    def test_distinct_platt_vs_isotonic_when_n_cal_ge_30(self, disjoint_data):
        d_tr, d_cal, d_te = disjoint_data["train"], disjoint_data["cal"], disjoint_data["test"]

        model_platt = calibrated_gb.CalibratedGBDTClassifier(
            method="platt", threshold=49.50, n_estimators=30, random_state=42
        )
        model_platt.fit(d_tr["X"], d_tr["y_nadir"], d_tr["y_secure"], X_cal=d_cal["X"], y_cal=d_cal["y_nadir"])

        model_iso = calibrated_gb.CalibratedGBDTClassifier(
            method="isotonic", threshold=49.50, n_estimators=30, random_state=42
        )
        model_iso.fit(d_tr["X"], d_tr["y_nadir"], d_tr["y_secure"], X_cal=d_cal["X"], y_cal=d_cal["y_nadir"])

        assert model_platt.method == "platt"
        assert model_platt.effective_method_ == "platt"
        assert model_platt.fallback_used_ is False
        assert model_platt.calibrator_ is not None

        assert model_iso.method == "isotonic"
        assert model_iso.effective_method_ == "isotonic"
        assert model_iso.fallback_used_ is False
        assert model_iso.n_cal_ == 50
        assert len(model_iso.isotonic_x) > 0

        probs_platt = model_platt.predict_proba(d_te["X"])[:, 1]
        probs_iso = model_iso.predict_proba(d_te["X"])[:, 1]
        assert float(np.max(np.abs(probs_platt - probs_iso))) > 1e-4

        bs_platt = calibration_metrics.brier_score(d_te["y_secure"], probs_platt, threshold=49.50)
        bs_iso = calibration_metrics.brier_score(d_te["y_secure"], probs_iso, threshold=49.50)
        assert bs_platt < 0.20
        assert bs_iso < 0.20

    def test_small_sample_isotonic_fallback_preserves_requested_method(self, disjoint_data):
        d_tr = disjoint_data["train"]
        X_small = d_tr["X"][:15]
        y_small = d_tr["y_nadir"][:15]

        model_iso = calibrated_gb.CalibratedGBDTClassifier(method="isotonic", threshold=49.50, n_estimators=15)
        with pytest.warns(UserWarning, match="N_cal=15 < 30"):
            model_iso.fit(X_small, y_small)

        assert model_iso.method == "isotonic"
        assert model_iso.effective_method_ == "platt"
        assert model_iso.fallback_used_ is True
        probs = model_iso.predict_proba(X_small)
        assert probs.shape == (15, 2)

    def test_quantile_conformal_intervals_widen_on_ood(self, disjoint_data):
        d_tr, d_cal, d_te = disjoint_data["train"], disjoint_data["cal"], disjoint_data["test"]
        model = calibrated_gb.CalibratedGBDTClassifier(
            method="platt", threshold=49.50, n_estimators=40, random_state=42, fit_full_after_cal=True
        )
        model.fit(d_tr["X"], d_tr["y_nadir"], d_tr["y_secure"], X_cal=d_cal["X"], y_cal=d_cal["y_nadir"])

        X_id = d_te["X"]
        feat_std = np.std(d_tr["X"], axis=0)
        X_ood = X_id - 4.5 * feat_std

        y_id_hat, y_id_low, y_id_high = model.predict_nadir_intervals(X_id, confidence_level=0.95)
        y_ood_hat, y_ood_low, y_ood_high = model.predict_nadir_intervals(X_ood, confidence_level=0.95)
        width_hz_id = float(np.mean(y_id_high - y_id_low))
        width_hz_ood = float(np.mean(y_ood_high - y_ood_low))

        assert np.all(y_id_low <= y_id_hat + 1e-9) and np.all(y_id_hat <= y_id_high + 1e-9)
        assert np.all(y_ood_low <= y_ood_hat + 1e-9) and np.all(y_ood_hat <= y_ood_high + 1e-9)
        assert width_hz_ood > width_hz_id, f"Expected wider OOD nadir interval: OOD={width_hz_ood:.4f} <= ID={width_hz_id:.4f}"

        p_id, p_low_id, p_high_id = model.predict_with_intervals(X_id, confidence_level=0.95)
        _, p_low_99, p_high_99 = model.predict_with_intervals(X_id, confidence_level=0.99)
        for p, lo95, hi95, lo99, hi99 in zip(p_id, p_low_id, p_high_id, p_low_99, p_high_99):
            assert 0.0 <= lo95 <= p + 1e-9
            assert p - 1e-9 <= hi95 <= 1.0
            assert (hi99 - lo99) >= (hi95 - lo95) - 1e-9


class TestCalibrationMetricsAndGenuineIntegration:

    def test_brier_ece_mce_and_reliability_diagram_10bins(self):
        y_true = [1.0, 0.0, 1.0, 1.0, 0.0]
        assert calibration_metrics.brier_score(y_true, [1.0, 0.0, 1.0, 1.0, 0.0]) == pytest.approx(0.0)
        assert calibration_metrics.brier_score(y_true, [0.0, 1.0, 0.0, 0.0, 1.0]) == pytest.approx(1.0)
        assert calibration_metrics.brier_score(y_true, [0.5] * 5) == pytest.approx(0.25)

        ece = calibration_metrics.expected_calibration_error([1, 1, 0, 0], [0.8, 0.8, 0.2, 0.2], n_bins=10)
        mce = calibration_metrics.maximum_calibration_error([1, 1, 0, 0], [0.8, 0.8, 0.2, 0.2], n_bins=10)
        assert ece == pytest.approx(0.20, abs=1e-6)
        assert 0.0 <= ece <= mce <= 1.0

        comp = calibration_metrics.compute_calibration_metrics([1, 0, 1, 0], [0.9, 0.1, 0.7, 0.3])
        assert comp["n_bins"] == 10
        assert len(comp["bin_counts"]) == 10
        assert len(comp["bin_edges"]) == 11
        assert sum(comp["bin_counts"]) == 4

    def test_genuine_50hz_scenarios_to_models(self):
        gen = ScenarioGenerator(fps=50, t_span=(0.0, 6.0))
        s1 = gen.generate_single_scenario(tripped_bus=30, loss_fraction=0.40, renewable_penetration=0.10, load_scale=1.0)
        s2 = gen.generate_single_scenario(tripped_bus=38, loss_fraction=1.00, renewable_penetration=0.30, load_scale=1.0)

        extractor = FeatureExtractor(t_obs=1.0, f_nominal=50.0)
        X_df, y_nadir, y_sec, _ = extractor.transform_batch([s1, s2], as_dataframe=True)
        assert X_df.shape == (2, 14)
        assert list(X_df.columns) == CANONICAL_FEATURE_NAMES
        assert int(y_sec[0]) == 1 and int(y_sec[1]) == 0

        for reg_cls in (
            regressors.MeanNadirRegressor,
            regressors.AnalyticalSwingNadirRegressor,
            regressors.LinearNadirRegressor,
        ):
            r = reg_cls().fit(X_df, y_nadir)
            preds = r.predict(X_df)
            assert np.all((preds >= 45.0) & (preds <= 50.5))

        cal = calibrated_gb.CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=15)
        cal.fit(X_df, y_nadir, y_sec)
        probs = cal.predict_proba(X_df)
        assert probs.shape == (2, 2)


def test_speed_budget():
    t0 = time.perf_counter()
    data = generate_synthetic_50hz_regression_dataset(n_samples=50, random_state=42)
    reg = regressors.LinearNadirRegressor().fit(data["X"], data["y_nadir"])
    _ = reg.predict(data["X"])
    assert (time.perf_counter() - t0) * 1000.0 < 50.0
