"""Tests for nadir regressors, security classifiers, and probability calibration."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from pmu_fsa.models import regressors
from pmu_fsa.models import classifiers
from pmu_fsa.models import calibrated_gb
from pmu_fsa.models import calibration_metrics
from pmu_fsa.simulation.scenario_generator import ScenarioGenerator
from pmu_fsa.features.extractor import FeatureExtractor


def generate_synthetic_regression_dataset(
    n_samples: int = 60,
    random_state: int = 42,
    noise_level: float = 0.01,
) -> Dict[str, np.ndarray]:
    rng = np.random.RandomState(random_state)
    loss_mw = rng.uniform(200.0, 1200.0, n_samples)
    ren_pen = rng.uniform(0.10, 0.45, n_samples)

    f0 = 60.0
    h_tot = 5.0 * (1.0 - 0.5 * ren_pen)
    p_loss_pu = loss_mw / 1000.0

    init_rocof = -f0 * p_loss_pu / (2.0 * h_tot) + rng.normal(0, noise_level, n_samples)
    pk_rocof = init_rocof * 1.15
    max_abs_rocof = np.abs(pk_rocof)
    mean_rocof = init_rocof * 0.70
    inst_drop = mean_rocof * 1.0
    max_drop = pk_rocof * 0.90
    integ_dev = 0.5 * inst_drop * 1.0
    integ_abs = np.abs(integ_dev)
    spread = 0.05 + 0.15 * p_loss_pu + rng.normal(0, 0.005, n_samples)
    spatial_var = 1e-4 * (p_loss_pu ** 2)
    max_ang = 5.0 + 20.0 * p_loss_pu
    tie_ang = 2.0 + 10.0 * p_loss_pu
    init_acc = init_rocof / 0.1
    mean_acc = init_rocof / 1.0

    X = np.column_stack([
        init_rocof, pk_rocof, max_abs_rocof, mean_rocof,
        inst_drop, max_drop, integ_dev, integ_abs,
        spread, spatial_var, max_ang, tie_ang, init_acc, mean_acc
    ])

    y_nadir = 60.0 - 1.25 * (p_loss_pu / (h_tot / 5.0)) + rng.normal(0, noise_level, n_samples)
    y_nadir = np.clip(y_nadir, 55.5, 59.95)

    return {
        "X": X,
        "y_nadir": y_nadir,
        "n_samples": n_samples,
        "n_features": 14,
    }


def generate_synthetic_classification_dataset(
    n_samples: int = 60,
    threshold: float = 59.5,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    data = generate_synthetic_regression_dataset(n_samples=n_samples, random_state=random_state)
    y_secure = (data["y_nadir"] >= threshold).astype(int)
    data["y_secure"] = y_secure
    data["threshold"] = threshold
    return data


def generate_synthetic_calibration_splits(
    n_train: int = 100,
    n_cal: int = 50,
    n_test: int = 50,
    threshold: float = 59.5,
    random_state: int = 42,
) -> Dict[str, Any]:
    total_n = n_train + n_cal + n_test
    data = generate_synthetic_classification_dataset(n_samples=total_n, threshold=threshold, random_state=random_state)
    X = data["X"]
    y_n = data["y_nadir"]
    y_s = data["y_secure"]

    return {
        "train": {"X": X[:n_train], "y_nadir": y_n[:n_train], "y_secure": y_s[:n_train]},
        "cal": {"X": X[n_train:n_train + n_cal], "y_nadir": y_n[n_train:n_train + n_cal], "y_secure": y_s[n_train:n_train + n_cal]},
        "test": {"X": X[n_train + n_cal:], "y_nadir": y_n[n_train + n_cal:], "y_secure": y_s[n_train + n_cal:]},
        "threshold": threshold,
    }


def _build_genuine_pmu_feature_batch():
    gen = ScenarioGenerator(fps=50)
    scen1 = gen.generate_single_scenario(tripped_bus=37, p_loss_mw=350.0, renewable_penetration=0.10, load_scale=1.0)
    scen2 = gen.generate_single_scenario(tripped_bus=35, p_loss_mw=650.0, renewable_penetration=0.10, load_scale=1.0)

    extractor = FeatureExtractor(t_obs=1.0)
    X, y_nadir, y_secure, meta = extractor.transform_batch([scen1, scen2], as_dataframe=True)
    return {
        "X": X.values if hasattr(X, "values") else np.array(X),
        "y_nadir": np.array(y_nadir, dtype=float),
        "y_secure": np.array(y_secure, dtype=int),
        "columns": list(X.columns) if hasattr(X, "columns") else [],
    }


@pytest.fixture(scope="module")
def genuine_pmu_feature_batch():
    return _build_genuine_pmu_feature_batch()


def calc_reg_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum((yt - yp) ** 2) / ss_tot) if ss_tot > 0 else 1.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def calc_cls_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    accuracy = float((tp + tn) / len(yt))
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "fnr": fnr, "accuracy": accuracy,
    }


class TestNadirRegressors:

    @pytest.fixture
    def synth_data(self):
        return generate_synthetic_regression_dataset(n_samples=50, random_state=42)

    def get_regressors(self):
        return [
            ("LinearNadirRegressor", regressors.LinearNadirRegressor()),
            ("DecisionTreeNadirRegressor", regressors.DecisionTreeNadirRegressor(max_depth=4)),
            ("RandomForestNadirRegressor", regressors.RandomForestNadirRegressor(n_estimators=10, max_depth=4)),
            ("GradientBoostingNadirRegressor", regressors.GradientBoostingNadirRegressor(n_estimators=10, max_depth=3)),
        ]

    def test_fit_and_predict_shapes(self, synth_data):
        X, y = synth_data["X"], synth_data["y_nadir"]
        for name, model in self.get_regressors():
            model.fit(X, y)

            pred_single = model.predict(X[:1])
            assert len(pred_single) == 1, f"{name}: Single sample prediction shape mismatch"
            assert isinstance(float(pred_single[0]), float)

            preds_batch = model.predict(X)
            assert len(preds_batch) == len(y), f"{name}: Batch prediction shape mismatch"
            for p in preds_batch:
                assert not math.isnan(p) and not math.isinf(p)

    def test_evaluation_metrics_and_accuracy(self, synth_data):
        X, y = synth_data["X"], synth_data["y_nadir"]
        for name, model in self.get_regressors():
            model.fit(X, y)
            preds = model.predict(X)
            m = calc_reg_metrics(y, preds)

            assert m["mae"] >= 0.0, f"{name}: MAE must be non-negative"
            assert m["rmse"] >= m["mae"] - 1e-9, f"{name}: RMSE must be >= MAE"
            assert m["r2"] <= 1.0 + 1e-9, f"{name}: R2 cannot exceed 1.0"
            assert m["mae"] < 0.60, f"{name}: Training MAE too high: {m['mae']:.4f} Hz"

            if name in ("DecisionTreeNadirRegressor", "RandomForestNadirRegressor", "GradientBoostingNadirRegressor"):
                assert m["r2"] > 0.80, f"{name}: Expected R2 > 0.80 on clean data, got {m['r2']:.4f}"

    def test_physical_frequency_bounds_clipping(self, synth_data):
        X, y = synth_data["X"], synth_data["y_nadir"]
        X_extreme = X * 10.0

        for name, model in self.get_regressors():
            model.fit(X, y)
            preds = model.predict(X_extreme)
            for p in preds:
                assert 55.0 <= p <= 60.5, f"{name}: Predicted nadir {p} outside physical bounds [55.0, 60.5] Hz"

    def test_prediction_monotonicity(self):
        X_train = np.array([
            [0.2, 0.1, -0.15, -0.1],
            [0.4, 0.1, -0.25, -0.2],
            [0.6, 0.1, -0.38, -0.35],
            [0.8, 0.1, -0.50, -0.5],
            [1.0, 0.1, -0.65, -0.65],
        ])
        y_train = np.array([59.8, 59.6, 59.4, 59.2, 59.0])

        model = regressors.LinearNadirRegressor()
        model.fit(X_train, y_train)

        X_test = np.array([
            [0.25, 0.1, -0.18, -0.12],
            [0.50, 0.1, -0.32, -0.28],
            [0.75, 0.1, -0.48, -0.45],
            [0.95, 0.1, -0.62, -0.60],
        ])
        preds = model.predict(X_test)
        for i in range(len(preds) - 1):
            assert preds[i] >= preds[i + 1] - 1e-4, f"Monotonicity violation: {preds[i]} < {preds[i+1]}"

    def test_zero_variance_feature_resilience(self):
        X = np.array([
            [1.0, 0.1, -0.2],
            [1.0, 0.2, -0.3],
            [1.0, 0.3, -0.4],
            [1.0, 0.4, -0.5],
        ])
        y = np.array([59.6, 59.4, 59.2, 59.0])
        model = regressors.LinearNadirRegressor(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        for p in preds:
            assert not math.isnan(p)
            assert 55.0 <= p <= 60.5

    def test_empty_input_and_dimension_mismatch(self, synth_data):
        X, y = synth_data["X"], synth_data["y_nadir"]
        model = regressors.LinearNadirRegressor()
        model.fit(X, y)

        empty_preds = model.predict([])
        assert len(empty_preds) == 0

        with pytest.raises(ValueError):
            model.predict(X[:, :3])


class TestSecurityClassifiers:

    @pytest.fixture
    def synth_cls_data(self):
        return generate_synthetic_classification_dataset(n_samples=60, threshold=59.5, random_state=42)

    def get_classifiers(self):
        return [
            ("BinarySecurityClassifier", classifiers.BinarySecurityClassifier(threshold=59.5)),
            ("LogisticSecurityClassifier", classifiers.LogisticSecurityClassifier(threshold=59.5)),
            ("DecisionTreeSecurityClassifier", classifiers.DecisionTreeSecurityClassifier(threshold=59.5, max_depth=4)),
            ("RandomForestSecurityClassifier", classifiers.RandomForestSecurityClassifier(threshold=59.5, n_estimators=10, max_depth=4)),
            ("GradientBoostingSecurityClassifier", classifiers.GradientBoostingSecurityClassifier(threshold=59.5, n_estimators=10, max_depth=3)),
        ]

    def test_fit_and_prediction_simplex(self, synth_cls_data):
        X, y_nadir, y_sec = synth_cls_data["X"], synth_cls_data["y_nadir"], synth_cls_data["y_secure"]

        for name, clf in self.get_classifiers():
            clf.fit(X, y_nadir)
            preds = clf.predict(X)
            probs = clf.predict_proba(X)

            assert len(preds) == len(y_sec)
            assert len(probs) == len(y_sec)

            for p_bin, row in zip(preds, probs):
                assert p_bin in (0, 1, 0.0, 1.0), f"{name}: Prediction {p_bin} not binary"
                assert len(row) == 2, f"{name}: Expected 2 probabilities"
                p_insec, p_sec = float(row[0]), float(row[1])
                assert 0.0 <= p_insec <= 1.0
                assert 0.0 <= p_sec <= 1.0
                assert abs((p_insec + p_sec) - 1.0) < 1e-6, f"{name}: Simplex sum != 1.0: {p_insec + p_sec}"

    def test_classification_metrics(self, synth_cls_data):
        X, y_nadir, y_sec = synth_cls_data["X"], synth_cls_data["y_nadir"], synth_cls_data["y_secure"]
        for name, clf in self.get_classifiers():
            clf.fit(X, y_nadir)
            preds = clf.predict(X)
            m = calc_cls_metrics(y_sec, preds)

            for key in ("precision", "recall", "f1", "fnr", "accuracy"):
                assert 0.0 <= m[key] <= 1.0, f"{name}: Metric {key} = {m[key]} out of [0, 1]"
            assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == len(y_sec)

    def test_threshold_boundary_case(self):
        clf = classifiers.BinarySecurityClassifier(threshold=59.5)

        y_sec = clf._prepare_y(np.array([59.500000]))
        assert int(y_sec[0]) == 1, f"Boundary f_nadir == 59.5 must map to Secure (1), got {y_sec[0]}"

        y_marginal = clf._prepare_y(np.array([59.499999]))
        assert int(y_marginal[0]) == 0, f"Marginal f_nadir < 59.5 must map to Insecure (0), got {y_marginal[0]}"

        mock_reg = MagicMock()
        mock_reg.predict.return_value = np.array([59.500000, 59.499999])
        clf.regressor_ = mock_reg
        clf.model_ = mock_reg

        preds = clf.predict(np.array([[1.0], [2.0]]))
        assert int(preds[0]) == 1, f"Boundary f_pred == 59.5 must predict Secure (1), got {preds[0]}"
        assert int(preds[1]) == 0, f"Marginal f_pred < 59.5 must predict Insecure (0), got {preds[1]}"

        probs = clf.predict_proba(np.array([[1.0], [2.0]]))
        assert np.isclose(probs[0, 1], 0.5000, atol=1e-5), f"P(secure) at 59.5 Hz must be 0.5, got {probs[0, 1]}"
        assert probs[1, 1] < 0.5000, f"P(secure) at 59.499999 Hz must be < 0.5, got {probs[1, 1]}"

    def test_extreme_class_imbalance(self):
        rng = np.random.RandomState(42)
        n = 100
        X = rng.normal(0, 1, (n, 4))
        y_nadir = np.array([59.8] * 95 + [58.5] * 5)
        clf = classifiers.BinarySecurityClassifier(threshold=59.5)
        clf.fit(X, y_nadir)

        preds = clf.predict(X)
        probs = clf.predict_proba(X)
        assert len(preds) == n
        assert len(probs) == n

        m = calc_cls_metrics([1] * 95 + [0] * 5, preds)
        assert 0.0 <= m["accuracy"] <= 1.0

    def test_single_class_edge_cases(self):
        X = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]])
        clf_pos = classifiers.BinarySecurityClassifier()
        clf_pos.fit(X, [59.8, 59.7, 59.9])
        probs_pos = clf_pos.predict_proba(X)
        for row in probs_pos:
            assert row[1] >= 0.5

        clf_neg = classifiers.BinarySecurityClassifier()
        clf_neg.fit(X, [58.5, 58.7, 58.2])
        probs_neg = clf_neg.predict_proba(X)
        for row in probs_neg:
            assert row[1] <= 0.5


class TestCalibratedGBDT:

    @pytest.fixture
    def disjoint_data(self):
        return generate_synthetic_calibration_splits(n_train=80, n_cal=40, n_test=40, threshold=59.5, random_state=42)

    def test_disjoint_split_protocol(self, disjoint_data):
        d_tr = disjoint_data["train"]
        d_cal = disjoint_data["cal"]
        d_te = disjoint_data["test"]

        model = calibrated_gb.CalibratedGBDTClassifier(method="platt")
        model.fit(d_tr["X"], d_tr["y_nadir"], X_cal=d_cal["X"], y_cal=d_cal["y_nadir"])

        probs = model.predict_proba(d_te["X"])
        assert len(probs) == len(d_te["y_nadir"])
        for row in probs:
            assert 0.0 <= row[1] <= 1.0
            assert abs(row[0] + row[1] - 1.0) < 1e-6

    def test_platt_and_isotonic_methods(self, disjoint_data):
        d_tr = disjoint_data["train"]
        d_te = disjoint_data["test"]

        model_platt = calibrated_gb.CalibratedGBDTClassifier(method="platt")
        model_platt.fit(d_tr["X"], d_tr["y_nadir"], d_tr["y_secure"])
        probs_platt = model_platt.predict_proba(d_te["X"])

        model_iso = calibrated_gb.CalibratedGBDTClassifier(method="isotonic")
        model_iso.fit(d_tr["X"], d_tr["y_nadir"], d_tr["y_secure"])
        probs_iso = model_iso.predict_proba(d_te["X"])

        assert len(probs_platt) == len(d_te["X"])
        assert len(probs_iso) == len(d_te["X"])

    def test_calibration_brier_score_compliance(self, disjoint_data):
        d_tr = disjoint_data["train"]
        d_te = disjoint_data["test"]

        model = calibrated_gb.CalibratedGBDTClassifier(method="platt")
        model.fit(d_tr["X"], d_tr["y_nadir"], d_tr["y_secure"])
        p_sec = [row[1] for row in model.predict_proba(d_te["X"])]

        bs = calibration_metrics.brier_score(d_te["y_secure"], p_sec)
        assert bs < 0.25, f"Calibrated Brier score {bs} exceeds benchmark threshold 0.25"

    def test_small_sample_fallback_resilience(self, disjoint_data):
        d_tr = disjoint_data["train"]
        X_small = d_tr["X"][:15]
        y_small = d_tr["y_nadir"][:15]

        model = calibrated_gb.CalibratedGBDTClassifier(method="isotonic")
        model.fit(X_small, y_small)
        probs = model.predict_proba(X_small)
        assert len(probs) == 15
        for row in probs:
            assert 0.0 <= row[1] <= 1.0

    def test_uncertainty_bounds_and_intervals(self, disjoint_data):
        d_tr = disjoint_data["train"]
        model = calibrated_gb.CalibratedGBDTClassifier(method="platt")
        model.fit(d_tr["X"], d_tr["y_nadir"])

        p_sec, p_low, p_high = model.predict_with_intervals(d_tr["X"], confidence_level=0.95)
        for p, low, high in zip(p_sec, p_low, p_high):
            assert 0.0 <= low <= high <= 1.0, f"Invalid interval: [{low}, {high}]"
            assert low <= p + 1e-6, f"Lower bound {low} > estimate {p}"
            assert p <= high + 1e-6, f"Estimate {p} > upper bound {high}"

    def test_extreme_score_saturation(self):
        model = calibrated_gb.CalibratedGBDTClassifier(method="platt")
        p_high = model._calibrate_single(120.0)
        p_low = model._calibrate_single(0.0)

        assert p_high == 1.0, f"Expected 1.0 for extreme positive, got {p_high}"
        assert p_low == 0.0, f"Expected 0.0 for extreme negative, got {p_low}"
        assert not math.isinf(p_high) and not math.isnan(p_low)

    def test_platt_scaling_genuine_optimization(self, disjoint_data):
        d_tr = disjoint_data["train"]
        d_cal = disjoint_data["cal"]

        model = calibrated_gb.CalibratedGBDTClassifier(method="platt", threshold=59.5)
        assert model.platt_a == 8.0
        assert model.calibrator_ is None

        model.fit(d_tr["X"], d_tr["y_nadir"], X_cal=d_cal["X"], y_cal=d_cal["y_nadir"])

        assert model.calibrator_ is not None, "calibrator_ must be instantiated after fit"
        assert model.platt_a != 8.0, f"platt_a must be optimized, got physics default {model.platt_a}"
        assert model.platt_a > 0.0, f"platt_a must be positive, got {model.platt_a}"
        assert math.isfinite(model.platt_a) and math.isfinite(model.platt_b)
        assert model.base_regressor_ is model.base_model


class TestCalibrationMetrics:

    def test_brier_score_analytical_bounds(self):
        y_true = [1.0, 0.0, 1.0, 1.0, 0.0]
        assert abs(calibration_metrics.brier_score(y_true, [1.0, 0.0, 1.0, 1.0, 0.0])) < 1e-9
        assert abs(calibration_metrics.brier_score(y_true, [0.0, 1.0, 0.0, 0.0, 1.0]) - 1.0) < 1e-9
        assert abs(calibration_metrics.brier_score(y_true, [0.5, 0.5, 0.5, 0.5, 0.5]) - 0.25) < 1e-9

    def test_ece_analytical_calculation(self):
        y_true = [1.0, 1.0, 0.0, 0.0]
        y_prob = [0.8, 0.8, 0.2, 0.2]
        ece = calibration_metrics.expected_calibration_error(y_true, y_prob, n_bins=5)
        assert abs(ece - 0.20) < 1e-6

    def test_ece_perfect_calibration(self):
        y_prob = [0.3] * 10 + [0.7] * 10
        y_true = [1.0] * 3 + [0.0] * 7 + [1.0] * 7 + [0.0] * 3
        ece = calibration_metrics.expected_calibration_error(y_true, y_prob, n_bins=5)
        assert abs(ece) < 1e-6

    def test_mce_upper_bound_invariant(self):
        y_true = [1.0, 0.0, 1.0, 0.0, 1.0]
        y_prob = [0.9, 0.1, 0.6, 0.4, 0.8]
        ece = calibration_metrics.expected_calibration_error(y_true, y_prob, n_bins=5)
        mce = calibration_metrics.maximum_calibration_error(y_true, y_prob, n_bins=5)
        assert 0.0 <= ece <= mce <= 1.0 + 1e-9

    def test_reliability_diagram_structure(self):
        y_true = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        y_prob = [0.95, 0.85, 0.75, 0.65, 0.35, 0.25, 0.15, 0.05]
        n_bins = 5
        diag = calibration_metrics.reliability_diagram(y_true, y_prob, n_bins=n_bins)

        assert len(diag["bin_accuracies"]) == n_bins
        assert len(diag["bin_confidences"]) == n_bins
        assert len(diag["bin_counts"]) == n_bins
        assert len(diag["bin_edges"]) == n_bins + 1
        assert sum(diag["bin_counts"]) == len(y_true)

    def test_empty_bins_handling(self):
        y_true = [1.0, 0.0]
        y_prob = [0.55, 0.55]
        ece = calibration_metrics.expected_calibration_error(y_true, y_prob, n_bins=10)
        assert 0.0 <= ece <= 1.0


class TestM1M2Integration:

    def test_genuine_features_to_models(self, genuine_pmu_feature_batch):
        X = genuine_pmu_feature_batch["X"]
        y_nadir = genuine_pmu_feature_batch["y_nadir"]
        y_sec = genuine_pmu_feature_batch["y_secure"]

        assert X.shape[0] == 2
        assert X.shape[1] >= 14
        assert len(y_nadir) == 2
        assert len(y_sec) == 2

        reg = regressors.LinearNadirRegressor()
        reg.fit(X, y_nadir)
        preds = reg.predict(X)
        assert len(preds) == 2
        for p in preds:
            assert 45.0 <= p <= 50.5

        clf = classifiers.BinarySecurityClassifier(threshold=49.5)
        clf.fit(X, y_nadir)
        cls_preds = clf.predict(X)
        probs = clf.predict_proba(X)
        assert len(cls_preds) == 2
        assert len(probs) == 2

        cal_gb = calibrated_gb.CalibratedGBDTClassifier(method="platt")
        cal_gb.fit(X, y_nadir, y_sec)
        cal_probs = cal_gb.predict_proba(X)
        assert len(cal_probs) == 2


def test_speed_budget():
    t0 = time.perf_counter()
    data = generate_synthetic_regression_dataset(n_samples=50, random_state=42)
    reg = regressors.LinearNadirRegressor()
    reg.fit(data["X"], data["y_nadir"])
    preds = reg.predict(data["X"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert elapsed_ms < 50.0, f"Synthetic test exceeded 50 ms budget: {elapsed_ms:.2f} ms"
