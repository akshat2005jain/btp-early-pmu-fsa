"""End-to-end tests for nadir regression, security classification, and calibration."""

import math
import copy
from typing import Dict, Any, List, Tuple

from tests.e2e.conftest import (
    np, HAS_NUMPY,
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
    _to_float_list_1d,
    _to_float_list_2d,
)

try:
    from pmu_fsa.models import regressors
    from pmu_fsa.models import classifiers
    from pmu_fsa.models import calibrated_gb
    from pmu_fsa.models import calibration_metrics
    HAS_MODEL_MODULES = True
except ImportError:
    HAS_MODEL_MODULES = False


def _calc_regression_metrics(y_true: List[float], y_pred: List[float]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot calculate metrics on empty targets")
    mae = sum(abs(y - p) for y, p in zip(y_true, y_pred)) / n
    mse = sum((y - p) ** 2 for y, p in zip(y_true, y_pred)) / n
    rmse = math.sqrt(mse)
    y_mean = sum(y_true) / n
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    r2 = 1.0 - (sum((y - p) ** 2 for y, p in zip(y_true, y_pred)) / ss_tot) if ss_tot > 0 else 1.0
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _calc_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot calculate metrics on empty targets")
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy = (tp + tn) / n

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "fnr": fnr, "accuracy": accuracy
    }


def test_f14_linear_nadir_regressor_fit_and_prediction_shape(sample_regression_dataset):
    data = sample_regression_dataset
    X = data["X"]
    y = data["y_nadir"]

    model = RefLinearNadirRegressor()
    model.fit(X, y)
    preds = model.predict(X)

    preds_list = _to_float_list_1d(preds)
    assert len(preds_list) == len(_to_float_list_1d(y)), \
        f"Prediction shape {len(preds_list)} does not match target count {len(_to_float_list_1d(y))}"

    if HAS_MODEL_MODULES and hasattr(regressors, "LinearNadirRegressor"):
        prod_model = regressors.LinearNadirRegressor()
        prod_model.fit(X, y)
        prod_preds = _to_float_list_1d(prod_model.predict(X))
        assert len(prod_preds) == len(preds_list)


def test_f14_baseline_regressor_comparison(sample_regression_dataset):
    data = sample_regression_dataset
    X = data["X"]
    y = data["y_nadir"]
    y_list = _to_float_list_1d(y)

    models = [
        ("LinearNadirRegressor", RefLinearNadirRegressor()),
        ("DecisionTreeNadirRegressor", RefDecisionTreeNadirRegressor(max_depth=3)),
        ("RandomForestNadirRegressor", RefRandomForestNadirRegressor(n_estimators=4, max_depth=3)),
        ("GradientBoostingNadirRegressor", RefGradientBoostingNadirRegressor(n_estimators=5, max_depth=3)),
    ]

    for name, model in models:
        model.fit(X, y)
        preds = model.predict(X)
        p_list = _to_float_list_1d(preds)
        assert len(p_list) == len(y_list), f"{name} returned {len(p_list)} predictions, expected {len(y_list)}"

        metrics = _calc_regression_metrics(y_list, p_list)
        assert metrics["mae"] < 0.6, f"{name} training MAE {metrics['mae']} exceeds upper bound 0.6 Hz"
        assert metrics["r2"] > -0.5, f"{name} R2 {metrics['r2']} is unacceptably negative"


def test_f14_nadir_prediction_physical_bounds(sample_regression_dataset):
    data = sample_regression_dataset
    X = data["X"]
    y = data["y_nadir"]

    model = RefGradientBoostingNadirRegressor()
    model.fit(X, y)
    preds = _to_float_list_1d(model.predict(X))

    for p in preds:
        assert 55.0 <= p <= 60.5, f"Predicted nadir {p} outside physical bounds [55.0, 60.5] Hz"


def test_f14_prediction_monotonicity_with_disturbance_size():
    # Columns: [loss_mw / 1000, renewable_penetration, rocof, delta_f]
    X_train = [
        [0.2, 0.1, -0.15, -0.1],
        [0.4, 0.1, -0.25, -0.2],
        [0.6, 0.1, -0.38, -0.35],
        [0.8, 0.1, -0.50, -0.5],
        [1.0, 0.1, -0.65, -0.65],
    ]
    y_train = [59.8, 59.6, 59.4, 59.2, 59.0]

    model = RefLinearNadirRegressor()
    model.fit(X_train, y_train)

    X_test = [
        [0.25, 0.1, -0.18, -0.12],
        [0.50, 0.1, -0.32, -0.28],
        [0.75, 0.1, -0.48, -0.45],
        [0.95, 0.1, -0.62, -0.60],
    ]
    preds = _to_float_list_1d(model.predict(X_test))

    for i in range(len(preds) - 1):
        assert preds[i] >= preds[i + 1] - 1e-4, \
            f"Monotonicity violation: loss {X_test[i][0]} produced {preds[i]} < loss {X_test[i+1][0]} {preds[i+1]}"


def test_f14_regression_metrics_evaluation(sample_regression_dataset):
    data = sample_regression_dataset
    X = data["X"]
    y = data["y_nadir"]
    y_list = _to_float_list_1d(y)

    model = RefDecisionTreeNadirRegressor(max_depth=4)
    model.fit(X, y)
    preds = _to_float_list_1d(model.predict(X))

    metrics = _calc_regression_metrics(y_list, preds)
    mae = metrics["mae"]
    rmse = metrics["rmse"]
    r2 = metrics["r2"]

    assert mae >= 0.0, f"MAE must be non-negative, got {mae}"
    assert rmse >= mae - 1e-9, f"RMSE ({rmse}) must be >= MAE ({mae}) by Cauchy-Schwarz"
    assert r2 <= 1.0 + 1e-9, f"R2 score ({r2}) cannot exceed 1.0"


def test_f14_regressor_residual_distribution_sanity(sample_regression_dataset):
    data = sample_regression_dataset
    X = data["X"]
    y = data["y_nadir"]
    y_list = _to_float_list_1d(y)

    model = RefLinearNadirRegressor(alpha=1e-4)
    model.fit(X, y)
    preds = _to_float_list_1d(model.predict(X))

    residuals = [y - p for y, p in zip(y_list, preds)]
    mean_residual = sum(residuals) / len(residuals)
    assert abs(mean_residual) < 0.05, f"Mean residual {mean_residual} indicates significant model bias"


def test_f15_security_classifier_threshold_contract(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = _to_float_list_1d(data["y_nadir"])
    y_sec_expected = [1 if val >= 59.5 else 0 for val in y_nadir]

    clf = RefBinarySecurityClassifier(threshold=59.5)
    clf.fit(X, y_nadir)
    binary_preds = _to_float_list_1d(clf.predict(X))

    assert len(binary_preds) == len(y_sec_expected)
    for b in binary_preds:
        assert b in (0, 1, 0.0, 1.0), f"Binary prediction {b} not in {{0, 1}}"

    if HAS_MODEL_MODULES and hasattr(classifiers, "BinarySecurityClassifier"):
        prod_clf = classifiers.BinarySecurityClassifier(threshold=59.5)
        prod_clf.fit(X, y_nadir)
        prod_preds = _to_float_list_1d(prod_clf.predict(X))
        assert len(prod_preds) == len(y_sec_expected)


def test_f15_classification_metrics_computation(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]
    y_sec = _to_float_list_1d(data["y_secure"])

    clf = RefBinarySecurityClassifier(threshold=59.5)
    clf.fit(X, y_nadir)
    preds = [int(p) for p in _to_float_list_1d(clf.predict(X))]

    metrics = _calc_classification_metrics([int(y) for y in y_sec], preds)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["fnr"] <= 1.0
    assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == len(y_sec)


def test_f15_predict_proba_simplex_bounds(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]

    clf = RefBinarySecurityClassifier(threshold=59.5)
    clf.fit(X, y_nadir)
    probs = _to_float_list_2d(clf.predict_proba(X))

    assert len(probs) == len(_to_float_list_1d(y_nadir))
    for row in probs:
        assert len(row) == 2, f"Expected 2 class probabilities, got {len(row)}"
        p_insecure, p_secure = row
        assert 0.0 <= p_insecure <= 1.0, f"P(insecure) out of bounds: {p_insecure}"
        assert 0.0 <= p_secure <= 1.0, f"P(secure) out of bounds: {p_secure}"
        assert abs((p_insecure + p_secure) - 1.0) < 1e-6, f"Simplex violation: {p_insecure} + {p_secure} != 1.0"


def test_f15_threshold_sensitivity_and_tradeoff(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]
    y_sec = [int(y) for y in _to_float_list_1d(data["y_secure"])]

    clf = RefBinarySecurityClassifier(threshold=59.5)
    clf.fit(X, y_nadir)

    recalls = []
    precisions = []
    for thresh in [0.2, 0.4, 0.6, 0.8]:
        preds = [int(p) for p in _to_float_list_1d(clf.predict(X, proba_thresh=thresh))]
        m = _calc_classification_metrics(y_sec, preds)
        recalls.append(m["recall"])
        precisions.append(m["precision"])

    for i in range(len(recalls) - 1):
        assert recalls[i] >= recalls[i + 1] - 1e-6, \
            f"Recall did not decrease with higher threshold: {recalls}"


def test_f15_pr_auc_calculation():
    y_true = [1, 1, 1, 1, 0, 0, 0, 0]
    y_scores = [0.9, 0.85, 0.75, 0.60, 0.40, 0.30, 0.20, 0.10]

    precisions = []
    recalls = []
    for s in sorted(set(y_scores), reverse=True):
        preds = [1 if val >= s else 0 for val in y_scores]
        m = _calc_classification_metrics(y_true, preds)
        precisions.append(m["precision"])
        recalls.append(m["recall"])

    points = sorted(zip(recalls, precisions), key=lambda x: x[0])
    pr_auc = 0.0
    for i in range(len(points) - 1):
        dr = points[i + 1][0] - points[i][0]
        avg_p = 0.5 * (points[i][1] + points[i + 1][1])
        pr_auc += dr * avg_p

    assert 0.0 <= pr_auc <= 1.0, f"PR-AUC {pr_auc} out of bounds"
    assert pr_auc > 0.5, f"Expected strong PR-AUC for well-separated data, got {pr_auc}"


def test_f15_false_negative_rate_safety_metric():
    y_true = [1, 1, 1, 1, 1]
    y_pred = [1, 1, 1, 0, 0]

    metrics = _calc_classification_metrics(y_true, y_pred)
    expected_fnr = 2.0 / 5.0
    assert abs(metrics["fnr"] - expected_fnr) < 1e-6, \
        f"FNR {metrics['fnr']} does not match expected {expected_fnr}"


def test_f16_calibrated_gbdt_fit_and_probability_output(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]

    cal_gbdt = RefCalibratedGBDTClassifier(method="platt")
    cal_gbdt.fit(X, y_nadir)
    probs = _to_float_list_2d(cal_gbdt.predict_proba(X))

    assert len(probs) == len(_to_float_list_1d(y_nadir))
    for p_insec, p_sec in probs:
        assert 0.0 <= p_sec <= 1.0, f"P(secure) {p_sec} out of [0, 1] bounds"
        assert 0.0 <= p_insec <= 1.0, f"P(insecure) {p_insec} out of [0, 1] bounds"
        assert abs(p_insec + p_sec - 1.0) < 1e-6

    if HAS_MODEL_MODULES and hasattr(calibrated_gb, "CalibratedGBDTClassifier"):
        prod_cal = calibrated_gb.CalibratedGBDTClassifier(method="platt")
        prod_cal.fit(X, y_nadir)
        prod_probs = _to_float_list_2d(prod_cal.predict_proba(X))
        assert len(prod_probs) == len(probs)


def test_f16_platt_scaling_sigmoid_transformation():
    scores = [58.5, 59.0, 59.4, 59.5, 59.6, 60.0]
    clf = RefCalibratedGBDTClassifier(method="platt")

    probs = [clf._calibrate_single(s) for s in scores]

    for i in range(len(probs) - 1):
        assert probs[i] <= probs[i + 1] + 1e-7, \
            f"Platt scaling non-monotonic: {scores[i]} -> {probs[i]}, {scores[i+1]} -> {probs[i+1]}"

    p_thresh = clf._calibrate_single(59.5)
    assert abs(p_thresh - 0.5) < 1e-3, f"Expected P(59.5)=0.5, got {p_thresh}"


def test_f16_isotonic_regression_monotonicity(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]

    cal_iso = RefCalibratedGBDTClassifier(method="isotonic")
    cal_iso.fit(X, y_nadir)

    test_scores = [58.8 + 0.1 * k for k in range(12)]
    cal_probs = [cal_iso._calibrate_single(s) for s in test_scores]

    for i in range(len(cal_probs) - 1):
        assert cal_probs[i] <= cal_probs[i + 1] + 1e-7, \
            f"Isotonic calibration violated monotonicity: {test_scores[i]} -> {cal_probs[i]}, {test_scores[i+1]} -> {cal_probs[i+1]}"


def test_f16_calibrated_confidence_intervals(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]

    cal_gbdt = RefCalibratedGBDTClassifier(method="platt")
    cal_gbdt.fit(X, y_nadir)

    probs, low_bounds, high_bounds = cal_gbdt.predict_with_intervals(X, confidence_level=0.95)
    p_list = _to_float_list_1d(probs)
    low_list = _to_float_list_1d(low_bounds)
    high_list = _to_float_list_1d(high_bounds)

    for p, low, high in zip(p_list, low_list, high_list):
        assert 0.0 <= low <= high <= 1.0, f"Invalid interval bounds: [{low}, {high}]"
        assert low <= p + 1e-6, f"Lower bound {low} exceeds estimate {p}"
        assert p <= high + 1e-6, f"Estimate {p} exceeds upper bound {high}"


def test_f16_calibration_improves_or_preserves_reliability(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]
    y_sec = _to_float_list_1d(data["y_secure"])

    cal_gbdt = RefCalibratedGBDTClassifier(method="platt")
    cal_gbdt.fit(X, y_nadir)
    probs_2d = _to_float_list_2d(cal_gbdt.predict_proba(X))
    p_sec = [p[1] for p in probs_2d]

    brier = ref_brier_score(y_sec, p_sec)
    assert brier < 0.25, f"Calibrated GBDT Brier score {brier} exceeds safety limit 0.25"


def test_f16_calibrated_gbdt_predict_with_custom_threshold(sample_classification_dataset):
    data = sample_classification_dataset
    X = data["X"]
    y_nadir = data["y_nadir"]

    cal_gbdt = RefCalibratedGBDTClassifier(method="platt")
    cal_gbdt.fit(X, y_nadir)

    preds_lenient = _to_float_list_1d(cal_gbdt.predict(X, threshold=0.2))
    preds_strict = _to_float_list_1d(cal_gbdt.predict(X, threshold=0.8))

    sum_lenient = sum(preds_lenient)
    sum_strict = sum(preds_strict)
    assert sum_lenient >= sum_strict, \
        f"Lenient threshold produced fewer secure predictions ({sum_lenient}) than strict ({sum_strict})"


def test_f17_brier_score_bounds_and_zero_error():
    y_true = [1.0, 0.0, 1.0, 1.0, 0.0]

    perfect_p = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert abs(ref_brier_score(y_true, perfect_p)) < 1e-9, "Perfect predictions must yield Brier score 0.0"

    wrong_p = [0.0, 1.0, 0.0, 0.0, 1.0]
    assert abs(ref_brier_score(y_true, wrong_p) - 1.0) < 1e-9, "Inverse predictions must yield Brier score 1.0"

    mid_p = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert abs(ref_brier_score(y_true, mid_p) - 0.25) < 1e-9, "p=0.5 must yield Brier score 0.25"

    if HAS_MODEL_MODULES and hasattr(calibration_metrics, "brier_score"):
        prod_brier = calibration_metrics.brier_score(y_true, perfect_p)
        assert abs(prod_brier) < 1e-9


def test_f17_expected_calibration_error_calculation():
    y_true = [1.0, 1.0, 0.0, 0.0]
    y_prob = [0.8, 0.8, 0.2, 0.2]

    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0, f"ECE {ece} outside [0, 1]"
    # 0.5 * |1.0 - 0.8| + 0.5 * |0.0 - 0.2| = 0.2
    assert abs(ece - 0.2) < 1e-6, f"Expected ECE 0.2, got {ece}"


def test_f17_maximum_calibration_error_calculation():
    y_true = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    y_prob = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]

    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=5)
    mce = ref_maximum_calibration_error(y_true, y_prob, n_bins=5)

    assert 0.0 <= mce <= 1.0, f"MCE {mce} outside [0, 1]"
    assert mce >= ece - 1e-9, f"MCE ({mce}) must be >= ECE ({ece})"


def test_f17_reliability_diagram_binning():
    y_true = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    y_prob = [0.95, 0.85, 0.75, 0.65, 0.35, 0.25, 0.15, 0.05]

    n_bins = 5
    diagram = ref_reliability_diagram(y_true, y_prob, n_bins=n_bins)

    assert "bin_accuracies" in diagram
    assert "bin_confidences" in diagram
    assert "bin_counts" in diagram
    assert "bin_edges" in diagram

    assert len(diagram["bin_accuracies"]) == n_bins
    assert len(diagram["bin_confidences"]) == n_bins
    assert len(diagram["bin_counts"]) == n_bins
    assert len(diagram["bin_edges"]) == n_bins + 1
    assert sum(diagram["bin_counts"]) == len(y_true), "Sum of bin counts must equal sample count"


def test_f17_perfectly_calibrated_uniform_synthetic_data():
    y_true = []
    y_prob = []
    # Bin [0.2, 0.4]: 3/10 positive; bin [0.6, 0.8]: 7/10 positive
    y_prob.extend([0.3] * 10)
    y_true.extend([1.0] * 3 + [0.0] * 7)
    y_prob.extend([0.7] * 10)
    y_true.extend([1.0] * 7 + [0.0] * 3)

    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=5)
    assert abs(ece) < 1e-6, f"Expected near-zero ECE for perfectly calibrated data, got {ece}"


def test_f17_miscalibrated_overconfident_detection():
    y_prob = [0.99] * 20
    y_true = [1.0] * 10 + [0.0] * 10

    ece = ref_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece > 0.4, f"Expected high ECE for overconfident model, got {ece}"
