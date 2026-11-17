"""Binary classifiers for post-disturbance frequency security assessment (50.0 Hz standard)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from pmu_fsa.models.regressors import LinearNadirRegressor

try:
    from lightgbm import LGBMClassifier as _LGBMClassifier
    _HAS_LIGHTGBM = True
except (ImportError, OSError):
    _LGBMClassifier = None
    _HAS_LIGHTGBM = False


DEFAULT_THRESHOLD_SWEEP: List[float] = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]


def compute_binary_security_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_secure: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute confusion counts, F1, PR-AUC, ROC-AUC, missed-insecure rate, and false-alarm rate.

    Convention: class 1 = secure (nadir >= threshold), class 0 = insecure (nadir < threshold).
    - FP (y_true=0, y_pred=1): dangerous missed insecure contingency -> fnr_insecure = FP / (TN + FP).
    - FN (y_true=1, y_pred=0): conservative false alarm on secure grid -> fpr_secure = FN / (TP + FN).
    """
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    if len(yt) != len(yp):
        raise ValueError(f"Sample count mismatch: y_true has {len(yt)}, y_pred has {len(yp)}")
    if len(yt) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fnr": 0.0,
            "fpr_secure": 0.0,
            "fnr_insecure": 0.0,
            "missed_insecure_count": 0,
            "false_alarm_count": 0,
            "roc_auc": 0.5,
            "pr_auc": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    tp = int(np.sum((yt == 1) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
    accuracy = float(accuracy_score(yt, yp))

    fpr_secure = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    total_insecure = tn + fp
    fnr_insecure = float(fp / total_insecure) if total_insecure > 0 else 0.0

    if p_secure is not None and len(p_secure) == len(yt) and len(np.unique(yt)) > 1:
        ps = np.asarray(p_secure, dtype=float).ravel()
        roc_auc = float(roc_auc_score(yt, ps))
        prec_arr, rec_arr, _ = precision_recall_curve(yt, ps)
        pr_auc = float(auc(rec_arr, prec_arr))
    else:
        roc_auc = 0.5
        pr_auc = 1.0 if int(yt[0]) == 1 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fnr": fpr_secure,
        "fpr_secure": fpr_secure,
        "fnr_insecure": fnr_insecure,
        "missed_insecure_count": fp,
        "false_alarm_count": fn,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_probability_threshold_sweep(
    y_true: np.ndarray,
    p_secure: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate classification & safety metrics across decision probability thresholds."""
    sweep = thresholds if thresholds is not None else list(DEFAULT_THRESHOLD_SWEEP)
    yt = np.asarray(y_true, dtype=int).ravel()
    ps = np.asarray(p_secure, dtype=float).ravel()

    rows: List[Dict[str, Any]] = []
    for tau in sweep:
        tau_f = float(tau)
        yp = np.where(ps >= tau_f, 1, 0).astype(int)
        m = compute_binary_security_metrics(yt, yp, p_secure=ps)
        rows.append({
            "proba_threshold": tau_f,
            "tp": m["tp"],
            "tn": m["tn"],
            "fp": m["fp"],
            "fn": m["fn"],
            "missed_insecure_count": m["missed_insecure_count"],
            "false_alarm_count": m["false_alarm_count"],
            "fnr_insecure": m["fnr_insecure"],
            "fpr_secure": m["fpr_secure"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "accuracy": m["accuracy"],
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
        })
    return rows


class BaseSecurityClassifier(BaseEstimator, ClassifierMixin):
    """Base estimator for binary frequency security classifiers."""

    def __init__(
        self,
        threshold: float = 49.5,
        scale_features: bool = False,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
    ) -> None:
        self.threshold = float(threshold)
        self.scale_features = scale_features
        self.class_weight = class_weight
        self.classes_ = np.array([0, 1], dtype=int)
        self.model_name = "base_security"
        self.model_: Any = None
        self.scaler_: Optional[StandardScaler] = None
        self.constant_class_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None

    def _prepare_X(self, X: Any, is_fit: bool = False) -> np.ndarray:
        if X is None:
            if is_fit:
                raise ValueError("Cannot fit on None dataset")
            return np.empty((0, self.n_features_in_ or 0), dtype=float)

        if hasattr(X, "shape") and len(X.shape) > 0 and X.shape[0] == 0:
            if is_fit:
                raise ValueError("Cannot fit on empty dataset")
            return np.empty((0, self.n_features_in_ or 0), dtype=float)

        if isinstance(X, (list, tuple)) and len(X) == 0:
            if is_fit:
                raise ValueError("Cannot fit on empty dataset")
            return np.empty((0, self.n_features_in_ or 0), dtype=float)

        if hasattr(X, "columns"):
            cols = [str(c) for c in X.columns]
            if is_fit:
                self.feature_names_in_ = cols
            X_arr = np.asarray(X.values, dtype=float)
        else:
            if is_fit:
                self.feature_names_in_ = None
            X_arr = np.asarray(X, dtype=float)

        if X_arr.ndim == 1:
            if is_fit:
                X_arr = X_arr.reshape(-1, 1)
            else:
                if self.n_features_in_ is not None and len(X_arr) == self.n_features_in_:
                    X_arr = X_arr.reshape(1, -1)
                else:
                    X_arr = X_arr.reshape(-1, 1)

        if is_fit:
            if X_arr.shape[0] == 0:
                raise ValueError("Cannot fit on empty dataset")
            self.n_features_in_ = X_arr.shape[1]
        else:
            if self.n_features_in_ is not None and X_arr.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Feature dimension mismatch: expected {self.n_features_in_}, got {X_arr.shape[1]}"
                )

        return X_arr

    def _prepare_y(self, y: Any) -> np.ndarray:
        if y is None:
            raise ValueError("Target vector y cannot be None")
        if hasattr(y, "values"):
            y_arr = np.asarray(y.values, dtype=float)
        else:
            y_arr = np.asarray(y, dtype=float)

        y_flat = y_arr.ravel()
        if len(y_flat) == 0:
            raise ValueError("Cannot process empty target vector")

        if np.any(y_flat > 1.5):
            self._effective_threshold_ = float(self.threshold)
            y_binary = np.where(y_flat >= self.threshold, 1, 0).astype(int)
        else:
            y_binary = np.asarray(y_flat, dtype=int)
            y_binary = np.where(y_binary >= 1, 1, 0)

        return y_binary

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> BaseSecurityClassifier:
        """Fit classifier on training data."""
        X_proc = self._prepare_X(X, is_fit=True)
        y_proc = self._prepare_y(y)

        if len(X_proc) != len(y_proc):
            raise ValueError(
                f"Sample count mismatch: X has {len(X_proc)} samples, y has {len(y_proc)}"
            )

        unique_classes = np.unique(y_proc)
        if len(unique_classes) == 1:
            self.constant_class_ = int(unique_classes[0])
            self.model_ = None
            return self
        else:
            self.constant_class_ = None

        if self.scale_features:
            self.scaler_ = StandardScaler()
            X_proc = self.scaler_.fit_transform(X_proc)

        self._fit_underlying_model(X_proc, y_proc, sample_weight=sample_weight)
        return self

    def _fit_underlying_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        raise NotImplementedError

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict [P(insecure), P(secure)] class probabilities."""
        X_proc = self._prepare_X(X, is_fit=False)
        n_samples = len(X_proc)
        if n_samples == 0:
            return np.empty((0, 2), dtype=float)

        if self.constant_class_ is not None:
            if self.constant_class_ == 1:
                return np.column_stack([np.zeros(n_samples), np.ones(n_samples)])
            return np.column_stack([np.ones(n_samples), np.zeros(n_samples)])

        if self.model_ is None:
            raise RuntimeError("Model is not fitted")

        if self.scale_features and self.scaler_ is not None:
            X_proc = self.scaler_.transform(X_proc)

        probs = self.model_.predict_proba(X_proc)

        if hasattr(self.model_, "classes_") and len(self.model_.classes_) == 2:
            if self.model_.classes_[0] != 0:
                probs = probs[:, ::-1]

        probs = np.clip(probs, 0.0, 1.0)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
        return probs

    def predict(
        self,
        X: Any,
        proba_thresh: float = 0.5,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary security labels (1 = secure, 0 = insecure)."""
        thresh = threshold if threshold is not None else proba_thresh
        probs = self.predict_proba(X)
        if len(probs) == 0:
            return np.array([], dtype=int)
        p_secure = probs[:, 1]
        return np.where(p_secure >= thresh, 1, 0).astype(int)

    def evaluate(
        self,
        X: Any,
        y: Any,
        proba_thresh: float = 0.5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Compute classification metrics including missed-insecurity rate (fnr_insecure)."""
        thresh = threshold if threshold is not None else proba_thresh
        y_true = self._prepare_y(y)
        y_pred = self.predict(X, proba_thresh=thresh)
        probs = self.predict_proba(X)
        p_sec = probs[:, 1] if len(probs) > 0 else np.array([], dtype=float)
        return compute_binary_security_metrics(y_true, y_pred, p_secure=p_sec)

    def evaluate_threshold_sweep(
        self,
        X: Any,
        y: Any,
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate confusion counts and FNR_insecure across probability thresholds."""
        y_true = self._prepare_y(y)
        probs = self.predict_proba(X)
        p_sec = probs[:, 1] if len(probs) > 0 else np.array([], dtype=float)
        return evaluate_probability_threshold_sweep(y_true, p_sec, thresholds=thresholds)


class BinarySecurityClassifier(BaseSecurityClassifier):
    """Classifier mapping continuous nadir predictions through a sigmoid around the threshold."""

    def __init__(
        self,
        threshold: float = 49.5,
        alpha: float = 1e-3,
        proba_scale: float = 8.0,
        backend: str = "regressor",
    ) -> None:
        super().__init__(threshold=threshold, scale_features=False)
        self.alpha = float(alpha)
        self.proba_scale = float(proba_scale)
        self.backend = backend
        self.model_name = "binary_security_reference"
        self.regressor_: Optional[LinearNadirRegressor] = None

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> BinarySecurityClassifier:
        X_proc = self._prepare_X(X, is_fit=True)
        if hasattr(y, "values"):
            y_arr = np.asarray(y.values, dtype=float).ravel()
        else:
            y_arr = np.asarray(y, dtype=float).ravel()

        if len(X_proc) != len(y_arr):
            raise ValueError(f"Sample count mismatch: X has {len(X_proc)}, y has {len(y_arr)}")

        eff_thresh = float(self.threshold)
        if np.any(y_arr > 1.5):
            y_binary = (y_arr >= eff_thresh).astype(int)
        else:
            y_binary = (y_arr >= 0.5).astype(int)
        self._effective_threshold_ = eff_thresh

        unique_classes = np.unique(y_binary)
        if len(unique_classes) == 1 and self.backend != "regressor":
            self.constant_class_ = int(unique_classes[0])
            self.model_ = None
            self.regressor_ = None
            return self
        elif len(unique_classes) == 1 and not np.any(y_arr > 1.5):
            self.constant_class_ = int(unique_classes[0])
            self.model_ = None
            self.regressor_ = None
            return self
        else:
            self.constant_class_ = None

        if self.backend == "regressor":
            if not np.any(y_arr > 1.5):
                y_reg = np.where(y_arr >= 1, eff_thresh + 0.3, eff_thresh - 0.5)
            else:
                y_reg = y_arr

            clip_low = min(45.0, float(np.min(y_reg)) - 2.0)
            clip_high = max(50.5, float(np.max(y_reg)) + 1.0)
            self.regressor_ = LinearNadirRegressor(
                alpha=self.alpha,
                clip_bounds=(clip_low, clip_high),
                scale_features=True,
            )
            self.regressor_.fit(X_proc, y_reg)
            self.model_ = self.regressor_
        else:
            backend_map = {
                "logistic": LogisticSecurityClassifier(threshold=eff_thresh, scale_features=True),
                "decision_tree": DecisionTreeSecurityClassifier(threshold=eff_thresh),
                "random_forest": RandomForestSecurityClassifier(threshold=eff_thresh),
                "gradient_boosting": GradientBoostingSecurityClassifier(threshold=eff_thresh),
                "lightgbm": LightGBMSecurityClassifier(threshold=eff_thresh),
            }
            if self.backend not in backend_map:
                raise ValueError(f"Unknown backend: '{self.backend}'")
            self.model_ = backend_map[self.backend]
            self.model_.fit(X_proc, y, sample_weight=sample_weight)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.constant_class_ is not None:
            X_proc = self._prepare_X(X, is_fit=False)
            n_samples = len(X_proc)
            if self.constant_class_ == 1:
                return np.column_stack([np.zeros(n_samples), np.ones(n_samples)])
            return np.column_stack([np.ones(n_samples), np.zeros(n_samples)])

        if self.backend == "regressor":
            if self.regressor_ is None:
                raise RuntimeError("Model is not fitted")
            preds = self.regressor_.predict(X)
            if len(preds) == 0:
                return np.empty((0, 2), dtype=float)

            eff_thresh = getattr(self, "_effective_threshold_", self.threshold)
            probs = []
            for pred in preds:
                z = (pred - eff_thresh) * self.proba_scale
                if z > 35.0:
                    p_sec = 1.0
                elif z < -35.0:
                    p_sec = 0.0
                else:
                    p_sec = 1.0 / (1.0 + math.exp(-z))
                probs.append([1.0 - p_sec, p_sec])
            return np.array(probs, dtype=float)

        return self.model_.predict_proba(X)


class LogisticSecurityClassifier(BaseSecurityClassifier):
    """L2-regularized logistic regression security classifier."""

    def __init__(
        self,
        threshold: float = 49.5,
        C: float = 1.0,
        class_weight: Optional[Union[str, Dict[Any, float]]] = "balanced",
        scale_features: bool = True,
        max_iter: int = 1000,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            threshold=threshold,
            scale_features=scale_features,
            class_weight=class_weight,
        )
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.model_name = "logistic_security"

    def _fit_underlying_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.model_ = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)


class DecisionTreeSecurityClassifier(BaseSecurityClassifier):
    """CART decision tree security classifier."""

    def __init__(
        self,
        threshold: float = 49.5,
        max_depth: Optional[int] = 6,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        scale_features: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            threshold=threshold,
            scale_features=scale_features,
            class_weight=class_weight,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_name = "decision_tree_security"

    def _fit_underlying_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.model_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)


class RandomForestSecurityClassifier(BaseSecurityClassifier):
    """Random forest security classifier."""

    def __init__(
        self,
        threshold: float = 49.5,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: Optional[Union[str, Dict[Any, float]]] = "balanced",
        scale_features: bool = False,
        random_state: Optional[int] = 42,
        n_jobs: Optional[int] = None,
    ) -> None:
        super().__init__(
            threshold=threshold,
            scale_features=scale_features,
            class_weight=class_weight,
        )
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_name = "random_forest_security"

    def _fit_underlying_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)


class GradientBoostingSecurityClassifier(BaseSecurityClassifier):
    """Gradient boosted decision tree security classifier."""

    def __init__(
        self,
        threshold: float = 49.5,
        n_estimators: int = 100,
        learning_rate: float = 0.08,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        class_weight: Optional[Union[str, Dict[Any, float]]] = None,
        scale_features: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            threshold=threshold,
            scale_features=scale_features,
            class_weight=class_weight,
        )
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = float(subsample)
        self.random_state = random_state
        self.model_name = "gradient_boosting_security"

    def _fit_underlying_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        if self.class_weight == "balanced" and sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)

        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self.model_.fit(X, y, sample_weight=sample_weight)


class LightGBMSecurityClassifier(BaseSecurityClassifier):
    """Leaf-wise histogram gradient boosting security classifier (LightGBM / HistGBDT)."""

    def __init__(
        self,
        threshold: float = 49.5,
        n_estimators: int = 100,
        learning_rate: float = 0.08,
        num_leaves: int = 15,
        max_depth: Optional[int] = 5,
        min_child_samples: int = 5,
        reg_lambda: float = 1e-2,
        class_weight: Optional[Union[str, Dict[Any, float]]] = "balanced",
        scale_features: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__(
            threshold=threshold,
            scale_features=scale_features,
            class_weight=class_weight,
        )
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.max_depth = max_depth
        self.min_child_samples = int(min_child_samples)
        self.reg_lambda = float(reg_lambda)
        self.random_state = random_state
        self.model_name = "lightgbm_security"
        self.backend_: str = "lightgbm" if _HAS_LIGHTGBM else "sklearn_hist_gbdt"

    def _fit_underlying_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None,
    ) -> None:
        eff_min_samples = max(1, min(self.min_child_samples, max(1, len(X) // 3)))
        if self.class_weight == "balanced" and sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)

        if _HAS_LIGHTGBM and _LGBMClassifier is not None:
            self.backend_ = "lightgbm"
            self.model_ = _LGBMClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=max(2, self.num_leaves),
                max_depth=-1 if self.max_depth is None else int(self.max_depth),
                min_child_samples=eff_min_samples,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                n_jobs=1,
                verbose=-1,
            )
            self.model_.fit(X, y, sample_weight=sample_weight)
        else:
            self.backend_ = "sklearn_hist_gbdt"
            self.model_ = HistGradientBoostingClassifier(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_leaf_nodes=max(2, self.num_leaves),
                max_depth=self.max_depth,
                min_samples_leaf=eff_min_samples,
                l2_regularization=self.reg_lambda,
                random_state=self.random_state,
            )
            self.model_.fit(X, y, sample_weight=sample_weight)


CLASSIFIER_REGISTRY: Dict[str, Type[BaseSecurityClassifier]] = {
    "binary": BinarySecurityClassifier,
    "binary_security": BinarySecurityClassifier,
    "logistic": LogisticSecurityClassifier,
    "logistic_regression": LogisticSecurityClassifier,
    "decision_tree": DecisionTreeSecurityClassifier,
    "cart": DecisionTreeSecurityClassifier,
    "dt": DecisionTreeSecurityClassifier,
    "random_forest": RandomForestSecurityClassifier,
    "rf": RandomForestSecurityClassifier,
    "gradient_boosting": GradientBoostingSecurityClassifier,
    "gbdt": GradientBoostingSecurityClassifier,
    "gbc": GradientBoostingSecurityClassifier,
    "lightgbm": LightGBMSecurityClassifier,
    "lgbm": LightGBMSecurityClassifier,
}


def get_security_classifier(name: str = "binary", **kwargs: Any) -> BaseSecurityClassifier:
    """Instantiate a security classifier by name."""
    key = name.strip().lower()
    if key not in CLASSIFIER_REGISTRY:
        raise ValueError(
            f"Unknown classifier name: '{name}'. Available: {sorted(list(CLASSIFIER_REGISTRY.keys()))}"
        )
    return CLASSIFIER_REGISTRY[key](**kwargs)


class SecurityClassifier:
    """Wrapper around registered security classifiers."""

    def __init__(
        self,
        model_type: str = "binary",
        threshold: float = 49.5,
        **kwargs: Any,
    ) -> None:
        init_kwargs = dict(kwargs)
        init_kwargs["threshold"] = threshold
        self.model_ = get_security_classifier(model_type, **init_kwargs)

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> SecurityClassifier:
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(
        self,
        X: Any,
        proba_thresh: float = 0.5,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        return self.model_.predict(X, proba_thresh=proba_thresh, threshold=threshold)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.model_.predict_proba(X)

    def evaluate(
        self,
        X: Any,
        y: Any,
        proba_thresh: float = 0.5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        return self.model_.evaluate(X, y, proba_thresh=proba_thresh, threshold=threshold)

    def evaluate_threshold_sweep(
        self,
        X: Any,
        y: Any,
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        return self.model_.evaluate_threshold_sweep(X, y, thresholds=thresholds)
