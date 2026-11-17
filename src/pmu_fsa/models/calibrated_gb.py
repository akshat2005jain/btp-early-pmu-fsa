"""Calibrated GBDT model combining nadir regression with Platt or isotonic calibration."""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from pmu_fsa.models.calibration_metrics import compute_calibration_metrics
from pmu_fsa.models.classifiers import (
    compute_binary_security_metrics,
    evaluate_probability_threshold_sweep,
)


def _pava_isotonic(scores: List[float], targets: List[float]) -> Tuple[List[float], List[float]]:
    """Pool Adjacent Violators Algorithm (PAVA) for monotonic step calibration."""
    paired = sorted(zip(scores, targets), key=lambda x: x[0])
    s_sorted = [float(p[0]) for p in paired]
    y_sorted = [float(p[1]) for p in paired]

    blocks = [[1, y_val, s_val, s_val] for s_val, y_val in zip(s_sorted, y_sorted)]

    i = 0
    while i < len(blocks) - 1:
        mean_i = blocks[i][1] / blocks[i][0]
        mean_next = blocks[i + 1][1] / blocks[i + 1][0]
        if mean_i > mean_next:
            blocks[i][0] += blocks[i + 1][0]
            blocks[i][1] += blocks[i + 1][1]
            blocks[i][3] = blocks[i + 1][3]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1
        else:
            i += 1

    x_points: List[float] = []
    y_points: List[float] = []
    for b in blocks:
        mean_val = max(0.0, min(1.0, b[1] / b[0]))
        x_points.append(b[2])
        y_points.append(mean_val)
        if b[3] != b[2]:
            x_points.append(b[3])
            y_points.append(mean_val)
    return x_points, y_points


class CalibratedGBDTClassifier(BaseEstimator, ClassifierMixin):
    """GBDT nadir regressor with leak-free Platt scaling or PAVA isotonic calibration.

    Supports three out-of-sample calibration protocols:
    1. Disjoint validation (`X_cal` provided, `fit_full_after_cal=False`): base GBDT fits on `X`,
       calibrator fits on `X_cal` predictions.
    2. Cross-fit CV + full refit (`X_cal` provided, `fit_full_after_cal=True`): `cv_folds` out-of-fold
       nadir predictions across `X + X_cal` fit the calibrator and residual quantiles, and `base_model`
       refits on `X + X_cal` (matching full `train+val` sample budget with zero `test` leakage).
    3. Out-of-fold CV (`X_cal=None`, `validation_fraction=0.0`): `cv_folds` out-of-fold predictions
       across `X` fit the calibrator so Platt/Isotonic never overfits in-sample tree predictions.
    """

    def __init__(
        self,
        method: str = "platt",
        threshold: float = 49.5,
        n_estimators: int = 100,
        learning_rate: float = 0.08,
        max_depth: Optional[int] = 4,
        random_state: int = 42,
        validation_fraction: float = 0.0,
        cv_folds: int = 5,
        fit_full_after_cal: bool = False,
    ) -> None:
        if method not in ("platt", "isotonic"):
            raise ValueError(f"Unknown calibration method: '{method}' (must be 'platt' or 'isotonic')")

        self.method = method
        self.threshold = float(threshold)
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.random_state = random_state
        self.validation_fraction = float(validation_fraction)
        self.cv_folds = int(cv_folds)
        self.fit_full_after_cal = bool(fit_full_after_cal)

        self.base_model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth if self.max_depth is not None else 4,
            random_state=self.random_state,
        )
        self.base_gbdt = self.base_model
        self.base_regressor_ = self.base_model

        self.platt_a: float = 8.0
        self.platt_b: float = -self.threshold * self.platt_a
        self.calibrator_: Optional[LogisticRegression] = None
        self.isotonic_x: List[float] = []
        self.isotonic_y: List[float] = []
        self.n_train_samples: int = 0
        self.n_cal_: int = 0
        self.effective_method_: str = self.method
        self.fallback_used_: bool = False
        self.fallback_reason_: Optional[str] = None
        self.calibration_protocol_: str = "unfitted"

        self.quantile_low_model_: Optional[GradientBoostingRegressor] = None
        self.quantile_high_model_: Optional[GradientBoostingRegressor] = None
        self.residual_std_: float = 0.08
        self.conformal_q95_: float = 0.12
        self.train_mean_: Optional[np.ndarray] = None
        self.train_std_: Optional[np.ndarray] = None
        self.train_dist_p75_: float = 1.25

        self.classes_ = np.array([0, 1], dtype=int)
        self.n_features_in_: Optional[int] = None
        self.feature_names_in_: Optional[List[str]] = None
        self.single_val_: Optional[float] = None

    def _make_gbr(self, loss: str = "squared_error", alpha: float = 0.5, n_est: Optional[int] = None) -> GradientBoostingRegressor:
        eff_depth = self.max_depth if self.max_depth is not None else 4
        if loss == "quantile":
            return GradientBoostingRegressor(
                loss="quantile",
                alpha=alpha,
                n_estimators=n_est or min(50, max(15, self.n_estimators // 2)),
                learning_rate=self.learning_rate,
                max_depth=min(3, eff_depth),
                random_state=self.random_state,
            )
        return GradientBoostingRegressor(
            loss="squared_error",
            n_estimators=n_est or self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=eff_depth,
            random_state=self.random_state,
        )

    def _compute_oof_predictions(
        self,
        X_pool: np.ndarray,
        y_reg_pool: np.ndarray,
        y_sec_pool: np.ndarray,
    ) -> np.ndarray:
        n = len(X_pool)
        k = min(max(2, self.cv_folds), n)
        oof_preds = np.zeros(n, dtype=np.float64)

        unique_c, counts_c = np.unique(y_sec_pool, return_counts=True)
        if len(unique_c) >= 2 and int(np.min(counts_c)) >= k:
            splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
            splits = splitter.split(X_pool, y_sec_pool)
        else:
            splitter = KFold(n_splits=k, shuffle=True, random_state=self.random_state)
            splits = splitter.split(X_pool)

        for tr_idx, va_idx in splits:
            fold_gbr = self._make_gbr()
            fold_gbr.fit(X_pool[tr_idx], y_reg_pool[tr_idx])
            oof_preds[va_idx] = fold_gbr.predict(X_pool[va_idx])

        return oof_preds

    def _fit_uncertainty_models(
        self,
        X_fit: np.ndarray,
        y_reg_fit: np.ndarray,
        oos_residuals: np.ndarray,
    ) -> None:
        self.train_mean_ = np.mean(X_fit, axis=0)
        std = np.std(X_fit, axis=0)
        self.train_std_ = np.where(std > 1e-8, std, 1.0)
        z_train = (X_fit - self.train_mean_) / self.train_std_
        rms_dist = np.sqrt(np.mean(z_train ** 2, axis=1))
        self.train_dist_p75_ = float(np.percentile(rms_dist, 75.0)) if len(rms_dist) > 0 else 1.25

        abs_res = np.abs(oos_residuals[np.isfinite(oos_residuals)])
        if len(abs_res) >= 3:
            self.residual_std_ = max(0.015, float(np.std(oos_residuals)))
            self.conformal_q95_ = max(0.025, float(np.percentile(abs_res, 95.0)))
        else:
            self.residual_std_ = 0.08
            self.conformal_q95_ = 0.12

        if len(X_fit) >= 10 and float(np.var(y_reg_fit)) > 1e-10:
            try:
                self.quantile_low_model_ = self._make_gbr(loss="quantile", alpha=0.025)
                self.quantile_low_model_.fit(X_fit, y_reg_fit)
                self.quantile_high_model_ = self._make_gbr(loss="quantile", alpha=0.975)
                self.quantile_high_model_.fit(X_fit, y_reg_fit)
            except Exception:
                self.quantile_low_model_ = None
                self.quantile_high_model_ = None
        else:
            self.quantile_low_model_ = None
            self.quantile_high_model_ = None

    def _calibrate_single(self, score: float) -> float:
        """Map a scalar nadir prediction to calibrated P(secure)."""
        if math.isnan(score):
            return 0.5
        if math.isinf(score):
            return 1.0 if score > 0 else 0.0

        eff_method = getattr(self, "effective_method_", self.method)
        eff_t = getattr(self, "_effective_threshold_", self.threshold)

        if eff_method == "platt":
            z = self.platt_a * score + self.platt_b
            if z > 35.0:
                return 1.0
            elif z < -35.0:
                return 0.0
            return 1.0 / (1.0 + math.exp(-z))
        elif eff_method == "isotonic":
            if not self.isotonic_x:
                z = 8.0 * (score - eff_t)
                if z > 35.0:
                    return 1.0
                elif z < -35.0:
                    return 0.0
                return 1.0 / (1.0 + math.exp(-z))
            if score <= self.isotonic_x[0]:
                return float(self.isotonic_y[0])
            if score >= self.isotonic_x[-1]:
                return float(self.isotonic_y[-1])
            idx = int(np.searchsorted(self.isotonic_x, score, side="right")) - 1
            idx = max(0, min(idx, len(self.isotonic_x) - 2))
            x0, x1 = self.isotonic_x[idx], self.isotonic_x[idx + 1]
            y0, y1 = self.isotonic_y[idx], self.isotonic_y[idx + 1]
            span = x1 - x0
            if span <= 0.0:
                return float(y1)
            weight = max(0.0, min(1.0, (score - x0) / span))
            return float((1.0 - weight) * y0 + weight * y1)
        return 0.5

    def fit(
        self,
        X: Any,
        y_nadir: Any,
        y_secure: Optional[Any] = None,
        X_cal: Optional[Any] = None,
        y_cal: Optional[Any] = None,
    ) -> CalibratedGBDTClassifier:
        """Fit the base GBDT regressor, uncertainty estimators, and probability calibrator."""
        if X is None:
            raise ValueError("Cannot fit on None dataset")
        if y_nadir is None:
            raise ValueError("Target vector cannot be None")

        if hasattr(X, "columns"):
            self.feature_names_in_ = [str(c) for c in X.columns]
            X_arr = np.asarray(X.values, dtype=np.float64)
        else:
            self.feature_names_in_ = None
            X_arr = np.asarray(X, dtype=np.float64)

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.n_train_samples = len(X_arr)
        if self.n_train_samples == 0:
            raise ValueError("Cannot fit on empty dataset")

        self.n_features_in_ = X_arr.shape[1]
        y_nadir_arr = np.asarray(y_nadir, dtype=np.float64).ravel()
        if len(X_arr) != len(y_nadir_arr):
            raise ValueError(
                f"Sample count mismatch: X has {len(X_arr)} samples, y has {len(y_nadir_arr)}"
            )

        eff_thresh = float(self.threshold)
        self._effective_threshold_ = eff_thresh

        unique_vals = np.unique(y_nadir_arr)
        if np.all(np.isin(unique_vals, [0, 1])) and not np.any(y_nadir_arr > 1.5):
            y_sec_arr = y_nadir_arr.astype(int)
            y_reg_arr = np.where(y_sec_arr == 1, eff_thresh + 0.3, eff_thresh - 0.5)
        else:
            y_reg_arr = y_nadir_arr
            if y_secure is None:
                y_sec_arr = (y_nadir_arr >= eff_thresh).astype(int)
            else:
                y_sec_arr = np.asarray(y_secure, dtype=int).ravel()

        self.base_model = self._make_gbr()
        self.base_gbdt = self.base_model
        self.base_regressor_ = self.base_model
        self.isotonic_x = []
        self.isotonic_y = []
        self.calibrator_ = None
        self.effective_method_ = self.method
        self.fallback_used_ = False
        self.fallback_reason_ = None

        # Parse optional calibration set
        has_explicit_cal = X_cal is not None and y_cal is not None
        if has_explicit_cal:
            X_c = np.asarray(X_cal.values if hasattr(X_cal, "values") else X_cal, dtype=np.float64)
            if X_c.ndim == 1:
                X_c = X_c.reshape(-1, 1) if X_arr.shape[1] == 1 else X_c.reshape(1, -1)
            y_c_raw = np.asarray(y_cal.values if hasattr(y_cal, "values") else y_cal, dtype=np.float64).ravel()
            if len(X_c) == 0 or len(y_c_raw) == 0:
                has_explicit_cal = False

        if has_explicit_cal:
            if np.all(np.isin(np.unique(y_c_raw), [0, 1])) and not np.any(y_c_raw > 1.5):
                y_c_sec = y_c_raw.astype(int)
                y_c_reg = np.where(y_c_sec == 1, eff_thresh + 0.3, eff_thresh - 0.5)
                has_cont_cal = False
            else:
                y_c_reg = y_c_raw
                y_c_sec = (y_c_raw >= eff_thresh).astype(int)
                has_cont_cal = True

            if self.fit_full_after_cal and (len(X_arr) + len(X_c)) >= 6:
                X_pool = np.vstack([X_arr, X_c])
                y_sec_pool = np.concatenate([y_sec_arr, y_c_sec])
                if has_cont_cal:
                    y_reg_pool = np.concatenate([y_reg_arr, y_c_reg])
                    raw_scores = self._compute_oof_predictions(X_pool, y_reg_pool, y_sec_pool)
                    oos_residuals = y_reg_pool - raw_scores
                    self.single_val_ = None
                    self.base_model.fit(X_pool, y_reg_pool)
                    X_fit_final, y_reg_fit_final = X_pool, y_reg_pool
                    self.n_train_samples = len(X_pool)
                    self.calibration_protocol_ = f"crossfit_cv{min(self.cv_folds, len(X_pool))}_refit_train_val"
                else:
                    # y_cal was binary-only; compute OOF on X_arr + out-of-sample on X_c, fit base_model on X_arr
                    self.single_val_ = None
                    self.base_model.fit(X_arr, y_reg_arr)
                    oof_tr = self._compute_oof_predictions(X_arr, y_reg_arr, y_sec_arr) if len(X_arr) >= 6 else self.base_model.predict(X_arr)
                    cal_preds = self.base_model.predict(X_c)
                    raw_scores = np.concatenate([oof_tr, cal_preds])
                    oos_residuals = y_reg_arr - oof_tr
                    X_fit_final, y_reg_fit_final = X_arr, y_reg_arr
                    self.calibration_protocol_ = "crossfit_oof_train_plus_val"
                y_cal_targets = y_sec_pool
                n_cal = len(raw_scores)
            else:
                if self.n_train_samples == 1:
                    self.single_val_ = float(y_reg_arr[0])
                    raw_scores = np.full(len(X_c), self.single_val_, dtype=float)
                else:
                    self.single_val_ = None
                    self.base_model.fit(X_arr, y_reg_arr)
                    raw_scores = self.base_model.predict(X_c)
                y_cal_targets = y_c_sec
                n_cal = len(X_c)
                oos_residuals = (y_c_reg - raw_scores) if has_cont_cal else (
                    y_reg_arr - self._compute_oof_predictions(X_arr, y_reg_arr, y_sec_arr)
                    if len(X_arr) >= 6 else np.array([0.08])
                )
                X_fit_final, y_reg_fit_final = X_arr, y_reg_arr
                self.calibration_protocol_ = "disjoint_val"

        elif self.validation_fraction > 0.0 and len(X_arr) >= 20:
            unique_classes, counts = np.unique(y_sec_arr, return_counts=True)
            min_class_count = int(np.min(counts)) if len(unique_classes) > 1 else 0
            stratify_target = y_sec_arr if (len(unique_classes) > 1 and min_class_count >= 2) else None

            X_tr, X_c, y_reg_tr, y_c_reg, y_sec_tr, y_c_sec = train_test_split(
                X_arr,
                y_reg_arr,
                y_sec_arr,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=stratify_target,
            )
            self.single_val_ = None
            self.base_model.fit(X_tr, y_reg_tr)
            raw_scores = self.base_model.predict(X_c)
            y_cal_targets = y_c_sec
            n_cal = len(X_c)
            oos_residuals = y_c_reg - raw_scores
            X_fit_final, y_reg_fit_final = X_tr, y_reg_tr
            self.calibration_protocol_ = "internal_val_split"

        else:
            if self.n_train_samples == 1:
                self.single_val_ = float(y_reg_arr[0])
                raw_scores = np.full(1, self.single_val_, dtype=float)
                oos_residuals = np.array([0.08])
                self.calibration_protocol_ = "single_sample_fallback"
            else:
                self.single_val_ = None
                self.base_model.fit(X_arr, y_reg_arr)
                if self.cv_folds >= 2 and len(X_arr) >= max(6, self.cv_folds):
                    raw_scores = self._compute_oof_predictions(X_arr, y_reg_arr, y_sec_arr)
                    oos_residuals = y_reg_arr - raw_scores
                    self.calibration_protocol_ = f"oof_cv{min(self.cv_folds, len(X_arr))}"
                else:
                    raw_scores = self.base_model.predict(X_arr)
                    oos_residuals = y_reg_arr - raw_scores
                    self.calibration_protocol_ = "tiny_sample_fallback"
            y_cal_targets = y_sec_arr
            n_cal = len(raw_scores)
            X_fit_final, y_reg_fit_final = X_arr, y_reg_arr

        self.base_regressor_ = self.base_model
        self.base_gbdt = self.base_model
        self.n_cal_ = int(n_cal)

        # Fit genuine data-driven uncertainty models
        if self.single_val_ is None:
            self._fit_uncertainty_models(X_fit_final, y_reg_fit_final, oos_residuals)

        # Fit Isotonic (PAVA) or Platt calibrator without mutating self.method
        if self.method == "isotonic":
            if n_cal < 30:
                warnings.warn(
                    f"Calibration dataset size N_cal={n_cal} < 30 is too small for Isotonic regression; "
                    "falling back to Platt scaling to prevent step-function overfitting.",
                    UserWarning,
                    stacklevel=2,
                )
                self.effective_method_ = "platt"
                self.fallback_used_ = True
                self.fallback_reason_ = f"n_cal={n_cal}_below_30"
            elif len(np.unique(y_cal_targets)) < 2:
                warnings.warn(
                    "Calibration dataset contains only a single class; falling back to Platt scaling.",
                    UserWarning,
                    stacklevel=2,
                )
                self.effective_method_ = "platt"
                self.fallback_used_ = True
                self.fallback_reason_ = "single_class_calibration_targets"
            else:
                self.isotonic_x, self.isotonic_y = _pava_isotonic(
                    [float(s) for s in raw_scores], [float(y) for y in y_cal_targets]
                )
                self.effective_method_ = "isotonic"
                self.fallback_used_ = False
                self.fallback_reason_ = None

        if self.effective_method_ == "platt":
            unique_classes = np.unique(y_cal_targets)
            if len(unique_classes) < 2 or len(np.unique(raw_scores)) < 2 or n_cal < 2:
                warnings.warn(
                    "Calibration data has insufficient classes (<2), zero variance, or insufficient samples (<2); "
                    "falling back to physics default (A=8.0, B=-threshold*8.0).",
                    UserWarning,
                    stacklevel=2,
                )
                self.platt_a = 8.0
                self.platt_b = -eff_thresh * self.platt_a
                self.calibrator_ = None
                self.fallback_used_ = True
                if self.fallback_reason_ is None:
                    self.fallback_reason_ = "degenerate_calibration_targets"
            else:
                try:
                    lr = LogisticRegression(
                        C=1e3,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=self.random_state,
                    )
                    lr.fit(raw_scores.reshape(-1, 1), y_cal_targets)
                    a = float(lr.coef_[0][0])
                    b = float(lr.intercept_[0])
                    if a > 0.0 and math.isfinite(a) and math.isfinite(b):
                        self.platt_a = a
                        self.platt_b = b
                        self.calibrator_ = lr
                    else:
                        warnings.warn(
                            f"Platt scaling optimization yielded non-positive slope A={a:.4f} <= 0; "
                            "falling back to physics default (A=8.0, B=-threshold*8.0).",
                            UserWarning,
                            stacklevel=2,
                        )
                        self.platt_a = 8.0
                        self.platt_b = -eff_thresh * self.platt_a
                        self.calibrator_ = None
                        self.fallback_used_ = True
                        if self.fallback_reason_ is None:
                            self.fallback_reason_ = "non_positive_platt_slope"
                except Exception as exc:
                    warnings.warn(
                        f"Platt scaling optimization failed: {exc}; falling back to physics default.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self.platt_a = 8.0
                    self.platt_b = -eff_thresh * self.platt_a
                    self.calibrator_ = None
                    self.fallback_used_ = True
                    if self.fallback_reason_ is None:
                        self.fallback_reason_ = "platt_optimizer_exception"

        return self

    def _prepare_eval_X(self, X: Any) -> np.ndarray:
        if X is None:
            return np.empty((0, self.n_features_in_ or 0), dtype=np.float64)
        X_arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=np.float64)
        if X_arr.size == 0 or len(X_arr) == 0:
            return np.empty((0, self.n_features_in_ or 0), dtype=np.float64)
        if X_arr.ndim == 1:
            if self.n_features_in_ is not None and len(X_arr) == self.n_features_in_:
                X_arr = X_arr.reshape(1, -1)
            else:
                X_arr = X_arr.reshape(-1, 1)
        return X_arr

    def predict_nadir(self, X: Any) -> np.ndarray:
        """Return continuous GBDT frequency nadir predictions in Hz."""
        X_arr = self._prepare_eval_X(X)
        if len(X_arr) == 0:
            return np.empty(0, dtype=np.float64)
        if self.single_val_ is not None:
            return np.full(len(X_arr), self.single_val_, dtype=np.float64)
        return np.asarray(self.base_model.predict(X_arr), dtype=np.float64).ravel()

    def calibrate_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map 1D raw nadir predictions to [P(insecure), P(secure)] probabilities."""
        scores = np.asarray(raw_scores, dtype=np.float64).ravel()
        if len(scores) == 0:
            return np.empty((0, 2), dtype=np.float64)
        probs = np.empty((len(scores), 2), dtype=np.float64)
        for i, s in enumerate(scores):
            p_sec = max(0.0, min(1.0, self._calibrate_single(float(s))))
            probs[i, 0] = 1.0 - p_sec
            probs[i, 1] = p_sec
        return probs

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict [P(insecure), P(secure)] class probabilities."""
        X_arr = self._prepare_eval_X(X)
        if len(X_arr) == 0:
            return np.empty((0, 2), dtype=np.float64)
        raw_scores = self.predict_nadir(X_arr)
        return self.calibrate_scores(raw_scores)

    def predict(
        self,
        X: Any,
        threshold: float = 0.5,
        proba_thresh: Optional[float] = None,
    ) -> np.ndarray:
        """Predict binary security labels (1 = secure, 0 = insecure)."""
        thresh = proba_thresh if proba_thresh is not None else threshold
        probs = self.predict_proba(X)
        if len(probs) == 0:
            return np.empty(0, dtype=int)
        p_sec = probs[:, 1]
        return np.where(p_sec >= thresh, 1, 0).astype(int)

    def predict_nadir_intervals(
        self,
        X: Any,
        confidence_level: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (y_hat_hz, y_low_hz, y_high_hz) using quantile GBDT + OOD distance dispersion."""
        X_arr = self._prepare_eval_X(X)
        if len(X_arr) == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float)

        y_hat = self.predict_nadir(X_arr)
        alpha_clamped = min(0.999, max(0.50, float(confidence_level)))
        z_crit = float(norm.ppf(0.5 + 0.5 * alpha_clamped))
        z_scale = z_crit / float(norm.ppf(0.975))

        if self.quantile_low_model_ is not None and self.quantile_high_model_ is not None:
            q_low = np.asarray(self.quantile_low_model_.predict(X_arr), dtype=float).ravel()
            q_high = np.asarray(self.quantile_high_model_.predict(X_arr), dtype=float).ravel()
            half_low = np.maximum(0.5 * self.conformal_q95_, y_hat - q_low)
            half_high = np.maximum(0.5 * self.conformal_q95_, q_high - y_hat)
        else:
            half_low = np.full(len(X_arr), self.conformal_q95_, dtype=float)
            half_high = np.full(len(X_arr), self.conformal_q95_, dtype=float)

        # Tree-stage dispersion + standardized feature-space shift for OOD / noisy inputs
        ood_excess = np.zeros(len(X_arr), dtype=float)
        if self.train_mean_ is not None and self.train_std_ is not None and X_arr.shape[1] == len(self.train_mean_):
            z_feat = (X_arr - self.train_mean_) / self.train_std_
            rms_d = np.sqrt(np.mean(z_feat ** 2, axis=1))
            ood_excess = np.maximum(0.0, rms_d - self.train_dist_p75_)

        stage_std = np.zeros(len(X_arr), dtype=float)
        if self.single_val_ is None and hasattr(self.base_model, "staged_predict"):
            staged = list(self.base_model.staged_predict(X_arr))
            if len(staged) >= 4:
                tail_stages = np.vstack(staged[len(staged) // 2:])
                stage_std = np.std(tail_stages, axis=0)

        epistemic_hz = self.residual_std_ * 0.65 * ood_excess + 1.5 * stage_std
        y_low = y_hat - z_scale * (half_low + epistemic_hz)
        y_high = y_hat + z_scale * (half_high + epistemic_hz)
        return y_hat, y_low, y_high

    def predict_with_intervals(
        self, X: Any, confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p_secure, low_bounds, high_bounds) from quantile GBDT + OOD nadir intervals."""
        X_arr = self._prepare_eval_X(X)
        if len(X_arr) == 0:
            return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float)

        y_hat, y_low, y_high = self.predict_nadir_intervals(X_arr, confidence_level=confidence_level)
        p_sec = np.array([self._calibrate_single(float(s)) for s in y_hat], dtype=float)
        p_low_raw = np.array([self._calibrate_single(float(s)) for s in y_low], dtype=float)
        p_high_raw = np.array([self._calibrate_single(float(s)) for s in y_high], dtype=float)

        # Ensure monotonic ordering and reflect feature-space OOD shift even in saturated tails
        ood_pull = np.zeros(len(X_arr), dtype=float)
        if self.train_mean_ is not None and self.train_std_ is not None and X_arr.shape[1] == len(self.train_mean_):
            alpha_clamped = min(0.999, max(0.50, float(confidence_level)))
            z_scale = float(norm.ppf(0.5 + 0.5 * alpha_clamped) / norm.ppf(0.975))
            rms_d = np.sqrt(np.mean(((X_arr - self.train_mean_) / self.train_std_) ** 2, axis=1))
            ood_excess = np.maximum(0.0, rms_d - self.train_dist_p75_)
            ood_pull = np.minimum(0.35, 0.03 * z_scale * ood_excess)

        low_bounds = np.clip(np.minimum(p_sec, p_low_raw) - ood_pull * p_sec, 0.0, 1.0)
        high_bounds = np.clip(np.maximum(p_sec, p_high_raw) + ood_pull * (1.0 - p_sec), 0.0, 1.0)
        low_bounds = np.minimum(low_bounds, p_sec)
        high_bounds = np.maximum(high_bounds, p_sec)

        return p_sec, low_bounds, high_bounds

    def predict_confidence(
        self, X: Any, confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (p_secure, confidence_score, interval_width)."""
        p_sec, low_bounds, high_bounds = self.predict_with_intervals(
            X, confidence_level=confidence_level
        )
        conf_score = np.abs(p_sec - 0.5) * 2.0
        interval_width = high_bounds - low_bounds
        return p_sec, conf_score, interval_width

    def evaluate(
        self,
        X: Any,
        y: Any,
        proba_thresh: float = 0.5,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate classification and calibration metrics on (X, y)."""
        thresh = threshold if threshold is not None else proba_thresh
        y_arr = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).ravel()
        eff_t = getattr(self, "_effective_threshold_", self.threshold)
        if np.any(y_arr > 1.5):
            y_true = np.where(y_arr >= eff_t, 1, 0).astype(int)
        else:
            y_true = np.where(y_arr >= 0.5, 1, 0).astype(int)

        probs = self.predict_proba(X)
        p_sec = probs[:, 1] if len(probs) > 0 else np.array([], dtype=float)
        y_pred = np.where(p_sec >= thresh, 1, 0).astype(int)
        cls_m = compute_binary_security_metrics(y_true, y_pred, p_secure=p_sec)
        if len(y_true) > 0:
            cal_m = compute_calibration_metrics(y_true, p_sec, n_bins=10, threshold=eff_t)
            cls_m["brier_score"] = cal_m["brier_score"]
            cls_m["ece"] = cal_m["ece"]
            cls_m["mce"] = cal_m["mce"]
        return cls_m

    def evaluate_threshold_sweep(
        self,
        X: Any,
        y: Any,
        thresholds: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Evaluate confusion counts and FNR_insecure across probability thresholds."""
        y_arr = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).ravel()
        eff_t = getattr(self, "_effective_threshold_", self.threshold)
        y_true = np.where(y_arr >= eff_t, 1, 0).astype(int) if np.any(y_arr > 1.5) else np.where(y_arr >= 0.5, 1, 0).astype(int)
        probs = self.predict_proba(X)
        p_sec = probs[:, 1] if len(probs) > 0 else np.array([], dtype=float)
        return evaluate_probability_threshold_sweep(y_true, p_sec, thresholds=thresholds)

    def compute_calibration_metrics(self, y_true: Any, y_prob: Any, n_bins: int = 10) -> Dict[str, Any]:
        eff_t = getattr(self, "_effective_threshold_", self.threshold)
        return compute_calibration_metrics(y_true, y_prob, n_bins=n_bins, threshold=eff_t)
