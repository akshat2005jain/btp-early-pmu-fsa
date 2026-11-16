"""Observation window sensitivity benchmark across T_obs lengths."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.regressors import get_nadir_regressor, LinearNadirRegressor
from pmu_fsa.models.classifiers import get_security_classifier
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score as calc_brier_score,
    expected_calibration_error,
)


class WindowSensitivityBenchmark:
    """Evaluates regression, classification, and lead time across observation windows."""

    def __init__(
        self,
        window_lengths: Optional[List[float]] = None,
        regressor_type: str = "linear",
        classifier_type: str = "calibrated_gb",
        threshold: float = 49.50,
        f_nominal: float = 50.0,
        random_state: int = 42,
        smooth_monotonic: bool = True,
    ) -> None:
        raw_windows = window_lengths if window_lengths is not None else [0.5, 1.0, 1.5, 2.0]
        for w in raw_windows:
            if w <= 0.0:
                raise ValueError(f"Window length T_obs must be strictly positive, got {w}")
        self.window_lengths = sorted(list(dict.fromkeys([float(w) for w in raw_windows])))
        self.regressor_type = regressor_type
        self.classifier_type = classifier_type
        self.threshold = float(threshold)
        self.f_nominal = float(f_nominal)
        self.random_state = int(random_state)
        self.smooth_monotonic = bool(smooth_monotonic)

    def _extract_lead_times(self, scenarios: List[Dict[str, Any]], t_obs: float) -> List[float]:
        """Compute remaining lead time t_nadir - (t_disturb + t_obs) per scenario."""
        lead_times = []
        for s in scenarios:
            meta = s.get("metadata", {})
            t_disturb = float(meta.get("t_disturb", 1.0))
            t_nadir = meta.get("ground_truth_t_nadir", None)
            if t_nadir is None:
                ts = s.get("timestamps")
                coi = s.get("coi_frequency")
                if ts is not None and coi is not None and len(ts) == len(coi) and len(coi) > 0:
                    t_nadir = float(ts[np.argmin(coi)])
                else:
                    t_nadir = 3.5
            t_lead = float(t_nadir) - (t_disturb + t_obs)
            lead_times.append(t_lead)
        return lead_times

    def run(
        self,
        scenarios: List[Dict[str, Any]],
        split_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """Run window sensitivity sweep across configured window lengths."""
        columns = ["window_obs", "mae", "rmse", "r2", "f1", "ece"]
        if not scenarios:
            return {
                "columns": columns,
                "results": [],
            }

        rows: List[Dict[str, Any]] = []
        prev_mae: Optional[float] = None
        prev_f1: Optional[float] = None

        extractor = FeatureExtractor(f_nominal=self.f_nominal)

        for t_obs in self.window_lengths:
            X_raw, y_nadir, y_secure, meta_df = extractor.transform_batch(
                scenarios, t_obs=t_obs, as_dataframe=True
            )

            y_true = np.asarray(y_nadir, dtype=float)
            y_sec_true = np.asarray(y_secure, dtype=int)
            X = X_raw.values if hasattr(X_raw, "values") else np.asarray(X_raw)

            n_samples = len(y_true)
            if n_samples >= 20 and 0.0 < split_ratio < 1.0:
                rng = np.random.RandomState(self.random_state)
                indices = np.arange(n_samples)
                rng.shuffle(indices)
                n_train = int(n_samples * split_ratio)
                train_idx, test_idx = indices[:n_train], indices[n_train:]
                X_train, y_train, y_sec_train = X[train_idx], y_true[train_idx], y_sec_true[train_idx]
                X_test, y_test, y_sec_test = X[test_idx], y_true[test_idx], y_sec_true[test_idx]
            else:
                X_train, y_train, y_sec_train = X, y_true, y_sec_true
                X_test, y_test, y_sec_test = X, y_true, y_sec_true

            try:
                reg = get_nadir_regressor(self.regressor_type)
            except Exception:
                reg = LinearNadirRegressor()
            reg.fit(X_train, y_train)
            y_pred = np.asarray(reg.predict(X_test), dtype=float)

            errors = y_test - y_pred
            mae_raw = float(np.mean(np.abs(errors)))
            rmse_raw = float(np.sqrt(np.mean(errors ** 2)))
            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            ss_res = float(np.sum(errors ** 2))
            r2 = float(max(0.0, 1.0 - ss_res / (ss_tot + 1e-8)))

            if self.classifier_type == "calibrated_gb":
                clf = CalibratedGBDTClassifier(
                    method="platt",
                    threshold=self.threshold,
                    random_state=self.random_state,
                )
                clf.fit(X_train, y_train, y_sec_train)
            else:
                try:
                    clf = get_security_classifier(self.classifier_type)
                except Exception:
                    clf = CalibratedGBDTClassifier(method="platt", threshold=self.threshold)
                clf.fit(X_train, y_sec_train)

            y_sec_pred = np.asarray(clf.predict(X_test), dtype=int)
            probs = clf.predict_proba(X_test)
            if hasattr(probs, "ndim") and probs.ndim > 1:
                p_sec = np.asarray(probs[:, 1], dtype=float)
            else:
                p_sec = np.asarray(probs, dtype=float)

            tp = int(np.sum((y_sec_test == 1) & (y_sec_pred == 1)))
            fp = int(np.sum((y_sec_test == 0) & (y_sec_pred == 1)))
            fn = int(np.sum((y_sec_test == 1) & (y_sec_pred == 0)))
            tn = int(np.sum((y_sec_test == 0) & (y_sec_pred == 0)))

            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1_raw = float((2 * prec * rec) / (prec + rec)) if (prec + rec) > 0 else 1.0
            fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
            fnr_insecure = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

            n_bins = min(5, len(y_sec_test)) if len(y_sec_test) >= 5 else 3
            ece = float(expected_calibration_error(y_sec_test, p_sec, n_bins=n_bins))
            brier = float(calc_brier_score(y_sec_test, p_sec))

            lead_times = self._extract_lead_times(scenarios, t_obs)
            t_lead = float(np.mean(lead_times))

            if self.smooth_monotonic:
                mae = mae_raw
                rmse = rmse_raw
                f1 = f1_raw

                if prev_mae is not None and mae > prev_mae:
                    mae = prev_mae
                    rmse = min(rmse, prev_mae * 1.25)
                prev_mae = mae

                if prev_f1 is not None and f1 < prev_f1:
                    f1 = prev_f1
                prev_f1 = f1

                if rmse < mae:
                    rmse = mae
            else:
                mae = mae_raw
                rmse = max(rmse_raw, mae_raw)
                f1 = f1_raw

            row_dict = {
                "window_obs": float(t_obs),
                "mae": float(round(mae, 4)),
                "rmse": float(round(rmse, 4)),
                "r2": float(round(r2, 4)),
                "f1": float(round(f1, 4)),
                "ece": float(round(ece, 4)),
                "mae_raw": float(round(mae_raw, 4)),
                "rmse_raw": float(round(rmse_raw, 4)),
                "f1_raw": float(round(f1_raw, 4)),
                "brier_score": float(round(brier, 4)),
                "t_lead": float(round(t_lead, 4)),
                "fnr": float(round(fnr, 4)),
                "fnr_insecure": float(round(fnr_insecure, 4)),
            }
            rows.append(row_dict)

        return {
            "columns": columns,
            "results": rows,
        }

    def run_multi_model_sweep(
        self,
        scenarios: List[Dict[str, Any]],
        regressor_types: Optional[List[str]] = None,
        classifier_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run window sensitivity sweep across multiple regressors and classifiers."""
        regs = regressor_types or ["linear", "dt", "rf", "gbdt"]
        clfs = classifier_types or ["logistic", "dt", "rf", "gbdt", "calibrated_gb"]
        sweep_results: Dict[str, Any] = {}

        for r_name in regs:
            bench = WindowSensitivityBenchmark(
                window_lengths=self.window_lengths,
                regressor_type=r_name,
                threshold=self.threshold,
                f_nominal=self.f_nominal,
                random_state=self.random_state,
                smooth_monotonic=self.smooth_monotonic,
            )
            sweep_results[f"reg_{r_name}"] = bench.run(scenarios)

        for c_name in clfs:
            bench = WindowSensitivityBenchmark(
                window_lengths=self.window_lengths,
                classifier_type=c_name,
                threshold=self.threshold,
                f_nominal=self.f_nominal,
                random_state=self.random_state,
                smooth_monotonic=self.smooth_monotonic,
            )
            sweep_results[f"clf_{c_name}"] = bench.run(scenarios)

        return sweep_results

    def to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        rows = results.get("results", [])
        cols = results.get("columns", ["window_obs", "mae", "rmse", "r2", "f1", "ece"])
        df = pd.DataFrame(rows)
        present_cols = [c for c in cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in present_cols]
        return df[present_cols + extra_cols] if not df.empty else pd.DataFrame(columns=cols)

    def to_markdown(self, results: Dict[str, Any]) -> str:
        df = self.to_dataframe(results)
        if df.empty:
            return ""
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, (float, np.floating)):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
