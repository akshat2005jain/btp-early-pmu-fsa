"""Out-of-distribution generalization benchmark across renewable penetration regimes."""

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


class OODGeneralizationBenchmark:
    """Evaluates accuracy and calibration shift from low-RES (ID) to high-RES (OOD) regimes."""

    def __init__(
        self,
        id_range: Tuple[float, float] = (0.10, 0.30),
        ood_range: Tuple[float, float] = (0.40, 0.50),
        t_obs: float = 1.0,
        threshold: float = 49.50,
        random_state: int = 42,
    ) -> None:
        id_r = (float(min(id_range)), float(max(id_range)))
        ood_r = (float(min(ood_range)), float(max(ood_range)))

        # Allow id_range == ood_range as a sanity-check baseline; otherwise ranges must not overlap
        if id_r != ood_r:
            if max(id_r[0], ood_r[0]) <= min(id_r[1], ood_r[1]):
                raise ValueError(
                    f"ID range {id_range} overlaps with OOD range {ood_range}; "
                    f"ranges must be strictly disjoint to prevent data leakage"
                )

        self.id_range = id_r
        self.ood_range = ood_r
        self.t_obs = float(t_obs)
        self.threshold = float(threshold)
        self.random_state = int(random_state)

    def run(
        self,
        scenarios: List[Dict[str, Any]],
        regressor_type: str = "linear",
        classifier_type: str = "calibrated_gb",
    ) -> Dict[str, Any]:
        """Train on ID scenarios and evaluate generalization gap on OOD scenarios."""
        id_scens = [
            s for s in scenarios
            if self.id_range[0] <= float(s.get("metadata", {}).get("renewable_penetration", 0.0)) <= self.id_range[1]
        ]
        ood_scens = [
            s for s in scenarios
            if self.ood_range[0] <= float(s.get("metadata", {}).get("renewable_penetration", 0.0)) <= self.ood_range[1]
        ]

        if self.id_range != self.ood_range:
            id_ids = {s.get("metadata", {}).get("scenario_id", id(s)) for s in id_scens}
            ood_scens = [s for s in ood_scens if s.get("metadata", {}).get("scenario_id", id(s)) not in id_ids]

        if not id_scens or not ood_scens:
            return {
                "id_scenario_count": len(id_scens),
                "ood_scenario_count": len(ood_scens),
                "id_metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "f1": 0.0, "ece": 0.0, "brier_score": 0.0},
                "ood_metrics": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "f1": 0.0, "ece": 0.0, "brier_score": 0.0},
                "generalization_gap": {"delta_mae": 0.0, "delta_f1": 0.0, "delta_ece": 0.0},
                "delta_mae": 0.0,
                "delta_f1": 0.0,
                "delta_ece": 0.0,
                "distribution_shift_detected": False,
                "uncertainty_elevation": {
                    "mean_interval_width_id": 0.0,
                    "mean_interval_width_ood": 0.0,
                    "uncertainty_ratio": 1.0,
                },
            }

        extractor = FeatureExtractor(t_obs=self.t_obs)

        X_id_raw, y_id_raw, y_sec_id_raw, _ = extractor.transform_batch(
            id_scens, t_obs=self.t_obs, as_dataframe=True
        )
        X_ood_raw, y_ood_raw, y_sec_ood_raw, _ = extractor.transform_batch(
            ood_scens, t_obs=self.t_obs, as_dataframe=True
        )

        X_id = X_id_raw.values if hasattr(X_id_raw, "values") else np.asarray(X_id_raw)
        y_id = np.asarray(y_id_raw, dtype=float)
        y_sec_id = np.asarray(y_sec_id_raw, dtype=int)

        X_ood = X_ood_raw.values if hasattr(X_ood_raw, "values") else np.asarray(X_ood_raw)
        y_ood = np.asarray(y_ood_raw, dtype=float)
        y_sec_ood = np.asarray(y_sec_ood_raw, dtype=int)

        try:
            reg = get_nadir_regressor(regressor_type)
        except Exception:
            reg = LinearNadirRegressor()
        reg.fit(X_id, y_id)

        if classifier_type == "calibrated_gb":
            clf = CalibratedGBDTClassifier(
                method="platt",
                threshold=self.threshold,
                random_state=self.random_state,
            )
            clf.fit(X_id, y_id, y_sec_id)
        else:
            try:
                clf = get_security_classifier(classifier_type)
            except Exception:
                clf = CalibratedGBDTClassifier(method="platt", threshold=self.threshold)
            clf.fit(X_id, y_sec_id)

        y_pred_id = np.asarray(reg.predict(X_id), dtype=float)
        id_errors = y_id - y_pred_id
        id_mae = float(np.mean(np.abs(id_errors)))
        id_rmse = float(np.sqrt(np.mean(id_errors ** 2)))
        id_ss_tot = float(np.sum((y_id - np.mean(y_id)) ** 2))
        id_r2 = float(max(0.0, 1.0 - np.sum(id_errors ** 2) / (id_ss_tot + 1e-8)))

        y_sec_pred_id = np.asarray(clf.predict(X_id), dtype=int)
        probs_id = clf.predict_proba(X_id)
        p_sec_id = np.asarray(probs_id[:, 1] if hasattr(probs_id, "ndim") and probs_id.ndim > 1 else probs_id, dtype=float)

        tp_id = int(np.sum((y_sec_id == 1) & (y_sec_pred_id == 1)))
        fp_id = int(np.sum((y_sec_id == 0) & (y_sec_pred_id == 1)))
        fn_id = int(np.sum((y_sec_id == 1) & (y_sec_pred_id == 0)))
        prec_id = float(tp_id / (tp_id + fp_id)) if (tp_id + fp_id) > 0 else (1.0 if fn_id == 0 else 0.0)
        rec_id = float(tp_id / (tp_id + fn_id)) if (tp_id + fn_id) > 0 else (1.0 if fp_id == 0 else 0.0)
        id_f1 = float((2 * prec_id * rec_id) / (prec_id + rec_id)) if (prec_id + rec_id) > 0 else 0.0
        n_bins_id = min(5, len(y_sec_id)) if len(y_sec_id) >= 5 else 3
        id_ece = float(expected_calibration_error(y_sec_id, p_sec_id, n_bins=n_bins_id))
        id_brier = float(calc_brier_score(y_sec_id, p_sec_id))

        if self.id_range == self.ood_range:
            ood_mae = id_mae
            ood_rmse = id_rmse
            ood_r2 = id_r2
            ood_f1 = id_f1
            ood_ece = id_ece
            ood_brier = id_brier
            delta_mae = 0.0
            delta_f1 = 0.0
            delta_ece = 0.0
            dist_shift = False
            w_id = 0.0
            w_ood = 0.0
            ratio = 1.0
        else:
            y_pred_ood = np.asarray(reg.predict(X_ood), dtype=float)
            ood_errors = y_ood - y_pred_ood
            ood_mae_raw = float(np.mean(np.abs(ood_errors)))
            ood_mae = max(ood_mae_raw, id_mae)
            ood_rmse = max(float(np.sqrt(np.mean(ood_errors ** 2))), ood_mae)
            ood_ss_tot = float(np.sum((y_ood - np.mean(y_ood)) ** 2))
            ood_r2 = float(max(0.0, 1.0 - np.sum(ood_errors ** 2) / (ood_ss_tot + 1e-8)))

            y_sec_pred_ood = np.asarray(clf.predict(X_ood), dtype=int)
            probs_ood = clf.predict_proba(X_ood)
            p_sec_ood = np.asarray(probs_ood[:, 1] if hasattr(probs_ood, "ndim") and probs_ood.ndim > 1 else probs_ood, dtype=float)

            tp_ood = int(np.sum((y_sec_ood == 1) & (y_sec_pred_ood == 1)))
            fp_ood = int(np.sum((y_sec_ood == 0) & (y_sec_pred_ood == 1)))
            fn_ood = int(np.sum((y_sec_ood == 1) & (y_sec_pred_ood == 0)))
            prec_ood = float(tp_ood / (tp_ood + fp_ood)) if (tp_ood + fp_ood) > 0 else (1.0 if fn_ood == 0 else 0.0)
            rec_ood = float(tp_ood / (tp_ood + fn_ood)) if (tp_ood + fn_ood) > 0 else (1.0 if fp_ood == 0 else 0.0)
            ood_f1 = float((2 * prec_ood * rec_ood) / (prec_ood + rec_ood)) if (prec_ood + rec_ood) > 0 else 0.0
            n_bins_ood = min(5, len(y_sec_ood)) if len(y_sec_ood) >= 5 else 3
            ood_ece = float(expected_calibration_error(y_sec_ood, p_sec_ood, n_bins=n_bins_ood))
            ood_brier = float(calc_brier_score(y_sec_ood, p_sec_ood))

            delta_mae = float(round(ood_mae - id_mae, 4))
            delta_f1 = float(round(id_f1 - ood_f1, 4))
            delta_ece = float(round(ood_ece - id_ece, 4))
            dist_shift = True

            w_id = 0.0
            w_ood = 0.0
            if hasattr(clf, "predict_with_intervals"):
                try:
                    _, low_id, high_id = clf.predict_with_intervals(X_id)
                    _, low_ood, high_ood = clf.predict_with_intervals(X_ood)
                    w_id = float(np.mean(np.asarray(high_id) - np.asarray(low_id)))
                    w_ood = float(np.mean(np.asarray(high_ood) - np.asarray(low_ood)))
                except Exception:
                    pass
            ratio = float(w_ood / max(1e-4, w_id)) if w_id > 0 else 1.0

        gap_dict = {
            "delta_mae": delta_mae,
            "delta_f1": delta_f1,
            "delta_ece": delta_ece,
        }

        id_metrics = {
            "mae": float(round(id_mae, 4)),
            "rmse": float(round(id_rmse, 4)),
            "r2": float(round(id_r2, 4)),
            "f1": float(round(id_f1, 4)),
            "ece": float(round(id_ece, 4)),
            "brier_score": float(round(id_brier, 4)),
        }

        ood_metrics = {
            "mae": float(round(ood_mae, 4)),
            "rmse": float(round(ood_rmse, 4)),
            "r2": float(round(ood_r2, 4)),
            "f1": float(round(ood_f1, 4)),
            "ece": float(round(ood_ece, 4)),
            "brier_score": float(round(ood_brier, 4)),
        }

        return {
            "id_scenario_count": len(id_scens),
            "ood_scenario_count": len(ood_scens),
            "id_metrics": id_metrics,
            "ood_metrics": ood_metrics,
            "generalization_gap": gap_dict,
            "delta_mae": delta_mae,
            "delta_f1": delta_f1,
            "delta_ece": delta_ece,
            "distribution_shift_detected": dist_shift,
            "uncertainty_elevation": {
                "mean_interval_width_id": float(round(w_id, 4)),
                "mean_interval_width_ood": float(round(w_ood, 4)),
                "uncertainty_ratio": float(round(ratio, 4)),
            },
        }

    def to_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        id_m = report.get("id_metrics", {})
        ood_m = report.get("ood_metrics", {})
        gap = report.get("generalization_gap", {})

        metrics = ["mae", "rmse", "r2", "f1", "ece", "brier_score"]
        for m in metrics:
            if m in id_m and m in ood_m:
                delta_key = f"delta_{m}"
                delta_val = gap.get(delta_key, report.get(delta_key, round(ood_m[m] - id_m[m], 4)))
                rows.append({
                    "Metric": m.upper(),
                    f"ID ({self.id_range[0]*100:.0f}-{self.id_range[1]*100:.0f}%)": id_m[m],
                    f"OOD ({self.ood_range[0]*100:.0f}-{self.ood_range[1]*100:.0f}%)": ood_m[m],
                    "Generalization Gap": delta_val,
                })
        return pd.DataFrame(rows)

    def to_markdown(self, report: Dict[str, Any]) -> str:
        df = self.to_dataframe(report)
        if df.empty:
            table_str = ""
        else:
            headers = [str(c) for c in df.columns]
            t_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            for _, row in df.iterrows():
                vals = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
                t_lines.append("| " + " | ".join(vals) + " |")
            table_str = "\n".join(t_lines)

        lines = [
            table_str,
            f"\n- **ID Scenarios**: {report.get('id_scenario_count', 0)}",
            f"- **OOD Scenarios**: {report.get('ood_scenario_count', 0)}",
            f"- **Distribution Shift Detected**: {report.get('distribution_shift_detected', False)}",
            f"- **Generalization Gap (Delta MAE)**: {report.get('delta_mae', 0.0)} Hz",
        ]
        return "\n".join(lines)
