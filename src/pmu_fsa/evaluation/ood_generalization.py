"""Out-of-distribution generalization benchmark across RES, load, and combined regimes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pmu_fsa.evaluation.sensitivity import split_scenarios_by_partition
from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score as calc_brier_score,
    expected_calibration_error,
    maximum_calibration_error,
)
from pmu_fsa.models.classifiers import compute_binary_security_metrics, get_security_classifier
from pmu_fsa.models.regressors import LinearNadirRegressor, get_nadir_regressor


class OODGeneralizationBenchmark:
    """Evaluates held-out ID test vs OOD-RES, OOD-Load, and OOD-Combined regimes without clamping."""

    def __init__(
        self,
        id_range: Tuple[float, float] = (0.10, 0.30),
        ood_range: Tuple[float, float] = (0.40, 0.50),
        id_load_range: Tuple[float, float] = (0.88, 1.12),
        t_obs: float = 1.0,
        threshold: float = 49.50,
        f_nominal: float = 50.0,
        random_state: int = 42,
    ) -> None:
        id_r = (float(min(id_range)), float(max(id_range)))
        ood_r = (float(min(ood_range)), float(max(ood_range)))

        if id_r != ood_r:
            if max(id_r[0], ood_r[0]) <= min(id_r[1], ood_r[1]):
                raise ValueError(
                    f"ID range {id_range} overlaps with OOD range {ood_range}; "
                    f"ranges must be strictly disjoint to prevent data leakage"
                )

        self.id_range = id_r
        self.ood_range = ood_r
        self.id_load_range = (float(min(id_load_range)), float(max(id_load_range)))
        self.t_obs = float(t_obs)
        self.threshold = float(threshold)
        self.f_nominal = float(f_nominal)
        self.random_state = int(random_state)

    def _classify_scenario_regime(self, scen: Dict[str, Any]) -> str:
        meta = scen.get("metadata", {})
        ood_type = str(meta.get("ood_type", "")).strip().lower()
        if ood_type in ("id", "ood_res", "ood_load", "ood_combined"):
            return ood_type

        pen = float(meta.get("renewable_penetration", meta.get("p_ren_requested", 0.0)))
        ld = float(meta.get("load_scale", 1.0))
        in_id_res = (self.id_range[0] - 1e-6) <= pen <= (self.id_range[1] + 0.02 + 1e-6)
        in_ood_res = (self.ood_range[0] - 1e-6) <= pen <= (self.ood_range[1] + 0.10)
        in_id_load = self.id_load_range[0] <= ld <= self.id_load_range[1]

        if in_id_res and in_id_load:
            return "id"
        if in_ood_res and in_id_load:
            return "ood_res"
        if in_id_res and not in_id_load:
            return "ood_load"
        if in_ood_res and not in_id_load:
            return "ood_combined"
        if str(meta.get("partition", "")) == "ood_test":
            return "ood_res"
        return "other"

    def _evaluate_subset(
        self,
        label: str,
        X_sub: np.ndarray,
        y_sub: np.ndarray,
        ys_sub: np.ndarray,
        reg: Any,
        clf: Any,
    ) -> Dict[str, Any]:
        if len(X_sub) == 0:
            return {
                "regime": label,
                "n_scenarios": 0,
                "mae": 0.0,
                "rmse": 0.0,
                "bias": 0.0,
                "median_ae": 0.0,
                "p95_ae": 0.0,
                "r2": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.5,
                "ece": 0.0,
                "mce": 0.0,
                "brier_score": 0.0,
                "fnr_insecure": 0.0,
                "missed_insecure_count": 0,
                "fpr_secure": 0.0,
                "mean_prob_interval_width": 0.0,
                "mean_nadir_interval_hz": 0.0,
                "error_quantiles_hz": {"p05": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0},
            }

        reg_m = reg.evaluate(X_sub, y_sub)
        y_pred = np.asarray(reg.predict(X_sub), dtype=float)
        signed_err = y_pred - y_sub

        probs = clf.predict_proba(X_sub)
        p_sec = np.asarray(probs[:, 1] if hasattr(probs, "ndim") and probs.ndim > 1 else probs, dtype=float)
        y_sec_pred = (p_sec >= 0.5).astype(int)
        cls_m = compute_binary_security_metrics(ys_sub, y_sec_pred, p_secure=p_sec)

        ece = float(expected_calibration_error(ys_sub, p_sec, n_bins=10, threshold=self.threshold))
        mce = float(maximum_calibration_error(ys_sub, p_sec, n_bins=10, threshold=self.threshold))
        brier = float(calc_brier_score(ys_sub, p_sec, threshold=self.threshold))

        w_prob = 0.0
        w_hz = 0.0
        if hasattr(clf, "predict_with_intervals"):
            try:
                _, low_p, high_p = clf.predict_with_intervals(X_sub)
                w_prob = float(np.mean(np.asarray(high_p) - np.asarray(low_p)))
            except Exception:
                w_prob = 0.0
        if hasattr(clf, "predict_nadir_intervals"):
            try:
                _, low_hz, high_hz = clf.predict_nadir_intervals(X_sub)
                w_hz = float(np.mean(np.asarray(high_hz) - np.asarray(low_hz)))
            except Exception:
                w_hz = 0.0

        q_vals = np.percentile(signed_err, [5.0, 25.0, 50.0, 75.0, 95.0])

        return {
            "regime": label,
            "n_scenarios": int(len(X_sub)),
            "mae": round(float(reg_m["mae"]), 4),
            "rmse": round(float(reg_m["rmse"]), 4),
            "bias": round(float(reg_m["bias"]), 4),
            "median_ae": round(float(reg_m["median_ae"]), 4),
            "p95_ae": round(float(reg_m["p95_ae"]), 4),
            "r2": round(float(reg_m["r2"]), 4),
            "precision": round(float(cls_m["precision"]), 4),
            "recall": round(float(cls_m["recall"]), 4),
            "f1": round(float(cls_m["f1"]), 4),
            "pr_auc": round(float(cls_m["pr_auc"]), 4),
            "roc_auc": round(float(cls_m["roc_auc"]), 4),
            "ece": round(ece, 4),
            "mce": round(mce, 4),
            "brier_score": round(brier, 4),
            "fnr_insecure": round(float(cls_m["fnr_insecure"]), 4),
            "missed_insecure_count": int(cls_m["missed_insecure_count"]),
            "fpr_secure": round(float(cls_m["fpr_secure"]), 4),
            "mean_prob_interval_width": round(w_prob, 4),
            "mean_nadir_interval_hz": round(w_hz, 4),
            "error_quantiles_hz": {
                "p05": round(float(q_vals[0]), 4),
                "p25": round(float(q_vals[1]), 4),
                "p50": round(float(q_vals[2]), 4),
                "p75": round(float(q_vals[3]), 4),
                "p95": round(float(q_vals[4]), 4),
            },
        }

    def run(
        self,
        scenarios: List[Dict[str, Any]],
        regressor_type: str = "gbdt",
        classifier_type: str = "calibrated_gb",
        regressor: Optional[Any] = None,
        classifier: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train on ID train (+ calibrate on ID val) and evaluate strictly on held-out ID test vs OOD."""
        if self.id_range == self.ood_range:
            id_scens = [
                s for s in scenarios
                if self.id_range[0] <= float(s.get("metadata", {}).get("renewable_penetration", 0.0)) <= self.id_range[1]
            ]
            ood_all_scens = list(id_scens)
            ood_res_scens, ood_load_scens, ood_comb_scens = [], [], []
        else:
            id_scens = []
            ood_res_scens = []
            ood_load_scens = []
            ood_comb_scens = []
            for s in scenarios:
                reg_tag = self._classify_scenario_regime(s)
                if reg_tag == "id":
                    id_scens.append(s)
                elif reg_tag == "ood_res":
                    ood_res_scens.append(s)
                elif reg_tag == "ood_load":
                    ood_load_scens.append(s)
                elif reg_tag == "ood_combined":
                    ood_comb_scens.append(s)
            id_ids = {s.get("metadata", {}).get("scenario_id", id(s)) for s in id_scens}
            ood_res_scens = [s for s in ood_res_scens if s.get("metadata", {}).get("scenario_id", id(s)) not in id_ids]
            ood_load_scens = [s for s in ood_load_scens if s.get("metadata", {}).get("scenario_id", id(s)) not in id_ids]
            ood_comb_scens = [s for s in ood_comb_scens if s.get("metadata", {}).get("scenario_id", id(s)) not in id_ids]
            ood_all_scens = ood_res_scens + ood_load_scens + ood_comb_scens

        if not id_scens or not ood_all_scens:
            empty_m = {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "f1": 0.0, "ece": 0.0, "brier_score": 0.0, "fnr_insecure": 0.0}
            return {
                "id_scenario_count": len(id_scens),
                "id_test_count": 0,
                "ood_scenario_count": len(ood_all_scens),
                "id_metrics": dict(empty_m),
                "ood_metrics": dict(empty_m),
                "regime_breakdown": {},
                "generalization_gap": {"delta_mae": 0.0, "delta_f1": 0.0, "delta_ece": 0.0, "delta_fnr_insecure": 0.0},
                "delta_mae": 0.0,
                "delta_f1": 0.0,
                "delta_ece": 0.0,
                "distribution_shift_detected": False,
                "uncertainty_elevation": {
                    "mean_interval_width_id": 0.0,
                    "mean_interval_width_ood": 0.0,
                    "mean_nadir_interval_hz_id": 0.0,
                    "mean_nadir_interval_hz_ood": 0.0,
                    "uncertainty_ratio": 1.0,
                },
            }

        # Strict held-out split within ID scenarios: NEVER evaluate ID metrics on training scenarios
        id_train, id_val, id_test, _, protocol = split_scenarios_by_partition(
            id_scens, random_state=self.random_state
        )

        extractor = FeatureExtractor(t_obs=self.t_obs, f_nominal=self.f_nominal)

        def _extract_np(scen_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not scen_list:
                return np.empty((0, 14), dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=int)
            X_df, y_raw, _, _ = extractor.transform_batch(scen_list, t_obs=self.t_obs, as_dataframe=True)
            X_np = X_df.values if hasattr(X_df, "values") else np.asarray(X_df, dtype=float)
            y_np = np.asarray(y_raw, dtype=float)
            ys_np = np.where(y_np >= self.threshold, 1, 0).astype(int)
            return X_np, y_np, ys_np

        X_tr, y_tr, ys_tr = _extract_np(id_train)
        X_va, y_va, _ = _extract_np(id_val)
        X_id_te, y_id_te, ys_id_te = _extract_np(id_test)

        if regressor is not None and getattr(regressor, "n_features_in_", X_id_te.shape[1]) == X_id_te.shape[1]:
            reg = regressor
        else:
            try:
                reg = get_nadir_regressor(regressor_type, random_state=self.random_state)
            except TypeError:
                reg = get_nadir_regressor(regressor_type)
            except Exception:
                reg = LinearNadirRegressor()
            reg.fit(X_tr, y_tr)

        if classifier is not None and getattr(classifier, "n_features_in_", X_id_te.shape[1]) == X_id_te.shape[1]:
            clf = classifier
        else:
            if classifier_type == "calibrated_gb":
                clf = CalibratedGBDTClassifier(
                    method="platt",
                    threshold=self.threshold,
                    random_state=self.random_state,
                )
                if len(X_va) >= 2:
                    clf.fit(X_tr, y_tr, ys_tr, X_cal=X_va, y_cal=y_va)
                else:
                    clf.fit(X_tr, y_tr, ys_tr)
            else:
                try:
                    clf = get_security_classifier(classifier_type, threshold=self.threshold)
                except Exception:
                    clf = CalibratedGBDTClassifier(method="platt", threshold=self.threshold)
                clf.fit(X_tr, ys_tr)

        id_test_eval = self._evaluate_subset("ID Held-Out Test", X_id_te, y_id_te, ys_id_te, reg, clf)

        if self.id_range == self.ood_range:
            ood_all_eval = dict(id_test_eval)
            ood_all_eval["regime"] = "Same-Distribution Sanity"
            regime_breakdown = {
                "id_test": id_test_eval,
                "ood_all": ood_all_eval,
            }
            delta_mae = 0.0
            delta_f1 = 0.0
            delta_ece = 0.0
            delta_fnr_insec = 0.0
            dist_shift = False
            w_id = id_test_eval["mean_prob_interval_width"]
            w_ood = w_id
            w_hz_id = id_test_eval["mean_nadir_interval_hz"]
            w_hz_ood = w_hz_id
            ratio = 1.0
        else:
            X_ood_all, y_ood_all, ys_ood_all = _extract_np(ood_all_scens)
            ood_all_eval = self._evaluate_subset("OOD-All", X_ood_all, y_ood_all, ys_ood_all, reg, clf)

            X_res, y_res, ys_res = _extract_np(ood_res_scens)
            ood_res_eval = self._evaluate_subset("OOD-RES (40-50%)", X_res, y_res, ys_res, reg, clf)

            X_ld, y_ld, ys_ld = _extract_np(ood_load_scens)
            ood_load_eval = self._evaluate_subset("OOD-Load (0.80, 1.20)", X_ld, y_ld, ys_ld, reg, clf)

            X_cb, y_cb, ys_cb = _extract_np(ood_comb_scens)
            ood_comb_eval = self._evaluate_subset("OOD-Combined (RES + Load)", X_cb, y_cb, ys_cb, reg, clf)

            regime_breakdown = {
                "id_test": id_test_eval,
                "ood_res": ood_res_eval,
                "ood_load": ood_load_eval,
                "ood_combined": ood_comb_eval,
                "ood_all": ood_all_eval,
            }

            # Unclamped, authentic generalization gap
            delta_mae = round(float(ood_all_eval["mae"] - id_test_eval["mae"]), 4)
            delta_f1 = round(float(id_test_eval["f1"] - ood_all_eval["f1"]), 4)
            delta_ece = round(float(ood_all_eval["ece"] - id_test_eval["ece"]), 4)
            delta_fnr_insec = round(float(ood_all_eval["fnr_insecure"] - id_test_eval["fnr_insecure"]), 4)
            dist_shift = True

            w_id = float(id_test_eval["mean_prob_interval_width"])
            w_ood = float(ood_all_eval["mean_prob_interval_width"])
            w_hz_id = float(id_test_eval["mean_nadir_interval_hz"])
            w_hz_ood = float(ood_all_eval["mean_nadir_interval_hz"])
            ratio = float(w_hz_ood / max(1e-4, w_hz_id)) if w_hz_id > 0 else (
                float(w_ood / max(1e-4, w_id)) if w_id > 0 else 1.0
            )

        gap_dict = {
            "delta_mae": delta_mae,
            "delta_f1": delta_f1,
            "delta_ece": delta_ece,
            "delta_fnr_insecure": delta_fnr_insec,
        }

        return {
            "partition_protocol": protocol,
            "id_scenario_count": len(id_scens),
            "id_train_count": len(id_train),
            "id_val_count": len(id_val),
            "id_test_count": len(id_test),
            "ood_scenario_count": len(ood_all_scens),
            "ood_res_count": len(ood_res_scens),
            "ood_load_count": len(ood_load_scens),
            "ood_combined_count": len(ood_comb_scens),
            "id_metrics": id_test_eval,
            "ood_metrics": ood_all_eval,
            "regime_breakdown": regime_breakdown,
            "generalization_gap": gap_dict,
            "delta_mae": delta_mae,
            "delta_f1": delta_f1,
            "delta_ece": delta_ece,
            "distribution_shift_detected": dist_shift,
            "uncertainty_elevation": {
                "mean_interval_width_id": round(w_id, 4),
                "mean_interval_width_ood": round(w_ood, 4),
                "mean_nadir_interval_hz_id": round(w_hz_id, 4),
                "mean_nadir_interval_hz_ood": round(w_hz_ood, 4),
                "uncertainty_ratio": round(ratio, 4),
            },
        }

    def to_regime_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        """Export multi-regime table (Held-Out ID Test, OOD-RES, OOD-Load, OOD-Combined, OOD-All)."""
        rb = report.get("regime_breakdown", {})
        ordered_keys = ["id_test", "ood_res", "ood_load", "ood_combined", "ood_all"]
        rows = []
        for k in ordered_keys:
            if k in rb and rb[k].get("n_scenarios", 0) > 0:
                m = rb[k]
                rows.append({
                    "Regime": m["regime"],
                    "N": m["n_scenarios"],
                    "MAE (Hz)": m["mae"],
                    "RMSE (Hz)": m["rmse"],
                    "Bias (Hz)": m["bias"],
                    "P95 AE (Hz)": m["p95_ae"],
                    "R2": m["r2"],
                    "F1": m["f1"],
                    "PR-AUC": m["pr_auc"],
                    "Brier": m["brier_score"],
                    "ECE": m["ece"],
                    "FNR_insecure": m["fnr_insecure"],
                    "Missed_Insecure": m["missed_insecure_count"],
                    "Interval_Hz": m["mean_nadir_interval_hz"],
                })
        return pd.DataFrame(rows)

    def to_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        rows = []
        id_m = report.get("id_metrics", {})
        ood_m = report.get("ood_metrics", {})
        gap = report.get("generalization_gap", {})

        metrics = ["mae", "rmse", "bias", "p95_ae", "r2", "f1", "pr_auc", "ece", "brier_score", "fnr_insecure"]
        for m in metrics:
            if m in id_m and m in ood_m:
                delta_key = f"delta_{m}"
                delta_val = gap.get(delta_key, round(float(ood_m[m]) - float(id_m[m]), 4))
                rows.append({
                    "Metric": m.upper(),
                    f"ID Held-Out Test ({self.id_range[0]*100:.0f}-{self.id_range[1]*100:.0f}%)": id_m[m],
                    f"OOD Test ({self.ood_range[0]*100:.0f}-{self.ood_range[1]*100:.0f}% / Load Shift)": ood_m[m],
                    "Generalization Gap": delta_val,
                })
        return pd.DataFrame(rows)

    def to_markdown(self, report: Dict[str, Any]) -> str:
        df_regimes = self.to_regime_dataframe(report)
        df_summary = self.to_dataframe(report)

        sections: List[str] = []
        if not df_regimes.empty:
            headers = [str(c) for c in df_regimes.columns]
            t_lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            for _, row in df_regimes.iterrows():
                vals = [
                    str(int(v)) if isinstance(v, (int, np.integer))
                    else (f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v))
                    for v in row
                ]
                t_lines.append("| " + " | ".join(vals) + " |")
            sections.append("\n".join(t_lines))

        if not df_summary.empty:
            headers = [str(c) for c in df_summary.columns]
            t_lines = ["\n| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
            for _, row in df_summary.iterrows():
                vals = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
                t_lines.append("| " + " | ".join(vals) + " |")
            sections.append("\n".join(t_lines))

        sections.extend([
            f"\n- **Partition Protocol**: `{report.get('partition_protocol', 'metadata_partitions')}` "
            f"(train={report.get('id_train_count', 0)}, val={report.get('id_val_count', 0)}, held-out test={report.get('id_test_count', 0)})",
            f"- **ID Scenarios**: {report.get('id_scenario_count', 0)} (evaluated strictly on {report.get('id_test_count', 0)} held-out test)",
            f"- **OOD Scenarios**: {report.get('ood_scenario_count', 0)} "
            f"(RES={report.get('ood_res_count', 0)}, Load={report.get('ood_load_count', 0)}, Combined={report.get('ood_combined_count', 0)})",
            f"- **Distribution Shift Detected**: {report.get('distribution_shift_detected', False)}",
            f"- **Generalization Gap (Delta MAE)**: {report.get('delta_mae', 0.0)} Hz",
        ])
        return "\n".join(sections)
