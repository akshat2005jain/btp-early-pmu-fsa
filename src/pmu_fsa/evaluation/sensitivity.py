"""Observation window sensitivity benchmark across T_obs lengths (strictly leak-free & unsmoothed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score as calc_brier_score,
    expected_calibration_error,
)
from pmu_fsa.models.classifiers import compute_binary_security_metrics, get_security_classifier
from pmu_fsa.models.regressors import LinearNadirRegressor, get_nadir_regressor


def split_scenarios_by_partition(
    scenarios: List[Dict[str, Any]],
    random_state: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], str]:
    """Partition scenarios into (train, val, test, ood_test, protocol_name) without data leakage.

    1. When explicit `"test"` and `"train"` metadata partitions exist, use them verbatim.
    2. Otherwise (e.g. synthetic test batches lacking `"test"`), isolate any `"ood_test"` scenarios
       first so OOD data NEVER enters training, and apply a deterministic 60/20/20 split to ID scenarios.
    """
    ood_scens = [s for s in scenarios if str(s.get("metadata", {}).get("partition", "")) == "ood_test"]
    id_pool = [s for s in scenarios if str(s.get("metadata", {}).get("partition", "")) != "ood_test"]

    train_scens = [s for s in id_pool if str(s.get("metadata", {}).get("partition", "")) == "train"]
    val_scens = [s for s in id_pool if str(s.get("metadata", {}).get("partition", "")) == "val"]
    test_scens = [s for s in id_pool if str(s.get("metadata", {}).get("partition", "")) == "test"]

    if len(train_scens) >= 1 and len(test_scens) >= 1:
        return train_scens, val_scens, test_scens, ood_scens, "metadata_partitions"

    # Fallback deterministic 60/20/20 split strictly on non-OOD scenarios
    pool = id_pool if len(id_pool) > 0 else list(scenarios)
    n = len(pool)
    if n == 0:
        return [], [], [], ood_scens, "empty"
    if n == 1:
        return pool, [], pool, ood_scens, "degenerate_single_sample"

    rng = np.random.RandomState(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)

    if n >= 5:
        n_train = max(1, int(round(0.60 * n)))
        n_val = max(1, int(round(0.20 * n)))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        tr_idx = indices[:n_train]
        va_idx = indices[n_train : n_train + n_val]
        te_idx = indices[n_train + n_val :]
        return (
            [pool[i] for i in tr_idx],
            [pool[i] for i in va_idx],
            [pool[i] for i in te_idx],
            ood_scens,
            "deterministic_60_20_20_fallback",
        )

    n_train = max(1, n - 1)
    tr_idx = indices[:n_train]
    te_idx = indices[n_train:]
    return (
        [pool[i] for i in tr_idx],
        [],
        [pool[i] for i in te_idx],
        ood_scens,
        "deterministic_small_split_fallback",
    )


class WindowSensitivityBenchmark:
    """Evaluates leak-free regression, classification, calibration, and lead time across T_obs."""

    def __init__(
        self,
        window_lengths: Optional[List[float]] = None,
        regressor_type: str = "gbdt",
        classifier_type: str = "calibrated_gb",
        threshold: float = 49.50,
        f_nominal: float = 50.0,
        random_state: int = 42,
        models_dir: Optional[Union[str, Path]] = None,
        strict_nadir: bool = True,
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
        self.models_dir = Path(models_dir) if models_dir is not None else None
        self.strict_nadir = bool(strict_nadir)

    @classmethod
    def filter_causal_scenarios(
        cls,
        scenarios: List[Dict[str, Any]],
        t_obs: float,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Filter scenarios strictly to those satisfying t_0 + t_obs < t_nadir (lead_time > 0)."""
        if not scenarios:
            return [], []
        lead_times = cls._extract_lead_times(scenarios, t_obs)
        eligible: List[Dict[str, Any]] = []
        excluded_ids: List[str] = []
        for idx, (scen, lt) in enumerate(zip(scenarios, lead_times)):
            if lt > 1e-6:
                eligible.append(scen)
            else:
                sid = str(scen.get("metadata", {}).get("scenario_id", f"scenario_{idx:04d}"))
                excluded_ids.append(sid)
        return eligible, excluded_ids

    @staticmethod
    def _extract_lead_times(scenarios: List[Dict[str, Any]], t_obs: float) -> List[float]:
        """Compute remaining lead time t_nadir - (t_disturb + t_obs) per scenario."""
        lead_times: List[float] = []
        for s in scenarios:
            meta = s.get("metadata", {})
            t_disturb = float(meta.get("t_disturb", 1.0))
            t_nadir = meta.get("ground_truth_t_nadir", None)
            if t_nadir is None:
                ts = s.get("timestamps")
                freq_ch = s.get("pmu_sys_frequency", s.get("coi_frequency"))
                if ts is not None and freq_ch is not None and len(ts) == len(freq_ch) and len(freq_ch) > 0:
                    post_mask = np.asarray(ts, dtype=float) >= t_disturb
                    if np.any(post_mask):
                        post_t = np.asarray(ts, dtype=float)[post_mask]
                        post_f = np.asarray(freq_ch, dtype=float)[post_mask]
                        t_nadir = float(post_t[int(np.argmin(post_f))])
                    else:
                        t_nadir = float(ts[int(np.argmin(freq_ch))])
                else:
                    t_nadir = 3.5
            lead_times.append(float(t_nadir) - (t_disturb + float(t_obs)))
        return lead_times

    def _evaluate_split_at_window(
        self,
        eval_scens: List[Dict[str, Any]],
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        y_sec_eval: np.ndarray,
        reg: Any,
        clf: Any,
        t_obs: float,
        n_population_total: Optional[int] = None,
        excluded_ids: Optional[List[str]] = None,
        population_mode: str = "window_specific_causal",
    ) -> Dict[str, Any]:
        excl = list(excluded_ids or [])
        n_eligible = len(eval_scens)
        n_total = int(n_population_total) if n_population_total is not None else n_eligible
        n_excluded = max(0, n_total - n_eligible)

        if n_eligible == 0:
            return {
                "window_obs": float(t_obs),
                "evaluation_population": population_mode,
                "n_causal": 0,
                "n_eligible": 0,
                "n_excluded": n_excluded,
                "n_total": n_total,
                "causal_valid_ratio": 0.0,
                "excluded_scenario_ids": ";".join(excl),
                "t_lead": float("nan"),
                "t_lead_median": float("nan"),
                "t_lead_min": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "bias": float("nan"),
                "median_ae": float("nan"),
                "p95_ae": float("nan"),
                "r2": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
                "f1": float("nan"),
                "pr_auc": float("nan"),
                "roc_auc": float("nan"),
                "brier_score": float("nan"),
                "ece": float("nan"),
                "fnr": float("nan"),
                "fpr_secure": float("nan"),
                "fnr_insecure": float("nan"),
                "missed_insecure_count": 0,
                "mae_raw": float("nan"),
                "rmse_raw": float("nan"),
                "f1_raw": float("nan"),
            }

        reg_m = reg.evaluate(X_eval, y_eval)
        probs = clf.predict_proba(X_eval)
        p_sec = np.asarray(probs[:, 1] if hasattr(probs, "ndim") and probs.ndim > 1 else probs, dtype=float)
        y_sec_pred = (p_sec >= 0.5).astype(int)
        cls_m = compute_binary_security_metrics(y_sec_eval, y_sec_pred, p_secure=p_sec)

        ece = float(expected_calibration_error(y_sec_eval, p_sec, n_bins=10, threshold=self.threshold))
        brier = float(calc_brier_score(y_sec_eval, p_sec, threshold=self.threshold))

        lead_arr = np.asarray(self._extract_lead_times(eval_scens, t_obs), dtype=float)
        n_causal = int(np.sum(lead_arr > 1e-6))
        causal_ratio = float(n_causal / n_total) if n_total > 0 else 0.0
        t_lead_mean = float(np.mean(lead_arr))
        t_lead_med = float(np.median(lead_arr))
        t_lead_min = float(np.min(lead_arr))

        mae = float(reg_m["mae"])
        rmse = float(reg_m["rmse"])
        r2 = float(reg_m["r2"])
        f1 = float(cls_m["f1"])

        return {
            "window_obs": float(t_obs),
            "evaluation_population": population_mode,
            "n_causal": n_causal,
            "n_eligible": n_eligible,
            "n_excluded": n_excluded,
            "n_total": n_total,
            "causal_valid_ratio": round(causal_ratio, 4),
            "excluded_scenario_ids": ";".join(excl),
            "t_lead": round(t_lead_mean, 4),
            "t_lead_median": round(t_lead_med, 4),
            "t_lead_min": round(t_lead_min, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "bias": round(float(reg_m["bias"]), 4),
            "median_ae": round(float(reg_m["median_ae"]), 4),
            "p95_ae": round(float(reg_m["p95_ae"]), 4),
            "r2": round(r2, 4),
            "precision": round(float(cls_m["precision"]), 4),
            "recall": round(float(cls_m["recall"]), 4),
            "f1": round(f1, 4),
            "pr_auc": round(float(cls_m["pr_auc"]), 4),
            "roc_auc": round(float(cls_m["roc_auc"]), 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            "fnr": round(float(cls_m["fpr_secure"]), 4),
            "fpr_secure": round(float(cls_m["fpr_secure"]), 4),
            "fnr_insecure": round(float(cls_m["fnr_insecure"]), 4),
            "missed_insecure_count": int(cls_m["missed_insecure_count"]),
            "mae_raw": round(mae, 4),
            "rmse_raw": round(rmse, 4),
            "f1_raw": round(f1, 4),
        }

    def run(
        self,
        scenarios: List[Dict[str, Any]],
        split_ratio: float = 0.70,
        default_window_models: Optional[Tuple[float, Any, Any]] = None,
    ) -> Dict[str, Any]:
        """Run leak-free, strictly pre-nadir window sensitivity sweep across configured window lengths."""
        base_columns = ["window_obs", "mae", "rmse", "r2", "f1", "ece"]
        if not scenarios:
            return {
                "columns": base_columns,
                "results": [],
                "ood_results": [],
                "common_causal_results": [],
                "ood_common_causal_results": [],
                "excluded_by_window": {},
                "partition_protocol": "empty",
            }

        train_scens, val_scens, test_scens, ood_scens, protocol = split_scenarios_by_partition(
            scenarios, random_state=self.random_state
        )

        extractor = FeatureExtractor(f_nominal=self.f_nominal, strict_nadir=self.strict_nadir)
        rows: List[Dict[str, Any]] = []
        ood_rows: List[Dict[str, Any]] = []
        common_id_rows: List[Dict[str, Any]] = []
        common_ood_rows: List[Dict[str, Any]] = []
        excluded_by_window: Dict[str, Dict[str, List[str]]] = {}

        max_w = max(self.window_lengths)
        test_common, test_common_excl = (
            self.filter_causal_scenarios(test_scens, max_w) if self.strict_nadir else (list(test_scens), [])
        )
        ood_common, ood_common_excl = (
            self.filter_causal_scenarios(ood_scens, max_w) if self.strict_nadir else (list(ood_scens), [])
        )

        for t_obs in self.window_lengths:
            if self.strict_nadir:
                tr_elig, tr_excl = self.filter_causal_scenarios(train_scens, t_obs)
                va_elig, va_excl = self.filter_causal_scenarios(val_scens, t_obs)
                te_elig, te_excl = self.filter_causal_scenarios(test_scens, t_obs)
                ood_elig, ood_excl = self.filter_causal_scenarios(ood_scens, t_obs)
            else:
                tr_elig, tr_excl = list(train_scens), []
                va_elig, va_excl = list(val_scens), []
                te_elig, te_excl = list(test_scens), []
                ood_elig, ood_excl = list(ood_scens), []

            excluded_by_window[f"{t_obs:.1f}s"] = {
                "train": tr_excl,
                "val": va_excl,
                "id_test": te_excl,
                "ood_test": ood_excl,
            }

            if len(tr_elig) == 0:
                continue

            X_tr_df, y_tr_raw, _, _ = extractor.transform_batch(tr_elig, t_obs=t_obs, as_dataframe=True)
            X_train = X_tr_df.values if hasattr(X_tr_df, "values") else np.asarray(X_tr_df, dtype=float)
            y_train = np.asarray(y_tr_raw, dtype=float)
            ys_train = np.where(y_train >= self.threshold, 1, 0).astype(int)

            if len(va_elig) > 0:
                X_va_df, y_va_raw, _, _ = extractor.transform_batch(va_elig, t_obs=t_obs, as_dataframe=True)
                X_val = X_va_df.values if hasattr(X_va_df, "values") else np.asarray(X_va_df, dtype=float)
                y_val = np.asarray(y_va_raw, dtype=float)
            else:
                X_val, y_val = None, None

            if len(te_elig) > 0:
                X_te_df, y_te_raw, _, _ = extractor.transform_batch(te_elig, t_obs=t_obs, as_dataframe=True)
                X_test = X_te_df.values if hasattr(X_te_df, "values") else np.asarray(X_te_df, dtype=float)
                y_test = np.asarray(y_te_raw, dtype=float)
                ys_test = np.where(y_test >= self.threshold, 1, 0).astype(int)
            else:
                X_test = np.empty((0, X_train.shape[1]), dtype=float)
                y_test = np.empty(0, dtype=float)
                ys_test = np.empty(0, dtype=int)

            # Reuse pre-saved model if provided for its exact training window; otherwise fit strictly on train (+ val)
            if (
                default_window_models is not None
                and abs(float(default_window_models[0]) - t_obs) < 1e-6
                and default_window_models[1] is not None
                and default_window_models[2] is not None
                and getattr(default_window_models[1], "n_features_in_", X_train.shape[1]) == X_train.shape[1]
            ):
                reg, clf = default_window_models[1], default_window_models[2]
            else:
                try:
                    reg = get_nadir_regressor(self.regressor_type, random_state=self.random_state)
                except TypeError:
                    reg = get_nadir_regressor(self.regressor_type)
                except Exception:
                    reg = LinearNadirRegressor()
                reg.fit(X_train, y_train)

                if self.classifier_type == "calibrated_gb":
                    clf = CalibratedGBDTClassifier(
                        method="platt",
                        threshold=self.threshold,
                        random_state=self.random_state,
                    )
                    if X_val is not None and y_val is not None and len(X_val) >= 2:
                        clf.fit(X_train, y_train, ys_train, X_cal=X_val, y_cal=y_val)
                    else:
                        clf.fit(X_train, y_train, ys_train)
                else:
                    try:
                        clf = get_security_classifier(self.classifier_type, threshold=self.threshold)
                    except Exception:
                        clf = CalibratedGBDTClassifier(method="platt", threshold=self.threshold)
                    clf.fit(X_train, ys_train)

            row_id = self._evaluate_split_at_window(
                te_elig,
                X_test,
                y_test,
                ys_test,
                reg,
                clf,
                t_obs,
                n_population_total=len(test_scens),
                excluded_ids=te_excl,
                population_mode="window_specific_causal" if self.strict_nadir else "all_scenarios",
            )
            rows.append(row_id)

            if len(test_common) > 0:
                X_tc_df, y_tc_raw, _, _ = extractor.transform_batch(test_common, t_obs=t_obs, as_dataframe=True)
                X_tc = X_tc_df.values if hasattr(X_tc_df, "values") else np.asarray(X_tc_df, dtype=float)
                y_tc = np.asarray(y_tc_raw, dtype=float)
                ys_tc = np.where(y_tc >= self.threshold, 1, 0).astype(int)
                common_id_rows.append(
                    self._evaluate_split_at_window(
                        test_common,
                        X_tc,
                        y_tc,
                        ys_tc,
                        reg,
                        clf,
                        t_obs,
                        n_population_total=len(test_scens),
                        excluded_ids=test_common_excl,
                        population_mode="common_causal_intersection",
                    )
                )

            if len(ood_scens) > 0:
                if len(ood_elig) > 0:
                    X_ood_df, y_ood_raw, _, _ = extractor.transform_batch(ood_elig, t_obs=t_obs, as_dataframe=True)
                    X_ood = X_ood_df.values if hasattr(X_ood_df, "values") else np.asarray(X_ood_df, dtype=float)
                    y_ood = np.asarray(y_ood_raw, dtype=float)
                    ys_ood = np.where(y_ood >= self.threshold, 1, 0).astype(int)
                else:
                    X_ood = np.empty((0, X_train.shape[1]), dtype=float)
                    y_ood = np.empty(0, dtype=float)
                    ys_ood = np.empty(0, dtype=int)
                row_ood = self._evaluate_split_at_window(
                    ood_elig,
                    X_ood,
                    y_ood,
                    ys_ood,
                    reg,
                    clf,
                    t_obs,
                    n_population_total=len(ood_scens),
                    excluded_ids=ood_excl,
                    population_mode="window_specific_causal" if self.strict_nadir else "all_scenarios",
                )
                ood_rows.append(row_ood)

                if len(ood_common) > 0:
                    X_oc_df, y_oc_raw, _, _ = extractor.transform_batch(ood_common, t_obs=t_obs, as_dataframe=True)
                    X_oc = X_oc_df.values if hasattr(X_oc_df, "values") else np.asarray(X_oc_df, dtype=float)
                    y_oc = np.asarray(y_oc_raw, dtype=float)
                    ys_oc = np.where(y_oc >= self.threshold, 1, 0).astype(int)
                    common_ood_rows.append(
                        self._evaluate_split_at_window(
                            ood_common,
                            X_oc,
                            y_oc,
                            ys_oc,
                            reg,
                            clf,
                            t_obs,
                            n_population_total=len(ood_scens),
                            excluded_ids=ood_common_excl,
                            population_mode="common_causal_intersection",
                        )
                    )

        return {
            "columns": base_columns,
            "results": rows,
            "ood_results": ood_rows,
            "common_causal_results": common_id_rows,
            "ood_common_causal_results": common_ood_rows,
            "excluded_by_window": excluded_by_window,
            "partition_protocol": protocol,
            "strict_nadir": self.strict_nadir,
            "n_train": len(train_scens),
            "n_val": len(val_scens),
            "n_test": len(test_scens),
            "n_ood_test": len(ood_scens),
            "n_test_common_causal": len(test_common),
            "n_ood_common_causal": len(ood_common),
        }

    def run_multi_model_sweep(
        self,
        scenarios: List[Dict[str, Any]],
        regressor_types: Optional[List[str]] = None,
        classifier_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run window sensitivity sweep across multiple regressors and classifiers."""
        regs = regressor_types or ["mean", "analytical", "linear", "dt", "rf", "gbdt", "lgbm"]
        clfs = classifier_types or ["logistic", "dt", "rf", "gbdt", "lgbm", "calibrated_gb"]
        sweep_results: Dict[str, Any] = {}

        for r_name in regs:
            bench = WindowSensitivityBenchmark(
                window_lengths=self.window_lengths,
                regressor_type=r_name,
                threshold=self.threshold,
                f_nominal=self.f_nominal,
                random_state=self.random_state,
                strict_nadir=self.strict_nadir,
            )
            sweep_results[f"reg_{r_name}"] = bench.run(scenarios)

        for c_name in clfs:
            bench = WindowSensitivityBenchmark(
                window_lengths=self.window_lengths,
                classifier_type=c_name,
                threshold=self.threshold,
                f_nominal=self.f_nominal,
                random_state=self.random_state,
                strict_nadir=self.strict_nadir,
            )
            sweep_results[f"clf_{c_name}"] = bench.run(scenarios)

        return sweep_results

    def to_dataframe(self, results: Dict[str, Any], split: str = "id_test") -> pd.DataFrame:
        key_map = {
            "id_test": "results",
            "ood_test": "ood_results",
            "id_common": "common_causal_results",
            "ood_common": "ood_common_causal_results",
        }
        key = key_map.get(split, "results")
        rows = results.get(key, [])
        cols = results.get("columns", ["window_obs", "mae", "rmse", "r2", "f1", "ece"])
        df = pd.DataFrame(rows)
        present_cols = [c for c in cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in present_cols]
        return df[present_cols + extra_cols] if not df.empty else pd.DataFrame(columns=cols)

    def to_markdown(self, results: Dict[str, Any], split: str = "id_test") -> str:
        df = self.to_dataframe(results, split=split)
        if df.empty:
            return ""
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, (int, np.integer)):
                    vals.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
