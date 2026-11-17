"""Physical interpretability, permutation/impurity importance, correlations, and ablations."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark, split_scenarios_by_partition
from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score as calc_brier_score,
    expected_calibration_error,
)
from pmu_fsa.models.classifiers import compute_binary_security_metrics, get_security_classifier
from pmu_fsa.models.regressors import GradientBoostingNadirRegressor, get_nadir_regressor
from pmu_fsa.simulation.pmu_recorder import (
    CANONICAL_13BUS_PMU_PLACEMENT,
    compute_causal_rocof,
    estimate_pmu_system_frequency,
)


CANONICAL_14_FEATURES: List[str] = [
    "rocof_init",
    "rocof_peak_neg",
    "rocof_end",
    "rocof_mean",
    "freq_drop_end",
    "freq_drop_max",
    "freq_dev_integral",
    "bus_drop_worst",
    "area_freq_spread_max",
    "bus_freq_std_mean",
    "angle_divergence_max",
    "tie_line_angle_excursion",
    "freq_curvature_init",
    "freq_curvature_mean",
]

# Positional column subsets on the 14 canonical features
FEATURE_GROUP_INDICES: Dict[str, List[int]] = {
    "all_14": list(range(14)),
    "freq_only": [4, 5, 6],
    "freq_plus_rocof": [0, 1, 2, 3, 4, 5, 6],
    "no_spatial_angle": [0, 1, 2, 3, 4, 5, 6, 12, 13],
    "spatial_angle_only": [7, 8, 9, 10, 11],
    "no_rocof": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "no_freq_level": [0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13],
    "no_curvature": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}


def _resolve_feature_names(X: Any, feature_names: Optional[Sequence[str]] = None) -> List[str]:
    if feature_names is not None and len(feature_names) > 0:
        return [str(f) for f in feature_names]
    if hasattr(X, "columns"):
        return [str(c) for c in X.columns]
    arr = np.asarray(X)
    n_cols = arr.shape[1] if arr.ndim > 1 else 1
    if n_cols == 14:
        return list(CANONICAL_14_FEATURES)
    return [f"feature_{i}" for i in range(n_cols)]


class InterpretabilityAnalyzer:
    """Computes feature importance, collinearity diagnostics, subgroup shifts, and ablations."""

    def __init__(
        self,
        t_obs: float = 1.0,
        threshold: float = 49.50,
        f_nominal: float = 50.0,
        seed: int = 42,
    ) -> None:
        self.t_obs = float(t_obs)
        self.threshold = float(threshold)
        self.f_nominal = float(f_nominal)
        self.seed = int(seed)

    @staticmethod
    def compute_permutation_importance(
        model: Any,
        X: Any,
        y: Any,
        feature_names: Optional[Sequence[str]] = None,
        task: str = "regression",
        n_repeats: int = 15,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Compute permutation importance (mean & std increase in MAE or Brier loss across repeats)."""
        names = _resolve_feature_names(X, feature_names)
        X_arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        y_arr = np.asarray(y.values if hasattr(y, "values") else y, dtype=float).ravel()

        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n_samples, n_features = X_arr.shape
        if n_samples == 0 or n_features == 0:
            return pd.DataFrame(columns=["feature", "importance_mean", "importance_std", "rank"])

        if task == "regression":
            base_pred = np.asarray(model.predict(X_arr), dtype=float).ravel()
            base_loss = float(np.mean(np.abs(y_arr - base_pred)))
        else:
            y_bin = np.where(y_arr > 1.5, (y_arr >= 49.50).astype(float), (y_arr >= 0.5).astype(float))
            base_prob = np.asarray(model.predict_proba(X_arr)[:, 1], dtype=float).ravel()
            base_loss = float(np.mean((base_prob - y_bin) ** 2))

        rng = np.random.default_rng(seed)
        deltas = np.zeros((n_features, max(1, int(n_repeats))), dtype=float)
        for j in range(n_features):
            for r in range(deltas.shape[1]):
                X_perm = X_arr.copy()
                X_perm[:, j] = rng.permutation(X_perm[:, j])
                if task == "regression":
                    p_pred = np.asarray(model.predict(X_perm), dtype=float).ravel()
                    perm_loss = float(np.mean(np.abs(y_arr - p_pred)))
                else:
                    p_prob = np.asarray(model.predict_proba(X_perm)[:, 1], dtype=float).ravel()
                    perm_loss = float(np.mean((p_prob - y_bin) ** 2))
                deltas[j, r] = perm_loss - base_loss

        means = np.mean(deltas, axis=1)
        stds = np.std(deltas, axis=1, ddof=1) if deltas.shape[1] > 1 else np.zeros(n_features)
        df = pd.DataFrame({
            "feature": names[:n_features],
            "importance_mean": means,
            "importance_std": stds,
        })
        df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1, dtype=int)
        return df

    @staticmethod
    def compute_impurity_importance(
        model: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Extract normalized tree impurity (or standardized coefficient) feature importances."""
        imp: Optional[np.ndarray] = None
        if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
            imp = np.asarray(model.feature_importances_, dtype=float)
        elif hasattr(model, "base_model") and hasattr(model.base_model, "feature_importances_"):
            imp = np.asarray(model.base_model.feature_importances_, dtype=float)
        elif hasattr(model, "model_") and hasattr(model.model_, "feature_importances_"):
            imp = np.asarray(model.model_.feature_importances_, dtype=float)
        elif hasattr(model, "weights") and model.weights is not None:
            imp = np.abs(np.asarray(model.weights, dtype=float)).ravel()

        if imp is None or len(imp) == 0:
            n_f = getattr(model, "n_features_in_", 14) or 14
            imp = np.zeros(n_f, dtype=float)

        total = float(np.sum(imp))
        if total > 0:
            imp = imp / total

        names = list(feature_names) if feature_names is not None and len(feature_names) == len(imp) else (
            list(CANONICAL_14_FEATURES) if len(imp) == 14 else [f"feature_{i}" for i in range(len(imp))]
        )
        df = pd.DataFrame({
            "feature": names,
            "impurity_importance": imp,
        }).sort_values("impurity_importance", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1, dtype=int)
        return df

    @staticmethod
    def compute_correlation_matrix(
        X: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Compute 14x14 Pearson and Spearman correlation matrices plus rank/condition diagnostics."""
        names = _resolve_feature_names(X, feature_names)
        X_arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        df_X = pd.DataFrame(X_arr, columns=names[: X_arr.shape[1]])
        pearson_df = df_X.corr(method="pearson").fillna(0.0)
        spearman_df = df_X.corr(method="spearman").fillna(0.0)

        p_vals = np.abs(pearson_df.values.copy())
        s_vals = np.abs(spearman_df.values.copy())
        np.fill_diagonal(p_vals, 0.0)
        np.fill_diagonal(s_vals, 0.0)

        # Standardize columns before condition number so physical unit scale differences don't mask collinearity
        stds = np.std(X_arr, axis=0)
        X_std = (X_arr - np.mean(X_arr, axis=0)) / np.where(stds > 1e-12, stds, 1.0)
        rank = int(np.linalg.matrix_rank(X_std))
        cond = float(np.linalg.cond(X_std)) if X_std.size > 0 else 1.0

        return {
            "pearson_df": pearson_df,
            "spearman_df": spearman_df,
            "matrix_rank": rank,
            "n_features": int(X_arr.shape[1]),
            "standardized_condition_number": cond,
            "max_off_diag_abs_pearson": float(np.max(p_vals)) if p_vals.size > 0 else 0.0,
            "max_off_diag_abs_spearman": float(np.max(s_vals)) if s_vals.size > 0 else 0.0,
        }

    @staticmethod
    def _mask_partial_pmu_scenarios(
        scenarios: List[Dict[str, Any]],
        observed_buses_1based: Sequence[int] = CANONICAL_13BUS_PMU_PLACEMENT,
    ) -> List[Dict[str, Any]]:
        """Mask unobserved buses to NaN and recompute wide-area PMU frequency strictly from observed buses."""
        obs_zero_idx = np.array([b - 1 for b in observed_buses_1based if 1 <= b <= 39], dtype=np.intp)
        masked_list: List[Dict[str, Any]] = []
        for s in scenarios:
            sc = copy.deepcopy(s)
            bf = np.asarray(sc["bus_frequencies"], dtype=float).copy()
            va = np.asarray(sc["voltage_angles"], dtype=float).copy()
            vm = np.asarray(sc["voltage_magnitudes"], dtype=float).copy()
            br = np.asarray(sc.get("bus_rocof", np.zeros_like(bf)), dtype=float).copy()

            n_buses = bf.shape[1]
            unobs_mask = np.ones(n_buses, dtype=bool)
            valid_obs = obs_zero_idx[obs_zero_idx < n_buses]
            unobs_mask[valid_obs] = False

            bf[:, unobs_mask] = np.nan
            va[:, unobs_mask] = np.nan
            vm[:, unobs_mask] = np.nan
            br[:, unobs_mask] = np.nan

            f_sys_partial = estimate_pmu_system_frequency(bf, observed_bus_indices=valid_obs)
            sc["bus_frequencies"] = bf
            sc["bus_rocof"] = br
            sc["voltage_magnitudes"] = vm
            sc["voltage_angles"] = va
            sc["pmu_sys_frequency"] = f_sys_partial
            sc["coi_frequency"] = f_sys_partial
            meta = dict(sc.get("metadata", {}))
            meta["observed_buses"] = list(observed_buses_1based)
            sc["metadata"] = meta
            masked_list.append(sc)
        return masked_list

    @staticmethod
    def _downsample_scenarios_to_25fps(scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Downsample 50 fps PMU streams to 25 fps (stride=2) and recompute causal 25-fps ROCOF."""
        downsampled: List[Dict[str, Any]] = []
        for s in scenarios:
            sc = copy.deepcopy(s)
            ts = np.asarray(sc["timestamps"], dtype=float)
            meta = dict(sc.get("metadata", {}))
            orig_fps = float(meta.get("fps", 50.0))
            stride = 2 if orig_fps >= 45.0 else 1

            ts_sub = ts[::stride].copy()
            bf_sub = np.asarray(sc["bus_frequencies"], dtype=float)[::stride].copy()
            vm_sub = np.asarray(sc["voltage_magnitudes"], dtype=float)[::stride].copy()
            va_sub = np.asarray(sc["voltage_angles"], dtype=float)[::stride].copy()
            new_fps = orig_fps / float(stride)

            br_sub = compute_causal_rocof(bf_sub, fps=new_fps, window_length=5)
            f_sys_sub = estimate_pmu_system_frequency(bf_sub)

            sc["timestamps"] = ts_sub
            sc["bus_frequencies"] = bf_sub
            sc["bus_rocof"] = br_sub
            sc["voltage_magnitudes"] = vm_sub
            sc["voltage_angles"] = va_sub
            sc["pmu_sys_frequency"] = f_sys_sub
            sc["coi_frequency"] = f_sys_sub
            for extra_k in ("soc", "fracsec"):
                if extra_k in sc and isinstance(sc[extra_k], np.ndarray) and len(sc[extra_k]) == len(ts):
                    sc[extra_k] = sc[extra_k][::stride].copy()
            meta["fps"] = int(round(new_fps))
            sc["metadata"] = meta
            downsampled.append(sc)
        return downsampled

    def run_feature_group_ablations(
        self,
        scenarios: List[Dict[str, Any]],
        regressor_type: str = "gbdt",
        classifier_type: str = "calibrated_gb",
        include_single_family_drops: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate feature-group & observability ablations with strict pre-nadir and partition isolation."""
        if not scenarios:
            return {"results": [], "columns": []}

        train_raw, val_raw, test_raw, ood_raw, protocol = split_scenarios_by_partition(
            scenarios, random_state=self.seed
        )
        train_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(train_raw, self.t_obs)
        val_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(val_raw, self.t_obs)
        test_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(test_raw, self.t_obs)
        ood_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(ood_raw, self.t_obs)
        extractor = FeatureExtractor(t_obs=self.t_obs, f_nominal=self.f_nominal, strict_nadir=True)

        def _extract_full(scen_list: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
            if not scen_list:
                return np.empty((0, 14), dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=int), list(CANONICAL_14_FEATURES)
            X_df, y_raw, _, _ = extractor.transform_batch(scen_list, t_obs=self.t_obs, as_dataframe=True)
            cols = list(X_df.columns) if hasattr(X_df, "columns") else list(CANONICAL_14_FEATURES)
            X_np = X_df.values if hasattr(X_df, "values") else np.asarray(X_df, dtype=float)
            y_np = np.asarray(y_raw, dtype=float)
            ys_np = np.where(y_np >= self.threshold, 1, 0).astype(int)
            return X_np, y_np, ys_np, cols

        X_tr, y_tr, ys_tr, feat_cols = _extract_full(train_scens)
        X_va, y_va, _, _ = _extract_full(val_scens)
        X_te, y_te, ys_te, _ = _extract_full(test_scens)
        X_ood, y_ood, ys_ood, _ = _extract_full(ood_scens)

        def _fit_and_eval(
            ablation_id: str,
            description: str,
            X_tr_c: np.ndarray,
            X_va_c: np.ndarray,
            X_te_c: np.ndarray,
            X_ood_c: np.ndarray,
        ) -> Dict[str, Any]:
            try:
                reg = get_nadir_regressor(regressor_type, random_state=self.seed)
            except TypeError:
                reg = get_nadir_regressor(regressor_type)
            reg.fit(X_tr_c, y_tr)
            id_reg = reg.evaluate(X_te_c, y_te)

            if classifier_type == "calibrated_gb":
                clf = CalibratedGBDTClassifier(method="platt", threshold=self.threshold, random_state=self.seed)
                if len(X_va_c) >= 2:
                    clf.fit(X_tr_c, y_tr, ys_tr, X_cal=X_va_c, y_cal=y_va)
                else:
                    clf.fit(X_tr_c, y_tr, ys_tr)
            else:
                clf = get_security_classifier(classifier_type, threshold=self.threshold)
                clf.fit(X_tr_c, ys_tr)

            p_te = clf.predict_proba(X_te_c)[:, 1]
            cls_te = compute_binary_security_metrics(ys_te, (p_te >= 0.5).astype(int), p_secure=p_te)
            ece_te = expected_calibration_error(ys_te, p_te, n_bins=10, threshold=self.threshold)
            brier_te = calc_brier_score(ys_te, p_te, threshold=self.threshold)

            ood_mae, ood_f1, ood_ece, ood_fnr = float("nan"), float("nan"), float("nan"), float("nan")
            if len(X_ood_c) > 0:
                ood_reg = reg.evaluate(X_ood_c, y_ood)
                p_ood = clf.predict_proba(X_ood_c)[:, 1]
                cls_ood = compute_binary_security_metrics(ys_ood, (p_ood >= 0.5).astype(int), p_secure=p_ood)
                ood_mae = float(ood_reg["mae"])
                ood_f1 = float(cls_ood["f1"])
                ood_ece = float(expected_calibration_error(ys_ood, p_ood, n_bins=10, threshold=self.threshold))
                ood_fnr = float(cls_ood["fnr_insecure"])

            return {
                "ablation": ablation_id,
                "description": description,
                "n_features": int(X_tr_c.shape[1]),
                "id_mae": round(float(id_reg["mae"]), 4),
                "id_rmse": round(float(id_reg["rmse"]), 4),
                "id_r2": round(float(id_reg["r2"]), 4),
                "id_f1": round(float(cls_te["f1"]), 4),
                "id_brier": round(float(brier_te), 4),
                "id_ece": round(float(ece_te), 4),
                "id_fnr_insecure": round(float(cls_te["fnr_insecure"]), 4),
                "ood_mae": round(ood_mae, 4) if np.isfinite(ood_mae) else float("nan"),
                "ood_f1": round(ood_f1, 4) if np.isfinite(ood_f1) else float("nan"),
                "ood_ece": round(ood_ece, 4) if np.isfinite(ood_ece) else float("nan"),
                "ood_fnr_insecure": round(ood_fnr, 4) if np.isfinite(ood_fnr) else float("nan"),
            }

        ablation_specs = [
            ("all_14", "All 14 Canonical PMU Features (50 fps, 39-bus)", FEATURE_GROUP_INDICES["all_14"]),
            ("freq_only", "Aggregate Frequency Deviation Only (3 features)", FEATURE_GROUP_INDICES["freq_only"]),
            ("freq_plus_rocof", "Frequency + First-Order ROCOF (7 features)", FEATURE_GROUP_INDICES["freq_plus_rocof"]),
            ("no_spatial_angle", "Temporal Frequency + ROCOF + Curvature, No Spatial/Angle (9 features)", FEATURE_GROUP_INDICES["no_spatial_angle"]),
            ("spatial_angle_only", "Multi-Area Spatial Dispersion & Phase Angles Only (5 features)", FEATURE_GROUP_INDICES["spatial_angle_only"]),
        ]
        if include_single_family_drops:
            ablation_specs.extend([
                ("no_rocof", "Drop First-Order ROCOF Family (10 features)", FEATURE_GROUP_INDICES["no_rocof"]),
                ("no_freq_level", "Drop Aggregate Frequency Level & Integral Family (11 features)", FEATURE_GROUP_INDICES["no_freq_level"]),
                ("no_curvature", "Drop Second-Order Curvature Family (12 features)", FEATURE_GROUP_INDICES["no_curvature"]),
            ])

        rows: List[Dict[str, Any]] = []
        n_avail = X_tr.shape[1]
        for ab_id, desc, col_idx in ablation_specs:
            valid_cols = [c for c in col_idx if c < n_avail]
            if not valid_cols:
                valid_cols = list(range(n_avail))
            rows.append(
                _fit_and_eval(
                    ab_id,
                    desc,
                    X_tr[:, valid_cols],
                    X_va[:, valid_cols] if len(X_va) > 0 else np.empty((0, len(valid_cols))),
                    X_te[:, valid_cols],
                    X_ood[:, valid_cols] if len(X_ood) > 0 else np.empty((0, len(valid_cols))),
                )
            )

        # Ablation 6: Partial 13-bus spatial PMU observability (33.3% bus coverage)
        tr_13 = self._mask_partial_pmu_scenarios(train_scens)
        va_13 = self._mask_partial_pmu_scenarios(val_scens)
        te_13 = self._mask_partial_pmu_scenarios(test_scens)
        ood_13 = self._mask_partial_pmu_scenarios(ood_scens)
        X_tr_13, _, _, _ = _extract_full(tr_13)
        X_va_13, _, _, _ = _extract_full(va_13)
        X_te_13, _, _, _ = _extract_full(te_13)
        X_ood_13, _, _, _ = _extract_full(ood_13)
        # Replace any residual all-unobserved feature NaNs with training column medians
        col_med = np.nanmedian(X_tr_13, axis=0)
        col_med = np.where(np.isfinite(col_med), col_med, 0.0)
        X_tr_13 = np.where(np.isfinite(X_tr_13), X_tr_13, col_med)
        if len(X_va_13) > 0:
            X_va_13 = np.where(np.isfinite(X_va_13), X_va_13, col_med)
        X_te_13 = np.where(np.isfinite(X_te_13), X_te_13, col_med)
        if len(X_ood_13) > 0:
            X_ood_13 = np.where(np.isfinite(X_ood_13), X_ood_13, col_med)

        rows.append(
            _fit_and_eval(
                "partial_13bus_pmu",
                f"Partial Spatial Observability ({len(CANONICAL_13BUS_PMU_PLACEMENT)}/39 PMU Buses, 33.3%)",
                X_tr_13,
                X_va_13,
                X_te_13,
                X_ood_13,
            )
        )

        # Ablation 7: 25 fps vs 50 fps synchrophasor reporting rate
        tr_25 = self._downsample_scenarios_to_25fps(train_scens)
        va_25 = self._downsample_scenarios_to_25fps(val_scens)
        te_25 = self._downsample_scenarios_to_25fps(test_scens)
        ood_25 = self._downsample_scenarios_to_25fps(ood_scens)
        X_tr_25, _, _, _ = _extract_full(tr_25)
        X_va_25, _, _, _ = _extract_full(va_25)
        X_te_25, _, _, _ = _extract_full(te_25)
        X_ood_25, _, _, _ = _extract_full(ood_25)

        row_25 = _fit_and_eval(
            "fps_25_vs_50",
            "Reduced PMU Reporting Rate (25 frames/s vs 50 frames/s baseline)",
            X_tr_25,
            X_va_25,
            X_te_25,
            X_ood_25,
        )
        row_25["delta_id_mae_vs_50fps"] = round(float(row_25["id_mae"] - rows[0]["id_mae"]), 4)
        rows.append(row_25)

        return {
            "partition_protocol": protocol,
            "feature_names": feat_cols,
            "columns": list(rows[0].keys()) if rows else [],
            "results": rows,
        }

    @staticmethod
    def compute_multiseed_rank_stability(
        X_train: Any,
        y_train: Any,
        X_eval: Any,
        y_eval: Any,
        feature_names: Optional[Sequence[str]] = None,
        seeds: Sequence[int] = (42, 43, 44, 45, 46),
    ) -> Dict[str, Any]:
        """Evaluate feature-importance rank stability across random seeds."""
        names = _resolve_feature_names(X_train, feature_names)
        X_tr = np.asarray(X_train.values if hasattr(X_train, "values") else X_train, dtype=float)
        y_tr = np.asarray(y_train.values if hasattr(y_train, "values") else y_train, dtype=float).ravel()
        X_ev = np.asarray(X_eval.values if hasattr(X_eval, "values") else X_eval, dtype=float)
        y_ev = np.asarray(y_eval.values if hasattr(y_eval, "values") else y_eval, dtype=float).ravel()

        perm_matrix = np.zeros((len(seeds), X_tr.shape[1]), dtype=float)
        imp_matrix = np.zeros((len(seeds), X_tr.shape[1]), dtype=float)
        top5_sets: List[set] = []

        for idx, s in enumerate(seeds):
            model = GradientBoostingNadirRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, subsample=0.85, random_state=int(s)
            )
            model.fit(X_tr, y_tr)
            imp_df = InterpretabilityAnalyzer.compute_impurity_importance(model, feature_names=names)
            perm_df = InterpretabilityAnalyzer.compute_permutation_importance(
                model, X_ev, y_ev, feature_names=names, task="regression", n_repeats=8, seed=int(s)
            )

            imp_map = dict(zip(imp_df["feature"], imp_df["impurity_importance"]))
            perm_map = dict(zip(perm_df["feature"], perm_df["importance_mean"]))
            for j, fn in enumerate(names):
                imp_matrix[idx, j] = imp_map.get(fn, 0.0)
                perm_matrix[idx, j] = perm_map.get(fn, 0.0)
            top5_sets.append(set(perm_df["feature"].iloc[: min(5, len(names))].tolist()))

        # Pairwise Spearman rank correlations across seeds
        pairwise_corrs: List[float] = []
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                if np.std(perm_matrix[i]) > 1e-12 and np.std(perm_matrix[j]) > 1e-12:
                    rho, _ = spearmanr(perm_matrix[i], perm_matrix[j])
                    if np.isfinite(rho):
                        pairwise_corrs.append(float(rho))
                else:
                    pairwise_corrs.append(1.0)

        # Jaccard top-5 overlap across seeds
        jaccards: List[float] = []
        for i in range(len(top5_sets)):
            for j in range(i + 1, len(top5_sets)):
                union = len(top5_sets[i] | top5_sets[j])
                jaccards.append(len(top5_sets[i] & top5_sets[j]) / max(1, union))

        summary_df = pd.DataFrame({
            "feature": names,
            "perm_importance_mean": np.mean(perm_matrix, axis=0),
            "perm_importance_std": np.std(perm_matrix, axis=0),
            "impurity_importance_mean": np.mean(imp_matrix, axis=0),
            "impurity_importance_std": np.std(imp_matrix, axis=0),
        }).sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)
        summary_df["mean_rank"] = np.arange(1, len(summary_df) + 1, dtype=int)

        return {
            "seeds": list(seeds),
            "mean_pairwise_spearman": float(np.mean(pairwise_corrs)) if pairwise_corrs else 1.0,
            "mean_top5_jaccard": float(np.mean(jaccards)) if jaccards else 1.0,
            "table": summary_df,
        }

    def compute_importance_across_windows(
        self,
        scenarios: List[Dict[str, Any]],
        windows: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    ) -> pd.DataFrame:
        """Track permutation and impurity importance evolution across observation windows T_obs."""
        if not scenarios:
            return pd.DataFrame()

        train_raw, _, test_raw, _, _ = split_scenarios_by_partition(scenarios, random_state=self.seed)
        extractor = FeatureExtractor(f_nominal=self.f_nominal, strict_nadir=True)
        rows: List[Dict[str, Any]] = []

        for w in windows:
            w_flt = float(w)
            train_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(train_raw, w_flt)
            test_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(test_raw, w_flt)
            if not train_scens or not test_scens:
                continue
            X_tr_df, y_tr, _, _ = extractor.transform_batch(train_scens, t_obs=w_flt, as_dataframe=True)
            X_te_df, y_te, _, _ = extractor.transform_batch(test_scens, t_obs=w_flt, as_dataframe=True)
            names = _resolve_feature_names(X_tr_df)

            model = GradientBoostingNadirRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=self.seed)
            model.fit(X_tr_df, y_tr)

            perm_df = self.compute_permutation_importance(
                model, X_te_df, y_te, feature_names=names, task="regression", n_repeats=10, seed=self.seed
            )
            imp_df = self.compute_impurity_importance(model, feature_names=names)
            imp_map = dict(zip(imp_df["feature"], imp_df["impurity_importance"]))

            for _, r in perm_df.iterrows():
                fn = str(r["feature"])
                rows.append({
                    "window_obs": float(w),
                    "feature": fn,
                    "perm_importance_mean": round(float(r["importance_mean"]), 5),
                    "perm_importance_std": round(float(r["importance_std"]), 5),
                    "perm_rank": int(r["rank"]),
                    "impurity_importance": round(float(imp_map.get(fn, 0.0)), 5),
                })

        return pd.DataFrame(rows)

    @staticmethod
    def compute_subgroup_feature_shifts(
        X: Any,
        y_nadir: Any,
        y_secure: Any,
        meta: Any,
        feature_names: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """Quantify physical feature distribution shifts across RES %, outage MW, and security status."""
        names = _resolve_feature_names(X, feature_names)
        X_arr = np.asarray(X.values if hasattr(X, "values") else X, dtype=float)
        yn = np.asarray(y_nadir, dtype=float).ravel()
        ys = np.asarray(y_secure, dtype=int).ravel()
        meta_df = meta if isinstance(meta, pd.DataFrame) else pd.DataFrame(meta)

        df = pd.DataFrame(X_arr, columns=names[: X_arr.shape[1]])
        df["y_nadir"] = yn
        df["is_secure"] = ys
        df["renewable_penetration"] = (
            meta_df["renewable_penetration"].astype(float).values
            if "renewable_penetration" in meta_df.columns
            else np.zeros(len(df))
        )
        df["active_loss_mw"] = (
            meta_df["active_loss_mw"].astype(float).values
            if "active_loss_mw" in meta_df.columns
            else np.zeros(len(df))
        )

        key_feats = [f for f in ["rocof_init", "freq_drop_end", "area_freq_spread_max", "angle_divergence_max", "freq_curvature_mean"] if f in df.columns]
        if not key_feats:
            key_feats = list(df.columns[: min(5, X_arr.shape[1])])

        group_defs: List[Tuple[str, str, pd.Series]] = [
            ("Security Status", "Secure (>=49.5 Hz)", df["is_secure"] == 1),
            ("Security Status", "Insecure (<49.5 Hz)", df["is_secure"] == 0),
            ("RES Penetration", "Low RES (<=20%)", df["renewable_penetration"] <= 0.21),
            ("RES Penetration", "Medium RES (25-35%)", (df["renewable_penetration"] > 0.21) & (df["renewable_penetration"] <= 0.35)),
            ("RES Penetration", "High RES OOD (>=40%)", df["renewable_penetration"] > 0.35),
            ("Outage Severity", "Small Loss (<400 MW)", df["active_loss_mw"] < 400.0),
            ("Outage Severity", "Medium Loss (400-700 MW)", (df["active_loss_mw"] >= 400.0) & (df["active_loss_mw"] <= 700.0)),
            ("Outage Severity", "Large Loss (>700 MW)", df["active_loss_mw"] > 700.0),
        ]

        rows: List[Dict[str, Any]] = []
        for axis_name, subgroup_label, mask in group_defs:
            sub = df.loc[mask]
            if sub.empty:
                continue
            entry: Dict[str, Any] = {
                "Grouping": axis_name,
                "Subgroup": subgroup_label,
                "N": int(len(sub)),
                "Mean_Nadir_Hz": round(float(sub["y_nadir"].mean()), 4),
            }
            for kf in key_feats:
                entry[f"{kf}_mean"] = round(float(sub[kf].mean()), 4)
                entry[f"{kf}_std"] = round(float(sub[kf].std(ddof=0)), 4)
            rows.append(entry)

        return pd.DataFrame(rows)

    @staticmethod
    def dataframe_to_markdown(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return ""
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, (int, np.integer)):
                    vals.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    vals.append("-" if np.isnan(v) else f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
