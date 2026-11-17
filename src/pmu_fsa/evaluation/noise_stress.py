"""PMU measurement noise stress benchmark (unsmoothed, held-out test, all 14 canonical features)."""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark, split_scenarios_by_partition
from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import (
    brier_score as calc_brier_score,
    expected_calibration_error,
)
from pmu_fsa.models.classifiers import compute_binary_security_metrics, get_security_classifier
from pmu_fsa.models.regressors import LinearNadirRegressor, get_nadir_regressor
from pmu_fsa.simulation.pmu_recorder import (
    PMUNoiseConfig,
    apply_pmu_measurement_noise,
    compute_causal_rocof,
    estimate_pmu_system_frequency,
)


class MeasurementNoiseBenchmark:
    """Evaluates prediction and calibration degradation under additive Gaussian PMU noise."""

    def __init__(
        self,
        noise_levels: Optional[List[float]] = None,
        rocof_noise_levels: Optional[List[float]] = None,
        t_obs: float = 1.0,
        threshold: float = 49.50,
        f_nominal: float = 50.0,
        seed: int = 42,
    ) -> None:
        raw_levels = noise_levels if noise_levels is not None else [0.0, 0.005, 0.01, 0.02, 0.05]
        for s in raw_levels:
            if s < 0.0:
                raise ValueError(f"Noise standard deviation must be non-negative, got {s}")

        self.noise_levels = sorted(list(dict.fromkeys([float(s) for s in raw_levels])))
        self.rocof_noise_levels = rocof_noise_levels
        self.t_obs = float(t_obs)
        self.threshold = float(threshold)
        self.f_nominal = float(f_nominal)
        self.seed = int(seed)

    def _inject_noise_into_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        noise_std: float,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """Inject reproducible Gaussian noise into PMU bus frequencies, magnitudes, and angles,
        then recompute causal ROCOF and PMU system frequency through the production PMURecorder functions.
        """
        if noise_std <= 0.0:
            return scenarios

        scale_ratio = float(noise_std / 0.005)
        sigma_v = min(0.02, 0.0005 * scale_ratio)
        sigma_theta_deg = min(2.0, 0.05 * scale_ratio)

        corrupted: List[Dict[str, Any]] = []
        for idx, s in enumerate(scenarios):
            s_copy = copy.deepcopy(s)
            meta = dict(s_copy.get("metadata", {}))
            fps = float(meta.get("fps", 50.0))

            bf = np.asarray(s_copy["bus_frequencies"], dtype=float)
            br = np.asarray(s_copy.get("bus_rocof", np.zeros_like(bf)), dtype=float)
            vm = np.asarray(s_copy["voltage_magnitudes"], dtype=float)
            va = np.asarray(s_copy["voltage_angles"], dtype=float)

            cfg = PMUNoiseConfig(
                sigma_f=float(noise_std),
                sigma_rocof=0.0,
                sigma_v=sigma_v,
                sigma_theta_deg=sigma_theta_deg,
                seed=int(seed),
            )
            f_noisy, rocof_noisy, v_noisy, theta_noisy = apply_pmu_measurement_noise(
                bf, br, vm, va, noise_config=cfg, seed_offset=idx, fps=fps
            )
            f_sys_noisy = estimate_pmu_system_frequency(f_noisy)

            s_copy["bus_frequencies"] = f_noisy
            s_copy["bus_rocof"] = compute_causal_rocof(f_noisy, fps=fps, window_length=5)
            s_copy["voltage_magnitudes"] = v_noisy
            s_copy["voltage_angles"] = theta_noisy
            s_copy["pmu_sys_frequency"] = f_sys_noisy
            # Overwrite coi_frequency with noisy PMU estimate so zero uncorrupted simulator COI can leak
            s_copy["coi_frequency"] = f_sys_noisy
            meta["is_noisy"] = True
            meta["noise_sigma_f_hz"] = float(noise_std)
            s_copy["metadata"] = meta
            corrupted.append(s_copy)
        return corrupted

    def run(
        self,
        scenarios: List[Dict[str, Any]],
        regressor_type: str = "gbdt",
        classifier_type: str = "calibrated_gb",
        regressor: Optional[Any] = None,
        classifier: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train on clean train (+ val) and evaluate across PMU noise levels strictly on held-out test."""
        base_columns = ["noise_std_hz", "mae", "rmse", "ece"]
        if not scenarios:
            return {
                "columns": base_columns,
                "results": [],
                "partition_protocol": "empty",
            }

        train_raw, val_raw, test_raw, _, protocol = split_scenarios_by_partition(
            scenarios, random_state=self.seed
        )
        train_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(train_raw, self.t_obs)
        val_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(val_raw, self.t_obs)
        test_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(test_raw, self.t_obs)

        extractor = FeatureExtractor(t_obs=self.t_obs, f_nominal=self.f_nominal, strict_nadir=True)

        X_tr_df, y_tr_raw, _, _ = extractor.transform_batch(train_scens, t_obs=self.t_obs, as_dataframe=True)
        X_train = X_tr_df.values if hasattr(X_tr_df, "values") else np.asarray(X_tr_df, dtype=float)
        y_train = np.asarray(y_tr_raw, dtype=float)
        ys_train = np.where(y_train >= self.threshold, 1, 0).astype(int)

        if len(val_scens) > 0:
            X_va_df, y_va_raw, _, _ = extractor.transform_batch(val_scens, t_obs=self.t_obs, as_dataframe=True)
            X_val = X_va_df.values if hasattr(X_va_df, "values") else np.asarray(X_va_df, dtype=float)
            y_val = np.asarray(y_va_raw, dtype=float)
        else:
            X_val, y_val = None, None

        # Exact same 14 canonical features on clean held-out test scenarios (ZERO dropped columns)
        X_te_clean_df, y_te_raw, _, _ = extractor.transform_batch(test_scens, t_obs=self.t_obs, as_dataframe=True)
        X_test_clean = X_te_clean_df.values if hasattr(X_te_clean_df, "values") else np.asarray(X_te_clean_df, dtype=float)
        y_test = np.asarray(y_te_raw, dtype=float)
        ys_test = np.where(y_test >= self.threshold, 1, 0).astype(int)
        feature_names = list(X_te_clean_df.columns) if hasattr(X_te_clean_df, "columns") else []

        if regressor is not None and getattr(regressor, "n_features_in_", X_test_clean.shape[1]) == X_test_clean.shape[1]:
            reg = regressor
        else:
            try:
                reg = get_nadir_regressor(regressor_type, random_state=self.seed)
            except TypeError:
                reg = get_nadir_regressor(regressor_type)
            except Exception:
                reg = LinearNadirRegressor()
            reg.fit(X_train, y_train)

        if classifier is not None and getattr(classifier, "n_features_in_", X_test_clean.shape[1]) == X_test_clean.shape[1]:
            clf = classifier
        else:
            if classifier_type == "calibrated_gb":
                clf = CalibratedGBDTClassifier(
                    method="platt",
                    threshold=self.threshold,
                    random_state=self.seed,
                )
                if X_val is not None and y_val is not None and len(X_val) >= 2:
                    clf.fit(X_train, y_train, ys_train, X_cal=X_val, y_cal=y_val)
                else:
                    clf.fit(X_train, y_train, ys_train)
            else:
                try:
                    clf = get_security_classifier(classifier_type, threshold=self.threshold)
                except Exception:
                    clf = CalibratedGBDTClassifier(method="platt", threshold=self.threshold)
                clf.fit(X_train, ys_train)

        # Clean held-out reference probabilities for probability-stability tracking
        probs_clean_ref = clf.predict_proba(X_test_clean)
        p_clean_sec = np.asarray(
            probs_clean_ref[:, 1] if hasattr(probs_clean_ref, "ndim") and probs_clean_ref.ndim > 1 else probs_clean_ref,
            dtype=float,
        )

        rows: List[Dict[str, Any]] = []
        for idx, noise in enumerate(self.noise_levels):
            if noise == 0.0:
                X_eval = X_test_clean
            else:
                noisy_test_scens = self._inject_noise_into_scenarios(
                    test_scens,
                    noise_std=noise,
                    seed=self.seed + int(round(noise * 100000)) + idx * 1000,
                )
                X_eval_df, _, _, _ = extractor.transform_batch(
                    noisy_test_scens, t_obs=self.t_obs, as_dataframe=True
                )
                X_eval = X_eval_df.values if hasattr(X_eval_df, "values") else np.asarray(X_eval_df, dtype=float)

            reg_m = reg.evaluate(X_eval, y_test)
            probs = clf.predict_proba(X_eval)
            p_sec = np.asarray(probs[:, 1] if hasattr(probs, "ndim") and probs.ndim > 1 else probs, dtype=float)
            y_sec_pred = (p_sec >= 0.5).astype(int)
            cls_m = compute_binary_security_metrics(ys_test, y_sec_pred, p_secure=p_sec)

            ece = float(expected_calibration_error(ys_test, p_sec, n_bins=10, threshold=self.threshold))
            brier = float(calc_brier_score(ys_test, p_sec, threshold=self.threshold))
            prob_stability = float(np.mean(np.abs(p_sec - p_clean_sec))) if len(p_sec) > 0 else 0.0

            mae = float(reg_m["mae"])
            rmse = float(reg_m["rmse"])
            f1 = float(cls_m["f1"])

            rows.append({
                "noise_std_hz": float(noise),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "bias": round(float(reg_m["bias"]), 4),
                "median_ae": round(float(reg_m["median_ae"]), 4),
                "p95_ae": round(float(reg_m["p95_ae"]), 4),
                "r2": round(float(reg_m["r2"]), 4),
                "precision": round(float(cls_m["precision"]), 4),
                "recall": round(float(cls_m["recall"]), 4),
                "f1": round(f1, 4),
                "fnr_insecure": round(float(cls_m["fnr_insecure"]), 4),
                "missed_insecure_count": int(cls_m["missed_insecure_count"]),
                "fpr_secure": round(float(cls_m["fpr_secure"]), 4),
                "brier_score": round(brier, 4),
                "ece": round(ece, 4),
                "prob_stability_mad": round(prob_stability, 4),
                "n_features_used": int(X_eval.shape[1]),
                "mae_raw": round(mae, 4),
                "rmse_raw": round(rmse, 4),
                "f1_raw": round(f1, 4),
            })

        return {
            "columns": base_columns,
            "results": rows,
            "partition_protocol": protocol,
            "n_train": len(train_scens),
            "n_val": len(val_scens),
            "n_test": len(test_scens),
            "feature_names": feature_names,
        }

    @staticmethod
    def compare_ols_vs_endpoint_rocof(
        scenarios: Optional[List[Dict[str, Any]]] = None,
        noise_std: float = 0.03,
        n_trials: int = 50,
        seed: int = 99,
    ) -> Dict[str, Any]:
        """Compare Monte Carlo slope variance of windowed OLS vs 2-point finite difference."""
        if n_trials <= 0:
            raise ValueError("n_trials must be >= 1")

        rng = random.Random(seed)
        true_slope = -0.50
        dt = 0.02
        n_points = 26
        t_vals = [i * dt for i in range(n_points)]

        ols_slopes: List[float] = []
        fd_slopes: List[float] = []

        for _ in range(n_trials):
            noisy_f = [50.0 + true_slope * t + rng.gauss(0.0, noise_std) for t in t_vals]

            fd = (noisy_f[1] - noisy_f[0]) / dt
            fd_slopes.append(fd)

            n = len(t_vals)
            mean_t = sum(t_vals) / n
            mean_f = sum(noisy_f) / n
            ss_tt = sum((t - mean_t) ** 2 for t in t_vals)
            ss_tf = sum((t - mean_t) * (f - mean_f) for t, f in zip(t_vals, noisy_f))
            ols = ss_tf / ss_tt if ss_tt > 0 else 0.0
            ols_slopes.append(ols)

        mean_fd = sum(fd_slopes) / len(fd_slopes)
        var_fd = sum((v - mean_fd) ** 2 for v in fd_slopes) / len(fd_slopes)

        mean_ols = sum(ols_slopes) / len(ols_slopes)
        var_ols = sum((v - mean_ols) ** 2 for v in ols_slopes) / len(ols_slopes)

        variance_ratio = float(var_ols / var_fd) if var_fd > 0 else 0.0
        verified = bool(variance_ratio < 0.15)

        return {
            "var_ols": float(round(var_ols, 6)),
            "var_endpoint": float(round(var_fd, 6)),
            "variance_ratio": float(round(variance_ratio, 4)),
            "noise_resilience_verified": verified,
        }

    def to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        rows = results.get("results", [])
        cols = results.get("columns", ["noise_std_hz", "mae", "rmse", "ece"])
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
                if isinstance(v, (int, np.integer)):
                    vals.append(str(int(v)))
                elif isinstance(v, (float, np.floating)):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
