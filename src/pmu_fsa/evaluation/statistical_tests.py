"""Statistical significance tests, paired bootstrap CIs, learning curves, and multi-seed stability."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import r2_score

from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import brier_score, expected_calibration_error
from pmu_fsa.models.classifiers import compute_binary_security_metrics, get_security_classifier
from pmu_fsa.models.regressors import get_nadir_regressor


def _resolve_metric_fn(metric: str | Callable[[np.ndarray, np.ndarray], float]) -> Callable[[np.ndarray, np.ndarray], float]:
    if callable(metric):
        return metric
    key = metric.strip().lower()
    if key == "mae":
        return lambda yt, yp: float(np.mean(np.abs(yt - yp)))
    if key == "rmse":
        return lambda yt, yp: float(math.sqrt(float(np.mean((yt - yp) ** 2))))
    if key == "bias":
        return lambda yt, yp: float(np.mean(yp - yt))
    if key == "r2":
        def _r2(yt: np.ndarray, yp: np.ndarray) -> float:
            if float(np.var(yt)) < 1e-12:
                return 1.0 if np.allclose(yt, yp, atol=1e-4) else 0.0
            return float(r2_score(yt, yp))
        return _r2
    if key == "f1":
        return lambda yt, yp: float(compute_binary_security_metrics(yt, (yp >= 0.5).astype(int))["f1"])
    if key == "fnr_insecure":
        return lambda yt, yp: float(compute_binary_security_metrics(yt, (yp >= 0.5).astype(int))["fnr_insecure"])
    if key in ("brier", "brier_score"):
        return lambda yt, yp: float(brier_score(yt, yp))
    if key == "ece":
        return lambda yt, yp: float(expected_calibration_error(yt, yp, n_bins=10))
    raise ValueError(f"Unsupported metric '{metric}'")


def paired_bootstrap_ci(
    y_true: Any,
    y_pred: Any,
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "mae",
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute scenario-level bootstrap confidence interval for a metric."""
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)}, len(y_pred)={len(yp)}")

    n = len(yt)
    if n == 0:
        return {
            "estimate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "std_error": 0.0,
            "n_boot": int(n_boot),
            "n_samples": 0,
        }

    fn = _resolve_metric_fn(metric)
    point_est = float(fn(yt, yp))
    if n < 2 or n_boot < 1:
        return {
            "estimate": point_est,
            "ci_low": point_est,
            "ci_high": point_est,
            "std_error": 0.0,
            "n_boot": int(n_boot),
            "n_samples": n,
        }

    rng = np.random.default_rng(seed)
    boot_vals = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot_vals[b] = fn(yt[idx], yp[idx])

    alpha = 1.0 - float(confidence_level)
    low_pct = 100.0 * (0.5 * alpha)
    high_pct = 100.0 * (1.0 - 0.5 * alpha)
    return {
        "estimate": point_est,
        "ci_low": float(np.percentile(boot_vals, low_pct)),
        "ci_high": float(np.percentile(boot_vals, high_pct)),
        "std_error": float(np.std(boot_vals, ddof=1)),
        "n_boot": int(n_boot),
        "n_samples": n,
    }


def holm_adjust_pvalues(p_values: Sequence[float]) -> List[float]:
    """Apply Holm (1979) step-down FWER adjustment to a family of raw p-values."""
    m = len(p_values)
    if m == 0:
        return []
    raw = np.clip(np.asarray(p_values, dtype=float), 0.0, 1.0)
    order = np.argsort(raw, kind="mergesort")
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        mult = m - rank_idx
        val = min(1.0, float(mult * raw[orig_idx]))
        running_max = max(running_max, val)
        adjusted_sorted[rank_idx] = running_max
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return [float(x) for x in adjusted]


def _sign_flip_permutation_pvalue(
    diff_per_scen: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided paired sign-flip permutation p-value under H0: E[d_i] = 0 (symmetric d_i)."""
    d = np.asarray(diff_per_scen, dtype=float).ravel()
    n = len(d)
    if n == 0 or not np.any(np.abs(d) > 1e-12):
        return 1.0

    obs_abs_mean = abs(float(np.mean(d)))
    if obs_abs_mean <= 1e-15:
        return 1.0

    # Exact enumeration for small paired sample sizes; Monte Carlo Rademacher flips otherwise
    if n <= 16:
        bits = ((np.arange(1 << n, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)) & 1)
        signs = np.where(bits == 1, 1.0, -1.0)
        null_abs_means = np.abs((signs @ d) / float(n))
        return float(np.mean(null_abs_means >= obs_abs_mean - 1e-15))

    m = max(1000, int(n_perm))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(m, n))
    null_abs_means = np.abs((signs @ d) / float(n))
    exceed = int(np.sum(null_abs_means >= obs_abs_mean - 1e-15))
    return float((1 + exceed) / (1 + m))


def paired_model_comparison_test(
    y_true: Any,
    pred_a: Any,
    pred_b: Any,
    task: str = "regression",
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare Model A vs Model B on identical held-out scenarios via paired sign-flip permutation,
    null-centered paired bootstrap, and separate Wilcoxon signed-rank diagnostic.

    Primary test: two-sided paired sign-flip permutation test on scenario loss differences
    d_i = loss_i(A) - loss_i(B) under H0: E[d_i] = 0 (negative delta = Model A lower loss).
    Wilcoxon signed-rank is reported separately as a rank-symmetry diagnostic and is never
    combined via an unadjusted OR rule.
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    pa = np.asarray(pred_a, dtype=float).ravel()
    pb = np.asarray(pred_b, dtype=float).ravel()
    if not (len(yt) == len(pa) == len(pb)):
        raise ValueError("y_true, pred_a, and pred_b must have identical length")

    n = len(yt)
    if n == 0:
        return {
            "n_samples": 0,
            "metric_name": "mae" if task == "regression" else "brier_score",
            "direction": "lower_is_better",
            "model_a_estimate": 0.0,
            "model_b_estimate": 0.0,
            "delta_a_minus_b": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "test_statistic_mean_diff": 0.0,
            "wilcoxon_statistic": 0.0,
            "p_value_permutation": 1.0,
            "p_value_bootstrap": 1.0,
            "p_value_wilcoxon": 1.0,
            "significant_at_005": False,
            "test_disagreement_note": "none",
            "practical_significance": "none",
        }

    if task == "regression":
        loss_a = np.abs(yt - pa)
        loss_b = np.abs(yt - pb)
        metric_fn = _resolve_metric_fn("mae")
        metric_name = "mae"
    else:
        yt_bin = np.where(yt > 1.5, (yt >= 49.50).astype(float), (yt >= 0.5).astype(float))
        yt = yt_bin
        loss_a = (pa - yt) ** 2
        loss_b = (pb - yt) ** 2
        metric_fn = _resolve_metric_fn("brier_score")
        metric_name = "brier_score"

    est_a = float(metric_fn(yt, pa))
    est_b = float(metric_fn(yt, pb))
    delta_obs = float(est_a - est_b)
    diff_per_scen = loss_a - loss_b

    if n >= 5 and np.any(np.abs(diff_per_scen) > 1e-12):
        try:
            w_stat, p_wilcoxon = wilcoxon(loss_a, loss_b, zero_method="wilcox", alternative="two-sided")
            w_stat = float(w_stat)
            p_wilcoxon = float(p_wilcoxon)
        except Exception:
            w_stat, p_wilcoxon = 0.0, 1.0
    else:
        w_stat, p_wilcoxon = 0.0, 1.0

    p_perm = _sign_flip_permutation_pvalue(
        diff_per_scen, n_perm=max(5000, int(n_boot) * 2), seed=seed + 10007
    )

    if n < 2 or n_boot < 1 or not np.any(np.abs(diff_per_scen) > 1e-12):
        ci_low, ci_high = delta_obs, delta_obs
        p_boot = 1.0 if not np.any(np.abs(diff_per_scen) > 1e-12) else p_perm
    else:
        rng = np.random.default_rng(seed)
        m_boot = int(n_boot)
        idx_mat = rng.integers(0, n, size=(m_boot, n))
        boot_deltas = np.mean(diff_per_scen[idx_mat], axis=1)

        alpha = 1.0 - float(confidence_level)
        ci_low = float(np.percentile(boot_deltas, 100.0 * (0.5 * alpha)))
        ci_high = float(np.percentile(boot_deltas, 100.0 * (1.0 - 0.5 * alpha)))

        # Null-centered bootstrap p-value under H0: E[d_i] = 0
        centered_d = diff_per_scen - np.mean(diff_per_scen)
        null_boot_abs = np.abs(np.mean(centered_d[idx_mat], axis=1))
        exceed_boot = int(np.sum(null_boot_abs >= abs(delta_obs) - 1e-15))
        p_boot = float((1 + exceed_boot) / (1 + m_boot))

    sig_primary = bool(p_perm < 0.05)
    sig_wilcox = bool(p_wilcoxon < 0.05)
    if sig_primary != sig_wilcox:
        disagreement = (
            "Wilcoxon tests rank symmetry of non-zero differences (sensitive to many tiny probability/residual "
            "offsets vs rare large errors), whereas primary sign-flip permutation tests mean loss difference E[d_i]=0."
        )
    else:
        disagreement = "consistent"

    abs_delta = abs(delta_obs)
    if task == "regression":
        if abs_delta >= 0.020:
            prac = "material (>=20 mHz MAE difference)"
        elif abs_delta >= 0.005:
            prac = "moderate (5-20 mHz MAE difference)"
        else:
            prac = "negligible (<5 mHz; below typical PMU frequency noise floor)"
    else:
        if abs_delta >= 0.010:
            prac = "material (>=0.010 Brier difference)"
        elif abs_delta >= 0.002:
            prac = "small (0.002-0.010 Brier difference)"
        else:
            prac = "negligible (<0.002 Brier difference)"

    return {
        "n_samples": n,
        "metric_name": metric_name,
        "direction": "lower_is_better",
        "model_a_estimate": est_a,
        "model_b_estimate": est_b,
        "delta_a_minus_b": delta_obs,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "test_statistic_mean_diff": delta_obs,
        "wilcoxon_statistic": w_stat,
        "p_value_permutation": p_perm,
        "p_value_bootstrap": p_boot,
        "p_value_wilcoxon": p_wilcoxon,
        "significant_at_005": sig_primary,
        "test_disagreement_note": disagreement,
        "practical_significance": prac,
    }


def compare_all_models_statistical(
    y_true: Any,
    predictions: Dict[str, np.ndarray],
    reference_model: str,
    task: str = "regression",
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare a reference model against every baseline model on identical held-out targets
    with Holm step-down multiple-comparison adjustment across the comparison family.
    """
    if reference_model not in predictions:
        raise KeyError(f"Reference model '{reference_model}' not found in predictions dict")

    ref_pred = predictions[reference_model]
    yt_arr = np.asarray(y_true, dtype=float).ravel()
    rows: List[Dict[str, Any]] = []
    non_ref_indices: List[int] = []
    raw_perm_pvals: List[float] = []

    for idx, (name, pred) in enumerate(predictions.items()):
        if name == reference_model:
            ci = paired_bootstrap_ci(
                y_true,
                ref_pred,
                metric="mae" if task == "regression" else "brier_score",
                n_boot=n_boot,
                seed=seed,
            )
            rows.append({
                "reference_model": reference_model,
                "baseline_model": name,
                "n_paired": int(len(yt_arr)),
                "metric": "mae" if task == "regression" else "brier_score",
                "direction": "lower_is_better",
                "reference_score": round(ci["estimate"], 5),
                "baseline_score": round(ci["estimate"], 5),
                "delta_ref_minus_base": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
                "test_statistic": 0.0,
                "p_value_permutation": 1.0,
                "p_holm_adjusted": 1.0,
                "p_value_bootstrap": 1.0,
                "p_value_wilcoxon": 1.0,
                "significant_unadjusted_005": False,
                "significant_holm_005": False,
                "significant_p005": False,
                "practical_significance": "self_reference",
                "test_disagreement_note": "self_reference",
            })
            continue

        comp = paired_model_comparison_test(
            y_true,
            ref_pred,
            pred,
            task=task,
            n_boot=n_boot,
            seed=seed + idx + 1,
        )
        non_ref_indices.append(len(rows))
        raw_perm_pvals.append(float(comp["p_value_permutation"]))
        rows.append({
            "reference_model": reference_model,
            "baseline_model": name,
            "n_paired": int(comp["n_samples"]),
            "metric": comp["metric_name"],
            "direction": comp["direction"],
            "reference_score": round(comp["model_a_estimate"], 5),
            "baseline_score": round(comp["model_b_estimate"], 5),
            "delta_ref_minus_base": round(comp["delta_a_minus_b"], 5),
            "ci95_low": round(comp["ci_low"], 5),
            "ci95_high": round(comp["ci_high"], 5),
            "test_statistic": round(comp["test_statistic_mean_diff"], 5),
            "p_value_permutation": round(comp["p_value_permutation"], 6),
            "p_holm_adjusted": 1.0,
            "p_value_bootstrap": round(comp["p_value_bootstrap"], 6),
            "p_value_wilcoxon": round(comp["p_value_wilcoxon"], 6),
            "significant_unadjusted_005": bool(comp["significant_at_005"]),
            "significant_holm_005": False,
            "significant_p005": False,
            "practical_significance": comp["practical_significance"],
            "test_disagreement_note": comp["test_disagreement_note"],
        })

    if non_ref_indices:
        holm_pvals = holm_adjust_pvalues(raw_perm_pvals)
        for row_idx, p_adj in zip(non_ref_indices, holm_pvals):
            rows[row_idx]["p_holm_adjusted"] = round(float(p_adj), 6)
            sig_holm = bool(p_adj < 0.05)
            rows[row_idx]["significant_holm_005"] = sig_holm
            rows[row_idx]["significant_p005"] = sig_holm

    return pd.DataFrame(rows)


class LearningCurveEvaluator:
    """Evaluates held-out test performance across training set fractions [20%, 40%, 60%, 80%, 100%]."""

    def __init__(
        self,
        train_fractions: Optional[Sequence[float]] = None,
        regressor_type: str = "gbdt",
        classifier_type: str = "calibrated_gb",
        threshold: float = 49.50,
        n_repeats: int = 3,
        seed: int = 42,
    ) -> None:
        fracs = list(train_fractions) if train_fractions is not None else [0.20, 0.40, 0.60, 0.80, 1.00]
        for f in fracs:
            if not (0.0 < float(f) <= 1.0):
                raise ValueError(f"train_fractions entries must be in (0, 1], got {f}")
        self.train_fractions = sorted(list(dict.fromkeys([float(f) for f in fracs])))
        self.regressor_type = regressor_type
        self.classifier_type = classifier_type
        self.threshold = float(threshold)
        self.n_repeats = max(1, int(n_repeats))
        self.seed = int(seed)

    def run(
        self,
        X_train: np.ndarray,
        y_train_nadir: np.ndarray,
        y_train_sec: np.ndarray,
        X_test: np.ndarray,
        y_test_nadir: np.ndarray,
        y_test_sec: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val_nadir: Optional[np.ndarray] = None,
        y_val_sec: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train_nadir, dtype=float).ravel()
        ys_tr = np.asarray(y_train_sec, dtype=int).ravel()
        X_te = np.asarray(X_test, dtype=float)
        y_te = np.asarray(y_test_nadir, dtype=float).ravel()
        ys_te = np.asarray(y_test_sec, dtype=int).ravel()

        n_total = len(X_tr)
        rows: List[Dict[str, Any]] = []

        for frac in self.train_fractions:
            n_sub = max(2, int(round(frac * n_total))) if n_total >= 2 else n_total
            n_sub = min(n_sub, n_total)
            reps = 1 if frac >= 0.999 else self.n_repeats

            maes, rmses, r2s, f1s, briers, eces, fnrs = [], [], [], [], [], [], []
            for r_idx in range(reps):
                rng = np.random.default_rng(self.seed + int(frac * 1000) + r_idx)
                if n_sub < n_total:
                    sub_idx = rng.choice(n_total, size=n_sub, replace=False)
                    X_sub, y_sub, ys_sub = X_tr[sub_idx], y_tr[sub_idx], ys_tr[sub_idx]
                else:
                    X_sub, y_sub, ys_sub = X_tr, y_tr, ys_tr

                reg = get_nadir_regressor(self.regressor_type, random_state=self.seed + r_idx) \
                    if self.regressor_type not in ("mean", "analytical", "linear", "ridge", "linear_nadir") \
                    else get_nadir_regressor(self.regressor_type)
                reg.fit(X_sub, y_sub)
                reg_m = reg.evaluate(X_te, y_te)
                maes.append(reg_m["mae"])
                rmses.append(reg_m["rmse"])
                r2s.append(reg_m["r2"])

                if self.classifier_type == "calibrated_gb":
                    clf = CalibratedGBDTClassifier(
                        method="platt",
                        threshold=self.threshold,
                        random_state=self.seed + r_idx,
                    )
                    if X_val is not None and y_val_nadir is not None and len(X_val) >= 2:
                        clf.fit(X_sub, y_sub, ys_sub, X_cal=X_val, y_cal=y_val_nadir)
                    else:
                        clf.fit(X_sub, y_sub, ys_sub)
                else:
                    clf = get_security_classifier(self.classifier_type, threshold=self.threshold)
                    clf.fit(X_sub, ys_sub)

                probs = clf.predict_proba(X_te)[:, 1]
                preds_sec = (probs >= 0.5).astype(int)
                cls_m = compute_binary_security_metrics(ys_te, preds_sec, p_secure=probs)
                f1s.append(cls_m["f1"])
                fnrs.append(cls_m["fnr_insecure"])
                briers.append(brier_score(ys_te, probs, threshold=self.threshold))
                eces.append(expected_calibration_error(ys_te, probs, n_bins=10, threshold=self.threshold))

            rows.append({
                "train_fraction": float(frac),
                "n_train_samples": int(n_sub),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "rmse_mean": float(np.mean(rmses)),
                "r2_mean": float(np.mean(r2s)),
                "f1_mean": float(np.mean(f1s)),
                "f1_std": float(np.std(f1s)),
                "brier_mean": float(np.mean(briers)),
                "ece_mean": float(np.mean(eces)),
                "fnr_insecure_mean": float(np.mean(fnrs)),
            })

        return {
            "columns": list(rows[0].keys()) if rows else [],
            "results": rows,
        }

    def to_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(report.get("results", []))

    def to_markdown(self, report: Dict[str, Any]) -> str:
        df = self.to_dataframe(report)
        if df.empty:
            return ""
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(int(v) if isinstance(v, (int, np.integer)) else v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)


class MultiSeedStabilityEvaluator:
    """Evaluates multi-seed training stability across seeds [42, 43, 44, 45, 46] on held-out test."""

    def __init__(
        self,
        seeds: Optional[Sequence[int]] = None,
        threshold: float = 49.50,
    ) -> None:
        self.seeds = [int(s) for s in (seeds if seeds is not None else [42, 43, 44, 45, 46])]
        self.threshold = float(threshold)

    def run(
        self,
        X_train: np.ndarray,
        y_train_nadir: np.ndarray,
        y_train_sec: np.ndarray,
        X_test: np.ndarray,
        y_test_nadir: np.ndarray,
        y_test_sec: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val_nadir: Optional[np.ndarray] = None,
        regressor_types: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        regs = list(regressor_types) if regressor_types is not None else ["rf", "gbdt", "lgbm"]
        X_tr = np.asarray(X_train, dtype=float)
        y_tr = np.asarray(y_train_nadir, dtype=float).ravel()
        ys_tr = np.asarray(y_train_sec, dtype=int).ravel()
        X_te = np.asarray(X_test, dtype=float)
        y_te = np.asarray(y_test_nadir, dtype=float).ravel()
        ys_te = np.asarray(y_test_sec, dtype=int).ravel()

        per_seed_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []

        for m_name in regs:
            maes, rmses, r2s = [], [], []
            for s in self.seeds:
                # Subsample 90% without replacement across seeds when model is deterministic tree (subsample<1 sensitivity)
                reg = get_nadir_regressor(m_name, random_state=s, subsample=0.85) \
                    if m_name in ("gbdt", "gradient_boosting", "gbr") \
                    else get_nadir_regressor(m_name, random_state=s)
                reg.fit(X_tr, y_tr)
                ev = reg.evaluate(X_te, y_te)
                maes.append(ev["mae"])
                rmses.append(ev["rmse"])
                r2s.append(ev["r2"])
                per_seed_rows.append({
                    "model": m_name,
                    "seed": s,
                    "mae": ev["mae"],
                    "rmse": ev["rmse"],
                    "r2": ev["r2"],
                })
            summary_rows.append({
                "model": f"reg_{m_name}",
                "n_seeds": len(self.seeds),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes)),
                "rmse_mean": float(np.mean(rmses)),
                "rmse_std": float(np.std(rmses)),
                "r2_mean": float(np.mean(r2s)),
                "f1_mean": float("nan"),
                "f1_std": float("nan"),
                "ece_mean": float("nan"),
                "ece_std": float("nan"),
            })

        # Calibrated GBDT multi-seed stability
        f1s, eces, briers, fnrs = [], [], [], []
        for s in self.seeds:
            clf = CalibratedGBDTClassifier(
                method="platt",
                threshold=self.threshold,
                random_state=s,
            )
            # Slight bootstrap perturbation of tree subsample via random_state + CV folds
            if X_val is not None and y_val_nadir is not None and len(X_val) >= 2:
                clf.fit(X_tr, y_tr, ys_tr, X_cal=X_val, y_cal=y_val_nadir)
            else:
                clf.fit(X_tr, y_tr, ys_tr)
            ev_c = clf.evaluate(X_te, ys_te)
            f1s.append(ev_c["f1"])
            eces.append(ev_c["ece"])
            briers.append(ev_c["brier_score"])
            fnrs.append(ev_c["fnr_insecure"])
            per_seed_rows.append({
                "model": "calibrated_gb_platt",
                "seed": s,
                "f1": ev_c["f1"],
                "ece": ev_c["ece"],
                "brier_score": ev_c["brier_score"],
                "fnr_insecure": ev_c["fnr_insecure"],
            })

        summary_rows.append({
            "model": "calibrated_gb_platt",
            "n_seeds": len(self.seeds),
            "mae_mean": float("nan"),
            "mae_std": float("nan"),
            "rmse_mean": float("nan"),
            "rmse_std": float("nan"),
            "r2_mean": float("nan"),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "ece_mean": float(np.mean(eces)),
            "ece_std": float(np.std(eces)),
        })

        return {
            "seeds": self.seeds,
            "per_seed": per_seed_rows,
            "summary": summary_rows,
        }

    def to_dataframe(self, report: Dict[str, Any]) -> pd.DataFrame:
        return pd.DataFrame(report.get("summary", []))

    def to_markdown(self, report: Dict[str, Any]) -> str:
        df = self.to_dataframe(report)
        if df.empty:
            return ""
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for v in row:
                if isinstance(v, (float, np.floating)):
                    vals.append("-" if math.isnan(float(v)) else f"{v:.4f}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
