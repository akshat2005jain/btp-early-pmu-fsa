#!/usr/bin/env python3
"""Train leak-free nadir regressors and calibrated security classifiers (50.0 Hz standard)."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import scipy
import sklearn

try:
    import joblib
except ImportError:
    import pickle as joblib

from pmu_fsa.evaluation.interpretability import CANONICAL_14_FEATURES
from pmu_fsa.evaluation.statistical_tests import paired_bootstrap_ci
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import compute_calibration_metrics
from pmu_fsa.models.classifiers import (
    DecisionTreeSecurityClassifier,
    GradientBoostingSecurityClassifier,
    LightGBMSecurityClassifier,
    LogisticSecurityClassifier,
    RandomForestSecurityClassifier,
    compute_binary_security_metrics,
    evaluate_probability_threshold_sweep,
)
from pmu_fsa.models.regressors import (
    AnalyticalSwingNadirRegressor,
    DecisionTreeNadirRegressor,
    GradientBoostingNadirRegressor,
    LightGBMNadirRegressor,
    LinearNadirRegressor,
    MeanNadirRegressor,
    RandomForestNadirRegressor,
)

logger = logging.getLogger("train_models")


def bootstrap_mae_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute 95% scenario-level bootstrap confidence interval for MAE."""
    res = paired_bootstrap_ci(y_true, y_pred, metric="mae", n_boot=n_boot, seed=seed)
    return float(res["ci_low"]), float(res["ci_high"])


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for v in r:
            if isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                vals.append("-" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _compute_subgroup_breakdown(
    y_true_nadir: np.ndarray,
    y_pred_nadir: np.ndarray,
    y_true_sec: np.ndarray,
    p_pred_sec: np.ndarray,
    meta_dict: Dict[str, np.ndarray],
    threshold: float = 49.50,
) -> List[Dict[str, Any]]:
    """Compute held-out test regression & classification metrics across physical subgroups."""
    n = len(y_true_nadir)
    if n == 0:
        return []

    y_pred_sec = (p_pred_sec >= 0.5).astype(int)
    rows: List[Dict[str, Any]] = []

    def _eval_mask(axis: str, group_name: str, mask: np.ndarray) -> None:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return
        yt_n, yp_n = y_true_nadir[idx], y_pred_nadir[idx]
        yt_s, yp_s, ps = y_true_sec[idx], y_pred_sec[idx], p_pred_sec[idx]
        diff = yp_n - yt_n
        cls_m = compute_binary_security_metrics(yt_s, yp_s, p_secure=ps)
        cal_m = compute_calibration_metrics(yt_s, ps, n_bins=10, threshold=threshold)
        rows.append({
            "Axis": axis,
            "Subgroup": group_name,
            "N": int(len(idx)),
            "MAE_Hz": round(float(np.mean(np.abs(diff))), 4),
            "RMSE_Hz": round(float(np.sqrt(np.mean(diff ** 2))), 4),
            "Bias_Hz": round(float(np.mean(diff)), 4),
            "F1": round(float(cls_m["f1"]), 4),
            "FNR_insecure": round(float(cls_m["fnr_insecure"]), 4),
            "Missed_Insecure": int(cls_m["missed_insecure_count"]),
            "Brier": round(float(cal_m["brier_score"]), 4),
            "ECE": round(float(cal_m["ece"]), 4),
        })

    if "renewable_penetration" in meta_dict and len(meta_dict["renewable_penetration"]) == n:
        ren = np.asarray(meta_dict["renewable_penetration"], dtype=float)
        for val in sorted(np.unique(np.round(ren, 2))):
            _eval_mask("RES Penetration", f"RES {val*100:.0f}%", np.isclose(ren, val, atol=0.015))

    if "active_loss_mw" in meta_dict and len(meta_dict["active_loss_mw"]) == n:
        mw = np.asarray(meta_dict["active_loss_mw"], dtype=float)
        _eval_mask("Outage MW", "Small (<400 MW)", mw < 400.0)
        _eval_mask("Outage MW", "Medium (400-700 MW)", (mw >= 400.0) & (mw <= 700.0))
        _eval_mask("Outage MW", "Large (>700 MW)", mw > 700.0)

    if "load_scale" in meta_dict and len(meta_dict["load_scale"]) == n:
        ld = np.asarray(meta_dict["load_scale"], dtype=float)
        for val in sorted(np.unique(np.round(ld, 2))):
            _eval_mask("Load Scale", f"Load {val:.2f}x", np.isclose(ld, val, atol=0.02))

    return rows


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Frequency Security Model Trainer (50.0 Hz)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--features", type=str, default="data/processed/features.npz", help="Path to features.npz")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing features.npz")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["mean", "analytical", "linear", "dt", "rf", "gbdt", "lgbm", "calibrated_gb", "all"],
        default="all",
        help="Model family to train",
    )
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Explicit list of models to train")
    parser.add_argument(
        "--calibration",
        type=str,
        choices=["platt", "isotonic"],
        default="platt",
        help="Primary calibration method for calibrated_gb.joblib",
    )
    parser.add_argument(
        "--strict-disjoint-cal",
        action="store_true",
        help="Keep base GBDT trained strictly on train without cross-fit refit on train+val",
    )
    parser.add_argument("--threshold", type=float, default=49.50, help="Security threshold (Hz)")
    parser.add_argument("--output-dir", type=str, default="models/", help="Directory to save models")
    parser.add_argument("--tables-dir", type=str, default=None, help="Directory to save regression & classification tables")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    feat_path = Path(parsed.features)
    if parsed.data_dir is not None and not feat_path.exists():
        feat_path = Path(parsed.data_dir) / "features.npz"

    if not feat_path.exists():
        logger.error("Features file not found at: %s", feat_path)
        return 1

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prevent overwriting live results/tables when --output-dir points to a temp directory without --tables-dir
    if parsed.tables_dir is not None:
        tables_dir = Path(parsed.tables_dir)
    elif out_dir.resolve() == (PROJECT_ROOT / "models").resolve():
        tables_dir = PROJECT_ROOT / "results" / "tables"
    else:
        tables_dir = out_dir.parent / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    with np.load(feat_path, allow_pickle=True) as data:
        X = np.asarray(data["X"], dtype=float)
        y_nadir = np.asarray(data["y_nadir"], dtype=float).ravel()
        y_secure = (
            np.asarray(data["y_secure"], dtype=int).ravel()
            if "y_secure" in data
            else np.where(y_nadir >= parsed.threshold, 1, 0).astype(int)
        )
        partitions = (
            np.asarray([str(p) for p in data["partitions"]])
            if "partitions" in data
            else np.array(["train"] * len(y_nadir))
        )
        scenario_ids = (
            np.asarray([str(s) for s in data["scenario_ids"]])
            if "scenario_ids" in data
            else np.array([f"scen_{i:04d}" for i in range(len(y_nadir))])
        )
        if "feature_names" in data:
            feature_names = [str(f) for f in data["feature_names"]]
        elif X.shape[1] == 14:
            feature_names = list(CANONICAL_14_FEATURES)
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        extra_meta: Dict[str, np.ndarray] = {}
        for mk in ("renewable_penetration", "p_ren_realized", "active_loss_mw", "load_scale", "ood_type"):
            if mk in data and len(data[mk]) == len(y_nadir):
                extra_meta[mk] = np.asarray(data[mk])

    # Load optional metadata from sibling CSV if NPZ omitted subgroup columns
    csv_candidate = feat_path.with_suffix(".csv")
    if csv_candidate.exists() and len(extra_meta) < 3:
        try:
            df_csv = pd.read_csv(csv_candidate)
            if len(df_csv) == len(y_nadir):
                for mk in ("renewable_penetration", "p_ren_realized", "active_loss_mw", "load_scale", "ood_type"):
                    if mk in df_csv.columns and mk not in extra_meta:
                        extra_meta[mk] = df_csv[mk].values
        except Exception:
            pass

    # Partition masks — NEVER allow ood_test or test into training or validation!
    train_mask = partitions == "train"
    val_mask = partitions == "val"
    test_mask = partitions == "test"
    ood_mask = partitions == "ood_test"

    if np.sum(train_mask) >= 1 and np.sum(test_mask) >= 1:
        split_protocol = "strict_metadata_partitions"
        idx_train = np.where(train_mask)[0]
        idx_val = np.where(val_mask)[0]
        idx_test = np.where(test_mask)[0]
    else:
        # Fallback deterministic 60/20/20 split strictly on non-OOD scenarios
        split_protocol = "deterministic_60_20_20_fallback"
        non_ood_idx = np.where(~ood_mask)[0]
        if len(non_ood_idx) == 0:
            non_ood_idx = np.arange(len(X))
        rng = np.random.RandomState(parsed.seed)
        perm = non_ood_idx.copy()
        rng.shuffle(perm)
        n_id = len(perm)
        if n_id >= 5:
            n_tr = max(1, int(round(0.60 * n_id)))
            n_va = max(1, int(round(0.20 * n_id)))
            if n_tr + n_va >= n_id:
                n_va = max(0, n_id - n_tr - 1)
            idx_train = perm[:n_tr]
            idx_val = perm[n_tr : n_tr + n_va]
            idx_test = perm[n_tr + n_va :]
        else:
            n_tr = max(1, n_id - 1)
            idx_train = perm[:n_tr]
            idx_val = np.array([], dtype=int)
            idx_test = perm[n_tr:] if n_tr < n_id else perm[:1]

    X_train, y_train_nadir, y_train_sec = X[idx_train], y_nadir[idx_train], y_secure[idx_train]
    X_val = X[idx_val] if len(idx_val) > 0 else None
    y_val_nadir = y_nadir[idx_val] if len(idx_val) > 0 else None
    y_val_sec = y_secure[idx_val] if len(idx_val) > 0 else None
    X_test, y_test_nadir, y_test_sec = X[idx_test], y_nadir[idx_test], y_secure[idx_test]

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_val_df = pd.DataFrame(X_val, columns=feature_names) if X_val is not None and len(X_val) > 0 else None
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Validation hyperparameter selection for Ridge alpha (strictly on val, NEVER test)
    selected_ridge_alpha = 1e-2
    if X_val_df is not None and len(X_val_df) >= 3:
        best_a_mae = float("inf")
        for cand_a in (1e-3, 1e-2, 1e-1, 1.0, 5.0):
            cand_m = LinearNadirRegressor(alpha=cand_a).fit(X_train_df, y_train_nadir)
            v_mae = cand_m.evaluate(X_val_df, y_val_nadir)["mae"]
            if v_mae < best_a_mae:
                best_a_mae = v_mae
                selected_ridge_alpha = cand_a

    all_reg_keys = ["mean", "analytical", "linear", "dt", "rf", "gbdt", "lgbm"]
    if parsed.models is not None:
        models_to_train = list(parsed.models)
    elif parsed.model_type == "all":
        models_to_train = all_reg_keys + ["calibrated_gb"]
    else:
        models_to_train = [parsed.model_type]

    def _build_regressor(key: str) -> Tuple[str, Any, Dict[str, Any]]:
        if key == "mean":
            return "Constant Mean Baseline", MeanNadirRegressor(), {}
        if key == "analytical":
            return "Analytical Swing-Equation Baseline", AnalyticalSwingNadirRegressor(alpha=1e-2), {"alpha": 1e-2}
        if key == "linear":
            return "Ridge Linear Regression", LinearNadirRegressor(alpha=selected_ridge_alpha), {"alpha": selected_ridge_alpha}
        if key == "dt":
            return "Decision Tree (CART)", DecisionTreeNadirRegressor(max_depth=5, random_state=parsed.seed), {"max_depth": 5}
        if key == "rf":
            return "Random Forest", RandomForestNadirRegressor(n_estimators=100, max_depth=10, random_state=parsed.seed), {"n_estimators": 100, "max_depth": 10}
        if key == "gbdt":
            return "Gradient Boosting (GBDT)", GradientBoostingNadirRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=parsed.seed), {"n_estimators": 100, "learning_rate": 0.08, "max_depth": 4}
        if key == "lgbm":
            m = LightGBMNadirRegressor(n_estimators=100, learning_rate=0.08, num_leaves=15, max_depth=5, random_state=parsed.seed)
            return f"LightGBM ({m.backend_})", m, {"n_estimators": 100, "learning_rate": 0.08, "num_leaves": 15, "backend": m.backend_}
        raise ValueError(f"Unknown regressor key '{key}'")

    val_metrics_reg: Dict[str, Any] = {}
    test_metrics_reg: Dict[str, Any] = {}
    val_metrics_clf: Dict[str, Any] = {}
    test_metrics_clf: Dict[str, Any] = {}
    hyperparams_record: Dict[str, Any] = {}

    reg_rows: List[Dict[str, Any]] = []
    clf_rows: List[Dict[str, Any]] = []
    trained_reg_objects: Dict[str, Any] = {}
    trained_clf_objects: Dict[str, Any] = {}

    # Combined train+val pool used ONLY after validation selection for equalized final model refit
    use_full_pool = (not parsed.strict_disjoint_cal) and (X_val_df is not None and len(X_val_df) >= 2)
    if use_full_pool:
        X_full_df = pd.concat([X_train_df, X_val_df], ignore_index=True)
        y_full_nadir = np.concatenate([y_train_nadir, y_val_nadir])
        y_full_sec = np.concatenate([y_train_sec, y_val_sec])
    else:
        X_full_df = X_train_df
        y_full_nadir = y_train_nadir
        y_full_sec = y_train_sec

    for m_type in models_to_train:
        if m_type in all_reg_keys:
            disp, reg_val_model, hp = _build_regressor(m_type)
            hyperparams_record[m_type] = hp

            # 1. Fit strictly on train and evaluate on val (for leak-free model selection)
            reg_val_model.fit(X_train_df, y_train_nadir)
            if X_val_df is not None and len(X_val_df) > 0:
                v_eval = reg_val_model.evaluate(X_val_df, y_val_nadir)
                val_metrics_reg[m_type] = {k: round(float(v), 5) for k, v in v_eval.items()}
            else:
                val_metrics_reg[m_type] = {}

            # 2. Final fit (on train+val if equalized budget enabled, else train) & evaluate on held-out test
            _, reg_final, _ = _build_regressor(m_type)
            reg_final.fit(X_full_df if use_full_pool else X_train_df, y_full_nadir if use_full_pool else y_train_nadir)

            t0 = time.perf_counter_ns()
            preds_test = reg_final.predict(X_test_df)
            infer_ms = (time.perf_counter_ns() - t0) / (max(1, len(X_test_df)) * 1e6)

            t_eval = reg_final.evaluate(X_test_df, y_test_nadir)
            ci_low, ci_high = bootstrap_mae_ci(y_test_nadir, preds_test, n_boot=1000, seed=parsed.seed)

            test_metrics_reg[m_type] = {
                "mae": round(t_eval["mae"], 4),
                "rmse": round(t_eval["rmse"], 4),
                "bias": round(t_eval["bias"], 4),
                "median_ae": round(t_eval["median_ae"], 4),
                "p95_ae": round(t_eval["p95_ae"], 4),
                "r2": round(t_eval["r2"], 4),
                "mae_ci95": [round(ci_low, 4), round(ci_high, 4)],
                "infer_ms": round(infer_ms, 4),
            }
            reg_rows.append({
                "Model": disp,
                "Val MAE (Hz)": round(val_metrics_reg[m_type].get("mae", float("nan")), 4) if val_metrics_reg[m_type] else float("nan"),
                "MAE (Hz)": round(t_eval["mae"], 4),
                "RMSE (Hz)": round(t_eval["rmse"], 4),
                "Bias (Hz)": round(t_eval["bias"], 4),
                "Median AE (Hz)": round(t_eval["median_ae"], 4),
                "P95 AE (Hz)": round(t_eval["p95_ae"], 4),
                "R2": round(t_eval["r2"], 4),
                "95% Bootstrap CI (Hz)": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "Inference (ms/scen)": round(infer_ms, 4),
            })
            trained_reg_objects[m_type] = reg_final
            joblib.dump(reg_final, out_dir / f"{m_type}.joblib")

        elif m_type == "calibrated_gb":
            if parsed.model_type == "all":
                base_clf_builders = [
                    ("Logistic Regression", "logistic_clf", lambda: LogisticSecurityClassifier(threshold=parsed.threshold, random_state=parsed.seed)),
                    ("Decision Tree", "dt_clf", lambda: DecisionTreeSecurityClassifier(threshold=parsed.threshold, max_depth=5, random_state=parsed.seed)),
                    ("Random Forest", "rf_clf", lambda: RandomForestSecurityClassifier(threshold=parsed.threshold, n_estimators=100, max_depth=10, random_state=parsed.seed)),
                    ("Uncalibrated GBDT", "gbdt_clf", lambda: GradientBoostingSecurityClassifier(threshold=parsed.threshold, n_estimators=100, learning_rate=0.08, max_depth=4, random_state=parsed.seed)),
                    ("LightGBM Classifier", "lgbm_clf", lambda: LightGBMSecurityClassifier(threshold=parsed.threshold, n_estimators=100, learning_rate=0.08, num_leaves=15, random_state=parsed.seed)),
                ]
                for disp_c, key_c, fn_c in base_clf_builders:
                    # Validation pass (strictly fit on train, evaluated on val)
                    if X_val_df is not None and len(X_val_df) > 0:
                        clf_v = fn_c().fit(X_train_df, y_train_sec)
                        ev_v = clf_v.evaluate(X_val_df, y_val_sec)
                        p_v = clf_v.predict_proba(X_val_df)[:, 1]
                        cal_v = compute_calibration_metrics(y_val_sec, p_v, n_bins=10, threshold=parsed.threshold)
                        val_metrics_clf[key_c] = {
                            "f1": round(ev_v["f1"], 4),
                            "brier_score": round(cal_v["brier_score"], 4),
                            "ece": round(cal_v["ece"], 4),
                            "fnr_insecure": round(ev_v["fnr_insecure"], 4),
                        }
                    else:
                        val_metrics_clf[key_c] = {}

                    # Final fit (equalized sample budget on train+val, zero test exposure)
                    clf_obj = fn_c().fit(X_full_df if use_full_pool else X_train_df, y_full_sec if use_full_pool else y_train_sec)
                    c_eval = clf_obj.evaluate(X_test_df, y_test_sec)
                    c_probs = clf_obj.predict_proba(X_test_df)[:, 1]
                    c_cal = compute_calibration_metrics(y_test_sec, c_probs, n_bins=10, threshold=parsed.threshold)

                    row_c = {
                        "Model": disp_c,
                        "Val Brier": val_metrics_clf[key_c].get("brier_score", float("nan")),
                        "Precision": round(c_eval["precision"], 4),
                        "Recall": round(c_eval["recall"], 4),
                        "F1": round(c_eval["f1"], 4),
                        "PR-AUC": round(c_eval["pr_auc"], 4),
                        "ROC-AUC": round(c_eval["roc_auc"], 4),
                        "Brier": round(c_cal["brier_score"], 4),
                        "ECE": round(c_cal["ece"], 4),
                        "MCE": round(c_cal["mce"], 4),
                        "FNR_insecure": round(c_eval["fnr_insecure"], 4),
                        "Missed_Insecure": int(c_eval["missed_insecure_count"]),
                        "FPR_secure": round(c_eval["fpr_secure"], 4),
                        "Effective_Calibration": "none",
                    }
                    clf_rows.append(row_c)
                    test_metrics_clf[key_c] = row_c
                    trained_clf_objects[key_c] = clf_obj
                    joblib.dump(clf_obj, out_dir / f"{key_c}.joblib")

            cal_methods = ["platt", "isotonic"] if parsed.model_type == "all" else [parsed.calibration]
            for cal_m in cal_methods:
                key_cal = f"calibrated_gb_{'iso' if cal_m == 'isotonic' else 'platt'}"
                # Validation evaluation (fit on train with OOF calibration, evaluate strictly out-of-sample on val)
                if X_val_df is not None and len(X_val_df) > 0:
                    cal_val_obj = CalibratedGBDTClassifier(
                        method=cal_m,
                        threshold=parsed.threshold,
                        n_estimators=100,
                        learning_rate=0.08,
                        max_depth=4,
                        random_state=parsed.seed,
                        cv_folds=5,
                        fit_full_after_cal=False,
                    )
                    cal_val_obj.fit(X_train_df, y_train_nadir, y_train_sec)
                    ev_cv = cal_val_obj.evaluate(X_val_df, y_val_sec)
                    val_metrics_clf[key_cal] = {
                        "f1": round(ev_cv["f1"], 4),
                        "brier_score": round(ev_cv["brier_score"], 4),
                        "ece": round(ev_cv["ece"], 4),
                        "fnr_insecure": round(ev_cv["fnr_insecure"], 4),
                        "effective_method": cal_val_obj.effective_method_,
                    }
                else:
                    val_metrics_clf[key_cal] = {}

                # Final calibrated GBDT: when X_val is present & use_full_pool=True, cross-fits Platt/Isotonic
                # across train+val via 5-fold OOF and refits base GBDT on train+val (equal sample budget to baselines!)
                cal_final = CalibratedGBDTClassifier(
                    method=cal_m,
                    threshold=parsed.threshold,
                    n_estimators=100,
                    learning_rate=0.08,
                    max_depth=4,
                    random_state=parsed.seed,
                    cv_folds=5,
                    fit_full_after_cal=use_full_pool,
                )
                if X_val_df is not None and len(X_val_df) >= 2:
                    cal_final.fit(
                        X_train_df,
                        y_train_nadir,
                        y_train_sec,
                        X_cal=X_val_df,
                        y_cal=y_val_nadir,
                    )
                else:
                    cal_final.fit(X_train_df, y_train_nadir, y_train_sec)

                ev_te = cal_final.evaluate(X_test_df, y_test_sec)
                p_te = cal_final.predict_proba(X_test_df)[:, 1]
                cal_te = compute_calibration_metrics(y_test_sec, p_te, n_bins=10, threshold=parsed.threshold)

                disp_label = f"Proposed Calibrated GBDT ({cal_m.capitalize()})"
                if cal_final.fallback_used_:
                    disp_label += f" [fallback->{cal_final.effective_method_}]"

                row_cal = {
                    "Model": disp_label,
                    "Val Brier": val_metrics_clf[key_cal].get("brier_score", float("nan")),
                    "Precision": round(ev_te["precision"], 4),
                    "Recall": round(ev_te["recall"], 4),
                    "F1": round(ev_te["f1"], 4),
                    "PR-AUC": round(ev_te["pr_auc"], 4),
                    "ROC-AUC": round(ev_te["roc_auc"], 4),
                    "Brier": round(cal_te["brier_score"], 4),
                    "ECE": round(cal_te["ece"], 4),
                    "MCE": round(cal_te["mce"], 4),
                    "FNR_insecure": round(ev_te["fnr_insecure"], 4),
                    "Missed_Insecure": int(ev_te["missed_insecure_count"]),
                    "FPR_secure": round(ev_te["fpr_secure"], 4),
                    "Effective_Calibration": cal_final.effective_method_,
                }
                clf_rows.append(row_cal)
                test_metrics_clf[key_cal] = {
                    **row_cal,
                    "requested_method": cal_final.method,
                    "effective_method": cal_final.effective_method_,
                    "fallback_used": cal_final.fallback_used_,
                    "fallback_reason": cal_final.fallback_reason_,
                    "n_cal": cal_final.n_cal_,
                    "calibration_protocol": cal_final.calibration_protocol_,
                    "isotonic_knots": len(cal_final.isotonic_x),
                    "platt_a": round(float(cal_final.platt_a), 5),
                    "platt_b": round(float(cal_final.platt_b), 5),
                }
                trained_clf_objects[key_cal] = cal_final
                joblib.dump(cal_final, out_dir / f"{key_cal}.joblib")
                if cal_m == "isotonic":
                    joblib.dump(cal_final, out_dir / "calibrated_gb_isotonic.joblib")

            primary_cal_key = f"calibrated_gb_{'iso' if parsed.calibration == 'isotonic' else 'platt'}"
            primary_cal_model = trained_clf_objects[primary_cal_key]
            joblib.dump(primary_cal_model, out_dir / "calibrated_gb.joblib")
            test_metrics_clf["calibrated_gb"] = test_metrics_clf[primary_cal_key]

    # Select best regressor & classifier strictly on validation metrics (never test!)
    best_reg_key = "gbdt" if "gbdt" in trained_reg_objects else (next(iter(trained_reg_objects.keys()), None))
    if val_metrics_reg:
        valid_reg_candidates = {k: v["mae"] for k, v in val_metrics_reg.items() if "mae" in v and k in trained_reg_objects}
        if valid_reg_candidates:
            best_reg_key = min(valid_reg_candidates, key=valid_reg_candidates.get)
    if best_reg_key and best_reg_key in trained_reg_objects:
        joblib.dump(trained_reg_objects[best_reg_key], out_dir / "best_regressor.joblib")

    best_clf_key = "calibrated_gb_platt" if "calibrated_gb_platt" in trained_clf_objects else (next(iter(trained_clf_objects.keys()), None))
    if val_metrics_clf:
        valid_clf_candidates = {k: v["brier_score"] for k, v in val_metrics_clf.items() if "brier_score" in v and k in trained_clf_objects}
        if valid_clf_candidates:
            best_clf_key = min(valid_clf_candidates, key=valid_clf_candidates.get)
    if best_clf_key and best_clf_key in trained_clf_objects:
        joblib.dump(trained_clf_objects[best_clf_key], out_dir / "best_classifier.joblib")

    # Subgroup breakdown & probability threshold safety sweep on held-out test
    subgroup_rows: List[Dict[str, Any]] = []
    threshold_sweep_rows: List[Dict[str, Any]] = []
    ref_reg = trained_reg_objects.get("gbdt") or (trained_reg_objects.get(best_reg_key) if best_reg_key else None)
    ref_clf = trained_clf_objects.get("calibrated_gb_platt") or (trained_clf_objects.get(best_clf_key) if best_clf_key else None)

    if ref_reg is not None and ref_clf is not None and len(idx_test) > 0:
        test_meta = {k: v[idx_test] for k, v in extra_meta.items()}
        y_pred_te_nadir = ref_reg.predict(X_test_df)
        p_pred_te_sec = ref_clf.predict_proba(X_test_df)[:, 1]
        subgroup_rows = _compute_subgroup_breakdown(
            y_test_nadir, y_pred_te_nadir, y_test_sec, p_pred_te_sec, test_meta, threshold=parsed.threshold
        )
        threshold_sweep_rows = evaluate_probability_threshold_sweep(
            y_test_sec, p_pred_te_sec, thresholds=[0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

    if reg_rows:
        df_reg = pd.DataFrame(reg_rows)
        df_reg.to_csv(tables_dir / "nadir_regression.csv", index=False)
        (tables_dir / "nadir_regression.md").write_text(_df_to_markdown(df_reg), encoding="utf-8")

    if clf_rows:
        df_clf = pd.DataFrame(clf_rows)
        df_clf.to_csv(tables_dir / "security_classification.csv", index=False)
        (tables_dir / "security_classification.md").write_text(_df_to_markdown(df_clf), encoding="utf-8")

    if subgroup_rows:
        df_sub = pd.DataFrame(subgroup_rows)
        df_sub.to_csv(tables_dir / "subgroup_performance.csv", index=False)
        (tables_dir / "subgroup_performance.md").write_text(_df_to_markdown(df_sub), encoding="utf-8")

    if threshold_sweep_rows:
        df_thr = pd.DataFrame(threshold_sweep_rows)
        df_thr.to_csv(tables_dir / "threshold_sweep_safety.csv", index=False)
        (tables_dir / "threshold_sweep_safety.md").write_text(_df_to_markdown(df_thr), encoding="utf-8")

    primary_cal_info = test_metrics_clf.get("calibrated_gb", {})
    training_metadata: Dict[str, Any] = {
        "protocol": split_protocol,
        "equalized_train_val_refit": bool(use_full_pool),
        "seed": int(parsed.seed),
        "threshold_hz": float(parsed.threshold),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "partition_counts": {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)),
            "test": int(len(idx_test)),
            "ood_test": int(np.sum(ood_mask)),
        },
        "train_ids": [str(s) for s in scenario_ids[idx_train]],
        "val_ids": [str(s) for s in scenario_ids[idx_val]],
        "test_ids": [str(s) for s in scenario_ids[idx_test]],
        "hyperparameters": hyperparams_record,
        "val_selected_best_regressor": best_reg_key,
        "val_selected_best_classifier": best_clf_key,
        "calibration_method": parsed.calibration,
        "effective_calibration_method": primary_cal_info.get("effective_method", parsed.calibration),
        "calibration_fallback_used": primary_cal_info.get("fallback_used", False),
        "n_cal": primary_cal_info.get("n_cal", int(len(idx_val))),
        "software_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "val_metrics": {
            "regression": val_metrics_reg,
            "classification": val_metrics_clf,
        },
        "test_metrics": {
            "regression": test_metrics_reg,
            "classification": test_metrics_clf,
        },
        "subgroup_test_metrics": subgroup_rows,
        "threshold_sweep_test": threshold_sweep_rows,
        # Top-level model keys preserved for backward-compatible callers
        **test_metrics_reg,
        **{k: v for k, v in test_metrics_clf.items()},
    }

    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2)

    logger.info(
        "Training completed cleanly (train=%d, val=%d, test=%d; best_reg=%s, best_clf=%s).",
        len(idx_train),
        len(idx_val),
        len(idx_test),
        best_reg_key,
        best_clf_key,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
