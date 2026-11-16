#!/usr/bin/env python3
"""Train nadir regressors and security classifiers on extracted PMU features."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    import pickle as joblib
    HAS_JOBLIB = False

from pmu_fsa.models.regressors import (
    LinearNadirRegressor,
    DecisionTreeNadirRegressor,
    RandomForestNadirRegressor,
    GradientBoostingNadirRegressor,
)
from pmu_fsa.models.classifiers import (
    LogisticSecurityClassifier,
    DecisionTreeSecurityClassifier,
    RandomForestSecurityClassifier,
    GradientBoostingSecurityClassifier,
)
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import compute_calibration_metrics

logger = logging.getLogger("train_models")


def bootstrap_mae_ci(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 500, seed: int = 42) -> Tuple[float, float]:
    """Compute 95% paired bootstrap confidence interval for MAE."""
    errors = np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))
    n = len(errors)
    if n < 2:
        val = float(errors[0]) if n == 1 else 0.0
        return val, val
    rng = np.random.default_rng(seed)
    boot_means = [float(np.mean(errors[rng.integers(0, n, size=n)])) for _ in range(n_boot)]
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


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
        choices=["linear", "dt", "rf", "gbdt", "calibrated_gb", "all"],
        default="calibrated_gb",
        help="Model family to train",
    )
    parser.add_argument("--models", type=str, nargs="+", default=None, help="List of models to train")
    parser.add_argument(
        "--calibration",
        type=str,
        choices=["platt", "isotonic"],
        default="platt",
        help="Calibration method for calibrated GBDT",
    )
    parser.add_argument("--threshold", type=float, default=49.50, help="Security threshold (Hz)")
    parser.add_argument("--output-dir", type=str, default="models/", help="Directory to save models")
    parser.add_argument("--tables-dir", type=str, default="results/tables/", help="Directory to save Table 3 & Table 4")
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
    tables_dir = Path(parsed.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    with np.load(feat_path, allow_pickle=True) as data:
        X = data["X"]
        y_nadir = data["y_nadir"]
        y_secure = data["y_secure"]
        partitions = data["partitions"] if "partitions" in data else np.array(["train"] * len(y_nadir))

    only_train_mask = (partitions == "train")
    val_mask = (partitions == "val")
    test_mask = (partitions == "test")

    if not np.any(test_mask) or np.sum(only_train_mask | val_mask) < 2:
        n_samples = len(X)
        indices = np.arange(n_samples)
        np.random.RandomState(parsed.seed).shuffle(indices)
        n_train = max(1, int(n_samples * 0.8))
        train_idx, test_idx = indices[:n_train], indices[n_train:]
        if len(test_idx) == 0:
            test_idx = train_idx
        X_train, y_train_nadir, y_train_sec = X[train_idx], y_nadir[train_idx], y_secure[train_idx]
        X_val, y_val_nadir, y_val_sec = None, None, None
        X_test, y_test_nadir, y_test_sec = X[test_idx], y_nadir[test_idx], y_secure[test_idx]
    else:
        if np.sum(val_mask) >= 2 and np.sum(only_train_mask) >= 2:
            X_train, y_train_nadir, y_train_sec = X[only_train_mask], y_nadir[only_train_mask], y_secure[only_train_mask]
            X_val, y_val_nadir, y_val_sec = X[val_mask], y_nadir[val_mask], y_secure[val_mask]
        else:
            comb_mask = only_train_mask | val_mask
            X_train, y_train_nadir, y_train_sec = X[comb_mask], y_nadir[comb_mask], y_secure[comb_mask]
            X_val, y_val_nadir, y_val_sec = None, None, None
        X_test, y_test_nadir, y_test_sec = X[test_mask], y_nadir[test_mask], y_secure[test_mask]

    X_train_full = np.vstack([X_train, X_val]) if X_val is not None else X_train
    y_train_full_nadir = np.concatenate([y_train_nadir, y_val_nadir]) if y_val_nadir is not None else y_train_nadir
    y_train_full_sec = np.concatenate([y_train_sec, y_val_sec]) if y_val_sec is not None else y_train_sec

    models_to_train = parsed.models or (
        [parsed.model_type] if parsed.model_type != "all" else ["linear", "dt", "rf", "gbdt", "calibrated_gb"]
    )

    metrics_summary: Dict[str, Any] = {}
    reg_rows: List[Dict[str, Any]] = []
    clf_rows: List[Dict[str, Any]] = []

    for m_type in models_to_train:
        logger.info("Training model: %s...", m_type)
        if m_type in ("linear", "dt", "rf", "gbdt"):
            if m_type == "linear":
                model = LinearNadirRegressor()
                disp = "Ridge Linear Regression"
            elif m_type == "dt":
                model = DecisionTreeNadirRegressor(max_depth=5, random_state=parsed.seed)
                disp = "Decision Tree (CART)"
            elif m_type == "rf":
                model = RandomForestNadirRegressor(n_estimators=50, random_state=parsed.seed)
                disp = "Random Forest"
            else:
                model = GradientBoostingNadirRegressor(n_estimators=50, random_state=parsed.seed)
                disp = "Gradient Boosting (GBDT)"

            model.fit(X_train_full, y_train_full_nadir)
            t0 = time.perf_counter_ns()
            preds = model.predict(X_test)
            infer_ms = (time.perf_counter_ns() - t0) / (max(1, len(X_test)) * 1e6)
            eval_m = model.evaluate(X_test, y_test_nadir)
            ci_low, ci_high = bootstrap_mae_ci(y_test_nadir, preds, seed=parsed.seed)

            metrics_summary[m_type] = {
                "mae": round(eval_m["mae"], 4),
                "rmse": round(eval_m["rmse"], 4),
                "r2": round(eval_m["r2"], 4),
                "mae_ci95": [round(ci_low, 4), round(ci_high, 4)],
                "infer_ms": round(infer_ms, 4),
            }
            reg_rows.append({
                "Model": disp,
                "MAE (Hz)": round(eval_m["mae"], 4),
                "RMSE (Hz)": round(eval_m["rmse"], 4),
                "R2": round(eval_m["r2"], 4),
                "95% Bootstrap CI (Hz)": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "Inference (ms/scen)": round(infer_ms, 4),
            })

        elif m_type == "calibrated_gb":
            if parsed.model_type == "all":
                base_clfs = [
                    ("Logistic Regression", "logistic_clf", LogisticSecurityClassifier(threshold=parsed.threshold, random_state=parsed.seed)),
                    ("Decision Tree", "dt_clf", DecisionTreeSecurityClassifier(threshold=parsed.threshold, max_depth=5, random_state=parsed.seed)),
                    ("Random Forest", "rf_clf", RandomForestSecurityClassifier(threshold=parsed.threshold, n_estimators=50, random_state=parsed.seed)),
                    ("Uncalibrated GBDT", "gbdt_clf", GradientBoostingSecurityClassifier(threshold=parsed.threshold, n_estimators=50, random_state=parsed.seed)),
                ]
                for disp_c, key_c, clf_obj in base_clfs:
                    clf_obj.fit(X_train_full, y_train_full_sec)
                    c_eval = clf_obj.evaluate(X_test, y_test_sec)
                    c_probs = clf_obj.predict_proba(X_test)[:, 1]
                    c_cal = compute_calibration_metrics(y_test_sec, c_probs)
                    clf_rows.append({
                        "Model": disp_c,
                        "Precision": round(c_eval["precision"], 4),
                        "Recall": round(c_eval["recall"], 4),
                        "F1": round(c_eval["f1"], 4),
                        "PR-AUC": round(c_eval["pr_auc"], 4),
                        "Brier": round(c_cal["brier_score"], 4),
                        "ECE": round(c_cal["ece"], 4),
                        "FNR_insecure": round(c_eval["fnr_insecure"], 4),
                    })
                    metrics_summary[key_c] = clf_rows[-1]

            model = CalibratedGBDTClassifier(
                method=parsed.calibration,
                threshold=parsed.threshold,
                random_state=parsed.seed,
            )
            model.fit(X_train, y_train_nadir, y_train_sec, X_cal=X_val, y_cal=y_val_sec)
            probs = model.predict_proba(X_test)
            p_sec = probs[:, 1] if hasattr(probs, "ndim") and probs.ndim > 1 else probs
            cal_metrics = compute_calibration_metrics(y_test_sec, p_sec)
            y_pred_sec = model.predict(X_test)

            tp = int(np.sum((y_test_sec == 1) & (y_pred_sec == 1)))
            tn = int(np.sum((y_test_sec == 0) & (y_pred_sec == 0)))
            fp = int(np.sum((y_test_sec == 0) & (y_pred_sec == 1)))
            fn = int(np.sum((y_test_sec == 1) & (y_pred_sec == 0)))
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 1.0
            fnr_insec = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0

            from sklearn.metrics import auc, precision_recall_curve
            if len(np.unique(y_test_sec)) > 1:
                pr_p, pr_r, _ = precision_recall_curve(y_test_sec, p_sec)
                pr_auc = float(auc(pr_r, pr_p))
            else:
                pr_auc = 1.0

            cal_metrics["precision"] = round(prec, 4)
            cal_metrics["recall"] = round(rec, 4)
            cal_metrics["f1"] = round(f1, 4)
            cal_metrics["pr_auc"] = round(pr_auc, 4)
            cal_metrics["fnr_insecure"] = round(fnr_insec, 4)
            metrics_summary["calibrated_gb"] = cal_metrics

            clf_rows.append({
                "Model": f"Proposed Calibrated GBDT ({model.method.capitalize()})",
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1": round(f1, 4),
                "PR-AUC": round(pr_auc, 4),
                "Brier": round(cal_metrics["brier_score"], 4),
                "ECE": round(cal_metrics["ece"], 4),
                "FNR_insecure": round(fnr_insec, 4),
            })

            if parsed.model_type == "all":
                platt_m = CalibratedGBDTClassifier(method="platt", threshold=parsed.threshold, random_state=parsed.seed)
                platt_m.fit(X_train, y_train_nadir, y_train_sec, X_cal=X_val, y_cal=y_val_sec)
                joblib.dump(platt_m, out_dir / "calibrated_gb_platt.joblib")

                iso_m = CalibratedGBDTClassifier(method="isotonic", threshold=parsed.threshold, random_state=parsed.seed)
                iso_m.fit(X_train, y_train_nadir, y_train_sec, X_cal=X_val, y_cal=y_val_sec)
                joblib.dump(iso_m, out_dir / "calibrated_gb_iso.joblib")
        else:
            logger.warning("Unrecognized model type: %s", m_type)
            continue

        save_path = out_dir / f"{m_type}.joblib"
        try:
            joblib.dump(model, save_path)
            logger.info("Saved model artifact to %s", save_path)
        except Exception as e:
            logger.warning("Could not save model artifact %s: %s", save_path, e)

    if reg_rows:
        df_reg = pd.DataFrame(reg_rows)
        df_reg.to_csv(tables_dir / "nadir_regression.csv", index=False)
        headers = list(df_reg.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, r in df_reg.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")
        (tables_dir / "nadir_regression.md").write_text("\n".join(lines), encoding="utf-8")

    if clf_rows:
        df_clf = pd.DataFrame(clf_rows)
        df_clf.to_csv(tables_dir / "security_classification.csv", index=False)
        headers = list(df_clf.columns)
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, r in df_clf.iterrows():
            lines.append("| " + " | ".join(str(v) for v in r) + " |")
        (tables_dir / "security_classification.md").write_text("\n".join(lines), encoding="utf-8")

    with open(out_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    logger.info("Training completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
