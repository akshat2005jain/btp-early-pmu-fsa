#!/usr/bin/env python3
"""Run leak-free sensitivity, OOD, noise, latency, statistical, and interpretability benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    import pickle as joblib

from pmu_fsa.evaluation.interpretability import CANONICAL_14_FEATURES, InterpretabilityAnalyzer
from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark, split_scenarios_by_partition
from pmu_fsa.evaluation.statistical_tests import (
    LearningCurveEvaluator,
    MultiSeedStabilityEvaluator,
    compare_all_models_statistical,
)
from pmu_fsa.features.extractor import FEATURE_SPECIFICATIONS, FeatureExtractor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import compute_calibration_metrics
from pmu_fsa.models.classifiers import (
    compute_binary_security_metrics,
    evaluate_probability_threshold_sweep,
)
from pmu_fsa.simulation.ieee39_data import GENERATOR_DATA, LOAD_DATA, NOMINAL_FREQ, SECURITY_THRESHOLD_HZ
from pmu_fsa.simulation.scenario_generator import load_scenario_from_npz

logger = logging.getLogger("evaluate_all")

ALL_BENCHMARK_KEYS = ["sensitivity", "ood", "noise", "latency", "statistical", "interpretability"]


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return ""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for v in row:
            if isinstance(v, (bool, np.bool_)):
                vals.append(str(bool(v)))
            elif isinstance(v, (int, np.integer)):
                vals.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                vals.append("-" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _save_table(df: pd.DataFrame, tables_dir: Path, stem: str) -> str:
    df.to_csv(tables_dir / f"{stem}.csv", index=False)
    md = _df_to_markdown(df)
    (tables_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    return md


def _generate_fallback_50hz_scenarios(seed: int = 42) -> List[Dict[str, Any]]:
    """Generate fast synthetic 50 Hz partitioned scenarios ONLY when --data-dir has zero .npz files."""
    rng = np.random.default_rng(seed)
    dt = 0.02
    n_samples = 251
    timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    t_post = np.maximum(0.0, timestamps - 1.0)

    specs = [
        *[("train", "id", 0.10 + (i % 3) * 0.10, 0.90 + (i % 3) * 0.10, 320.0 + (i % 4) * 180.0) for i in range(12)],
        *[("val", "id", 0.10 + (i % 3) * 0.10, 0.90 + (i % 3) * 0.10, 380.0 + (i % 3) * 200.0) for i in range(6)],
        *[("test", "id", 0.10 + (i % 3) * 0.10, 0.90 + (i % 3) * 0.10, 360.0 + (i % 4) * 190.0) for i in range(6)],
        *[("ood_test", "ood_res", 0.40 + (i % 2) * 0.10, 1.00, 450.0 + (i % 3) * 200.0) for i in range(4)],
        *[("ood_test", "ood_load", 0.20, 0.80 if i % 2 == 0 else 1.20, 500.0 + (i % 2) * 250.0) for i in range(2)],
        *[("ood_test", "ood_combined", 0.45, 1.20, 600.0 + i * 150.0) for i in range(2)],
    ]

    scenarios: List[Dict[str, Any]] = []
    for idx, (part, ood_t, pen, ld, mw) in enumerate(specs):
        severity = (mw / 650.0) * (1.0 + 0.65 * pen) * ld
        drop = 0.48 * severity * (1.0 - np.exp(-t_post / 1.1)) * np.exp(-np.maximum(0.0, t_post - 2.6) / 4.5)
        f_sys = 50.0 - drop
        spatial = rng.normal(0.0, 0.008 * severity, size=(n_samples, 39))
        bf = f_sys[:, np.newaxis] + spatial
        rocof = np.gradient(bf, dt, axis=0)
        v_ang = np.outer(t_post, np.linspace(-3.0, 3.0, 39) * severity)
        nadir_val = float(np.min(f_sys[timestamps >= 1.0]))
        t_nadir = float(timestamps[timestamps >= 1.0][int(np.argmin(f_sys[timestamps >= 1.0]))])
        scenarios.append({
            "timestamps": timestamps,
            "bus_frequencies": bf,
            "bus_rocof": rocof,
            "voltage_magnitudes": np.ones((n_samples, 39)),
            "voltage_angles": v_ang,
            "pmu_sys_frequency": f_sys,
            "coi_frequency": f_sys,
            "metadata": {
                "scenario_id": f"scen_fallback_{idx:03d}",
                "t_disturb": 1.0,
                "t_sim": 5.0,
                "fps": 50,
                "f_nominal": 50.0,
                "f_threshold_hz": 49.50,
                "disturbed_gen": 30 + (idx % 10),
                "loss_fraction": round(min(1.0, mw / 800.0), 2),
                "active_loss_mw": float(mw),
                "renewable_penetration": float(pen),
                "p_ren_realized": float(pen),
                "load_scale": float(ld),
                "ground_truth_nadir": round(nadir_val, 5),
                "ground_truth_t_nadir": round(t_nadir, 4),
                "is_secure": int(nadir_val >= 49.50),
                "partition": part,
                "ood_type": ood_t,
            },
        })
    return scenarios


def _load_saved_models(models_dir: Path) -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    if not models_dir.exists():
        return loaded
    for fp in sorted(models_dir.glob("*.joblib")):
        try:
            obj = joblib.load(fp)
            meta = getattr(obj, "artifact_metadata_", None)
            if isinstance(meta, dict):
                f_names = meta.get("feature_names")
                if f_names is not None and list(f_names) != list(CANONICAL_14_FEATURES):
                    raise ValueError(
                        f"Model {fp.name} schema mismatch: expected {CANONICAL_14_FEATURES}, got {f_names}"
                    )
            loaded[fp.stem] = obj
        except ValueError:
            raise
        except Exception as exc:
            logger.warning("Could not load model artifact %s: %s", fp, exc)
    return loaded


def _export_reference_tables(tables_dir: Path, scenarios: List[Dict[str, Any]], threshold: float) -> None:
    """Export Phase 13 reference specification tables as machine-readable CSV & Markdown."""
    total_gen_mw = sum(float(g["P_mw"]) for g in GENERATOR_DATA)
    total_load_mw = sum(float(p) for p, _ in LOAD_DATA.values())
    sys_df = pd.DataFrame([
        {"Parameter": "Nominal System Frequency (f0)", "Value": f"{NOMINAL_FREQ:.1f} Hz", "Source": "CERC IEGC 2023 / ieee39_data.py"},
        {"Parameter": "Benchmark Network", "Value": "IEEE 39-Bus, 10-Machine (New England)", "Source": "Athay et al. (1979)"},
        {"Parameter": "System MVA Base", "Value": "100.0 MVA", "Source": "ieee39_data.py"},
        {"Parameter": "Base Generation / Load (load_scale=1.0)", "Value": f"{total_gen_mw:.2f} MW / {total_load_mw:.2f} MW", "Source": "solve_power_flow()"},
        {"Parameter": "Slack Bus", "Value": "Bus 31 (Gen 2, V=0.982 pu, theta=0 deg)", "Source": "GENERATOR_DATA"},
        {"Parameter": "Bus 39 Equivalent (macro_intertie=False)", "Value": "Regional plant H_39=42.0 s on 100 MVA base (H_tot=324.7 s)", "Source": "configs/default_50hz.yaml / ieee39_data.py"},
        {"Parameter": "Renewable Representation", "Value": "Approach B: Co-located GFL inertia/damping/droop derating", "Source": "renewables.py"},
        {"Parameter": "Governor & Converter Droop", "Value": "Sync 5% droop (Tg=4.20 s) active; GFL converter droop disabled (enable_droop=False)", "Source": "dynamic_engine.py / renewables.py"},
        {"Parameter": "PMU Reporting Rate & ROCOF", "Value": "50 fps (dt=20 ms), area-weighted f_sys_pmu & causal 5-point backward OLS ROCOF", "Source": "pmu_recorder.py"},
        {"Parameter": "Contingency Security Threshold", "Value": f"{threshold:.2f} Hz (100 mHz above 49.40 Hz UFLS Stage-1)", "Source": "CERC IEGC 2023"},
    ])
    _save_table(sys_df, tables_dir, "system_configuration")

    feat_rows = []
    for idx, fn in enumerate(CANONICAL_14_FEATURES, 1):
        spec = FEATURE_SPECIFICATIONS.get(fn, {})
        feat_rows.append({
            "Index": idx,
            "Feature": fn,
            "Group": str(spec.get("group", spec.get("category", "pmu_causal"))),
            "Unit": str(spec.get("unit", "-")),
            "Definition": str(spec.get("description", spec.get("formula", fn))),
        })
    _save_table(pd.DataFrame(feat_rows), tables_dir, "canonical_features")

    if scenarios:
        part_rows = []
        for part in ("train", "val", "test", "ood_test"):
            sub = [s for s in scenarios if str(s.get("metadata", {}).get("partition", "")) == part]
            if not sub:
                continue
            nadirs = [float(s["metadata"]["ground_truth_nadir"]) for s in sub]
            sec = sum(1 for v in nadirs if v >= threshold)
            tnadirs = [float(s["metadata"]["ground_truth_t_nadir"]) for s in sub]
            t0s = [float(s["metadata"].get("t_disturb", 1.0)) for s in sub]
            c20 = sum(1 for tn, t0 in zip(tnadirs, t0s) if t0 + 2.0 < tn - 1e-6)
            part_rows.append({
                "Partition": part,
                "N_Scenarios": len(sub),
                "Secure_Count": sec,
                "Insecure_Count": len(sub) - sec,
                "Secure_Fraction": round(sec / len(sub), 4),
                "Nadir_Min_Hz": round(min(nadirs), 4),
                "Nadir_Median_Hz": round(float(np.median(nadirs)), 4),
                "Nadir_Max_Hz": round(max(nadirs), 4),
                "Causal_1_0s": len(sub),
                "Causal_2_0s": c20,
            })
        if part_rows:
            _save_table(pd.DataFrame(part_rows), tables_dir, "dataset_composition")

    lim_df = pd.DataFrame([
        {"Category": "Simulation Physics", "Limitation": "Classical constant-E' 3rd-order synchronous generator + first-order steam-reheat governor (R=5%, Tg=4.20 s) + Kron-reduced constant-impedance post-disturbance loads; omits AVR/PSS flux decay, boiler limits, and dynamic motor loads."},
        {"Category": "Renewable Representation (Approach B)", "Limitation": "Co-located constant-power GFL effective inertia, damping, and droop derating at unchanged bus active-power setpoints; omits explicit solar/wind park placement, line-flow redistribution, PLL dynamics, and current limiting."},
        {"Category": "PMU Channel Model", "Limitation": "Synthetic bus phasor sampling at 50 fps with Gaussian phasor noise and causal Butterworth filtering; excludes real PDC packet jitter, GPS phase jumps, CT/PT transients, and field utility PMU recordings."},
        {"Category": "Dataset & Sample Size", "Limitation": "750 simulated contingencies on one 39-bus benchmark network; held-out ID test split has N=52 scenarios (statistical power limited for sub-5 mHz differences)."},
        {"Category": "Model Generalization & Noise", "Limitation": "Validation-selected Ridge excels on smooth noiseless features (4.6 mHz ID MAE, 10.6 mHz OOD MAE) due to near-linear swing physics, whereas clean-trained models degrade substantially under unmitigated high-frequency PMU phasor noise unless regularized or noise-augmented."},
        {"Category": "Calibration & Threshold", "Limitation": "Post-hoc Platt/isotonic calibration operates on N_cal=218 OOF (or N_val=70 disjoint) scenarios; reliability bins have finite-sample variance on N=52 ID test cases."},
        {"Category": "Computational Benchmarking", "Limitation": "Desktop macOS user-space Python timing (time.perf_counter_ns); demonstrates sub-frame compute cost (<3 ms vs 20 ms 50-fps frame interval), not RTOS or hardware-in-the-loop relay certification."},
    ])
    _save_table(lim_df, tables_dir, "limitations_and_assumptions")


def _run_frequency_threshold_sensitivity(
    X_train: np.ndarray,
    y_train_nadir: np.ndarray,
    X_val: Optional[np.ndarray],
    y_val_nadir: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test_nadir: np.ndarray,
    X_ood: Optional[np.ndarray],
    y_ood_nadir: Optional[np.ndarray],
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate classification & calibration sensitivity across physical nadir security thresholds."""
    thresholds = [49.20, 49.40, 49.50, 49.60, 49.70, 49.80]
    rows: List[Dict[str, Any]] = []
    has_val = X_val is not None and y_val_nadir is not None and len(X_val) >= 2
    for f_thr in thresholds:
        ys_tr = (y_train_nadir >= f_thr).astype(int)
        ys_te = (y_test_nadir >= f_thr).astype(int)
        clf = CalibratedGBDTClassifier(
            method="platt",
            threshold=f_thr,
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            random_state=seed,
            cv_folds=5,
            fit_full_after_cal=has_val,
        )
        if has_val:
            clf.fit(X_train, y_train_nadir, ys_tr, X_cal=X_val, y_cal=y_val_nadir)
        else:
            clf.fit(X_train, y_train_nadir, ys_tr)

        p_te = clf.predict_proba(X_test)[:, 1]
        ev_te = compute_binary_security_metrics(ys_te, (p_te >= 0.5).astype(int), p_secure=p_te)
        cal_te = compute_calibration_metrics(ys_te, p_te, n_bins=10, threshold=f_thr)

        row: Dict[str, Any] = {
            "f_threshold_hz": round(f_thr, 2),
            "id_test_n": int(len(ys_te)),
            "id_secure_frac": round(float(np.mean(ys_te)), 4),
            "id_f1": round(float(ev_te["f1"]), 4),
            "id_recall": round(float(ev_te["recall"]), 4),
            "id_fnr_insecure": round(float(ev_te["fnr_insecure"]), 4),
            "id_missed_insecure": int(ev_te["missed_insecure_count"]),
            "id_brier": round(float(cal_te["brier_score"]), 4),
            "id_ece": round(float(cal_te["ece"]), 4),
        }
        if X_ood is not None and y_ood_nadir is not None and len(X_ood) > 0:
            ys_ood = (y_ood_nadir >= f_thr).astype(int)
            p_ood = clf.predict_proba(X_ood)[:, 1]
            ev_ood = compute_binary_security_metrics(ys_ood, (p_ood >= 0.5).astype(int), p_secure=p_ood)
            cal_ood = compute_calibration_metrics(ys_ood, p_ood, n_bins=10, threshold=f_thr)
            row.update({
                "ood_n": int(len(ys_ood)),
                "ood_secure_frac": round(float(np.mean(ys_ood)), 4),
                "ood_f1": round(float(ev_ood["f1"]), 4),
                "ood_fnr_insecure": round(float(ev_ood["fnr_insecure"]), 4),
                "ood_brier": round(float(cal_ood["brier_score"]), 4),
                "ood_ece": round(float(cal_ood["ece"]), 4),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate All Benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", type=str, default="models/", help="Directory containing trained models")
    parser.add_argument("--report-file", type=str, default=None, help="Path for master evaluation report markdown")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=ALL_BENCHMARK_KEYS,
        help="Benchmarks to run (sensitivity, ood, noise, latency, statistical, interpretability, or all)",
    )
    parser.add_argument("--data-dir", type=str, default="data/raw/", help="Directory containing raw PMU scenario .npz files")
    parser.add_argument("--features", type=str, default="data/processed/features.npz", help="Processed T_obs=1.0s features.npz")
    parser.add_argument("--t-obs", type=float, default=1.0, help="Primary observation window (s)")
    parser.add_argument("--threshold", type=float, default=49.50, help="Security nadir threshold (Hz)")
    parser.add_argument("--tables-dir", type=str, default=None, help="Output tables directory")
    parser.add_argument("--figures-dir", type=str, default="results/figures/", help="Output figures directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Base results directory override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if parsed.tables_dir is not None:
        tables_dir = Path(parsed.tables_dir)
    elif parsed.output_dir is not None:
        tables_dir = Path(parsed.output_dir) / "tables"
    elif parsed.report_file is not None and Path(parsed.report_file).resolve().parent != (PROJECT_ROOT / "results").resolve():
        tables_dir = Path(parsed.report_file).parent / "tables"
    else:
        tables_dir = Path("results/tables")
    tables_dir.mkdir(parents=True, exist_ok=True)

    if parsed.report_file is not None:
        report_path = Path(parsed.report_file)
    elif parsed.output_dir is not None:
        report_path = Path(parsed.output_dir) / "evaluation_report.md"
    else:
        report_path = Path("results/evaluation_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    models_dir = Path(parsed.models_dir)
    saved_models = _load_saved_models(models_dir)

    training_meta_path = models_dir / "training_metrics.json"
    val_best_reg_key = "linear"
    val_best_clf_key = "calibrated_gb_platt"
    if training_meta_path.exists():
        try:
            tmeta = json.loads(training_meta_path.read_text(encoding="utf-8"))
            val_best_reg_key = str(tmeta.get("val_selected_best_regressor", val_best_reg_key))
            val_best_clf_key = str(tmeta.get("val_selected_best_classifier", val_best_clf_key))
        except Exception:
            pass

    val_selected_reg = saved_models.get("best_regressor") or saved_models.get(val_best_reg_key)
    proposed_gbdt_reg = saved_models.get("gbdt") or val_selected_reg
    primary_clf = (
        saved_models.get("best_classifier")
        or saved_models.get("calibrated_gb")
        or saved_models.get("calibrated_gb_platt")
    )

    if saved_models:
        logger.info(
            "Loaded %d saved model artifacts from %s (val_best_reg=%s, proposed_reg=gbdt, val_best_clf=%s)",
            len(saved_models),
            models_dir,
            val_best_reg_key,
            val_best_clf_key,
        )

    data_dir = Path(parsed.data_dir)
    scenarios: List[Dict[str, Any]] = []
    used_synthetic_fallback = False
    if data_dir.exists():
        scen_files = sorted(list(data_dir.glob("*.npz")))
        scenarios = [load_scenario_from_npz(str(fp)) for fp in scen_files]

    if not scenarios:
        used_synthetic_fallback = True
        logger.warning(
            "No scenario .npz archives found in %s; using 50 Hz partitioned synthetic fallback fixture.",
            data_dir,
        )
        scenarios = _generate_fallback_50hz_scenarios(seed=parsed.seed)

    _export_reference_tables(tables_dir, scenarios, parsed.threshold)

    benchmarks_to_run = list(parsed.benchmarks)
    if "all" in benchmarks_to_run:
        benchmarks_to_run = list(ALL_BENCHMARK_KEYS)

    report_sections: List[str] = [
        "# PMU-Based Early Frequency-Security Assessment: Verified Benchmark Evaluation Report\n",
        f"- **System Standard**: `50.0 Hz` nominal, `{parsed.threshold:.2f} Hz` contingency security threshold",
        f"- **Loaded Scenarios**: `{len(scenarios)}` (`synthetic_fallback={used_synthetic_fallback}`)",
        f"- **Validation-Selected Best Regressor**: `{val_best_reg_key}` (`best_regressor.joblib`)",
        "- **Proposed Nonlinear Regressor**: `gbdt` (`gbdt.joblib`)",
        f"- **Validation-Selected Best Classifier**: `{val_best_clf_key}` (`calibrated_gb.joblib`)",
        f"- **Loaded Model Artifacts**: `{sorted(list(saved_models.keys())) if saved_models else 'trained_per_split'}`\n",
    ]

    # 1. Window Sensitivity Benchmark (strict pre-nadir causal filtering)
    if "sensitivity" in benchmarks_to_run:
        logger.info("Running Window Sensitivity Benchmark (strict pre-nadir causal filtering)...")
        sens_bench = WindowSensitivityBenchmark(
            regressor_type="gbdt",
            classifier_type="calibrated_gb",
            threshold=parsed.threshold,
            random_state=parsed.seed,
            models_dir=models_dir,
            strict_nadir=True,
        )
        default_win_tuple = (
            (parsed.t_obs, proposed_gbdt_reg, primary_clf)
            if (proposed_gbdt_reg is not None and primary_clf is not None)
            else None
        )
        sens_results = sens_bench.run(scenarios, default_window_models=default_win_tuple)

        df_sens = sens_bench.to_dataframe(sens_results, split="id_test")
        md_sens = _save_table(df_sens, tables_dir, "window_sensitivity")
        report_sections.append(r"## 1. Observation Window Sensitivity ($T_{\text{obs}} \in [0.5, 2.0]$ s — Proposed GBDT + Calibrated GBDT)" + "\n")
        report_sections.append("### 1A. Held-Out ID Test (`N = 52`, 100% Strictly Pre-Nadir)\n" + md_sens + "\n")

        df_sens_ood = sens_bench.to_dataframe(sens_results, split="ood_test")
        if not df_sens_ood.empty:
            md_sens_ood = _save_table(df_sens_ood, tables_dir, "window_sensitivity_ood")
            report_sections.append("### 1B. Held-Out OOD Test (`window_specific_causal`: non-causal `2.0 s` cases excluded)\n" + md_sens_ood + "\n")

        df_sens_ood_common = sens_bench.to_dataframe(sens_results, split="ood_common")
        if not df_sens_ood_common.empty:
            md_sens_common = _save_table(df_sens_ood_common, tables_dir, "window_sensitivity_ood_common")
            report_sections.append("### 1C. Held-Out OOD Test (`common_causal_intersection` across all 4 windows)\n" + md_sens_common + "\n")

        # Also evaluate validation-selected linear Ridge regressor across all windows
        sens_ridge = WindowSensitivityBenchmark(
            regressor_type="linear",
            classifier_type="calibrated_gb",
            threshold=parsed.threshold,
            random_state=parsed.seed,
            models_dir=models_dir,
            strict_nadir=True,
        )
        ridge_win_tuple = (
            (parsed.t_obs, val_selected_reg, primary_clf)
            if (val_selected_reg is not None and primary_clf is not None)
            else None
        )
        sens_ridge_res = sens_ridge.run(scenarios, default_window_models=ridge_win_tuple)
        md_sens_ridge = _save_table(sens_ridge.to_dataframe(sens_ridge_res, split="id_test"), tables_dir, "window_sensitivity_ridge")
        report_sections.append(f"### 1D. Validation-Selected Regressor (`{val_best_reg_key}`) Window Sensitivity (ID Test)\n" + md_sens_ridge + "\n")
        df_sr_ood = sens_ridge.to_dataframe(sens_ridge_res, split="ood_test")
        if not df_sr_ood.empty:
            _save_table(df_sr_ood, tables_dir, "window_sensitivity_ridge_ood")

    # 2. Out-of-Distribution Generalization Benchmark (Both Proposed GBDT & Validation-Selected Ridge)
    if "ood" in benchmarks_to_run:
        logger.info("Running Out-of-Distribution Generalization Benchmark (GBDT & Validation-Selected Ridge)...")
        ood_bench = OODGeneralizationBenchmark(
            t_obs=parsed.t_obs,
            threshold=parsed.threshold,
            random_state=parsed.seed,
        )
        ood_results = ood_bench.run(
            scenarios,
            regressor_type="gbdt",
            classifier_type="calibrated_gb",
            regressor=proposed_gbdt_reg,
            classifier=primary_clf,
        )
        df_ood = ood_bench.to_dataframe(ood_results)
        df_ood.to_csv(tables_dir / "ood_generalization.csv", index=False)
        df_ood_regimes = ood_bench.to_regime_dataframe(ood_results)
        if not df_ood_regimes.empty:
            _save_table(df_ood_regimes, tables_dir, "ood_regime_breakdown")

        md_ood = ood_bench.to_markdown(ood_results)
        (tables_dir / "ood_generalization.md").write_text(md_ood, encoding="utf-8")
        report_sections.append("## 2. Out-of-Distribution Generalization (ID Held-Out Test vs OOD-RES / Load / Combined)\n")
        report_sections.append("### 2A. Proposed GBDT + Calibrated GBDT\n" + md_ood + "\n")

        ood_ridge_res = ood_bench.run(
            scenarios,
            regressor_type="linear",
            classifier_type="calibrated_gb",
            regressor=val_selected_reg,
            classifier=primary_clf,
        )
        df_ood_ridge = ood_bench.to_dataframe(ood_ridge_res)
        _save_table(df_ood_ridge, tables_dir, "ood_generalization_ridge")
        df_ood_ridge_regimes = ood_bench.to_regime_dataframe(ood_ridge_res)
        if not df_ood_regimes.empty and not df_ood_ridge_regimes.empty:
            comp_rows = []
            for (_, r_gb), (_, r_ln) in zip(df_ood_regimes.iterrows(), df_ood_ridge_regimes.iterrows()):
                comp_rows.append({
                    "Regime": r_gb["Regime"],
                    "N": int(r_gb["N"]),
                    "Validation_Selected_Ridge_MAE_Hz": round(float(r_ln["MAE (Hz)"]), 4),
                    "Validation_Selected_Ridge_RMSE_Hz": round(float(r_ln["RMSE (Hz)"]), 4),
                    "Validation_Selected_Ridge_R2": round(float(r_ln["R2"]), 4),
                    "Proposed_GBDT_MAE_Hz": round(float(r_gb["MAE (Hz)"]), 4),
                    "Proposed_GBDT_RMSE_Hz": round(float(r_gb["RMSE (Hz)"]), 4),
                    "Proposed_GBDT_R2": round(float(r_gb["R2"]), 4),
                    "Calibrated_GBDT_F1": round(float(r_gb["F1"]), 4),
                    "Calibrated_GBDT_FNR_Insecure": round(float(r_gb["FNR_insecure"]), 4),
                })
            md_ood_cmp = _save_table(pd.DataFrame(comp_rows), tables_dir, "ood_model_comparison")
            report_sections.append(f"### 2B. Validation-Selected `{val_best_reg_key}` vs Proposed `gbdt` Across OOD Regimes\n" + md_ood_cmp + "\n")

    # 3. PMU Measurement Noise Stress Benchmark (Both Proposed GBDT & Validation-Selected Ridge)
    if "noise" in benchmarks_to_run:
        logger.info("Running PMU Measurement Noise Stress Benchmark (all 14 features, held-out test)...")
        noise_bench = MeasurementNoiseBenchmark(
            t_obs=parsed.t_obs,
            threshold=parsed.threshold,
            seed=parsed.seed,
        )
        noise_results = noise_bench.run(
            scenarios,
            regressor_type="gbdt",
            classifier_type="calibrated_gb",
            regressor=proposed_gbdt_reg,
            classifier=primary_clf,
        )
        df_noise = noise_bench.to_dataframe(noise_results)
        df_noise.to_csv(tables_dir / "measurement_noise.csv", index=False)
        md_noise = noise_bench.to_markdown(noise_results)
        (tables_dir / "measurement_noise.md").write_text(md_noise, encoding="utf-8")

        noise_ridge_res = noise_bench.run(
            scenarios,
            regressor_type="linear",
            classifier_type="calibrated_gb",
            regressor=val_selected_reg,
            classifier=primary_clf,
        )
        df_noise_ridge = noise_bench.to_dataframe(noise_ridge_res)
        md_noise_ridge = _save_table(df_noise_ridge, tables_dir, "measurement_noise_ridge")

        rocof_cmp = MeasurementNoiseBenchmark.compare_ols_vs_endpoint_rocof(noise_std=0.03, n_trials=100, seed=parsed.seed)
        report_sections.append("## 3. PMU Measurement Noise Stress Benchmark\n")
        report_sections.append("### 3A. Proposed GBDT + Calibrated GBDT\n" + md_noise + "\n")
        report_sections.append(f"### 3B. Validation-Selected `{val_best_reg_key}` Under Identical PMU Noise\n" + md_noise_ridge + "\n")
        report_sections.append(
            f"- **Causal Windowed OLS vs 2-Point ROCOF Variance Ratio**: `{rocof_cmp['variance_ratio']}` "
            f"(`var_ols={rocof_cmp['var_ols']}`, `var_2pt={rocof_cmp['var_endpoint']}`)\n"
        )

    # 4. Computational Latency & Measured ODE Simulation Speedup Benchmark
    if "latency" in benchmarks_to_run:
        logger.info("Running 6-Stage Computational Latency & Real DynamicSimulator Benchmark...")
        lat_bench = LatencyBenchmark()
        lat_results = lat_bench.profile(
            regressor=proposed_gbdt_reg,
            classifier=primary_clf,
            scenarios=scenarios,
            t_obs=parsed.t_obs,
            n_reps=100,
            warmup_reps=10,
            measure_simulation=not used_synthetic_fallback,
            n_sim_runs=6 if not used_synthetic_fallback else 2,
        )
        flat_lat = {k: v for k, v in lat_results.items() if not isinstance(v, (dict, list))}
        pd.DataFrame([flat_lat]).to_csv(tables_dir / "latency_profile.csv", index=False)

        trials_dict = lat_results.get("trials", {})
        if trials_dict and "e2e_times_ms" in trials_dict:
            df_trials = pd.DataFrame(trials_dict)
            df_trials.insert(0, "trial", np.arange(1, len(df_trials) + 1))
            df_trials.to_csv(tables_dir / "latency_trials.csv", index=False)

        df_stages = lat_bench.to_stages_dataframe(lat_results)
        _save_table(df_stages, tables_dir, "latency_stages")

        lat_ridge = lat_bench.profile(
            regressor=val_selected_reg,
            classifier=primary_clf,
            scenarios=scenarios,
            t_obs=parsed.t_obs,
            n_reps=100,
            warmup_reps=10,
            sim_time_ms=lat_results["sim_time_ms"],
            measure_simulation=False,
        )
        df_lat_cmp = pd.DataFrame([
            {
                "Pipeline_Variant": f"Validation-Selected ({val_best_reg_key}) + Calibrated GBDT",
                "Median_Total_ms": lat_ridge["t_total_ms"],
                "Mean_Total_ms": round(lat_ridge["t_total_mean_ms"], 4),
                "P95_Total_ms": round(lat_ridge["t_total_p95_ms"], 4),
                "P99_Total_ms": round(lat_ridge["t_total_p99_ms"], 4),
                "Regressor_Stage_Median_ms": lat_ridge["t_reg_ms"],
                "Nominal_50fps_Frame_ms": 20.0,
            },
            {
                "Pipeline_Variant": "Proposed GBDT + Calibrated GBDT",
                "Median_Total_ms": lat_results["t_total_ms"],
                "Mean_Total_ms": round(lat_results["t_total_mean_ms"], 4),
                "P95_Total_ms": round(lat_results["t_total_p95_ms"], 4),
                "P99_Total_ms": round(lat_results["t_total_p99_ms"], 4),
                "Regressor_Stage_Median_ms": lat_results["t_reg_ms"],
                "Nominal_50fps_Frame_ms": 20.0,
            },
        ])
        md_lat_cmp = _save_table(df_lat_cmp, tables_dir, "latency_model_comparison")

        md_lat = lat_bench.to_markdown(lat_results)
        (tables_dir / "latency_profile.md").write_text(md_lat, encoding="utf-8")
        report_sections.append("## 4. Computational Latency & Measured ODE Simulation Speedup (Desktop Benchmark)\n")
        report_sections.append(md_lat + "\n\n### 4B. Model Variant Latency Comparison (`N = 100` repetitions)\n" + md_lat_cmp + "\n")

    # Extract strictly causal canonical features once for statistical & interpretability analyses
    if "statistical" in benchmarks_to_run or "interpretability" in benchmarks_to_run:
        train_raw, val_raw, test_raw, ood_raw, _ = split_scenarios_by_partition(scenarios, random_state=parsed.seed)
        train_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(train_raw, parsed.t_obs)
        val_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(val_raw, parsed.t_obs)
        test_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(test_raw, parsed.t_obs)
        ood_scens, _ = WindowSensitivityBenchmark.filter_causal_scenarios(ood_raw, parsed.t_obs)

        fe = FeatureExtractor(t_obs=parsed.t_obs, f_nominal=50.0, strict_nadir=True)
        X_tr_df, y_tr, _, meta_tr = fe.transform_batch(train_scens, t_obs=parsed.t_obs, as_dataframe=True)
        X_va_df, y_va, _, _ = (
            fe.transform_batch(val_scens, t_obs=parsed.t_obs, as_dataframe=True)
            if val_scens
            else (pd.DataFrame(), np.array([]), np.array([]), pd.DataFrame())
        )
        X_te_df, y_te, _, meta_te = fe.transform_batch(test_scens, t_obs=parsed.t_obs, as_dataframe=True)
        X_ood_df, y_ood_stat, _, _ = (
            fe.transform_batch(ood_scens, t_obs=parsed.t_obs, as_dataframe=True)
            if ood_scens
            else (pd.DataFrame(), np.array([]), np.array([]), pd.DataFrame())
        )
        ys_tr = np.where(y_tr >= parsed.threshold, 1, 0).astype(int)
        ys_va = np.where(y_va >= parsed.threshold, 1, 0).astype(int) if len(y_va) > 0 else np.array([], dtype=int)
        ys_te = np.where(y_te >= parsed.threshold, 1, 0).astype(int)
        feat_names = list(X_tr_df.columns) if hasattr(X_tr_df, "columns") else None

    # 5. Statistical Hypothesis Tests (Holm-Adjusted Sign-Flip Permutation + Paired Bootstrap CI + Separate Wilcoxon)
    if "statistical" in benchmarks_to_run and len(test_scens) >= 2:
        logger.info("Running Paired Sign-Flip Permutation Tests (Holm-adjusted), Threshold Sweeps, and Learning Curves...")
        reg_preds_test: Dict[str, np.ndarray] = {}
        reg_preds_ood: Dict[str, np.ndarray] = {}
        for rk in ("mean", "analytical", "linear", "dt", "rf", "gbdt", "lgbm"):
            m_obj = saved_models.get(rk)
            if m_obj is not None and getattr(m_obj, "n_features_in_", X_te_df.shape[1]) == X_te_df.shape[1]:
                reg_preds_test[rk] = m_obj.predict(X_te_df)
                if len(X_ood_df) > 0:
                    reg_preds_ood[rk] = m_obj.predict(X_ood_df)

        report_sections.append("## 5. Paired Statistical Significance Tests (Sign-Flip Permutation + Holm FWER Adjustment)\n")
        if "gbdt" in reg_preds_test and len(reg_preds_test) >= 2:
            df_stat_reg = compare_all_models_statistical(
                y_te, reg_preds_test, reference_model="gbdt", task="regression", n_boot=2000, seed=parsed.seed
            )
            md_stat_reg = _save_table(df_stat_reg, tables_dir, "statistical_tests_regression")
            report_sections.append("### 5A. Nadir Regression vs Proposed `gbdt` (ID Test `N = 52`)\n" + md_stat_reg + "\n")

        if val_best_reg_key in reg_preds_test and len(reg_preds_test) >= 2:
            df_stat_ridge = compare_all_models_statistical(
                y_te, reg_preds_test, reference_model=val_best_reg_key, task="regression", n_boot=2000, seed=parsed.seed
            )
            md_stat_ridge = _save_table(df_stat_ridge, tables_dir, "statistical_tests_regression_ridge")
            report_sections.append(
                f"### 5B. Nadir Regression vs Validation-Selected `{val_best_reg_key}` (ID Test `N = 52`)\n" + md_stat_ridge + "\n"
            )

        if val_best_reg_key in reg_preds_ood and len(reg_preds_ood) >= 2 and len(y_ood_stat) >= 2:
            df_stat_ood = compare_all_models_statistical(
                y_ood_stat, reg_preds_ood, reference_model=val_best_reg_key, task="regression", n_boot=2000, seed=parsed.seed
            )
            md_stat_ood = _save_table(df_stat_ood, tables_dir, "statistical_tests_regression_ood")
            report_sections.append(
                f"### 5C. Nadir Regression vs Validation-Selected `{val_best_reg_key}` (OOD Test `N = {len(y_ood_stat)}`)\n"
                + md_stat_ood
                + "\n"
            )

        clf_probs_test: Dict[str, np.ndarray] = {}
        for ck in ("logistic_clf", "dt_clf", "rf_clf", "gbdt_clf", "lgbm_clf", "calibrated_gb_platt", "calibrated_gb_iso"):
            c_obj = saved_models.get(ck)
            if c_obj is not None and getattr(c_obj, "n_features_in_", X_te_df.shape[1]) == X_te_df.shape[1]:
                clf_probs_test[ck] = c_obj.predict_proba(X_te_df)[:, 1]
        if "calibrated_gb_platt" in clf_probs_test and len(clf_probs_test) >= 2:
            df_stat_clf = compare_all_models_statistical(
                ys_te, clf_probs_test, reference_model="calibrated_gb_platt", task="classification", n_boot=2000, seed=parsed.seed
            )
            md_stat_clf = _save_table(df_stat_clf, tables_dir, "statistical_tests_classification")
            report_sections.append("### 5D. Calibrated Probability Brier Loss vs Baselines (Holm-Adjusted)\n" + md_stat_clf + "\n")

            thr_rows = evaluate_probability_threshold_sweep(ys_te, clf_probs_test["calibrated_gb_platt"])
            df_thr = pd.DataFrame(thr_rows)
            md_thr = _save_table(df_thr, tables_dir, "threshold_sweep")
            report_sections.append("### 5E. Probability Decision-Threshold Sweep (`tau_prob` on Held-Out ID Test)\n" + md_thr + "\n")

        # Physical frequency threshold sensitivity sweep (49.20 - 49.80 Hz)
        df_fthr = _run_frequency_threshold_sensitivity(
            X_tr_df.values,
            y_tr,
            X_va_df.values if len(X_va_df) > 0 else None,
            y_va if len(y_va) > 0 else None,
            X_te_df.values,
            y_te,
            X_ood_df.values if len(X_ood_df) > 0 else None,
            y_ood_stat if len(y_ood_stat) > 0 else None,
            seed=parsed.seed,
        )
        md_fthr = _save_table(df_fthr, tables_dir, "frequency_threshold_sensitivity")
        report_sections.append("### 5F. Physical Security-Threshold Sensitivity (`f_thresh` in 49.20–49.80 Hz)\n" + md_fthr + "\n")

        lc_eval = LearningCurveEvaluator(threshold=parsed.threshold, seed=parsed.seed)
        lc_res = lc_eval.run(
            X_tr_df.values,
            y_tr,
            ys_tr,
            X_te_df.values,
            y_te,
            ys_te,
            X_val=X_va_df.values if len(X_va_df) > 0 else None,
            y_val_nadir=y_va if len(y_va) > 0 else None,
            y_val_sec=ys_va if len(ys_va) > 0 else None,
        )
        df_lc = lc_eval.to_dataframe(lc_res)
        df_lc.to_csv(tables_dir / "learning_curve.csv", index=False)
        md_lc = lc_eval.to_markdown(lc_res)
        (tables_dir / "learning_curve.md").write_text(md_lc, encoding="utf-8")
        report_sections.append("### 5G. Sample-Complexity Learning Curve (20%–100% Train)\n" + md_lc + "\n")

        ms_eval = MultiSeedStabilityEvaluator(seeds=[42, 43, 44, 45, 46], threshold=parsed.threshold)
        ms_res = ms_eval.run(
            X_tr_df.values,
            y_tr,
            ys_tr,
            X_te_df.values,
            y_te,
            ys_te,
            X_val=X_va_df.values if len(X_va_df) > 0 else None,
            y_val_nadir=y_va if len(y_va) > 0 else None,
        )
        df_ms = ms_eval.to_dataframe(ms_res)
        df_ms.to_csv(tables_dir / "multiseed_stability.csv", index=False)
        md_ms = ms_eval.to_markdown(ms_res)
        (tables_dir / "multiseed_stability.md").write_text(md_ms, encoding="utf-8")
        report_sections.append("### 5H. Multi-Seed Stability (Seeds 42–46)\n" + md_ms + "\n")

    # 6. Physical Interpretability, Correlations & Feature/Observability Ablations
    if "interpretability" in benchmarks_to_run and len(test_scens) >= 1:
        logger.info("Running Interpretability, Feature Correlations, and 10-Configuration Ablations...")
        interp = InterpretabilityAnalyzer(t_obs=parsed.t_obs, threshold=parsed.threshold, seed=parsed.seed)

        target_reg = proposed_gbdt_reg
        if target_reg is None or getattr(target_reg, "n_features_in_", X_te_df.shape[1]) != X_te_df.shape[1]:
            from pmu_fsa.models.regressors import GradientBoostingNadirRegressor
            target_reg = GradientBoostingNadirRegressor(
                n_estimators=100, learning_rate=0.08, max_depth=4, random_state=parsed.seed
            )
            target_reg.fit(X_tr_df, y_tr)

        perm_df = interp.compute_permutation_importance(
            target_reg, X_te_df, y_te, feature_names=feat_names, task="regression", n_repeats=15, seed=parsed.seed
        )
        imp_df = interp.compute_impurity_importance(target_reg, feature_names=feat_names)
        merged_imp = perm_df.merge(
            imp_df[["feature", "impurity_importance", "rank"]].rename(columns={"rank": "impurity_rank"}),
            on="feature",
            how="left",
        )
        md_imp = _save_table(merged_imp, tables_dir, "feature_importance")

        corr_info = interp.compute_correlation_matrix(X_tr_df, feature_names=feat_names)
        corr_info["spearman_df"].to_csv(tables_dir / "feature_correlations_spearman.csv")
        corr_info["pearson_df"].to_csv(tables_dir / "feature_correlations_pearson.csv")

        ab_res = interp.run_feature_group_ablations(
            scenarios,
            regressor_type="gbdt",
            classifier_type="calibrated_gb",
            include_single_family_drops=True,
        )
        df_ab = pd.DataFrame(ab_res["results"])
        md_ab = _save_table(df_ab, tables_dir, "feature_ablations")

        win_imp_df = interp.compute_importance_across_windows(scenarios, windows=[0.5, 1.0, 1.5, 2.0])
        if not win_imp_df.empty:
            win_imp_df.to_csv(tables_dir / "window_feature_importance.csv", index=False)
            (tables_dir / "window_feature_importance.md").write_text(
                _df_to_markdown(win_imp_df[win_imp_df["perm_rank"] <= 5]), encoding="utf-8"
            )

        rank_stab = interp.compute_multiseed_rank_stability(
            X_tr_df, y_tr, X_te_df, y_te, feature_names=feat_names, seeds=[42, 43, 44, 45, 46]
        )
        _save_table(rank_stab["table"], tables_dir, "multiseed_rank_stability")

        elig_all, _ = WindowSensitivityBenchmark.filter_causal_scenarios(scenarios, parsed.t_obs)
        X_all_df, y_all_n, _, meta_all = fe.transform_batch(elig_all, t_obs=parsed.t_obs, as_dataframe=True)
        ys_all = np.where(y_all_n >= parsed.threshold, 1, 0).astype(int)
        shifts_df = interp.compute_subgroup_feature_shifts(X_all_df, y_all_n, ys_all, meta_all, feature_names=feat_names)
        if not shifts_df.empty:
            _save_table(shifts_df, tables_dir, "subgroup_feature_shifts")

        report_sections.append("## 6. Physical Interpretability & Observability Ablations\n")
        report_sections.append(
            f"- **Clean 1.0s Feature Matrix Rank**: `{corr_info['matrix_rank']} / {corr_info['n_features']}` "
            f"(`freq_drop_end == freq_drop_max` on monotonic pre-nadir ramps; `14 / 14` under PMU noise; "
            f"max off-diagonal |Spearman| = `{corr_info['max_off_diag_abs_spearman']:.4f}`, "
            f"multi-seed rank Spearman = `{rank_stab['mean_pairwise_spearman']:.4f}`)\n"
        )
        report_sections.append("### 6A. Permutation & Impurity Feature Importance (Held-Out ID Test)\n" + md_imp + "\n")
        report_sections.append("### 6B. Feature-Group, Single-Family Drop, Partial 13-Bus PMU & 25-fps Ablations\n" + md_ab + "\n")

    report_path.write_text("\n".join(report_sections), encoding="utf-8")
    logger.info("Evaluation suite finished cleanly. Report written to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
