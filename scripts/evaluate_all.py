#!/usr/bin/env python3
"""Run leak-free sensitivity, OOD, noise, latency, statistical, and interpretability benchmarks."""

from __future__ import annotations

import argparse
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

from pmu_fsa.evaluation.interpretability import InterpretabilityAnalyzer
from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark, split_scenarios_by_partition
from pmu_fsa.evaluation.statistical_tests import (
    LearningCurveEvaluator,
    MultiSeedStabilityEvaluator,
    compare_all_models_statistical,
)
from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.classifiers import evaluate_probability_threshold_sweep
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


def _generate_fallback_50hz_scenarios(seed: int = 42) -> List[Dict[str, Any]]:
    """Generate fast synthetic 50 Hz partitioned scenarios ONLY when --data-dir has zero .npz files."""
    rng = np.random.default_rng(seed)
    dt = 0.02
    n_samples = 251
    timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
    t_post = np.maximum(0.0, timestamps - 1.0)

    specs = [
        # (partition, ood_type, pen, load_scale, loss_mw)
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
            loaded[fp.stem] = joblib.load(fp)
        except Exception as exc:
            logger.warning("Could not load model artifact %s: %s", fp, exc)
    return loaded


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

    # Resolve output paths cleanly without touching live results/ when custom --output-dir/--tables-dir is given
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
    primary_reg = saved_models.get("gbdt") or saved_models.get("best_regressor")
    primary_clf = (
        saved_models.get("calibrated_gb_platt")
        or saved_models.get("calibrated_gb")
        or saved_models.get("best_classifier")
    )
    if saved_models:
        logger.info("Loaded %d saved model artifacts from %s", len(saved_models), models_dir)

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

    benchmarks_to_run = list(parsed.benchmarks)
    if "all" in benchmarks_to_run:
        benchmarks_to_run = list(ALL_BENCHMARK_KEYS)

    report_sections: List[str] = [
        "# PMU-Based Early Frequency-Security Assessment: Honest Benchmark Evaluation Report\n",
        f"- **System Standard**: `50.0 Hz` nominal, `{parsed.threshold:.2f} Hz` security threshold",
        f"- **Loaded Scenarios**: `{len(scenarios)}` (`synthetic_fallback={used_synthetic_fallback}`)",
        f"- **Loaded Model Artifacts**: `{sorted(list(saved_models.keys())) if saved_models else 'trained_per_split'}`\n",
    ]

    # 1. Window Sensitivity Benchmark
    if "sensitivity" in benchmarks_to_run:
        logger.info("Running Window Sensitivity Benchmark (strictly held-out test & OOD)...")
        sens_bench = WindowSensitivityBenchmark(
            regressor_type="gbdt",
            classifier_type="calibrated_gb",
            threshold=parsed.threshold,
            random_state=parsed.seed,
            models_dir=models_dir,
        )
        default_win_tuple = (parsed.t_obs, primary_reg, primary_clf) if (primary_reg is not None and primary_clf is not None) else None
        sens_results = sens_bench.run(scenarios, default_window_models=default_win_tuple)

        df_sens = sens_bench.to_dataframe(sens_results, split="id_test")
        df_sens.to_csv(tables_dir / "window_sensitivity.csv", index=False)
        md_sens = sens_bench.to_markdown(sens_results, split="id_test")
        (tables_dir / "window_sensitivity.md").write_text(md_sens, encoding="utf-8")

        report_sections.append(r"## 1. Observation Window Sensitivity ($T_{\text{obs}} \in [0.5, 2.0]$ s — Held-Out ID Test)" + "\n")
        report_sections.append(md_sens + "\n")

        df_sens_ood = sens_bench.to_dataframe(sens_results, split="ood_test")
        if not df_sens_ood.empty:
            df_sens_ood.to_csv(tables_dir / "window_sensitivity_ood.csv", index=False)
            md_sens_ood = sens_bench.to_markdown(sens_results, split="ood_test")
            (tables_dir / "window_sensitivity_ood.md").write_text(md_sens_ood, encoding="utf-8")
            report_sections.append("### 1B. Observation Window Sensitivity on Held-Out OOD Test\n")
            report_sections.append(md_sens_ood + "\n")

    # 2. Out-of-Distribution Generalization Benchmark
    if "ood" in benchmarks_to_run:
        logger.info("Running Out-of-Distribution Generalization Benchmark...")
        ood_bench = OODGeneralizationBenchmark(
            t_obs=parsed.t_obs,
            threshold=parsed.threshold,
            random_state=parsed.seed,
        )
        ood_results = ood_bench.run(
            scenarios,
            regressor_type="gbdt",
            classifier_type="calibrated_gb",
            regressor=primary_reg,
            classifier=primary_clf,
        )
        df_ood = ood_bench.to_dataframe(ood_results)
        df_ood.to_csv(tables_dir / "ood_generalization.csv", index=False)
        df_ood_regimes = ood_bench.to_regime_dataframe(ood_results)
        if not df_ood_regimes.empty:
            df_ood_regimes.to_csv(tables_dir / "ood_regime_breakdown.csv", index=False)
            (tables_dir / "ood_regime_breakdown.md").write_text(_df_to_markdown(df_ood_regimes), encoding="utf-8")

        md_ood = ood_bench.to_markdown(ood_results)
        (tables_dir / "ood_generalization.md").write_text(md_ood, encoding="utf-8")
        report_sections.append("## 2. Out-of-Distribution Generalization (ID Held-Out Test vs OOD-RES / Load / Combined)\n")
        report_sections.append(md_ood + "\n")

    # 3. PMU Measurement Noise Stress Benchmark
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
            regressor=primary_reg,
            classifier=primary_clf,
        )
        df_noise = noise_bench.to_dataframe(noise_results)
        df_noise.to_csv(tables_dir / "measurement_noise.csv", index=False)
        md_noise = noise_bench.to_markdown(noise_results)
        (tables_dir / "measurement_noise.md").write_text(md_noise, encoding="utf-8")

        rocof_cmp = MeasurementNoiseBenchmark.compare_ols_vs_endpoint_rocof(noise_std=0.03, n_trials=100, seed=parsed.seed)
        report_sections.append("## 3. PMU Measurement Noise Stress Benchmark\n")
        report_sections.append(md_noise)
        report_sections.append(
            f"\n- **Causal Windowed OLS vs 2-Point ROCOF Variance Ratio**: `{rocof_cmp['variance_ratio']}` "
            f"(`var_ols={rocof_cmp['var_ols']}`, `var_2pt={rocof_cmp['var_endpoint']}`)\n"
        )

    # 4. Hardware Latency & Real ODE Simulation Speedup Benchmark
    if "latency" in benchmarks_to_run:
        logger.info("Running 6-Stage Hardware Latency & Real DynamicSimulator Benchmark...")
        lat_bench = LatencyBenchmark()
        lat_results = lat_bench.profile(
            regressor=primary_reg,
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

        df_stages = lat_bench.to_stages_dataframe(lat_results)
        df_stages.to_csv(tables_dir / "latency_stages.csv", index=False)
        (tables_dir / "latency_stages.md").write_text(_df_to_markdown(df_stages), encoding="utf-8")

        md_lat = lat_bench.to_markdown(lat_results)
        (tables_dir / "latency_profile.md").write_text(md_lat, encoding="utf-8")
        report_sections.append("## 4. Online Computational Latency & Measured ODE Simulation Speedup\n")
        report_sections.append(md_lat + "\n")

    # Extract canonical features once for statistical & interpretability analyses
    if "statistical" in benchmarks_to_run or "interpretability" in benchmarks_to_run:
        train_scens, val_scens, test_scens, ood_scens, _ = split_scenarios_by_partition(scenarios, random_state=parsed.seed)
        fe = FeatureExtractor(t_obs=parsed.t_obs, f_nominal=50.0)
        X_tr_df, y_tr, _, meta_tr = fe.transform_batch(train_scens, t_obs=parsed.t_obs, as_dataframe=True)
        X_va_df, y_va, _, _ = fe.transform_batch(val_scens, t_obs=parsed.t_obs, as_dataframe=True) if val_scens else (pd.DataFrame(), np.array([]), np.array([]), pd.DataFrame())
        X_te_df, y_te, _, meta_te = fe.transform_batch(test_scens, t_obs=parsed.t_obs, as_dataframe=True)
        ys_tr = np.where(y_tr >= parsed.threshold, 1, 0).astype(int)
        ys_va = np.where(y_va >= parsed.threshold, 1, 0).astype(int) if len(y_va) > 0 else np.array([], dtype=int)
        ys_te = np.where(y_te >= parsed.threshold, 1, 0).astype(int)
        feat_names = list(X_tr_df.columns) if hasattr(X_tr_df, "columns") else None

    # 5. Statistical Hypothesis Tests, Learning Curve, Threshold Sweep & Multi-Seed Stability
    if "statistical" in benchmarks_to_run and len(test_scens) >= 2:
        logger.info("Running Paired Statistical Tests, Learning Curves, and Multi-Seed Stability...")
        reg_preds_test: Dict[str, np.ndarray] = {}
        for rk in ("mean", "analytical", "linear", "dt", "rf", "gbdt", "lgbm"):
            m_obj = saved_models.get(rk)
            if m_obj is not None and getattr(m_obj, "n_features_in_", X_te_df.shape[1]) == X_te_df.shape[1]:
                reg_preds_test[rk] = m_obj.predict(X_te_df)
        if "gbdt" in reg_preds_test and len(reg_preds_test) >= 2:
            df_stat_reg = compare_all_models_statistical(
                y_te, reg_preds_test, reference_model="gbdt", task="regression", n_boot=1000, seed=parsed.seed
            )
            df_stat_reg.to_csv(tables_dir / "statistical_tests_regression.csv", index=False)
            md_stat_reg = _df_to_markdown(df_stat_reg)
            (tables_dir / "statistical_tests_regression.md").write_text(md_stat_reg, encoding="utf-8")
            report_sections.append("## 5. Paired Statistical Significance Tests (Bootstrap n=1000 & Wilcoxon)\n")
            report_sections.append("### 5A. Nadir Regression vs GBDT\n" + md_stat_reg + "\n")

        clf_probs_test: Dict[str, np.ndarray] = {}
        for ck in ("logistic_clf", "dt_clf", "rf_clf", "gbdt_clf", "lgbm_clf", "calibrated_gb_platt", "calibrated_gb_iso"):
            c_obj = saved_models.get(ck)
            if c_obj is not None and getattr(c_obj, "n_features_in_", X_te_df.shape[1]) == X_te_df.shape[1]:
                clf_probs_test[ck] = c_obj.predict_proba(X_te_df)[:, 1]
        if "calibrated_gb_platt" in clf_probs_test and len(clf_probs_test) >= 2:
            df_stat_clf = compare_all_models_statistical(
                ys_te, clf_probs_test, reference_model="calibrated_gb_platt", task="classification", n_boot=1000, seed=parsed.seed
            )
            df_stat_clf.to_csv(tables_dir / "statistical_tests_classification.csv", index=False)
            md_stat_clf = _df_to_markdown(df_stat_clf)
            (tables_dir / "statistical_tests_classification.md").write_text(md_stat_clf, encoding="utf-8")
            report_sections.append("### 5B. Calibrated Probability Brier Loss vs Baselines\n" + md_stat_clf + "\n")

            thr_rows = evaluate_probability_threshold_sweep(ys_te, clf_probs_test["calibrated_gb_platt"])
            df_thr = pd.DataFrame(thr_rows)
            df_thr.to_csv(tables_dir / "threshold_sweep.csv", index=False)
            (tables_dir / "threshold_sweep.md").write_text(_df_to_markdown(df_thr), encoding="utf-8")
            report_sections.append("### 5C. Safety Decision-Threshold Sweep (Held-Out Test)\n" + _df_to_markdown(df_thr) + "\n")

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
        report_sections.append("### 5D. Sample-Complexity Learning Curve (20%–100% Train)\n" + md_lc + "\n")

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
        report_sections.append("### 5E. Multi-Seed Stability (Seeds 42–46)\n" + md_ms + "\n")

    # 6. Physical Interpretability, Correlations & Feature/Observability Ablations
    if "interpretability" in benchmarks_to_run and len(test_scens) >= 1:
        logger.info("Running Interpretability, Feature Correlations, and Observability Ablations...")
        interp = InterpretabilityAnalyzer(t_obs=parsed.t_obs, threshold=parsed.threshold, seed=parsed.seed)

        target_reg = primary_reg
        if target_reg is None or getattr(target_reg, "n_features_in_", X_te_df.shape[1]) != X_te_df.shape[1]:
            from pmu_fsa.models.regressors import GradientBoostingNadirRegressor
            target_reg = GradientBoostingNadirRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=parsed.seed)
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
        merged_imp.to_csv(tables_dir / "feature_importance.csv", index=False)
        md_imp = _df_to_markdown(merged_imp)
        (tables_dir / "feature_importance.md").write_text(md_imp, encoding="utf-8")

        corr_info = interp.compute_correlation_matrix(X_tr_df, feature_names=feat_names)
        corr_info["spearman_df"].to_csv(tables_dir / "feature_correlations_spearman.csv")
        corr_info["pearson_df"].to_csv(tables_dir / "feature_correlations_pearson.csv")

        ab_res = interp.run_feature_group_ablations(scenarios, regressor_type="gbdt", classifier_type="calibrated_gb")
        df_ab = pd.DataFrame(ab_res["results"])
        df_ab.to_csv(tables_dir / "feature_ablations.csv", index=False)
        md_ab = _df_to_markdown(df_ab)
        (tables_dir / "feature_ablations.md").write_text(md_ab, encoding="utf-8")

        win_imp_df = interp.compute_importance_across_windows(scenarios, windows=[0.5, 1.0, 1.5, 2.0])
        if not win_imp_df.empty:
            win_imp_df.to_csv(tables_dir / "window_feature_importance.csv", index=False)
            (tables_dir / "window_feature_importance.md").write_text(
                _df_to_markdown(win_imp_df[win_imp_df["perm_rank"] <= 5]), encoding="utf-8"
            )

        rank_stab = interp.compute_multiseed_rank_stability(
            X_tr_df, y_tr, X_te_df, y_te, feature_names=feat_names, seeds=[42, 43, 44, 45, 46]
        )
        rank_stab["table"].to_csv(tables_dir / "multiseed_rank_stability.csv", index=False)
        (tables_dir / "multiseed_rank_stability.md").write_text(_df_to_markdown(rank_stab["table"]), encoding="utf-8")

        X_all_df, y_all_n, _, meta_all = fe.transform_batch(scenarios, t_obs=parsed.t_obs, as_dataframe=True)
        ys_all = np.where(y_all_n >= parsed.threshold, 1, 0).astype(int)
        shifts_df = interp.compute_subgroup_feature_shifts(X_all_df, y_all_n, ys_all, meta_all, feature_names=feat_names)
        if not shifts_df.empty:
            shifts_df.to_csv(tables_dir / "subgroup_feature_shifts.csv", index=False)
            (tables_dir / "subgroup_feature_shifts.md").write_text(_df_to_markdown(shifts_df), encoding="utf-8")

        report_sections.append("## 6. Physical Interpretability & Observability Ablations\n")
        report_sections.append(
            f"- **Feature Matrix Rank**: `{corr_info['matrix_rank']} / {corr_info['n_features']}` "
            f"(max off-diagonal |Spearman| = `{corr_info['max_off_diag_abs_spearman']:.4f}`, "
            f"multi-seed rank Spearman = `{rank_stab['mean_pairwise_spearman']:.4f}`)\n"
        )
        report_sections.append("### 6A. Permutation & Impurity Feature Importance (Held-Out Test)\n" + md_imp + "\n")
        report_sections.append("### 6B. Feature-Group, Partial 13-Bus PMU & 25-fps Reporting Ablations\n" + md_ab + "\n")

    report_path.write_text("\n".join(report_sections), encoding="utf-8")
    logger.info("Evaluation suite finished cleanly. Report written to %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
