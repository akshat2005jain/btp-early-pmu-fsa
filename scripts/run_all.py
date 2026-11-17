#!/usr/bin/env python3
"""Master end-to-end runner for the 50.0 Hz IEEE 39-bus PMU-FSA research pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from pmu_fsa.pipeline.run_experiments import ExperimentConfig
from pmu_fsa.simulation.dataset_validator import (
    validate_dataset,
    write_dataset_quality_report,
)
from pmu_fsa.simulation.ieee39_data import (
    BASE_MVA,
    GENERATOR_DATA,
    LOAD_DATA,
    N_BRANCHES,
    N_BUSES,
    N_GENERATORS,
)

logger = logging.getLogger("run_all")


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
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


def export_dataset_and_calibration_tables(
    features_csv_path: Path,
    manifest_path: Path,
    training_metrics_path: Path,
    tables_dir: Path,
) -> None:
    """Export dataset_composition, partition_summary, scenario_partitioning, and calibration_comparison tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if features_csv_path.exists():
        df_feat = pd.read_csv(features_csv_path)
        n_total = len(df_feat)
        total_load_mw = sum(p for p, _ in LOAD_DATA.values())
        total_gen_mw = sum(float(g["P_mw"]) for g in GENERATOR_DATA)
        h_regional = sum(42.0 if g["bus"] == 39 else float(g["H"]) for g in GENERATOR_DATA)

        comp_rows: List[Dict[str, Any]] = [
            {
                "Parameter / Dimension": "Benchmark Power System",
                "Specification": f"IEEE 39-Bus, {N_GENERATORS}-Machine, {N_BRANCHES}-Branch (3 Coherent Areas)",
                "Notes": f"Base MVA = {BASE_MVA:.0f}, Nominal Load = {total_load_mw:.1f} MW, Gen = {total_gen_mw:.1f} MW",
            },
            {
                "Parameter / Dimension": "System Frequency & Threshold",
                "Specification": "50.0 Hz nominal, 49.50 Hz security threshold",
                "Notes": f"Regional Bus-39 inertia H_39 = 42.0 s (pre-fault H_tot = {h_regional:.1f} s)",
            },
            {
                "Parameter / Dimension": "Outage Contingencies",
                "Specification": f"{df_feat['disturbed_gen'].nunique()} generators (Buses 30-39) x {df_feat['loss_fraction'].round(2).nunique()} loss fractions ({sorted(df_feat['loss_fraction'].round(2).unique().tolist())})",
                "Notes": f"Active outage range: [{df_feat['active_loss_mw'].min():.1f}, {df_feat['active_loss_mw'].max():.1f}] MW",
            },
            {
                "Parameter / Dimension": "Renewable Penetration Grid",
                "Specification": f"{sorted((df_feat['renewable_penetration'] * 100).round(0).astype(int).unique().tolist())} % of system load",
                "Notes": f"Realized RES range: [{df_feat['p_ren_realized'].min()*100:.2f}%, {df_feat['p_ren_realized'].max()*100:.2f}%]",
            },
            {
                "Parameter / Dimension": "System Load Scaling Grid",
                "Specification": f"{sorted(df_feat['load_scale'].round(2).unique().tolist())} x nominal PQ load",
                "Notes": "Newton-Raphson AC power flow solved per operating point",
            },
            {
                "Parameter / Dimension": "Total Unique Scenarios",
                "Specification": f"{n_total} trajectories (10.0 s horizon @ 50 fps, t_0 = 1.0 s)",
                "Notes": f"Secure: {int((df_feat['is_secure'] == 1).sum())} ({(df_feat['is_secure'] == 1).mean()*100:.1f}%), Insecure: {int((df_feat['is_secure'] == 0).sum())} ({(df_feat['is_secure'] == 0).mean()*100:.1f}%)",
            },
            {
                "Parameter / Dimension": "COI Frequency Nadir Range",
                "Specification": f"[{df_feat['ground_truth_nadir'].min():.4f}, {df_feat['ground_truth_nadir'].max():.4f}] Hz",
                "Notes": f"Median nadir = {df_feat['ground_truth_nadir'].median():.4f} Hz, t_nadir in [{df_feat['ground_truth_t_nadir'].min():.2f}, {df_feat['ground_truth_t_nadir'].max():.2f}] s",
            },
            {
                "Parameter / Dimension": "Causal Observation Windows",
                "Specification": "T_obs in {0.5, 1.0, 1.5, 2.0} s (primary T_obs = 1.0 s, 14 PMU features)",
                "Notes": f"Pre-nadir causal at 1.0 s: {int(df_feat['is_prior_to_nadir'].sum())}/{n_total} ({df_feat['is_prior_to_nadir'].mean()*100:.2f}%)",
            },
        ]
        df_comp = pd.DataFrame(comp_rows)
        df_comp.to_csv(tables_dir / "dataset_composition.csv", index=False)
        (tables_dir / "dataset_composition.md").write_text(_df_to_markdown(df_comp), encoding="utf-8")

        # Partition & OOD breakdown table
        part_rows: List[Dict[str, Any]] = []
        split_specs = [
            ("train", "id", "ID Train (RES <= 30%, Load 0.90-1.10x)", df_feat["partition"] == "train"),
            ("val", "id", "ID Validation (RES <= 30%, Load 0.90-1.10x)", df_feat["partition"] == "val"),
            ("test", "id", "ID Held-Out Test (RES <= 30%, Load 0.90-1.10x)", df_feat["partition"] == "test"),
            ("ood_test", "ood_res", "OOD-RES (RES 40-50%, Load 0.90-1.10x)", df_feat["ood_type"] == "ood_res"),
            ("ood_test", "ood_load", "OOD-Load (RES <= 30%, Load {0.80, 1.20}x)", df_feat["ood_type"] == "ood_load"),
            ("ood_test", "ood_combined", "OOD-Combined (RES 40-50%, Load {0.80, 1.20}x)", df_feat["ood_type"] == "ood_combined"),
            ("ood_test (all)", "all_ood", "All OOD Test Combined", df_feat["partition"] == "ood_test"),
            ("total", "all", "Full Dataset Total", np.ones(len(df_feat), dtype=bool)),
        ]
        for p_name, ood_name, desc, mask in split_specs:
            sub = df_feat[mask]
            if sub.empty:
                continue
            n_s = len(sub)
            n_sec = int((sub["is_secure"] == 1).sum())
            n_insec = int((sub["is_secure"] == 0).sum())
            part_rows.append({
                "Partition": p_name,
                "Regime": ood_name,
                "Operating Regime Description": desc,
                "N_Scenarios": n_s,
                "Share_Pct": round(100.0 * n_s / max(1, n_total), 2),
                "Secure_Count": n_sec,
                "Insecure_Count": n_insec,
                "Secure_Rate": round(n_sec / max(1, n_s), 4),
                "Mean_Nadir_Hz": round(float(sub["ground_truth_nadir"].mean()), 4),
                "Min_Nadir_Hz": round(float(sub["ground_truth_nadir"].min()), 4),
                "Mean_Lead_Time_1s": round(float(sub["lead_time_s"].mean()), 4),
            })

        df_part = pd.DataFrame(part_rows)
        for alias in ("partition_summary", "scenario_partitioning"):
            df_part.to_csv(tables_dir / f"{alias}.csv", index=False)
            (tables_dir / f"{alias}.md").write_text(_df_to_markdown(df_part), encoding="utf-8")

    if training_metrics_path.exists():
        tm = json.loads(training_metrics_path.read_text(encoding="utf-8"))
        val_clf = tm.get("val_metrics", {}).get("classification", {})
        test_clf = tm.get("test_metrics", {}).get("classification", {})
        cal_rows: List[Dict[str, Any]] = []
        for key, label in (
            ("gbdt_clf", "Uncalibrated GBDT Classifier"),
            ("calibrated_gb_platt", "Proposed Calibrated GBDT (Platt Scaling)"),
            ("calibrated_gb_iso", "Proposed Calibrated GBDT (PAVA Isotonic)"),
        ):
            tc = test_clf.get(key, {})
            vc = val_clf.get(key, {})
            if not tc:
                continue
            cal_rows.append({
                "Model": label,
                "Effective_Method": tc.get("effective_method", tc.get("Effective_Calibration", "none")),
                "N_Cal": tc.get("n_cal", tm.get("partition_counts", {}).get("val", 0)),
                "Val_Brier": vc.get("brier_score", float("nan")),
                "Val_ECE": vc.get("ece", float("nan")),
                "Test_Brier": tc.get("Brier", float("nan")),
                "Test_ECE": tc.get("ECE", float("nan")),
                "Test_MCE": tc.get("MCE", float("nan")),
                "Test_F1": tc.get("F1", float("nan")),
                "Test_PR_AUC": tc.get("PR-AUC", float("nan")),
                "Test_FNR_insecure": tc.get("FNR_insecure", float("nan")),
                "Missed_Insecure": tc.get("Missed_Insecure", 0),
            })
        if cal_rows:
            df_cal = pd.DataFrame(cal_rows)
            df_cal.to_csv(tables_dir / "calibration_comparison.csv", index=False)
            (tables_dir / "calibration_comparison.md").write_text(_df_to_markdown(df_cal), encoding="utf-8")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Master Reproducible Pipeline Runner (50.0 Hz IEEE 39-Bus PMU-FSA)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default_50hz.yaml", help="Path to YAML/JSON configuration")
    parser.add_argument("--n-scenarios", type=int, default=None, help="Override scenario count")
    parser.add_argument("--skip-simulation", action="store_true", help="Reuse existing raw .npz scenarios in raw_data_dir")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training stage")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip benchmark evaluation stage")
    parser.add_argument("--with-figures", action="store_true", help="Also run scripts/generate_figures.py at the end")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet logging")
    return parser.parse_args(args)


def _run_cmd(cmd: List[str]) -> None:
    logger.info("Executing: %s", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {' '.join(cmd)}")


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ExperimentConfig.from_file(PROJECT_ROOT / parsed.config if not Path(parsed.config).is_absolute() else parsed.config)
    if parsed.n_scenarios is not None:
        cfg.n_scenarios = int(parsed.n_scenarios)

    py = sys.executable
    raw_dir = PROJECT_ROOT / cfg.raw_data_dir
    sim_dir = PROJECT_ROOT / cfg.raw_sim_dir
    proc_dir = PROJECT_ROOT / cfg.processed_data_dir
    models_dir = PROJECT_ROOT / cfg.models_dir
    tables_dir = PROJECT_ROOT / cfg.tables_dir
    report_file = PROJECT_ROOT / cfg.report_file
    manifest_path = PROJECT_ROOT / "data" / "manifest.json"
    quality_md_path = PROJECT_ROOT / "results" / "dataset_quality_report.md"

    q_flag = ["--quiet"] if parsed.quiet else []

    # Stage 1: Raw & PMU Simulation
    if not parsed.skip_simulation:
        sim_cmd = [
            py, str(PROJECT_ROOT / "scripts" / "run_simulations.py"),
            "--clean",
            "--n-scenarios", str(cfg.n_scenarios),
            "--fps", str(cfg.fps),
            "--seed", str(cfg.seed),
            "--t-disturb", str(cfg.t_disturb),
            "--t-max", str(cfg.t_max),
            "--f-nominal", str(cfg.f_nominal),
            "--threshold", str(cfg.security_threshold_hz),
            "--output-dir", str(raw_dir),
            "--raw-sim-dir", str(sim_dir),
            "--loss-fractions", *[str(x) for x in cfg.loss_fractions],
            "--penetrations", *[str(x) for x in cfg.penetrations],
            "--load-scales", *[str(x) for x in cfg.load_scales],
            "--validate",
            *q_flag,
        ]
        if cfg.gen_buses:
            sim_cmd.extend(["--gen-buses", *[str(b) for b in cfg.gen_buses]])
        if cfg.macro_intertie:
            sim_cmd.append("--macro-intertie")
        if cfg.noise_std > 0.0:
            sim_cmd.extend(["--inject-noise", "--noise-std", str(cfg.noise_std)])
        _run_cmd(sim_cmd)

    # Stage 2: Feature Extraction across all windows
    ext_cmd = [
        py, str(PROJECT_ROOT / "scripts" / "extract_features.py"),
        "--input-dir", str(raw_dir),
        "--t-obs", str(cfg.t_obs),
        "--f-nominal", str(cfg.f_nominal),
        "--threshold", str(cfg.security_threshold_hz),
        "--output-file", str(proc_dir / "features.npz"),
        "--all-windows",
        *q_flag,
    ]
    if cfg.strict_nadir:
        ext_cmd.append("--strict-nadir")
    if cfg.save_csv:
        ext_cmd.append("--save-csv")
    _run_cmd(ext_cmd)

    # Re-validate covering both raw and processed datasets
    val_report = validate_dataset(
        raw_pmu_dir=raw_dir,
        processed_dir=proc_dir,
        f_nominal=cfg.f_nominal,
        threshold=cfg.security_threshold_hz,
        raw_sim_dir=sim_dir if sim_dir.exists() else None,
    )
    write_dataset_quality_report(val_report, md_path=quality_md_path, manifest_path=manifest_path)

    # Stage 3: Leak-free Model Training
    if not parsed.skip_training:
        train_cmd = [
            py, str(PROJECT_ROOT / "scripts" / "train_models.py"),
            "--features", str(proc_dir / "features.npz"),
            "--model-type", "all",
            "--calibration", "platt",
            "--threshold", str(cfg.security_threshold_hz),
            "--output-dir", str(models_dir),
            "--tables-dir", str(tables_dir),
            "--seed", str(cfg.seed),
            *q_flag,
        ]
        _run_cmd(train_cmd)

    # Export dataset composition, partition summary, and calibration comparison tables
    export_dataset_and_calibration_tables(
        features_csv_path=proc_dir / "features.csv",
        manifest_path=manifest_path,
        training_metrics_path=models_dir / "training_metrics.json",
        tables_dir=tables_dir,
    )

    # Stage 4: Evaluation Benchmarks, Latency, Statistical CIs, and Ablations
    if not parsed.skip_evaluation:
        eval_cmd = [
            py, str(PROJECT_ROOT / "scripts" / "evaluate_all.py"),
            "--data-dir", str(raw_dir),
            "--models-dir", str(models_dir),
            "--features", str(proc_dir / "features.npz"),
            "--tables-dir", str(tables_dir),
            "--report-file", str(report_file),
            "--seed", str(cfg.seed),
            *q_flag,
        ]
        _run_cmd(eval_cmd)

    # Optional Stage 5: Publication Figures
    if parsed.with_figures and (PROJECT_ROOT / "scripts" / "generate_figures.py").exists():
        _run_cmd([py, str(PROJECT_ROOT / "scripts" / "generate_figures.py")])

    logger.info("Full 50 Hz PMU-FSA pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
