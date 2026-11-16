#!/usr/bin/env python3
"""Run sensitivity, OOD, noise, and latency benchmarks and write summary reports."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from pmu_fsa.simulation.scenario_generator import load_scenario_from_npz
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.latency import LatencyBenchmark

logger = logging.getLogger("evaluate_all")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Evaluate All Benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models-dir", type=str, default="models/", help="Directory containing trained models")
    parser.add_argument("--report-file", type=str, default="results/evaluation_report.md", help="Path for report markdown")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["sensitivity", "ood", "noise", "latency"],
        help="Benchmarks to run",
    )
    parser.add_argument("--data-dir", type=str, default="data/raw/", help="Directory containing raw scenarios")
    parser.add_argument("--features", type=str, default="data/processed/features.npz", help="Processed features path")
    parser.add_argument("--tables-dir", type=str, default="results/tables/", help="Output tables directory")
    parser.add_argument("--figures-dir", type=str, default="results/figures/", help="Output figures directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Base output directory override")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tables_dir = Path(parsed.output_dir or parsed.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path(parsed.report_file)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = Path(parsed.data_dir)
    scenarios = []
    if data_dir.exists():
        scen_files = sorted(list(data_dir.glob("*.npz")))
        scenarios = [load_scenario_from_npz(str(fp)) for fp in scen_files]

    if not scenarios:
        logger.warning("No scenarios found in %s; generating fast synthetic fixtures for benchmark evaluation.", data_dir)
        dt = 0.02
        n_samples = 250
        timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
        for i in range(12):
            pen = 0.15 + (i % 6) * 0.05
            f_drop = (0.40 + 0.40 * pen) * (1.0 - np.exp(-np.maximum(0.0, timestamps - 1.0) / 1.5))
            coi = 50.0 - f_drop
            bf = np.tile(coi[:, np.newaxis], (1, 39))
            rocof = np.gradient(bf, dt, axis=0)
            nadir_val = float(np.min(coi))
            scenarios.append({
                "timestamps": timestamps,
                "bus_frequencies": bf,
                "bus_rocof": rocof,
                "voltage_magnitudes": np.ones((n_samples, 39)),
                "voltage_angles": np.zeros((n_samples, 39)),
                "coi_frequency": coi,
                "metadata": {
                    "scenario_id": f"scen_eval_{i:02d}",
                    "t_disturb": 1.0,
                    "t_sim": 5.0,
                    "fps": 50,
                    "f_nominal": 50.0,
                    "f_threshold_hz": 49.50,
                    "renewable_penetration": pen,
                    "load_scale": 1.0,
                    "ground_truth_nadir": round(nadir_val, 4),
                    "ground_truth_t_nadir": 3.5,
                    "is_secure": 1 if nadir_val >= 49.50 else 0,
                    "partition": "ood_test" if pen >= 0.35 else "train",
                },
            })

    benchmarks_to_run = parsed.benchmarks
    if "all" in benchmarks_to_run:
        benchmarks_to_run = ["sensitivity", "ood", "noise", "latency"]

    report_sections = ["# PMU-FSA Evaluation & Robustness Benchmark Report\n"]

    if "sensitivity" in benchmarks_to_run:
        logger.info("Executing Window Sensitivity Benchmark...")
        sens_bench = WindowSensitivityBenchmark()
        sens_results = sens_bench.run(scenarios)
        df_sens = pd.DataFrame(sens_results["results"], columns=sens_results["columns"])
        df_sens.to_csv(tables_dir / "window_sensitivity.csv", index=False)
        with open(tables_dir / "window_sensitivity.md", "w", encoding="utf-8") as f:
            f.write(sens_bench.to_markdown(sens_results))
        report_sections.append("## 1. Window Sensitivity Benchmark ($T_{obs} \in [0.5, 2.0]$ s)\n")
        report_sections.append(sens_bench.to_markdown(sens_results))
        report_sections.append("\n")

    if "ood" in benchmarks_to_run:
        logger.info("Executing Out-of-Distribution Generalization Benchmark...")
        ood_bench = OODGeneralizationBenchmark()
        ood_results = ood_bench.run(scenarios)
        df_ood = ood_bench.to_dataframe(ood_results)
        df_ood.to_csv(tables_dir / "ood_generalization.csv", index=False)
        with open(tables_dir / "ood_generalization.md", "w", encoding="utf-8") as f:
            f.write(ood_bench.to_markdown(ood_results))
        report_sections.append("## 2. Out-of-Distribution Generalization Benchmark\n")
        report_sections.append(ood_bench.to_markdown(ood_results))
        report_sections.append("\n")

    if "noise" in benchmarks_to_run:
        logger.info("Executing PMU Measurement Noise Robustness Benchmark...")
        noise_bench = MeasurementNoiseBenchmark()
        noise_results = noise_bench.run(scenarios)
        df_noise = pd.DataFrame(noise_results["results"], columns=noise_results["columns"])
        df_noise.to_csv(tables_dir / "measurement_noise.csv", index=False)
        with open(tables_dir / "measurement_noise.md", "w", encoding="utf-8") as f:
            f.write(noise_bench.to_markdown(noise_results))
        report_sections.append("## 3. PMU Measurement Noise Stress Benchmark\n")
        report_sections.append(noise_bench.to_markdown(noise_results))
        report_sections.append("\n")

    if "latency" in benchmarks_to_run:
        logger.info("Executing Latency & Execution Speed Benchmark...")
        lat_bench = LatencyBenchmark()
        lat_results = lat_bench.profile()
        df_lat = pd.DataFrame([lat_results])
        df_lat.to_csv(tables_dir / "latency_profile.csv", index=False)
        with open(tables_dir / "latency_profile.md", "w", encoding="utf-8") as f:
            f.write(lat_bench.to_markdown(lat_results))
        report_sections.append("## 4. Latency & Inference Profiling Benchmark\n")
        report_sections.append(lat_bench.to_markdown(lat_results))
        report_sections.append("\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_sections))

    logger.info("Evaluation suite finished. Report generated at %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
