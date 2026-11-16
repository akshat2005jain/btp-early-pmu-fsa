#!/usr/bin/env python3
"""End-to-end runner for simulation, feature extraction, training, and benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ["MPLBACKEND"] = "Agg"

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    import pickle as joblib
    HAS_JOBLIB = False

from pmu_fsa.simulation.scenario_generator import (
    ScenarioGenerator,
    export_scenario_to_npz,
    load_scenario_from_npz,
)
from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.regressors import (
    LinearNadirRegressor,
    DecisionTreeNadirRegressor,
    RandomForestNadirRegressor,
    GradientBoostingNadirRegressor,
    get_nadir_regressor,
)
from pmu_fsa.models.classifiers import (
    LogisticSecurityClassifier,
    get_security_classifier,
)
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import compute_calibration_metrics
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.latency import LatencyBenchmark

logger = logging.getLogger("pmu_fsa.pipeline")

WORKFLOW_STEPS = ["simulate", "extract_features", "train_models", "evaluate_all"]


@dataclass
class ExperimentConfig:
    """Configuration for an end-to-end experiment run."""
    seed: int = 42
    n_scenarios: int = 30
    fps: int = 50
    t_obs: float = 1.0
    window_lengths: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.005, 0.01, 0.02, 0.05])
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    models_dir: str = "models"
    output_dir: str = "results"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"
    report_file: str = "results/evaluation_report.md"
    noise_std: float = 0.0
    skip_simulation: bool = False
    skip_training: bool = False
    skip_benchmarks: bool = False
    headless: bool = True

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from a JSON or YAML file."""
        p = Path(path)
        if not p.exists():
            return cls()

        data: Dict[str, Any] = {}
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except ImportError:
                # Simple key-value fallback when PyYAML isn't installed
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith("#") and ":" in line:
                                k, v = line.split(":", 1)
                                k, v = k.strip(), v.strip()
                                if v.lower() == "true":
                                    data[k] = True
                                elif v.lower() == "false":
                                    data[k] = False
                                else:
                                    try:
                                        data[k] = int(v) if v.isdigit() else float(v)
                                    except ValueError:
                                        data[k] = v.strip("\"'")
                except Exception as e:
                    logger.warning("Could not parse YAML file: %s; using defaults", e)
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class ExperimentPipeline:
    """Runs simulation, feature extraction, model training, and evaluation benchmarks."""

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        self.config = config or ExperimentConfig()
        self.workflow_steps = list(WORKFLOW_STEPS)
        self.scenarios: List[Dict[str, Any]] = []
        self.features_by_window: Dict[float, Any] = {}
        self.trained_models: Dict[str, Any] = {}
        self.benchmark_reports: Dict[str, Any] = {}

        np.random.seed(self.config.seed)

    def run_simulate(self) -> List[Dict[str, Any]]:
        """Generate scenarios or load existing raw .npz files."""
        raw_dir = Path(self.config.raw_data_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        existing_files = sorted(list(raw_dir.glob("*.npz")))
        if self.config.skip_simulation and existing_files:
            logger.info("Skipping simulation; loading %d existing scenarios from %s", len(existing_files), raw_dir)
            self.scenarios = [load_scenario_from_npz(str(fp)) for fp in existing_files]
            return self.scenarios

        logger.info("Step 1 [simulate]: Generating %d scenarios at %d fps...", self.config.n_scenarios, self.config.fps)
        gen = ScenarioGenerator(fps=self.config.fps)
        self.scenarios = gen.generate_scenario_batch(
            max_scenarios=self.config.n_scenarios,
            inject_noise=(self.config.noise_std > 0.0),
        )

        for s in self.scenarios:
            scen_id = s["metadata"]["scenario_id"]
            export_scenario_to_npz(s, str(raw_dir / f"{scen_id}.npz"))
        logger.info("Successfully generated and saved %d scenarios into %s", len(self.scenarios), raw_dir)
        return self.scenarios

    def run_extract_features(self) -> Dict[float, Any]:
        """Extract causal PMU features across configured observation windows."""
        proc_dir = Path(self.config.processed_data_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Step 2 [extract_features]: Extracting causal features across windows %s...", self.config.window_lengths)
        extractor = FeatureExtractor()

        for w in self.config.window_lengths:
            X, y_nadir, y_secure, meta_df = extractor.transform_batch(self.scenarios, t_obs=w, as_dataframe=True)
            self.features_by_window[w] = (X, y_nadir, y_secure, meta_df)

            w_out = proc_dir / f"features_t{w:.1f}s.npz"
            np.savez_compressed(
                w_out,
                X=X.values if hasattr(X, "values") else X,
                y_nadir=y_nadir,
                y_secure=y_secure,
                scenario_ids=meta_df["scenario_id"].values if hasattr(meta_df, "columns") else [m["scenario_id"] for m in meta_df],
                partitions=meta_df["partition"].values if hasattr(meta_df, "columns") else [m.get("partition", "train") for m in meta_df],
            )

            if abs(w - self.config.t_obs) < 1e-4:
                np.savez_compressed(
                    proc_dir / "features.npz",
                    X=X.values if hasattr(X, "values") else X,
                    y_nadir=y_nadir,
                    y_secure=y_secure,
                    scenario_ids=meta_df["scenario_id"].values if hasattr(meta_df, "columns") else [m["scenario_id"] for m in meta_df],
                    partitions=meta_df["partition"].values if hasattr(meta_df, "columns") else [m.get("partition", "train") for m in meta_df],
                )

        logger.info("Extracted features across %d windows into %s", len(self.features_by_window), proc_dir)
        return self.features_by_window

    def run_train_models(self) -> Dict[str, Any]:
        """Train baseline regressors and calibrated GBDT classifiers."""
        models_dir = Path(self.config.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Step 3 [train_models]: Training regressors, classifiers, and calibrated GBDT...")
        w_def = self.config.t_obs
        if w_def in self.features_by_window:
            X_df, y_nadir, y_secure, meta_df = self.features_by_window[w_def]
        elif self.features_by_window:
            X_df, y_nadir, y_secure, meta_df = list(self.features_by_window.values())[0]
        else:
            extractor = FeatureExtractor(t_obs=w_def)
            X_df, y_nadir, y_secure, meta_df = extractor.transform_batch(self.scenarios, as_dataframe=True)

        X = X_df.values if hasattr(X_df, "values") else np.asarray(X_df)
        y_nadir = np.asarray(y_nadir, dtype=float)
        y_secure = np.asarray(y_secure, dtype=int)

        self.trained_models["linear"] = LinearNadirRegressor().fit(X, y_nadir)
        self.trained_models["dt"] = DecisionTreeNadirRegressor(max_depth=5, random_state=self.config.seed).fit(X, y_nadir)
        self.trained_models["rf"] = RandomForestNadirRegressor(n_estimators=50, random_state=self.config.seed).fit(X, y_nadir)
        self.trained_models["gbdt"] = GradientBoostingNadirRegressor(n_estimators=50, random_state=self.config.seed).fit(X, y_nadir)

        self.trained_models["calibrated_gb_platt"] = CalibratedGBDTClassifier(
            method="platt", random_state=self.config.seed
        ).fit(X, y_nadir, y_secure)
        self.trained_models["calibrated_gb_iso"] = CalibratedGBDTClassifier(
            method="isotonic", random_state=self.config.seed
        ).fit(X, y_nadir, y_secure)

        for name, model_obj in self.trained_models.items():
            try:
                joblib.dump(model_obj, models_dir / f"{name}.joblib")
            except Exception as e:
                logger.warning("Could not serialize model %s: %s", name, e)

        summary_metrics = {}
        for name, m in self.trained_models.items():
            if hasattr(m, "predict"):
                preds = np.asarray(m.predict(X))
                if "calibrated" not in name:
                    mae = float(np.mean(np.abs(preds - y_nadir)))
                    summary_metrics[name] = {"mae": round(mae, 4)}
                else:
                    probs = m.predict_proba(X)
                    p_sec = probs[:, 1] if hasattr(probs, "ndim") and probs.ndim > 1 else probs
                    cal = compute_calibration_metrics(y_secure, p_sec)
                    summary_metrics[name] = cal

        with open(models_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        logger.info("Model training completed: %d models trained and saved to %s", len(self.trained_models), models_dir)
        return self.trained_models

    def run_evaluate_all(self) -> Dict[str, Any]:
        """Run all evaluation benchmarks and export tables, plots, and markdown report."""
        tables_dir = Path(self.config.tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        report_file = Path(self.config.report_file)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Step 4 [evaluate_all]: Executing evaluation benchmark suite...")

        sens_bench = WindowSensitivityBenchmark(window_lengths=self.config.window_lengths)
        sens_res = sens_bench.run(self.scenarios)

        ood_bench = OODGeneralizationBenchmark()
        ood_res = ood_bench.run(self.scenarios)

        noise_bench = MeasurementNoiseBenchmark(noise_levels=self.config.noise_levels)
        noise_res = noise_bench.run(self.scenarios)

        lat_bench = LatencyBenchmark()
        lat_res = lat_bench.profile()

        self.benchmark_reports = {
            "sensitivity": sens_res,
            "ood": ood_res,
            "noise": noise_res,
            "latency": lat_res,
        }

        df_sens = pd.DataFrame(sens_res["results"], columns=sens_res["columns"])
        df_sens.to_csv(tables_dir / "window_sensitivity.csv", index=False)
        with open(tables_dir / "window_sensitivity.md", "w", encoding="utf-8") as f:
            f.write(sens_bench.to_markdown(sens_res))

        df_ood = ood_bench.to_dataframe(ood_res)
        df_ood.to_csv(tables_dir / "ood_generalization.csv", index=False)
        with open(tables_dir / "ood_generalization.md", "w", encoding="utf-8") as f:
            f.write(ood_bench.to_markdown(ood_res))

        df_noise = pd.DataFrame(noise_res["results"], columns=noise_res["columns"])
        df_noise.to_csv(tables_dir / "measurement_noise.csv", index=False)
        with open(tables_dir / "measurement_noise.md", "w", encoding="utf-8") as f:
            f.write(noise_bench.to_markdown(noise_res))

        df_lat = pd.DataFrame([lat_res])
        df_lat.to_csv(tables_dir / "latency_profile.csv", index=False)
        with open(tables_dir / "latency_profile.md", "w", encoding="utf-8") as f:
            f.write(lat_bench.to_markdown(lat_res))

        report_lines = [
            "# PMU-Based Early Frequency-Security Assessment: Benchmark Evaluation Report\n",
            r"## 1. Window Sensitivity Benchmark ($T_{obs} \in [0.5, 2.0]$ s)" + "\n",
            sens_bench.to_markdown(sens_res),
            "\n## 2. Out-of-Distribution Generalization Benchmark\n",
            ood_bench.to_markdown(ood_res),
            "\n## 3. PMU Measurement Noise Stress Benchmark (IEEE C37.118 Class P/M)\n",
            noise_bench.to_markdown(noise_res),
            "\n## 4. Online Computational Latency & Simulation Speedup\n",
            lat_bench.to_markdown(lat_res),
            f"\n- **Assessment Latency Compliant (< 10 ms)**: {lat_res.get('latency_compliant_under_10ms', True)}",
            f"- **Speedup over Dynamic ODE Simulation**: {lat_res.get('speedup_factor', 0.0)}x\n",
        ]

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        if self.config.headless:
            figures_dir = Path(self.config.figures_dir)
            figures_dir.mkdir(parents=True, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                if not df_sens.empty and "window_obs" in df_sens.columns and "mae" in df_sens.columns:
                    ax1.plot(df_sens["window_obs"], df_sens["mae"], marker="o", color="blue", label="MAE (Hz)")
                    ax1.set_xlabel("Observation Window T_obs (s)")
                    ax1.set_ylabel("MAE (Hz)")
                    ax1.set_title("Nadir Error vs Observation Window")
                    ax1.grid(True)
                    ax1.legend()

                if not df_noise.empty and "noise_std_hz" in df_noise.columns and "mae" in df_noise.columns:
                    ax2.plot(df_noise["noise_std_hz"], df_noise["mae"], marker="s", color="crimson", label="MAE (Hz)")
                    ax2.set_xlabel("Noise Std Dev sigma_f (Hz)")
                    ax2.set_ylabel("MAE (Hz)")
                    ax2.set_title("Degradation Under PMU Noise")
                    ax2.grid(True)
                    ax2.legend()

                fig.tight_layout()
                fig.savefig(figures_dir / "window_sensitivity_curve.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                logger.warning("Could not generate headless benchmark figures: %s", e)

        logger.info("Evaluation benchmark reporting completed: %s", report_file)
        return self.benchmark_reports

    def execute_full_pipeline(self) -> Dict[str, Any]:
        """Run all enabled pipeline stages in order."""
        logger.info("Starting end-to-end PMU-FSA experiment pipeline...")
        self.run_simulate()
        self.run_extract_features()
        if not self.config.skip_training:
            self.run_train_models()
        if not self.config.skip_benchmarks:
            self.run_evaluate_all()
        return {
            "n_scenarios": len(self.scenarios),
            "features_windows": list(self.features_by_window.keys()),
            "models": list(self.trained_models.keys()),
            "benchmarks": list(self.benchmark_reports.keys()),
        }


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="PMU-FSA Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--t-obs", type=float, default=1.0, help="Observation window length in seconds")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to save outputs")
    parser.add_argument("--noise-std", type=float, default=0.0, help="PMU noise std in Hz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-scenarios", type=int, default=30, help="Number of scenarios to simulate")
    parser.add_argument("--skip-simulation", action="store_true", help="Skip simulation if raw data exists")
    parser.add_argument("--skip-training", action="store_true", help="Skip training step")
    parser.add_argument("--skip-benchmarks", action="store_true", help="Skip benchmark evaluation")
    parser.add_argument("--headless", action="store_true", default=True, help="Force headless plotting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose debug output")
    return parser.parse_args(args)


def run_experiments(config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    pipeline = ExperimentPipeline(config=config)
    return pipeline.execute_full_pipeline()


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = ExperimentConfig.from_file(parsed.config)
    config.t_obs = float(parsed.t_obs)
    config.output_dir = str(parsed.output_dir)
    config.tables_dir = str(Path(parsed.output_dir) / "tables")
    config.figures_dir = str(Path(parsed.output_dir) / "figures")
    config.report_file = str(Path(parsed.output_dir) / "evaluation_report.md")
    config.noise_std = float(parsed.noise_std)
    config.seed = int(parsed.seed)
    config.n_scenarios = int(parsed.n_scenarios)
    config.skip_simulation = bool(parsed.skip_simulation)
    config.skip_training = bool(parsed.skip_training)
    config.skip_benchmarks = bool(parsed.skip_benchmarks)
    config.headless = bool(parsed.headless)

    pipeline = ExperimentPipeline(config=config)
    pipeline.execute_full_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
