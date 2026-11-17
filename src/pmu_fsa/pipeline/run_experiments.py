#!/usr/bin/env python3
"""End-to-end leak-free runner for 50 Hz simulation, feature extraction, training, and benchmarks."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
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

from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.features.extractor import CANONICAL_FEATURE_NAMES, FeatureExtractor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.calibration_metrics import compute_calibration_metrics
from pmu_fsa.models.classifiers import compute_binary_security_metrics
from pmu_fsa.models.regressors import (
    AnalyticalSwingNadirRegressor,
    DecisionTreeNadirRegressor,
    GradientBoostingNadirRegressor,
    LightGBMNadirRegressor,
    LinearNadirRegressor,
    MeanNadirRegressor,
    RandomForestNadirRegressor,
)
from pmu_fsa.simulation.pmu_recorder import PMUNoiseConfig
from pmu_fsa.simulation.scenario_generator import (
    DEFAULT_LOAD_SCALES,
    DEFAULT_LOSS_FRACTIONS,
    DEFAULT_PENETRATIONS,
    ScenarioGenerator,
    export_raw_and_pmu_scenario,
    load_scenario_from_npz,
)

logger = logging.getLogger("pmu_fsa.pipeline")

WORKFLOW_STEPS = ["simulate", "extract_features", "train_models", "evaluate_all"]


def _parse_simple_yaml(path: Path) -> Dict[str, Any]:
    """Minimal dependency-free YAML parser for flat key-value and inline list configs."""
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()
            if not val:
                continue
            if val.lower() in ("true", "yes"):
                data[key] = True
            elif val.lower() in ("false", "no"):
                data[key] = False
            elif val.startswith("[") and val.endswith("]"):
                try:
                    data[key] = ast.literal_eval(val)
                except Exception:
                    data[key] = [x.strip() for x in val[1:-1].split(",") if x.strip()]
            else:
                try:
                    data[key] = int(val)
                except ValueError:
                    try:
                        data[key] = float(val)
                    except ValueError:
                        data[key] = val.strip("\"'")
    return data


@dataclass
class ExperimentConfig:
    """Configuration for a reproducible 50.0 Hz PMU-FSA experiment run."""

    seed: int = 42
    f_nominal: float = 50.0
    security_threshold_hz: float = 49.50
    macro_intertie: bool = False
    n_scenarios: int = 750
    fps: int = 50
    t_disturb: float = 1.0
    t_max: float = 10.0
    t_obs: float = 1.0
    strict_nadir: bool = True
    save_csv: bool = True
    gen_buses: Optional[List[int]] = None
    loss_fractions: List[float] = field(default_factory=lambda: list(DEFAULT_LOSS_FRACTIONS))
    penetrations: List[float] = field(default_factory=lambda: list(DEFAULT_PENETRATIONS))
    load_scales: List[float] = field(default_factory=lambda: list(DEFAULT_LOAD_SCALES))
    window_lengths: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    noise_levels: List[float] = field(default_factory=lambda: [0.0, 0.005, 0.01, 0.02, 0.05])
    raw_data_dir: str = "data/raw"
    raw_sim_dir: str = "data/raw_sim"
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
        """Load configuration from a YAML or JSON file."""
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
                data = _parse_simple_yaml(p)
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class ExperimentPipeline:
    """Executes simulation, 14-feature extraction, disjoint training, and honest benchmarks."""

    def __init__(self, config: Optional[ExperimentConfig] = None) -> None:
        self.config = config or ExperimentConfig()
        self._isolate_temp_workspace_paths()
        self.workflow_steps = list(WORKFLOW_STEPS)
        self.scenarios: List[Dict[str, Any]] = []
        self.features_by_window: Dict[float, Any] = {}
        self.trained_models: Dict[str, Any] = {}
        self.benchmark_reports: Dict[str, Any] = {}

        np.random.seed(self.config.seed)

    def _isolate_temp_workspace_paths(self) -> None:
        """Prevent callers using a temporary raw_data_dir from clobbering live workspace directories."""
        raw_p = Path(self.config.raw_data_dir).resolve()
        live_raw = (PROJECT_ROOT / "data" / "raw").resolve()
        if raw_p != live_raw and not str(raw_p).startswith(str((PROJECT_ROOT / "data").resolve())):
            base = raw_p.parent
            if self.config.raw_sim_dir == "data/raw_sim":
                self.config.raw_sim_dir = str(base / f"{raw_p.name}_sim")
            if self.config.processed_data_dir == "data/processed":
                self.config.processed_data_dir = str(base / "processed")
            if self.config.models_dir == "models":
                self.config.models_dir = str(base / "models")
            if self.config.output_dir in ("results", "results/"):
                self.config.output_dir = str(base / "results")
            if self.config.tables_dir in ("results/tables", "results/tables/"):
                self.config.tables_dir = str(base / "results" / "tables")
            if self.config.figures_dir in ("results/figures", "results/figures/"):
                self.config.figures_dir = str(base / "results" / "figures")
            if self.config.report_file == "results/evaluation_report.md":
                self.config.report_file = str(base / "results" / "evaluation_report.md")

    def run_simulate(self) -> List[Dict[str, Any]]:
        """Generate scenarios across the 50 Hz IEEE 39-bus grid or load existing raw .npz files."""
        raw_dir = Path(self.config.raw_data_dir)
        sim_dir = Path(self.config.raw_sim_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        sim_dir.mkdir(parents=True, exist_ok=True)

        existing_files = sorted(raw_dir.glob("*.npz"))
        if self.config.skip_simulation and existing_files:
            logger.info("Skipping simulation; loading %d scenarios from %s", len(existing_files), raw_dir)
            self.scenarios = [load_scenario_from_npz(str(fp)) for fp in existing_files]
            return self.scenarios

        inject_noise = self.config.noise_std > 0.0
        noise_cfg = PMUNoiseConfig(sigma_f=self.config.noise_std, seed=self.config.seed) if inject_noise else None

        logger.info("Step 1 [simulate]: Generating %d scenarios at %d fps...", self.config.n_scenarios, self.config.fps)
        gen = ScenarioGenerator(
            fps=self.config.fps,
            t_span=(0.0, self.config.t_max),
            t_disturb=self.config.t_disturb,
            security_threshold_hz=self.config.security_threshold_hz,
            macro_intertie=self.config.macro_intertie,
            f_nominal=self.config.f_nominal,
            noise_config=noise_cfg,
            seed=self.config.seed,
        )
        pairs = gen.generate_scenario_batch(
            gen_buses=self.config.gen_buses,
            load_scales=self.config.load_scales,
            penetrations=self.config.penetrations,
            loss_fractions=self.config.loss_fractions,
            max_scenarios=self.config.n_scenarios,
            inject_noise=inject_noise,
            noise_config=noise_cfg,
            seed=self.config.seed,
            return_raw=True,
        )

        self.scenarios = []
        for raw_traj, pmu_traj in pairs:
            export_raw_and_pmu_scenario(raw_traj, pmu_traj, sim_dir, raw_dir)
            self.scenarios.append(pmu_traj)

        logger.info("Saved %d PMU scenarios into %s", len(self.scenarios), raw_dir)
        return self.scenarios

    def run_extract_features(self) -> Dict[float, Any]:
        """Extract canonical 14 causal PMU features across configured observation windows."""
        if not self.scenarios:
            raw_dir = Path(self.config.raw_data_dir)
            if raw_dir.exists():
                self.scenarios = [load_scenario_from_npz(str(fp)) for fp in sorted(raw_dir.glob("*.npz"))]

        proc_dir = Path(self.config.processed_data_dir)
        proc_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Step 2 [extract_features]: Extracting 14 features across windows %s...", self.config.window_lengths)
        extractor = FeatureExtractor(f_nominal=self.config.f_nominal)

        for w in self.config.window_lengths:
            X, y_nadir, y_secure, meta_df = extractor.transform_batch(self.scenarios, t_obs=w, as_dataframe=True)
            X_df = X[CANONICAL_FEATURE_NAMES].copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=CANONICAL_FEATURE_NAMES)
            self.features_by_window[w] = (X_df, y_nadir, y_secure, meta_df)

            scen_ids = meta_df["scenario_id"].to_numpy(dtype=str) if isinstance(meta_df, pd.DataFrame) else np.asarray([m["scenario_id"] for m in meta_df], dtype=str)
            parts = meta_df["partition"].to_numpy(dtype=str) if isinstance(meta_df, pd.DataFrame) else np.asarray([m.get("partition", "train") for m in meta_df], dtype=str)

            payload: Dict[str, Any] = {
                "X": X_df.to_numpy(dtype=float),
                "y_nadir": np.asarray(y_nadir, dtype=float),
                "y_secure": np.asarray(y_secure, dtype=int),
                "feature_names": np.asarray(CANONICAL_FEATURE_NAMES, dtype=str),
                "scenario_ids": scen_ids,
                "partitions": parts,
            }
            if isinstance(meta_df, pd.DataFrame):
                for col in meta_df.columns:
                    payload[col] = meta_df[col].to_numpy()

            w_out = proc_dir / f"features_t{w:.1f}s.npz"
            np.savez_compressed(w_out, **payload)
            if self.config.save_csv and isinstance(meta_df, pd.DataFrame):
                pd.concat([X_df.reset_index(drop=True), meta_df.reset_index(drop=True)], axis=1).to_csv(
                    w_out.with_suffix(".csv"), index=False
                )

            if abs(w - self.config.t_obs) < 1e-4:
                np.savez_compressed(proc_dir / "features.npz", **payload)

        return self.features_by_window

    def run_train_models(self) -> Dict[str, Any]:
        """Train regressors and calibrated GBDT classifiers strictly on disjoint splits."""
        models_dir = Path(self.config.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)

        w_def = self.config.t_obs
        if w_def in self.features_by_window:
            X_df, y_nadir, y_secure, meta_df = self.features_by_window[w_def]
        elif self.features_by_window:
            X_df, y_nadir, y_secure, meta_df = next(iter(self.features_by_window.values()))
        else:
            extractor = FeatureExtractor(t_obs=w_def, f_nominal=self.config.f_nominal)
            X_df, y_nadir, y_secure, meta_df = extractor.transform_batch(self.scenarios, as_dataframe=True)

        X_mat = X_df.to_numpy(dtype=float) if hasattr(X_df, "to_numpy") else np.asarray(X_df, dtype=float)
        y_nadir = np.asarray(y_nadir, dtype=float)
        y_secure = np.asarray(y_secure, dtype=int)
        parts = (
            meta_df["partition"].to_numpy(dtype=str)
            if isinstance(meta_df, pd.DataFrame) and "partition" in meta_df.columns
            else np.array(["train"] * len(y_nadir))
        )

        tr_idx = np.where(parts == "train")[0]
        va_idx = np.where(parts == "val")[0]
        te_idx = np.where(parts == "test")[0]

        if len(tr_idx) == 0 or len(te_idx) == 0:
            non_ood = np.where(parts != "ood_test")[0]
            if len(non_ood) == 0:
                non_ood = np.arange(len(X_mat))
            rng = np.random.RandomState(self.config.seed)
            perm = non_ood.copy()
            rng.shuffle(perm)
            n_tr = max(1, int(len(perm) * 0.6))
            n_va = max(0, int(len(perm) * 0.2)) if len(perm) >= 5 else 0
            tr_idx = perm[:n_tr]
            va_idx = perm[n_tr : n_tr + n_va]
            te_idx = perm[n_tr + n_va :] if (n_tr + n_va) < len(perm) else perm[:1]

        X_tr, yn_tr, ys_tr = X_mat[tr_idx], y_nadir[tr_idx], y_secure[tr_idx]
        X_va = X_mat[va_idx] if len(va_idx) > 0 else None
        yn_va = y_nadir[va_idx] if len(va_idx) > 0 else None
        X_te, yn_te, ys_te = X_mat[te_idx], y_nadir[te_idx], y_secure[te_idx]

        X_fit = np.vstack([X_tr, X_va]) if (X_va is not None and len(X_va) > 0) else X_tr
        yn_fit = np.concatenate([yn_tr, yn_va]) if (yn_va is not None and len(yn_va) > 0) else yn_tr

        self.trained_models["mean"] = MeanNadirRegressor().fit(X_fit, yn_fit)
        self.trained_models["analytical"] = AnalyticalSwingNadirRegressor().fit(X_fit, yn_fit)
        self.trained_models["linear"] = LinearNadirRegressor().fit(X_fit, yn_fit)
        self.trained_models["dt"] = DecisionTreeNadirRegressor(max_depth=5, random_state=self.config.seed).fit(X_fit, yn_fit)
        self.trained_models["rf"] = RandomForestNadirRegressor(n_estimators=100, random_state=self.config.seed).fit(X_fit, yn_fit)
        self.trained_models["gbdt"] = GradientBoostingNadirRegressor(n_estimators=100, random_state=self.config.seed).fit(X_fit, yn_fit)
        self.trained_models["lgbm"] = LightGBMNadirRegressor(n_estimators=100, random_state=self.config.seed).fit(X_fit, yn_fit)

        platt_clf = CalibratedGBDTClassifier(
            method="platt",
            threshold=self.config.security_threshold_hz,
            random_state=self.config.seed,
            fit_full_after_cal=(X_va is not None and len(X_va) >= 2),
        )
        iso_clf = CalibratedGBDTClassifier(
            method="isotonic",
            threshold=self.config.security_threshold_hz,
            random_state=self.config.seed,
            fit_full_after_cal=(X_va is not None and len(X_va) >= 2),
        )
        if X_va is not None and len(X_va) >= 2:
            platt_clf.fit(X_tr, yn_tr, ys_tr, X_cal=X_va, y_cal=yn_va)
            iso_clf.fit(X_tr, yn_tr, ys_tr, X_cal=X_va, y_cal=yn_va)
        else:
            platt_clf.fit(X_tr, yn_tr, ys_tr)
            iso_clf.fit(X_tr, yn_tr, ys_tr)

        self.trained_models["calibrated_gb_platt"] = platt_clf
        self.trained_models["calibrated_gb_iso"] = iso_clf
        self.trained_models["calibrated_gb"] = platt_clf

        for name, model_obj in self.trained_models.items():
            joblib.dump(model_obj, models_dir / f"{name}.joblib")

        summary_metrics: Dict[str, Any] = {}
        for name, m in self.trained_models.items():
            if "calibrated" not in name:
                ev = m.evaluate(X_te, yn_te)
                summary_metrics[name] = {k: round(float(v), 4) for k, v in ev.items()}
            else:
                probs = m.predict_proba(X_te)[:, 1]
                preds_sec = m.predict(X_te)
                cls_m = compute_binary_security_metrics(ys_te, preds_sec, p_secure=probs)
                cal_m = compute_calibration_metrics(ys_te, probs, n_bins=10, threshold=self.config.security_threshold_hz)
                summary_metrics[name] = {**cal_m, **{k: round(float(v), 4) if isinstance(v, float) else v for k, v in cls_m.items()}}

        with open(models_dir / "training_metrics.json", "w", encoding="utf-8") as f:
            json.dump(summary_metrics, f, indent=2)

        return self.trained_models

    def run_evaluate_all(self) -> Dict[str, Any]:
        """Run leak-free evaluation benchmarks and export tables and report."""
        tables_dir = Path(self.config.tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        report_file = Path(self.config.report_file)
        report_file.parent.mkdir(parents=True, exist_ok=True)

        reg = self.trained_models.get("gbdt")
        clf = self.trained_models.get("calibrated_gb_platt") or self.trained_models.get("calibrated_gb")

        sens_bench = WindowSensitivityBenchmark(
            window_lengths=self.config.window_lengths,
            threshold=self.config.security_threshold_hz,
            random_state=self.config.seed,
        )
        sens_res = sens_bench.run(self.scenarios)
        df_sens = sens_bench.to_dataframe(sens_res) if hasattr(sens_bench, "to_dataframe") else pd.DataFrame(sens_res["results"], columns=sens_res["columns"])
        df_sens.to_csv(tables_dir / "window_sensitivity.csv", index=False)
        (tables_dir / "window_sensitivity.md").write_text(sens_bench.to_markdown(sens_res), encoding="utf-8")

        ood_bench = OODGeneralizationBenchmark(
            t_obs=self.config.t_obs,
            threshold=self.config.security_threshold_hz,
            random_state=self.config.seed,
        )
        ood_res = ood_bench.run(self.scenarios, regressor=reg, classifier=clf)
        df_ood = ood_bench.to_dataframe(ood_res)
        df_ood.to_csv(tables_dir / "ood_generalization.csv", index=False)
        (tables_dir / "ood_generalization.md").write_text(ood_bench.to_markdown(ood_res), encoding="utf-8")

        noise_bench = MeasurementNoiseBenchmark(
            noise_levels=self.config.noise_levels,
            t_obs=self.config.t_obs,
            threshold=self.config.security_threshold_hz,
            seed=self.config.seed,
        )
        noise_res = noise_bench.run(self.scenarios, regressor=reg, classifier=clf)
        df_noise = noise_bench.to_dataframe(noise_res) if hasattr(noise_bench, "to_dataframe") else pd.DataFrame(noise_res["results"], columns=noise_res["columns"])
        df_noise.to_csv(tables_dir / "measurement_noise.csv", index=False)
        (tables_dir / "measurement_noise.md").write_text(noise_bench.to_markdown(noise_res), encoding="utf-8")

        lat_bench = LatencyBenchmark()
        lat_res = lat_bench.profile(
            regressor=reg,
            classifier=clf,
            scenarios=self.scenarios,
            t_obs=self.config.t_obs,
            n_reps=50,
            warmup_reps=5,
            measure_simulation=False,
        )
        flat_lat = {k: v for k, v in lat_res.items() if not isinstance(v, (dict, list))}
        pd.DataFrame([flat_lat]).to_csv(tables_dir / "latency_profile.csv", index=False)
        (tables_dir / "latency_profile.md").write_text(lat_bench.to_markdown(lat_res), encoding="utf-8")

        self.benchmark_reports = {
            "sensitivity": sens_res,
            "ood": ood_res,
            "noise": noise_res,
            "latency": lat_res,
        }

        report_lines = [
            "# PMU-Based Early Frequency-Security Assessment: Benchmark Evaluation Report\n",
            r"## 1. Window Sensitivity Benchmark ($T_{obs} \in [0.5, 2.0]$ s)" + "\n",
            sens_bench.to_markdown(sens_res),
            "\n## 2. Out-of-Distribution Generalization Benchmark\n",
            ood_bench.to_markdown(ood_res),
            "\n## 3. PMU Measurement Noise Stress Benchmark\n",
            noise_bench.to_markdown(noise_res),
            "\n## 4. Online Computational Latency & Simulation Speedup\n",
            lat_bench.to_markdown(lat_res),
        ]
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
        return self.benchmark_reports

    def execute_full_pipeline(self) -> Dict[str, Any]:
        """Run all enabled pipeline stages in order."""
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
        description="PMU-FSA Reproducible 50 Hz Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/default_50hz.yaml", help="Path to YAML/JSON config file")
    parser.add_argument("--t-obs", type=float, default=None, help="Observation window length override (s)")
    parser.add_argument("--output-dir", type=str, default=None, help="Results directory override")
    parser.add_argument("--noise-std", type=float, default=None, help="PMU noise std override (Hz)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--n-scenarios", type=int, default=None, help="Scenario count override")
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
    if parsed.t_obs is not None:
        config.t_obs = float(parsed.t_obs)
    if parsed.output_dir is not None:
        config.output_dir = str(parsed.output_dir)
        config.tables_dir = str(Path(parsed.output_dir) / "tables")
        config.figures_dir = str(Path(parsed.output_dir) / "figures")
        config.report_file = str(Path(parsed.output_dir) / "evaluation_report.md")
    if parsed.noise_std is not None:
        config.noise_std = float(parsed.noise_std)
    if parsed.seed is not None:
        config.seed = int(parsed.seed)
    if parsed.n_scenarios is not None:
        config.n_scenarios = int(parsed.n_scenarios)
    if parsed.skip_simulation:
        config.skip_simulation = True
    if parsed.skip_training:
        config.skip_training = True
    if parsed.skip_benchmarks:
        config.skip_benchmarks = True
    config.headless = bool(parsed.headless)

    pipeline = ExperimentPipeline(config=config)
    pipeline.execute_full_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
