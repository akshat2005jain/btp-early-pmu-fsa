"""Hardware-measured 6-stage online latency profiler and real ODE simulation benchmark."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy
import sklearn

from pmu_fsa.features.extractor import FeatureExtractor, extract_features
from pmu_fsa.features.window_slicer import extract_causal_window
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.regressors import GradientBoostingNadirRegressor


def _summarize_timings_ms(samples_ms: List[float]) -> Dict[str, float]:
    if not samples_ms:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "min_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min_ms": float(np.min(arr)),
        "p95_ms": float(np.percentile(arr, 95.0)),
        "p99_ms": float(np.percentile(arr, 99.0)),
        "max_ms": float(np.max(arr)),
    }


def collect_hardware_software_info() -> Dict[str, Any]:
    """Collect CPU, OS, and scientific Python library versions for benchmark provenance."""
    cpu_brand = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=1.0,
            ).strip()
            if out:
                cpu_brand = out
        except Exception:
            pass

    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cpu": cpu_brand,
        "cpu_count": os.cpu_count() or 1,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
    }


class LatencyBenchmark:
    """Profiles 6-stage online prediction latency against real DynamicSimulator ODE execution."""

    def __init__(
        self,
        budget_total_ms: float = 10.0,
        budget_feat_ms: float = 5.0,
        budget_infer_ms: float = 2.0,
        sim_time_ms: Optional[float] = None,
    ) -> None:
        self.budget_total_ms = float(budget_total_ms)
        self.budget_feat_ms = float(budget_feat_ms)
        self.budget_infer_ms = float(budget_infer_ms)
        # Keep None unless caller explicitly injects a fixed reference (e.g. unit test)
        self.sim_time_ms = float(sim_time_ms) if sim_time_ms is not None else None

    def _generate_synthetic_sample(self, n_samples: int = 501, fps: int = 50) -> Dict[str, Any]:
        dt = 1.0 / float(fps)
        timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
        f_base = 50.0
        t_post = np.maximum(0.0, timestamps - 1.0)
        drop = 0.55 * (1.0 - np.exp(-t_post / 1.2)) * np.exp(-np.maximum(0.0, t_post - 2.5) / 5.0)
        f_sys = f_base - drop
        rng = np.random.default_rng(42)
        bus_freqs = f_sys[:, np.newaxis] + rng.normal(0.0, 0.005, size=(n_samples, 39))
        bus_rocof = np.gradient(bus_freqs, dt, axis=0)
        v_mags = np.ones((n_samples, 39), dtype=float)
        v_angles = rng.normal(0.0, 0.5, size=(n_samples, 39))

        return {
            "timestamps": timestamps,
            "bus_frequencies": bus_freqs,
            "bus_rocof": bus_rocof,
            "voltage_magnitudes": v_mags,
            "voltage_angles": v_angles,
            "pmu_sys_frequency": f_sys,
            "coi_frequency": f_sys,
            "metadata": {
                "scenario_id": "latency_bench_sample",
                "t_disturb": 1.0,
                "fps": fps,
                "f_nominal": 50.0,
                "ground_truth_nadir": float(np.min(f_sys)),
                "ground_truth_t_nadir": 3.5,
                "renewable_penetration": 0.20,
                "load_scale": 1.0,
                "disturbed_gen": 35,
                "loss_fraction": 1.0,
                "active_loss_mw": 650.0,
                "is_secure": 0,
            },
        }

    def measure_ode_simulation(
        self,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        n_sim_runs: int = 6,
        warmup_runs: int = 1,
    ) -> Dict[str, Any]:
        """Measure actual DynamicSimulator.simulate() runtime across representative contingencies."""
        from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
        from pmu_fsa.simulation.ieee39_data import GENERATOR_BUSES
        from pmu_fsa.simulation.renewables import RenewablePenetrationConfig

        contingencies: List[Dict[str, Any]] = []
        if scenarios:
            test_pool = [s for s in scenarios if s.get("metadata", {}).get("partition") == "test"]
            pool = test_pool if test_pool else list(scenarios)
            step = max(1, len(pool) // max(1, n_sim_runs))
            for s in pool[::step][:n_sim_runs]:
                m = s.get("metadata", {})
                gen_bus = int(m.get("disturbed_gen", 35))
                if gen_bus not in GENERATOR_BUSES:
                    gen_bus = 35
                contingencies.append({
                    "tripped_bus": gen_bus,
                    "p_loss_mw": float(m.get("active_loss_mw", 500.0)),
                    "loss_fraction": m.get("loss_fraction", None),
                    "p_ren": float(m.get("renewable_penetration", m.get("p_ren_requested", 0.20))),
                    "load_scale": float(m.get("load_scale", 1.0)),
                    "t_disturb": float(m.get("t_disturb", 1.0)),
                    "t_sim": float(m.get("t_sim", 10.0)),
                    "fps": int(m.get("fps", 50)),
                })

        if not contingencies:
            contingencies = [
                {"tripped_bus": 35, "p_loss_mw": 650.0, "loss_fraction": 1.0, "p_ren": 0.20, "load_scale": 1.0, "t_disturb": 1.0, "t_sim": 10.0, "fps": 50},
                {"tripped_bus": 38, "p_loss_mw": 581.0, "loss_fraction": 0.70, "p_ren": 0.30, "load_scale": 1.1, "t_disturb": 1.0, "t_sim": 10.0, "fps": 50},
                {"tripped_bus": 32, "p_loss_mw": 650.0, "loss_fraction": 1.0, "p_ren": 0.10, "load_scale": 0.9, "t_disturb": 1.0, "t_sim": 10.0, "fps": 50},
            ][: max(1, n_sim_runs)]

        # Warmup ODE integration
        for w_idx in range(max(0, warmup_runs)):
            c = contingencies[w_idx % len(contingencies)]
            sim = DynamicSimulator(
                load_scaling=c["load_scale"],
                renewable_config=RenewablePenetrationConfig(p_ren=c["p_ren"]),
                macro_intertie=False,
            )
            sim.simulate(
                tripped_bus=c["tripped_bus"],
                p_loss_mw=c["p_loss_mw"],
                t_disturb=c["t_disturb"],
                t_span=(0.0, max(5.0, c["t_sim"])),
                fps=c["fps"],
                loss_fraction=c["loss_fraction"],
            )

        sim_times_ms: List[float] = []
        for idx in range(max(1, n_sim_runs)):
            c = contingencies[idx % len(contingencies)]
            t0 = time.perf_counter_ns()
            sim = DynamicSimulator(
                load_scaling=c["load_scale"],
                renewable_config=RenewablePenetrationConfig(p_ren=c["p_ren"]),
                macro_intertie=False,
            )
            sim.simulate(
                tripped_bus=c["tripped_bus"],
                p_loss_mw=c["p_loss_mw"],
                t_disturb=c["t_disturb"],
                t_span=(0.0, max(5.0, c["t_sim"])),
                fps=c["fps"],
                loss_fraction=c["loss_fraction"],
            )
            t1 = time.perf_counter_ns()
            sim_times_ms.append((t1 - t0) / 1e6)

        stats = _summarize_timings_ms(sim_times_ms)
        stats["n_sim_runs"] = len(sim_times_ms)
        stats["measured_on_hardware"] = True
        stats["raw_times_ms"] = [round(t, 4) for t in sim_times_ms]
        return stats

    def profile(
        self,
        extractor_fn: Optional[Callable] = None,
        model: Optional[Any] = None,
        regressor: Optional[Any] = None,
        classifier: Optional[Any] = None,
        scenarios: Optional[List[Dict[str, Any]]] = None,
        sample_data: Optional[Any] = None,
        t_obs: float = 1.0,
        n_iter: Optional[int] = None,
        n_reps: int = 100,
        warmup_reps: int = 10,
        sim_time_ms: Optional[float] = None,
        measure_simulation: Optional[bool] = None,
        n_sim_runs: int = 6,
    ) -> Dict[str, Any]:
        """Profile all 6 online assessment stages (>=100 reps, 10 warmups) and ODE simulation."""
        eff_reps = max(100, int(n_iter)) if (scenarios is not None and n_iter is not None) else (
            max(1, int(n_iter)) if n_iter is not None else max(100, int(n_reps))
        )
        eff_warmup = max(1, int(warmup_reps))

        # Select evaluation scenario pool (strictly held-out test when available)
        is_raw_feature_vector = (
            sample_data is not None
            and not (isinstance(sample_data, dict) and "timestamps" in sample_data)
        )

        if scenarios:
            test_scens = [s for s in scenarios if s.get("metadata", {}).get("partition") == "test"]
            scen_pool = test_scens if test_scens else list(scenarios)
        elif sample_data is not None and isinstance(sample_data, dict) and "timestamps" in sample_data:
            scen_pool = [sample_data]
        else:
            scen_pool = [self._generate_synthetic_sample()]

        fe = FeatureExtractor(t_obs=t_obs, f_nominal=50.0)

        # Determine feature dimensionality from actual extraction
        if is_raw_feature_vector:
            arr_in = np.asarray(sample_data, dtype=float)
            infer_probe = arr_in[:1] if arr_in.ndim > 1 else arr_in.reshape(1, -1)
        else:
            cw0 = extract_causal_window(scen_pool[0], t_obs=t_obs)
            if extractor_fn is not None:
                try:
                    d0 = extractor_fn(scen_pool[0], t_obs)
                except TypeError:
                    d0 = extractor_fn(scen_pool[0])
            else:
                d0 = extract_features(cw0, t_obs=None, f_nominal=50.0)
            infer_probe = np.array([list(d0.values())], dtype=float)

        n_feat = infer_probe.shape[1]

        # Resolve regression & calibrated classification models
        reg_model = regressor if regressor is not None else (
            model if (model is not None and not hasattr(model, "predict_proba")) else None
        )
        clf_model = classifier if classifier is not None else (
            model if (model is not None and hasattr(model, "predict_proba")) else None
        )

        if reg_model is None or getattr(reg_model, "n_features_in_", n_feat) != n_feat:
            rng = np.random.RandomState(42)
            X_syn = rng.randn(60, n_feat)
            y_syn = 49.55 - 0.35 * X_syn[:, 0] + 0.05 * rng.randn(60)
            reg_model = GradientBoostingNadirRegressor(n_estimators=100, max_depth=4, random_state=42)
            reg_model.fit(X_syn, y_syn)

        if clf_model is None or getattr(clf_model, "n_features_in_", n_feat) != n_feat:
            rng = np.random.RandomState(42)
            X_syn = rng.randn(60, n_feat)
            y_syn = 49.55 - 0.35 * X_syn[:, 0] + 0.05 * rng.randn(60)
            clf_model = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=100, random_state=42)
            clf_model.fit(X_syn, y_syn)

        def _run_single_stage_sequence(scen: Dict[str, Any]) -> Tuple[float, float, float, float, float, float]:
            # Stage 1: Window slicing
            t0 = time.perf_counter_ns()
            cw = extract_causal_window(scen, t_obs=t_obs)
            t1 = time.perf_counter_ns()

            # Stage 2: 14-Feature extraction
            if extractor_fn is not None:
                try:
                    f_dict = extractor_fn(cw, None)
                except Exception:
                    try:
                        f_dict = extractor_fn(scen, t_obs)
                    except TypeError:
                        f_dict = extractor_fn(scen)
            else:
                f_dict = extract_features(cw, t_obs=None, f_nominal=50.0)
            x_vec = np.fromiter(f_dict.values(), dtype=np.float64, count=len(f_dict)).reshape(1, -1)
            t2 = time.perf_counter_ns()

            if is_raw_feature_vector:
                x_vec = infer_probe

            # Stage 3: Regression inference
            _ = reg_model.predict(x_vec)
            t3 = time.perf_counter_ns()

            # Stage 4: Classification raw score inference
            if isinstance(clf_model, CalibratedGBDTClassifier):
                raw_s = clf_model.predict_nadir(x_vec)
                t4 = time.perf_counter_ns()
                # Stage 5: Probability calibration + decision
                probs = clf_model.calibrate_scores(raw_s)
                _ = int(probs[0, 1] >= 0.5)
                t5 = time.perf_counter_ns()
            else:
                probs = clf_model.predict_proba(x_vec)
                t4 = time.perf_counter_ns()
                _ = int(probs[0, 1] >= 0.5)
                t5 = time.perf_counter_ns()

            # Stage 6: Total end-to-end online execution in one uninterrupted pass
            te0 = time.perf_counter_ns()
            if not is_raw_feature_vector:
                cw_e2e = extract_causal_window(scen, t_obs=t_obs)
                if extractor_fn is not None:
                    try:
                        fd_e2e = extractor_fn(cw_e2e, None)
                    except Exception:
                        try:
                            fd_e2e = extractor_fn(scen, t_obs)
                        except TypeError:
                            fd_e2e = extractor_fn(scen)
                else:
                    fd_e2e = extract_features(cw_e2e, t_obs=None, f_nominal=50.0)
                x_e2e = np.fromiter(fd_e2e.values(), dtype=np.float64, count=len(fd_e2e)).reshape(1, -1)
            else:
                x_e2e = infer_probe
            _ = reg_model.predict(x_e2e)
            p_e2e = clf_model.predict_proba(x_e2e)
            _ = int(p_e2e[0, 1] >= 0.5)
            te1 = time.perf_counter_ns()

            return (
                (t1 - t0) / 1e6,
                (t2 - t1) / 1e6,
                (t3 - t2) / 1e6,
                (t4 - t3) / 1e6,
                (t5 - t4) / 1e6,
                (te1 - te0) / 1e6,
            )

        # Warm-up passes
        for w in range(eff_warmup):
            _ = _run_single_stage_sequence(scen_pool[w % len(scen_pool)])

        s1_times, s2_times, s3_times, s4_times, s5_times, s6_times = [], [], [], [], [], []
        for i in range(eff_reps):
            dt1, dt2, dt3, dt4, dt5, dt6 = _run_single_stage_sequence(scen_pool[i % len(scen_pool)])
            s1_times.append(dt1)
            s2_times.append(dt2)
            s3_times.append(dt3)
            s4_times.append(dt4)
            s5_times.append(dt5)
            s6_times.append(dt6)

        st_slice = _summarize_timings_ms(s1_times)
        st_feat = _summarize_timings_ms(s2_times)
        st_feat_total = _summarize_timings_ms([a + b for a, b in zip(s1_times, s2_times)])
        st_reg = _summarize_timings_ms(s3_times)
        st_clf = _summarize_timings_ms(s4_times)
        st_cal = _summarize_timings_ms(s5_times)
        st_infer_total = _summarize_timings_ms([a + b + c for a, b, c in zip(s3_times, s4_times, s5_times)])
        st_e2e = _summarize_timings_ms(s6_times)

        # Determine ODE simulation baseline runtime
        explicit_sim = sim_time_ms if sim_time_ms is not None else self.sim_time_ms
        should_measure_sim = (
            bool(measure_simulation)
            if measure_simulation is not None
            else (explicit_sim is None and (scenarios is not None or not is_raw_feature_vector))
        )

        if explicit_sim is not None and not measure_simulation:
            sim_stats = {
                "median_ms": float(explicit_sim),
                "mean_ms": float(explicit_sim),
                "std_ms": 0.0,
                "min_ms": float(explicit_sim),
                "p95_ms": float(explicit_sim),
                "p99_ms": float(explicit_sim),
                "max_ms": float(explicit_sim),
                "n_sim_runs": 0,
                "measured_on_hardware": False,
            }
        elif should_measure_sim:
            runs = n_sim_runs if scenarios is not None else min(3, max(1, n_sim_runs))
            sim_stats = self.measure_ode_simulation(scenarios=scenarios, n_sim_runs=runs, warmup_runs=1)
        else:
            fallback_ms = 145.0
            sim_stats = {
                "median_ms": fallback_ms,
                "mean_ms": fallback_ms,
                "std_ms": 0.0,
                "min_ms": fallback_ms,
                "p95_ms": fallback_ms,
                "p99_ms": fallback_ms,
                "max_ms": fallback_ms,
                "n_sim_runs": 0,
                "measured_on_hardware": False,
            }

        median_total = float(st_e2e["median_ms"])
        median_sim = float(sim_stats["median_ms"])
        speedup = float(round(median_sim / max(0.001, median_total), 1))
        compliant = bool(median_total < self.budget_total_ms and st_e2e["p99_ms"] < self.budget_total_ms)
        env_info = collect_hardware_software_info()

        stages = {
            "stage_1_window_slicing": st_slice,
            "stage_2_feature_extraction_14": st_feat,
            "stage_3_regression_inference": st_reg,
            "stage_4_classification_inference": st_clf,
            "stage_5_probability_calibration": st_cal,
            "stage_6_end_to_end_online": st_e2e,
            "ode_dynamic_simulation": sim_stats,
        }

        return {
            "t_slice_ms": round(st_slice["median_ms"], 4),
            "t_feat_ms": round(st_feat_total["median_ms"], 3),
            "t_reg_ms": round(st_reg["median_ms"], 4),
            "t_clf_ms": round(st_clf["median_ms"], 4),
            "t_cal_ms": round(st_cal["median_ms"], 4),
            "t_infer_ms": round(st_infer_total["median_ms"], 3),
            "t_total_ms": round(median_total, 3),
            "t_total_mean_ms": round(st_e2e["mean_ms"], 4),
            "t_total_std_ms": round(st_e2e["std_ms"], 4),
            "t_total_p95_ms": round(st_e2e["p95_ms"], 4),
            "t_total_p99_ms": round(st_e2e["p99_ms"], 4),
            "sim_time_ms": round(median_sim, 2),
            "sim_time_mean_ms": round(sim_stats["mean_ms"], 2),
            "sim_time_std_ms": round(sim_stats["std_ms"], 2),
            "sim_time_p95_ms": round(sim_stats["p95_ms"], 2),
            "sim_measured_on_hardware": bool(sim_stats["measured_on_hardware"]),
            "speedup_factor": speedup,
            "latency_compliant_under_10ms": compliant,
            "n_reps": eff_reps,
            "warmup_reps": eff_warmup,
            "n_features": n_feat,
            "environment": env_info,
            "stages": stages,
            "trials": {
                "feat_times_ms": [round(t, 4) for t in s2_times[:25]],
                "infer_times_ms": [round(a + b + c, 4) for a, b, c in list(zip(s3_times, s4_times, s5_times))[:25]],
            },
        }

    def profile_batch_scaling(
        self,
        model: Optional[Any] = None,
        batch_sizes: Optional[List[int]] = None,
        n_features: int = 14,
        n_iter: int = 10,
    ) -> Dict[str, Any]:
        """Measure batch inference latency and throughput across batch sizes."""
        batches = batch_sizes or [1, 10, 50, 100]
        n_iter = max(1, int(n_iter))
        if model is None:
            target_model = GradientBoostingNadirRegressor(n_estimators=50, max_depth=4, random_state=42)
            rng = np.random.RandomState(42)
            X_dummy = rng.randn(40, n_features)
            y_dummy = 49.5 - 0.2 * rng.rand(40)
            target_model.fit(X_dummy, y_dummy)
        else:
            target_model = model
            if hasattr(target_model, "n_features_in_") and target_model.n_features_in_ is not None:
                n_features = target_model.n_features_in_

        results: Dict[int, Dict[str, float]] = {}
        rng_eval = np.random.RandomState(123)
        for b_size in batches:
            X_batch = rng_eval.randn(b_size, n_features)
            for _ in range(5):
                _ = target_model.predict(X_batch)

            trial_times_ms: List[float] = []
            for _ in range(10):
                t_start_ns = time.perf_counter_ns()
                for _ in range(n_iter):
                    _ = target_model.predict(X_batch)
                t_end_ns = time.perf_counter_ns()
                trial_batch_ms = (t_end_ns - t_start_ns) / (n_iter * 1e6)
                trial_times_ms.append(trial_batch_ms)

            total_batch_ms = float(np.median(trial_times_ms))
            per_sample_ms = total_batch_ms / b_size

            results[b_size] = {
                "batch_total_ms": round(total_batch_ms, 4),
                "per_sample_ms": round(per_sample_ms, 4),
                "throughput_samples_per_sec": round(1000.0 / max(1e-4, per_sample_ms), 1),
            }

        return {"batch_scaling": results}

    def to_stages_dataframe(self, profile_result: Dict[str, Any]) -> pd.DataFrame:
        stages = profile_result.get("stages", {})
        stage_labels = [
            ("stage_1_window_slicing", "1. Causal Window Slicing"),
            ("stage_2_feature_extraction_14", "2. 14-Feature Extraction"),
            ("stage_3_regression_inference", "3. Nadir Regression Inference"),
            ("stage_4_classification_inference", "4. Security Classification Inference"),
            ("stage_5_probability_calibration", "5. Probability Calibration & Decision"),
            ("stage_6_end_to_end_online", "6. Total End-to-End Online Prediction"),
            ("ode_dynamic_simulation", "Reference: 10s Time-Domain ODE Simulation"),
        ]
        rows = []
        for key, disp in stage_labels:
            if key in stages:
                st = stages[key]
                rows.append({
                    "Stage": disp,
                    "Median (ms)": round(float(st.get("median_ms", 0.0)), 4),
                    "Mean (ms)": round(float(st.get("mean_ms", 0.0)), 4),
                    "Std (ms)": round(float(st.get("std_ms", 0.0)), 4),
                    "P95 (ms)": round(float(st.get("p95_ms", 0.0)), 4),
                    "P99 (ms)": round(float(st.get("p99_ms", 0.0)), 4),
                })
        return pd.DataFrame(rows)

    def to_markdown(self, profile_result: Dict[str, Any]) -> str:
        df_stages = self.to_stages_dataframe(profile_result)
        lines: List[str] = []
        if not df_stages.empty:
            headers = list(df_stages.columns)
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for _, row in df_stages.iterrows():
                vals = [f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
                lines.append("| " + " | ".join(vals) + " |")
            lines.append("")

        env = profile_result.get("environment", {})
        summary_rows = [
            {"Metric": "End-to-End Median Online Latency", "Value": f"{profile_result.get('t_total_ms', 0.0):.3f} ms", "Target": f"< {self.budget_total_ms:.1f} ms"},
            {"Metric": "End-to-End P95 / P99 Tail Latency", "Value": f"{profile_result.get('t_total_p95_ms', 0.0):.3f} / {profile_result.get('t_total_p99_ms', 0.0):.3f} ms", "Target": f"< {self.budget_total_ms:.1f} ms"},
            {"Metric": "Measured 10s Dynamic ODE Simulation (Median)", "Value": f"{profile_result.get('sim_time_ms', 0.0):.2f} ms (hardware_measured={profile_result.get('sim_measured_on_hardware', False)})", "Target": "Hardware Reference"},
            {"Metric": "Computational Speedup Ratio (Median Sim / Median Online)", "Value": f"{profile_result.get('speedup_factor', 0.0):.1f}x", "Target": "Real-Time Screening"},
            {"Metric": "Repetitions / Warm-up Runs", "Value": f"{profile_result.get('n_reps', 100)} / {profile_result.get('warmup_reps', 10)}", "Target": ">= 100 / 10"},
            {"Metric": "Latency Compliant (< 10 ms)", "Value": str(profile_result.get("latency_compliant_under_10ms", True)), "Target": "PASS"},
        ]
        df_sum = pd.DataFrame(summary_rows)
        headers_s = list(df_sum.columns)
        lines.append("| " + " | ".join(headers_s) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers_s)) + " |")
        for _, row in df_sum.iterrows():
            lines.append("| " + " | ".join(str(v) for v in row) + " |")

        if env:
            lines.append(
                f"\n- **Hardware / OS**: `{env.get('cpu', 'unknown')}` ({env.get('cpu_count', 1)} cores), `{env.get('platform', 'unknown')}`"
            )
            lines.append(
                f"- **Software Stack**: Python `{env.get('python_version')}`, NumPy `{env.get('numpy_version')}`, "
                f"SciPy `{env.get('scipy_version')}`, scikit-learn `{env.get('sklearn_version')}`"
            )
        return "\n".join(lines)
