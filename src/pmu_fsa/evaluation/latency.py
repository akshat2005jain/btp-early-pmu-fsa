"""Per-sample feature extraction and model inference latency profiler."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from pmu_fsa.features.extractor import FeatureExtractor
from pmu_fsa.models.regressors import LinearNadirRegressor


class LatencyBenchmark:
    """Profiles online feature extraction and inference latency against ODE simulation."""

    def __init__(
        self,
        budget_total_ms: float = 10.0,
        budget_feat_ms: float = 5.0,
        budget_infer_ms: float = 2.0,
        sim_time_ms: float = 350.0,
    ) -> None:
        self.budget_total_ms = float(budget_total_ms)
        self.budget_feat_ms = float(budget_feat_ms)
        self.budget_infer_ms = float(budget_infer_ms)
        self.sim_time_ms = float(sim_time_ms)

    def _generate_synthetic_sample(self, n_samples: int = 250, fps: int = 50) -> Dict[str, Any]:
        dt = 1.0 / float(fps)
        timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
        f_base = 50.0
        drop = 0.5 * (1.0 - np.exp(-timestamps / 1.5))
        coi = f_base - drop
        bus_freqs = np.tile(coi[:, np.newaxis], (1, 39))
        bus_rocof = np.tile(np.gradient(coi, dt)[:, np.newaxis], (1, 39))
        v_mags = np.ones((n_samples, 39), dtype=float)
        v_angles = np.zeros((n_samples, 39), dtype=float)

        return {
            "timestamps": timestamps,
            "bus_frequencies": bus_freqs,
            "bus_rocof": bus_rocof,
            "voltage_magnitudes": v_mags,
            "voltage_angles": v_angles,
            "coi_frequency": coi,
            "metadata": {
                "scenario_id": "latency_bench_sample",
                "t_disturb": 1.0,
                "fps": fps,
                "ground_truth_nadir": float(np.min(coi)),
                "ground_truth_t_nadir": 3.5,
                "renewable_penetration": 0.20,
                "is_secure": 1,
            },
        }

    def profile(
        self,
        extractor_fn: Optional[Callable] = None,
        model: Optional[Any] = None,
        sample_data: Optional[Any] = None,
        n_iter: int = 20,
        sim_time_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Measure single-sample feature extraction and inference latency across trials."""
        n_iter = max(1, int(n_iter))

        if extractor_fn is not None:
            fn = extractor_fn
        else:
            default_fe = FeatureExtractor()
            fn = lambda stream, t_obs=1.0: default_fe.extract_features_dict(stream, t_obs=t_obs)

        if sample_data is not None and isinstance(sample_data, dict) and "timestamps" in sample_data:
            scenario_sample = sample_data
        else:
            scenario_sample = self._generate_synthetic_sample()

        for _ in range(3):
            try:
                _ = fn(scenario_sample, 1.0)
            except TypeError:
                _ = fn(scenario_sample)

        feat_trial_times_ms: List[float] = []
        for _ in range(5):
            t_start_ns = time.perf_counter_ns()
            for _ in range(n_iter):
                try:
                    _ = fn(scenario_sample, 1.0)
                except TypeError:
                    _ = fn(scenario_sample)
            t_end_ns = time.perf_counter_ns()
            per_sample_ms = (t_end_ns - t_start_ns) / (n_iter * 1e6)
            feat_trial_times_ms.append(per_sample_ms)

        # Use min across trials to filter out OS scheduling jitter
        t_feat_ms = min(feat_trial_times_ms)

        if sample_data is not None and not (isinstance(sample_data, dict) and "timestamps" in sample_data):
            if hasattr(sample_data, "shape") and len(sample_data.shape) > 1:
                infer_input = np.asarray(sample_data[:1], dtype=float)
            elif isinstance(sample_data, (list, tuple)) and len(sample_data) > 0 and isinstance(sample_data[0], (list, tuple)):
                infer_input = np.asarray([sample_data[0]], dtype=float)
            else:
                infer_input = np.atleast_2d(np.asarray(sample_data, dtype=float))
        else:
            try:
                feat_dict = fn(scenario_sample, 1.0)
            except TypeError:
                feat_dict = fn(scenario_sample)
            infer_input = np.array([list(feat_dict.values())], dtype=float)

        if infer_input.ndim == 1:
            infer_input = infer_input.reshape(1, -1)

        n_feat = infer_input.shape[1]

        if model is not None:
            target_model = model
        else:
            target_model = LinearNadirRegressor(scale_features=False)
            rng = np.random.RandomState(42)
            X_dummy = rng.randn(10, n_feat)
            y_dummy = 49.5 - 0.2 * rng.rand(10)
            target_model.fit(X_dummy, y_dummy)

        for _ in range(3):
            _ = target_model.predict(infer_input)

        infer_trial_times_ms: List[float] = []
        for _ in range(5):
            t_start_ns = time.perf_counter_ns()
            for _ in range(n_iter):
                _ = target_model.predict(infer_input)
            t_end_ns = time.perf_counter_ns()
            per_sample_ms = (t_end_ns - t_start_ns) / (n_iter * 1e6)
            infer_trial_times_ms.append(per_sample_ms)

        t_infer_ms = min(infer_trial_times_ms)

        t_total_ms = float(round(t_feat_ms + t_infer_ms, 3))
        sim_time = float(sim_time_ms if sim_time_ms is not None else self.sim_time_ms)
        speedup = float(round(sim_time / max(0.01, t_total_ms), 1))
        compliant = bool(t_total_ms < self.budget_total_ms)

        return {
            "t_feat_ms": float(round(t_feat_ms, 3)),
            "t_infer_ms": float(round(t_infer_ms, 3)),
            "t_total_ms": t_total_ms,
            "sim_time_ms": float(round(sim_time, 1)),
            "speedup_factor": speedup,
            "latency_compliant_under_10ms": compliant,
            "trials": {
                "feat_times_ms": [float(round(t, 4)) for t in feat_trial_times_ms],
                "infer_times_ms": [float(round(t, 4)) for t in infer_trial_times_ms],
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
            target_model = LinearNadirRegressor(scale_features=False)
            rng = np.random.RandomState(42)
            X_dummy = rng.randn(20, n_features)
            y_dummy = 49.5 - 0.2 * rng.rand(20)
            target_model.fit(X_dummy, y_dummy)
        else:
            target_model = model
            if hasattr(target_model, "n_features_in_") and target_model.n_features_in_ is not None:
                n_features = target_model.n_features_in_

        results: Dict[int, Dict[str, float]] = {}
        for b_size in batches:
            X_batch = np.random.randn(b_size, n_features)
            for _ in range(3):
                _ = target_model.predict(X_batch)

            trial_times_ms: List[float] = []
            for _ in range(5):
                t_start_ns = time.perf_counter_ns()
                for _ in range(n_iter):
                    _ = target_model.predict(X_batch)
                t_end_ns = time.perf_counter_ns()
                trial_batch_ms = (t_end_ns - t_start_ns) / (n_iter * 1e6)
                trial_times_ms.append(trial_batch_ms)

            total_batch_ms = min(trial_times_ms)
            per_sample_ms = total_batch_ms / b_size

            results[b_size] = {
                "batch_total_ms": float(round(total_batch_ms, 4)),
                "per_sample_ms": float(round(per_sample_ms, 4)),
                "throughput_samples_per_sec": float(round(1000.0 / max(1e-4, per_sample_ms), 1)),
            }

        return {"batch_scaling": results}

    def to_markdown(self, profile_result: Dict[str, Any]) -> str:
        rows = [
            {"Metric": "Feature Extraction Latency (t_feat)", "Value": f"{profile_result.get('t_feat_ms', 0.0):.3f} ms", "Budget": f"< {self.budget_feat_ms:.1f} ms"},
            {"Metric": "Model Inference Latency (t_infer)", "Value": f"{profile_result.get('t_infer_ms', 0.0):.3f} ms", "Budget": f"< {self.budget_infer_ms:.1f} ms"},
            {"Metric": "Total Online Execution (t_total)", "Value": f"{profile_result.get('t_total_ms', 0.0):.3f} ms", "Budget": f"< {self.budget_total_ms:.1f} ms"},
            {"Metric": "Numerical Dynamic ODE Simulation", "Value": f"{profile_result.get('sim_time_ms', 0.0):.1f} ms", "Budget": "Baseline Reference"},
            {"Metric": "Computational Speedup Ratio", "Value": f"{profile_result.get('speedup_factor', 0.0):.1f}x", "Budget": "> 200x"},
            {"Metric": "Latency Compliant (< 10 ms)", "Value": str(profile_result.get('latency_compliant_under_10ms', True)), "Budget": "PASS"},
        ]
        df = pd.DataFrame(rows)
        headers = [str(c) for c in df.columns]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for _, row in df.iterrows():
            vals = [str(v) for v in row]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)
