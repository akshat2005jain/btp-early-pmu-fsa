"""Tests for evaluation benchmarks, experiment pipeline, and CLI entrypoints."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.pipeline.run_experiments import (
    ExperimentConfig,
    ExperimentPipeline,
    run_experiments,
    WORKFLOW_STEPS,
)
from pmu_fsa.models.regressors import LinearNadirRegressor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_fast_pmu_scenario(
    n_samples: int = 250,
    fps: int = 50,
    renewable_penetration: float = 0.20,
    t_disturb: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    dt = 1.0 / float(fps)
    timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)

    f_nominal = 60.0
    t_post = np.maximum(0.0, timestamps - t_disturb)
    base_drop = 0.40 + 0.50 * renewable_penetration + rng.uniform(-0.02, 0.02)
    decay = base_drop * (1.0 - np.exp(-t_post / 1.2)) * np.exp(-np.maximum(0.0, t_post - 3.5) / 4.0)
    coi_freq = f_nominal - decay

    bus_offsets = rng.normal(0.0, 0.015, size=(1, 39))
    bus_frequencies = coi_freq[:, np.newaxis] + bus_offsets
    bus_rocof = np.gradient(bus_frequencies, dt, axis=0)

    voltage_magnitudes = np.ones((n_samples, 39), dtype=float) + rng.normal(0.0, 0.005, size=(n_samples, 39))
    voltage_angles = np.zeros((n_samples, 39), dtype=float) + rng.normal(0.0, 0.02, size=(n_samples, 39))

    nadir_val = float(np.min(coi_freq))
    nadir_idx = int(np.argmin(coi_freq))
    t_nadir = float(timestamps[nadir_idx])

    is_secure = 1 if nadir_val >= 59.50 else 0

    return {
        "timestamps": timestamps,
        "bus_frequencies": bus_frequencies,
        "bus_rocof": bus_rocof,
        "voltage_magnitudes": voltage_magnitudes,
        "voltage_angles": voltage_angles,
        "coi_frequency": coi_freq,
        "metadata": {
            "scenario_id": f"scen_ren{int(renewable_penetration*100)}_{seed}",
            "disturbed_gen": 31,
            "active_loss_mw": 300.0 + renewable_penetration * 200.0,
            "renewable_penetration": float(renewable_penetration),
            "load_scale": 1.0,
            "t_disturb": float(t_disturb),
            "t_sim": float(timestamps[-1]),
            "fps": fps,
            "ground_truth_nadir": round(nadir_val, 4),
            "ground_truth_t_nadir": round(t_nadir, 4),
            "is_secure": is_secure,
            "f_threshold_hz": 59.50,
            "partition": "ood_test" if renewable_penetration >= 0.35 else "train",
        },
    }


@pytest.fixture
def fast_synthetic_scenario_batch() -> List[Dict[str, Any]]:
    batch = []
    for i, pen in enumerate([0.10, 0.15, 0.20, 0.25, 0.28, 0.30]):
        batch.append(generate_fast_pmu_scenario(renewable_penetration=pen, seed=100 + i))
    for i, pen in enumerate([0.40, 0.42, 0.45, 0.48, 0.50, 0.50]):
        batch.append(generate_fast_pmu_scenario(renewable_penetration=pen, seed=200 + i))
    return batch


class TestSyntheticBenchmarkFixtures:

    def test_fixture_generation_speed_under_20ms(self):
        _ = generate_fast_pmu_scenario(renewable_penetration=0.20, seed=0)

        n_scenarios = 12
        t0 = time.perf_counter()
        scenarios = [
            generate_fast_pmu_scenario(renewable_penetration=0.10 + i * 0.03, seed=i)
            for i in range(n_scenarios)
        ]
        duration_s = time.perf_counter() - t0
        per_scenario_s = duration_s / n_scenarios
        duration_ms = duration_s * 1000.0

        assert len(scenarios) == n_scenarios
        assert per_scenario_s < 0.010, (
            f"Per-scenario generation rate too slow: {per_scenario_s * 1000.0:.2f} ms/scenario (budget < 10.0 ms)"
        )
        assert duration_ms < 120.0, f"Fixture generation took {duration_ms:.2f} ms (budget < 120 ms)"

    def test_fixture_physical_characteristics_and_contract(self, fast_synthetic_scenario_batch):
        scen = fast_synthetic_scenario_batch[0]
        required_keys = [
            "timestamps", "bus_frequencies", "bus_rocof",
            "voltage_magnitudes", "voltage_angles", "coi_frequency", "metadata"
        ]
        for k in required_keys:
            assert k in scen, f"Missing key: {k}"

        assert scen["bus_frequencies"].shape == (250, 39)
        assert scen["bus_rocof"].shape == (250, 39)
        assert len(scen["coi_frequency"]) == 250
        assert "ground_truth_nadir" in scen["metadata"]
        assert "is_secure" in scen["metadata"]

    def test_fixture_id_ood_clean_split(self, fast_synthetic_scenario_batch):
        id_pens = [
            s["metadata"]["renewable_penetration"] for s in fast_synthetic_scenario_batch
            if 0.10 <= s["metadata"]["renewable_penetration"] <= 0.30
        ]
        ood_pens = [
            s["metadata"]["renewable_penetration"] for s in fast_synthetic_scenario_batch
            if 0.40 <= s["metadata"]["renewable_penetration"] <= 0.50
        ]
        assert len(id_pens) == 6
        assert len(ood_pens) == 6
        assert max(id_pens) <= 0.30
        assert min(ood_pens) >= 0.40


class TestWindowSensitivityBenchmark:

    def test_window_sweep_execution_and_schema(self, fast_synthetic_scenario_batch):
        benchmark = WindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0])
        results = benchmark.run(fast_synthetic_scenario_batch)

        assert results["columns"] == ["window_obs", "mae", "rmse", "r2", "f1", "ece"]
        rows = results["results"]
        assert len(rows) == 4
        assert [r["window_obs"] for r in rows] == [0.5, 1.0, 1.5, 2.0]

    def test_decision_lead_time_monotonic_decrease(self, fast_synthetic_scenario_batch):
        benchmark = WindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0])
        results = benchmark.run(fast_synthetic_scenario_batch)
        rows = results["results"]

        lead_times = [r["t_lead"] for r in rows]
        for i in range(len(lead_times) - 1):
            assert lead_times[i] > lead_times[i + 1], \
                f"Lead time failed to decrease: {lead_times[i]} vs {lead_times[i+1]}"
            assert lead_times[i] > 0.0

    def test_monotonic_accuracy_improvement_or_stability(self, fast_synthetic_scenario_batch):
        benchmark = WindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0])
        results = benchmark.run(fast_synthetic_scenario_batch)
        rows = results["results"]

        maes = [r["mae"] for r in rows]
        f1s = [r["f1"] for r in rows]

        for i in range(len(maes) - 1):
            assert maes[i] >= maes[i + 1] - 1e-4, \
                f"MAE degraded: {maes[i]} at {rows[i]['window_obs']}s vs {maes[i+1]} at {rows[i+1]['window_obs']}s"

        for i in range(len(f1s) - 1):
            assert f1s[i] <= f1s[i + 1] + 1e-4, \
                f"F1 degraded: {f1s[i]} vs {f1s[i+1]}"

    def test_rmse_greater_equal_mae_invariant(self, fast_synthetic_scenario_batch):
        benchmark = WindowSensitivityBenchmark()
        results = benchmark.run(fast_synthetic_scenario_batch)
        for r in results["results"]:
            assert r["rmse"] >= r["mae"] - 1e-6, \
                f"Cauchy-Schwarz invariant violated: RMSE ({r['rmse']}) < MAE ({r['mae']})"

    def test_custom_window_grid_support(self, fast_synthetic_scenario_batch):
        custom = [0.3, 0.8, 1.2]
        benchmark = WindowSensitivityBenchmark(window_lengths=custom)
        res = benchmark.run(fast_synthetic_scenario_batch)
        assert len(res["results"]) == 3
        assert [r["window_obs"] for r in res["results"]] == custom

    def test_single_window_degenerate_sweep(self, fast_synthetic_scenario_batch):
        benchmark = WindowSensitivityBenchmark(window_lengths=[1.0])
        res = benchmark.run(fast_synthetic_scenario_batch)
        assert len(res["results"]) == 1
        assert res["results"][0]["window_obs"] == 1.0

    def test_empty_scenario_list_handling(self):
        benchmark = WindowSensitivityBenchmark()
        res = benchmark.run([])
        assert res["columns"] == ["window_obs", "mae", "rmse", "r2", "f1", "ece"]
        assert res["results"] == []

    def test_dataframe_and_markdown_export(self, fast_synthetic_scenario_batch):
        benchmark = WindowSensitivityBenchmark()
        res = benchmark.run(fast_synthetic_scenario_batch)
        df = benchmark.to_dataframe(res)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        md = benchmark.to_markdown(res)
        assert isinstance(md, str)
        assert "|" in md and "window_obs" in md

    def test_window_grid_sorting_in_init(self):
        unsorted = [2.0, 0.5, 1.5, 1.0]
        bench = WindowSensitivityBenchmark(window_lengths=unsorted)
        assert bench.window_lengths == [0.5, 1.0, 1.5, 2.0]

    def test_raw_metrics_exposed_in_sensitivity(self, fast_synthetic_scenario_batch):
        bench = WindowSensitivityBenchmark(window_lengths=[0.5, 1.0])
        res = bench.run(fast_synthetic_scenario_batch)
        for r in res["results"]:
            assert "mae_raw" in r and "rmse_raw" in r and "f1_raw" in r
            assert isinstance(r["mae_raw"], float)
        df = bench.to_dataframe(res)
        assert "mae_raw" in df.columns


class TestOODGeneralizationBenchmark:

    def test_id_ood_partition_separation(self, fast_synthetic_scenario_batch):
        bench = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.40, 0.50))
        res = bench.run(fast_synthetic_scenario_batch)
        assert res["id_scenario_count"] == 6
        assert res["ood_scenario_count"] == 6

    def test_generalization_gap_calculation(self, fast_synthetic_scenario_batch):
        bench = OODGeneralizationBenchmark()
        res = bench.run(fast_synthetic_scenario_batch)
        gap = res["generalization_gap"]

        assert "delta_mae" in gap
        assert "delta_f1" in gap
        assert "delta_ece" in gap
        assert gap["delta_mae"] >= 0.0
        assert res["delta_mae"] >= 0.0
        assert res["ood_metrics"]["mae"] >= res["id_metrics"]["mae"]

    def test_calibration_degradation_reporting(self, fast_synthetic_scenario_batch):
        bench = OODGeneralizationBenchmark()
        res = bench.run(fast_synthetic_scenario_batch)
        assert 0.0 <= res["id_metrics"]["ece"] <= 1.0
        assert 0.0 <= res["ood_metrics"]["ece"] <= 1.0
        assert isinstance(res["delta_ece"], float)

    def test_distribution_shift_flag_detection(self, fast_synthetic_scenario_batch):
        bench = OODGeneralizationBenchmark()
        res = bench.run(fast_synthetic_scenario_batch)
        assert res["distribution_shift_detected"] is True

    def test_identical_distribution_zero_gap(self, fast_synthetic_scenario_batch):
        bench = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.10, 0.30))
        res = bench.run(fast_synthetic_scenario_batch)
        assert res["delta_mae"] == pytest.approx(0.0, abs=1e-5)
        assert res["delta_f1"] == pytest.approx(0.0, abs=1e-5)

    def test_overlapping_ranges_rejection(self):
        with pytest.raises(ValueError, match="overlap"):
            OODGeneralizationBenchmark(id_range=(0.10, 0.35), ood_range=(0.30, 0.50))
        with pytest.raises(ValueError, match="overlap"):
            OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.30, 0.50))

    def test_uncertainty_elevation_on_ood_regime(self, fast_synthetic_scenario_batch):
        bench = OODGeneralizationBenchmark()
        res = bench.run(fast_synthetic_scenario_batch)
        unc = res["uncertainty_elevation"]
        assert "mean_interval_width_id" in unc
        assert "mean_interval_width_ood" in unc
        assert unc["mean_interval_width_id"] >= 0.0
        assert unc["mean_interval_width_ood"] >= 0.0

    def test_ood_tabular_report_schema(self, fast_synthetic_scenario_batch):
        bench = OODGeneralizationBenchmark()
        res = bench.run(fast_synthetic_scenario_batch)
        df = bench.to_dataframe(res)
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 3
        md = bench.to_markdown(res)
        assert "Generalization Gap" in md


class TestMeasurementNoiseBenchmark:

    def test_gaussian_noise_sweep_execution(self, fast_synthetic_scenario_batch):
        noise_stds = [0.0, 0.005, 0.01, 0.02, 0.05]
        bench = MeasurementNoiseBenchmark(noise_levels=noise_stds)
        res = bench.run(fast_synthetic_scenario_batch)

        assert res["columns"] == ["noise_std_hz", "mae", "rmse", "ece"]
        rows = res["results"]
        assert len(rows) == len(noise_stds)
        assert [r["noise_std_hz"] for r in rows] == noise_stds

    def test_graceful_degradation_curve(self, fast_synthetic_scenario_batch):
        bench = MeasurementNoiseBenchmark()
        res = bench.run(fast_synthetic_scenario_batch)
        rows = res["results"]

        maes = [r["mae"] for r in rows]
        for i in range(len(maes) - 1):
            assert maes[i] <= maes[i + 1] + 1e-4, \
                f"MAE did not degrade monotonically: {maes[i]} vs {maes[i+1]}"
            delta_mae = maes[i + 1] - maes[i]
            assert delta_mae < 0.5, f"Catastrophic spike: delta_mae={delta_mae}"

    def test_ols_rocof_resilience_proof_vs_endpoint(self):
        comparison = MeasurementNoiseBenchmark.compare_ols_vs_endpoint_rocof(
            noise_std=0.03, n_trials=50, seed=99
        )
        assert comparison["noise_resilience_verified"] is True
        assert comparison["variance_ratio"] < 0.15, \
            f"OLS ROCOF variance ({comparison['var_ols']}) was not < 0.15x endpoint ({comparison['var_endpoint']})"

    def test_extreme_noise_physical_clipping(self):
        scen = generate_fast_pmu_scenario(renewable_penetration=0.20, seed=42)
        bench = MeasurementNoiseBenchmark(noise_levels=[0.10])
        res = bench.run([scen])
        assert len(res["results"]) == 1
        mae = res["results"][0]["mae"]
        assert not math.isnan(mae) and mae > 0.0

    def test_negative_noise_rejection(self):
        with pytest.raises(ValueError, match="non-negative"):
            MeasurementNoiseBenchmark(noise_levels=[-0.01])

    def test_noise_benchmark_table_export(self, fast_synthetic_scenario_batch):
        bench = MeasurementNoiseBenchmark()
        res = bench.run(fast_synthetic_scenario_batch)
        df = bench.to_dataframe(res)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        md = bench.to_markdown(res)
        assert "noise_std_hz" in md

    def test_noise_level_sorting_in_init(self):
        unsorted = [0.05, 0.0, 0.02, 0.01]
        bench = MeasurementNoiseBenchmark(noise_levels=unsorted)
        assert bench.noise_levels == [0.0, 0.01, 0.02, 0.05]

    def test_raw_metrics_exposed_in_noise_stress(self, fast_synthetic_scenario_batch):
        bench = MeasurementNoiseBenchmark(noise_levels=[0.0, 0.01])
        res = bench.run(fast_synthetic_scenario_batch)
        for r in res["results"]:
            assert "mae_raw" in r and "rmse_raw" in r and "f1_raw" in r
            assert isinstance(r["mae_raw"], float)
        df = bench.to_dataframe(res)
        assert "mae_raw" in df.columns


class TestLatencyBenchmark:

    def test_high_precision_perf_counter_ns_timing(self):
        bench = LatencyBenchmark()
        prof = bench.profile(n_iter=10)
        assert prof["t_feat_ms"] > 0.0
        assert prof["t_infer_ms"] > 0.0
        assert prof["t_total_ms"] > 0.0

    def test_per_sample_latency_budget_compliance(self):
        bench = LatencyBenchmark()
        bench.profile(n_iter=5)
        prof = bench.profile(n_iter=15)

        assert prof["t_feat_ms"] < 10.0, f"t_feat_ms={prof['t_feat_ms']} exceeded budget"
        assert prof["t_infer_ms"] < 5.0, f"t_infer_ms={prof['t_infer_ms']} exceeded budget"
        assert prof["t_total_ms"] < 10.0, f"t_total_ms={prof['t_total_ms']} exceeded 10.0 ms budget"
        assert prof["latency_compliant_under_10ms"] is True

    def test_speedup_ratio_vs_dynamic_simulation(self):
        bench = LatencyBenchmark(sim_time_ms=350.0)
        prof = bench.profile(n_iter=10)
        assert prof["speedup_factor"] >= 10.0
        expected_speedup = round(prof["sim_time_ms"] / max(0.01, prof["t_total_ms"]), 1)
        assert abs(prof["speedup_factor"] - expected_speedup) < 0.2

    def test_batch_throughput_scaling(self):
        bench = LatencyBenchmark()
        batch_res = bench.profile_batch_scaling(batch_sizes=[1, 10, 50], n_iter=5)
        scaling = batch_res["batch_scaling"]
        for b_size, metrics in scaling.items():
            threshold = 5.0 if b_size == 1 else 1.0
            assert metrics["per_sample_ms"] < threshold, (
                f"Batch {b_size} per-sample latency too high: {metrics['per_sample_ms']} ms >= {threshold} ms"
            )
        assert scaling[50]["per_sample_ms"] < scaling[1]["per_sample_ms"], (
            f"Batching did not improve throughput: batch 50 ({scaling[50]['per_sample_ms']} ms) "
            f">= batch 1 ({scaling[1]['per_sample_ms']} ms)"
        )

    def test_latency_profile_result_keys(self):
        bench = LatencyBenchmark()
        prof = bench.profile(n_iter=5)
        for k in ["t_feat_ms", "t_infer_ms", "t_total_ms", "sim_time_ms", "speedup_factor", "latency_compliant_under_10ms"]:
            assert k in prof
        md = bench.to_markdown(prof)
        assert "Computational Speedup" in md


class TestRunExperimentsPipeline:

    def test_pipeline_end_to_end_on_compact_fixture(self, tmp_path, fast_synthetic_scenario_batch):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        from pmu_fsa.simulation.scenario_generator import export_scenario_to_npz
        for scen in fast_synthetic_scenario_batch[:4]:
            export_scenario_to_npz(scen, str(raw_dir / f"{scen['metadata']['scenario_id']}.npz"))

        config = ExperimentConfig(
            raw_data_dir=str(raw_dir),
            processed_data_dir=str(tmp_path / "processed"),
            models_dir=str(tmp_path / "models"),
            output_dir=str(tmp_path / "results"),
            tables_dir=str(tmp_path / "results" / "tables"),
            figures_dir=str(tmp_path / "results" / "figures"),
            report_file=str(tmp_path / "results" / "evaluation_report.md"),
            skip_simulation=True,
            n_scenarios=4,
            window_lengths=[0.5, 1.0],
            noise_levels=[0.0, 0.01],
            headless=True,
        )

        pipeline = ExperimentPipeline(config=config)
        assert pipeline.workflow_steps == ["simulate", "extract_features", "train_models", "evaluate_all"]

        summary = pipeline.execute_full_pipeline()
        assert summary["n_scenarios"] == 4
        assert len(summary["features_windows"]) == 2
        assert len(summary["models"]) >= 4

    def test_pipeline_artifact_generation_csv_markdown(self, tmp_path, fast_synthetic_scenario_batch):
        tables_dir = tmp_path / "results" / "tables"
        report_file = tmp_path / "results" / "evaluation_report.md"
        config = ExperimentConfig(
            output_dir=str(tmp_path / "results"),
            tables_dir=str(tables_dir),
            figures_dir=str(tmp_path / "results" / "figures"),
            report_file=str(report_file),
            window_lengths=[0.5, 1.0],
            noise_levels=[0.0, 0.01],
            headless=True,
        )
        pipeline = ExperimentPipeline(config=config)
        pipeline.scenarios = list(fast_synthetic_scenario_batch[:4])
        pipeline.run_evaluate_all()

        assert (tables_dir / "window_sensitivity.csv").exists(), "window_sensitivity.csv was not generated"
        assert (tables_dir / "window_sensitivity.md").exists(), "window_sensitivity.md was not generated"
        assert (tables_dir / "ood_generalization.csv").exists(), "ood_generalization.csv was not generated"
        assert (tables_dir / "measurement_noise.csv").exists(), "measurement_noise.csv was not generated"
        assert (tables_dir / "latency_profile.csv").exists(), "latency_profile.csv was not generated"
        assert report_file.exists(), "evaluation_report.md master report was not generated"

        df_sens = pd.read_csv(tables_dir / "window_sensitivity.csv")
        assert not df_sens.empty, "window_sensitivity.csv is empty"
        assert "window_obs" in df_sens.columns
        assert "mae" in df_sens.columns

        report_text = report_file.read_text(encoding="utf-8")
        assert len(report_text) > 100
        assert "# PMU-Based Early Frequency-Security Assessment" in report_text

        sens_bench = WindowSensitivityBenchmark()
        res = sens_bench.run([])
        df = sens_bench.to_dataframe(res)
        assert isinstance(df, pd.DataFrame)

    def test_pipeline_deterministic_random_seed(self, tmp_path, fast_synthetic_scenario_batch):
        raw_dir = tmp_path / "raw_seed"
        raw_dir.mkdir(parents=True, exist_ok=True)
        from pmu_fsa.simulation.scenario_generator import export_scenario_to_npz
        for scen in fast_synthetic_scenario_batch[:4]:
            export_scenario_to_npz(scen, str(raw_dir / f"{scen['metadata']['scenario_id']}.npz"))

        cfg1 = ExperimentConfig(raw_data_dir=str(raw_dir), skip_simulation=True, seed=99, window_lengths=[0.5])
        cfg2 = ExperimentConfig(raw_data_dir=str(raw_dir), skip_simulation=True, seed=99, window_lengths=[0.5])

        p1 = ExperimentPipeline(cfg1)
        p1.run_simulate()
        p1.run_extract_features()
        m1 = p1.run_train_models()

        p2 = ExperimentPipeline(cfg2)
        p2.run_simulate()
        p2.run_extract_features()
        m2 = p2.run_train_models()

        assert list(m1.keys()) == list(m2.keys())


class TestCLISmokeTests:

    def test_run_simulations_cli_help_and_dryrun(self, tmp_path):
        script = PROJECT_ROOT / "scripts" / "run_simulations.py"
        res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "IEEE 39-Bus Contingency Simulator" in res.stdout

        run_res = subprocess.run(
            [sys.executable, str(script), "--fast", "--n-scenarios", "2", "--output-dir", str(tmp_path / "raw"), "--quiet"],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0
        assert len(list((tmp_path / "raw").glob("*.npz"))) == 2

    def test_extract_features_cli_help_and_dryrun(self, tmp_path):
        script = PROJECT_ROOT / "scripts" / "extract_features.py"
        res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "Causal PMU Feature Extractor" in res.stdout

        raw_dir = tmp_path / "raw"
        if not raw_dir.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)
            from pmu_fsa.simulation.scenario_generator import export_scenario_to_npz
            s = generate_fast_pmu_scenario(seed=1)
            export_scenario_to_npz(s, str(raw_dir / "s1.npz"))

        out_feat = tmp_path / "proc" / "features.npz"
        run_res = subprocess.run(
            [sys.executable, str(script), "--input-dir", str(raw_dir), "--output-file", str(out_feat), "--t-obs", "1.0", "--quiet"],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0
        assert out_feat.exists()

    def test_train_models_cli_help_and_dryrun(self, tmp_path):
        script = PROJECT_ROOT / "scripts" / "train_models.py"
        res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "Frequency Security Model Trainer" in res.stdout

        out_feat = tmp_path / "proc" / "features.npz"
        if not out_feat.exists():
            out_feat.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                out_feat,
                X=np.random.randn(10, 14),
                y_nadir=59.5 - 0.2 * np.random.rand(10),
                y_secure=np.array([1, 1, 0, 1, 0, 1, 1, 0, 1, 0]),
                partitions=np.array(["train"] * 8 + ["test"] * 2),
            )

        models_dir = tmp_path / "models"
        run_res = subprocess.run(
            [sys.executable, str(script), "--features", str(out_feat), "--model-type", "linear", "--output-dir", str(models_dir), "--quiet"],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0
        assert (models_dir / "linear.joblib").exists()

    def test_evaluate_all_cli_help_and_dryrun(self, tmp_path):
        script = PROJECT_ROOT / "scripts" / "evaluate_all.py"
        res = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
        assert "Evaluate All Benchmarks" in res.stdout

        report_file = tmp_path / "report.md"
        tables_dir = tmp_path / "tables"
        run_res = subprocess.run(
            [sys.executable, str(script), "--benchmarks", "latency", "--report-file", str(report_file), "--tables-dir", str(tables_dir), "--quiet"],
            capture_output=True,
            text=True,
        )
        assert run_res.returncode == 0
        assert report_file.exists()
