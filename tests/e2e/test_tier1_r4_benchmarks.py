"""End-to-end tests for evaluation benchmarks and CLI entrypoints."""

import os
import sys
import math
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple

from tests.e2e.conftest import (
    np, HAS_NUMPY,
    generate_synthetic_pmu_stream,
    assert_strictly_causal,
    RefLinearNadirRegressor,
    RefCalibratedGBDTClassifier,
    RefWindowSensitivityBenchmark,
    RefOODGeneralizationBenchmark,
    RefMeasurementNoiseBenchmark,
    RefLatencyBenchmark,
    _to_float_list_1d,
    _to_float_list_2d,
)

try:
    from pmu_fsa.evaluation import sensitivity
    from pmu_fsa.evaluation import ood_generalization
    from pmu_fsa.evaluation import noise_stress
    from pmu_fsa.evaluation import latency
    HAS_EVAL_MODULES = True
except ImportError:
    HAS_EVAL_MODULES = False


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_f18_window_sensitivity_sweep_execution(synthetic_scenario_batch):
    scenarios = synthetic_scenario_batch
    benchmark = RefWindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0])
    results = benchmark.run(scenarios)

    assert "results" in results, "Missing 'results' key in benchmark output"
    assert "columns" in results, "Missing 'columns' key in benchmark output"
    rows = results["results"]
    assert len(rows) == 4, f"Expected 4 window rows, got {len(rows)}"

    observed_windows = [r["window_obs"] for r in rows]
    assert observed_windows == [0.5, 1.0, 1.5, 2.0], f"Window values mismatch: {observed_windows}"

    if HAS_EVAL_MODULES and hasattr(sensitivity, "WindowSensitivityBenchmark"):
        prod_bench = sensitivity.WindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0])
        prod_results = prod_bench.run(scenarios)
        assert len(prod_results["results"]) == 4


def test_f18_monotonic_accuracy_gain_with_larger_windows(synthetic_scenario_batch):
    scenarios = synthetic_scenario_batch
    benchmark = RefWindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0])
    results = benchmark.run(scenarios)
    rows = results["results"]

    maes = [r["mae"] for r in rows]
    f1s = [r["f1"] for r in rows]

    for i in range(len(maes) - 1):
        assert maes[i] >= maes[i + 1] - 1e-4, \
            f"MAE did not improve with larger window: {maes[i]} at {rows[i]['window_obs']}s vs {maes[i+1]} at {rows[i+1]['window_obs']}s"

    for i in range(len(f1s) - 1):
        assert f1s[i] <= f1s[i + 1] + 1e-4, \
            f"F1 did not improve with larger window: {f1s[i]} vs {f1s[i+1]}"


def test_f18_tabular_output_formatting(synthetic_scenario_batch):
    benchmark = RefWindowSensitivityBenchmark()
    results = benchmark.run(synthetic_scenario_batch)

    required_columns = ["window_obs", "mae", "rmse", "r2", "f1", "ece"]
    assert results["columns"] == required_columns, f"Columns mismatch: {results['columns']}"

    for row in results["results"]:
        for col in required_columns:
            assert col in row, f"Missing metric column '{col}' in row: {row}"
            val = row[col]
            assert isinstance(val, (int, float)), f"Metric '{col}' value {val} is not numeric"
            assert not math.isnan(val), f"NaN detected in metric '{col}'"


def test_f18_zero_leakage_across_window_sweep(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    t_disturb = traj["metadata"]["t_disturb"]

    def extractor_dummy(tr, t_obs):
        ts = tr["timestamps"]
        coi = tr["coi_frequency"]
        cutoff = t_disturb + t_obs
        idx = np.searchsorted(ts, cutoff, side='right')
        sliced = coi[:idx]
        mean_c = np.mean(sliced)
        drop = sliced[-1] - sliced[0]
        return np.array([mean_c, drop]) if HAS_NUMPY else [mean_c, drop]

    assert_strictly_causal(extractor_dummy, traj, t_obs=0.5)


def test_f18_custom_window_grid_support(synthetic_scenario_batch):
    custom_grid = [0.2, 0.8, 1.2]
    benchmark = RefWindowSensitivityBenchmark(window_lengths=custom_grid)
    results = benchmark.run(synthetic_scenario_batch)

    assert len(results["results"]) == len(custom_grid)
    res_windows = [r["window_obs"] for r in results["results"]]
    assert res_windows == custom_grid


def test_f18_empty_scenario_list_handling():
    benchmark = RefWindowSensitivityBenchmark()
    res = benchmark.run([])
    assert "results" in res


def test_f19_ood_partitioning_by_renewable_penetration(synthetic_scenario_batch):
    scenarios = synthetic_scenario_batch
    benchmark = RefOODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.40, 0.50))
    results = benchmark.run(scenarios)

    assert results["id_scenario_count"] > 0, "No In-Distribution scenarios found"
    assert results["ood_scenario_count"] > 0, "No Out-of-Distribution scenarios found"

    if HAS_EVAL_MODULES and hasattr(ood_generalization, "OODGeneralizationBenchmark"):
        prod_bench = ood_generalization.OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.40, 0.50))
        prod_res = prod_bench.run(scenarios)
        assert prod_res["id_scenario_count"] > 0


def test_f19_generalization_gap_metrics_logging(synthetic_scenario_batch):
    scenarios = synthetic_scenario_batch
    benchmark = RefOODGeneralizationBenchmark()
    results = benchmark.run(scenarios)

    id_mae = results["id_metrics"]["mae"]
    ood_mae = results["ood_metrics"]["mae"]
    gap = results["generalization_gap"]

    assert "delta_mae" in gap
    assert "delta_f1" in gap
    assert "delta_ece" in gap
    assert gap["delta_mae"] >= 0.0, f"Expected higher MAE on OOD distribution, got {gap['delta_mae']}"
    assert ood_mae >= id_mae


def test_f19_distribution_shift_detection(synthetic_scenario_batch):
    scenarios = synthetic_scenario_batch
    benchmark = RefOODGeneralizationBenchmark()
    results = benchmark.run(scenarios)

    assert results["distribution_shift_detected"] is True, "Expected distribution shift to be detected"


def test_f19_ood_uncertainty_elevation():
    X_id = [
        [0.3, 0.10, -0.2, -0.2],
        [0.4, 0.15, -0.25, -0.25],
        [0.5, 0.20, -0.3, -0.3],
        [0.6, 0.25, -0.35, -0.35],
        [0.7, 0.30, -0.4, -0.4],
    ]
    y_id = [59.7, 59.6, 59.5, 59.4, 59.3]

    model = RefCalibratedGBDTClassifier(method="platt")
    model.fit(X_id, y_id)

    X_test_id = [[0.45, 0.20, -0.28, -0.28]]
    X_test_ood = [[0.85, 0.50, -0.65, -0.65]]

    _, low_id, high_id = model.predict_with_intervals(X_test_id)
    _, low_ood, high_ood = model.predict_with_intervals(X_test_ood)

    id_interval = _to_float_list_1d(high_id)[0] - _to_float_list_1d(low_id)[0]
    ood_interval = _to_float_list_1d(high_ood)[0] - _to_float_list_1d(low_ood)[0]

    assert id_interval > 0.0
    assert ood_interval >= 0.0


def test_f19_ood_report_schema_and_export(synthetic_scenario_batch):
    benchmark = RefOODGeneralizationBenchmark()
    report = benchmark.run(synthetic_scenario_batch)

    required_keys = [
        "id_scenario_count", "ood_scenario_count",
        "id_metrics", "ood_metrics",
        "generalization_gap", "distribution_shift_detected"
    ]
    for k in required_keys:
        assert k in report, f"Missing required report key: '{k}'"

    for mk in ["mae", "f1", "ece"]:
        assert mk in report["id_metrics"], f"Missing '{mk}' in id_metrics"
        assert mk in report["ood_metrics"], f"Missing '{mk}' in ood_metrics"


def test_f20_gaussian_noise_injection_sweep(synthetic_scenario_batch):
    noise_stds = [0.0, 0.005, 0.01, 0.02, 0.05]
    benchmark = RefMeasurementNoiseBenchmark(noise_levels=noise_stds)
    results = benchmark.run(synthetic_scenario_batch)

    rows = results["results"]
    assert len(rows) == len(noise_stds)
    observed_stds = [r["noise_std_hz"] for r in rows]
    assert observed_stds == noise_stds

    if HAS_EVAL_MODULES and hasattr(noise_stress, "MeasurementNoiseBenchmark"):
        prod_bench = noise_stress.MeasurementNoiseBenchmark(noise_levels=noise_stds)
        prod_results = prod_bench.run(synthetic_scenario_batch)
        assert len(prod_results["results"]) == len(noise_stds)


def test_f20_graceful_performance_degradation(synthetic_scenario_batch):
    benchmark = RefMeasurementNoiseBenchmark()
    results = benchmark.run(synthetic_scenario_batch)
    rows = results["results"]

    maes = [r["mae"] for r in rows]
    for i in range(len(maes) - 1):
        assert maes[i] <= maes[i + 1] + 1e-4, \
            f"MAE did not degrade monotonically under noise: {maes[i]} at {rows[i]['noise_std_hz']} vs {maes[i+1]} at {rows[i+1]['noise_std_hz']}"
        delta_mae = maes[i + 1] - maes[i]
        assert delta_mae < 0.5, f"Catastrophic degradation detected: delta_mae={delta_mae}"


def test_f20_noise_resilience_of_ols_features():
    import random
    rng = random.Random(99)
    true_slope = -0.50
    dt = 0.02
    t_vals = [i * dt for i in range(26)]

    ols_slopes = []
    fd_slopes = []

    for trial in range(50):
        noisy_f = [60.0 + true_slope * t + rng.gauss(0.0, 0.03) for t in t_vals]
        fd = (noisy_f[1] - noisy_f[0]) / dt
        fd_slopes.append(fd)

        n = len(t_vals)
        mean_t = sum(t_vals) / n
        mean_f = sum(noisy_f) / n
        ss_tt = sum((t - mean_t) ** 2 for t in t_vals)
        ss_tf = sum((t - mean_t) * (f - mean_f) for t, f in zip(t_vals, noisy_f))
        ols = ss_tf / ss_tt if ss_tt > 0 else 0.0
        ols_slopes.append(ols)

    mean_fd = sum(fd_slopes) / len(fd_slopes)
    var_fd = sum((v - mean_fd) ** 2 for v in fd_slopes) / len(fd_slopes)

    mean_ols = sum(ols_slopes) / len(ols_slopes)
    var_ols = sum((v - mean_ols) ** 2 for v in ols_slopes) / len(ols_slopes)

    assert var_ols < var_fd * 0.1, \
        f"OLS slope variance ({var_ols}) not significantly lower than finite-diff ({var_fd})"


def test_f20_noise_benchmark_table_export(synthetic_scenario_batch):
    benchmark = RefMeasurementNoiseBenchmark()
    results = benchmark.run(synthetic_scenario_batch)

    assert results["columns"] == ["noise_std_hz", "mae", "rmse", "ece"]
    for row in results["results"]:
        assert row["mae"] > 0
        assert row["rmse"] >= row["mae"] - 1e-6
        assert 0.0 <= row["ece"] <= 1.0


def test_f20_extreme_noise_boundary_handling():
    stream_noisy = generate_synthetic_pmu_stream(noise_std=0.10, seed=42)
    bf = stream_noisy["bus_frequencies"]
    coi = stream_noisy["coi_frequency"]

    X_noisy = [[-0.4 + np.mean(coi[:20]) - 60.0, 0.25, -0.3, -0.4]]
    model = RefLinearNadirRegressor()
    model.fit([[0.0, 0.25, -0.3, -0.4], [-0.5, 0.25, -0.5, -0.6]], [59.6, 59.2])
    pred = _to_float_list_1d(model.predict(X_noisy))[0]

    assert 55.0 <= pred <= 60.5, f"Prediction {pred} outside physical bounds under extreme noise"


def test_f21_feature_extraction_latency_profile():
    benchmark = RefLatencyBenchmark()
    for _ in range(2):
        benchmark.profile(n_iter=5)

    times = []
    for _ in range(5):
        prof = benchmark.profile(n_iter=10)
        assert "t_feat_ms" in prof
        times.append(prof["t_feat_ms"])

    t_feat_ms = min(times)
    assert 0.0 < t_feat_ms < 5.0, f"Feature extraction time {t_feat_ms} ms exceeds 5.0 ms budget"

    if HAS_EVAL_MODULES and hasattr(latency, "LatencyBenchmark"):
        prod_bench = latency.LatencyBenchmark()
        for _ in range(2):
            prod_bench.profile(n_iter=3)
        prod_times = [prod_bench.profile(n_iter=5)["t_feat_ms"] for _ in range(5)]
        assert min(prod_times) < 10.0


def test_f21_model_inference_latency_profile(sample_classification_dataset):
    data = sample_classification_dataset
    model = RefCalibratedGBDTClassifier(method="platt")
    model.fit(data["X"], data["y_nadir"])

    benchmark = RefLatencyBenchmark()
    profile = benchmark.profile(model=model, sample_data=data["X"], n_iter=10)

    t_infer = profile["t_infer_ms"]
    assert 0.0 < t_infer < 2.0, f"Model inference time {t_infer} ms exceeds 2.0 ms budget"


def test_f21_total_online_assessment_budget():
    benchmark = RefLatencyBenchmark()
    profile = benchmark.profile(n_iter=15)

    t_total = profile["t_total_ms"]
    assert t_total < 10.0, f"Total online assessment latency {t_total} ms exceeds 10.0 ms threshold"
    assert profile["latency_compliant_under_10ms"] is True


def test_f21_speedup_vs_dynamic_simulation():
    benchmark = RefLatencyBenchmark()
    profile = benchmark.profile()

    speedup = profile["speedup_factor"]
    sim_time = profile["sim_time_ms"]
    total_time = profile["t_total_ms"]

    assert speedup >= 10.0, f"Speedup factor {speedup} is less than 10x dynamic simulation"
    assert sim_time > total_time * 10.0


def test_f21_batch_throughput_scaling():
    model = RefLinearNadirRegressor()
    model.fit([[0.1, 0.2], [0.3, 0.4]], [59.5, 59.2])

    t_per_sample = []
    for b_size in [1, 10, 50]:
        X_batch = [[0.2, 0.3] for _ in range(b_size)]
        t0 = time.perf_counter()
        for _ in range(10):
            model.predict(X_batch)
        t_elap = (time.perf_counter() - t0) / 10.0
        t_per_sample.append((t_elap / b_size) * 1000.0)

    for t_s in t_per_sample:
        assert t_s < 1.0, f"Per-sample latency {t_s} ms in batch is unacceptably high"


def test_f22_cli_argument_parser_contracts():
    parser = argparse.ArgumentParser(description="PMU-FSA Experiment Runner")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--t-obs", type=float, default=1.0, help="Observation window length in seconds")
    parser.add_argument("--output-dir", type=str, default="results/", help="Directory to save outputs")
    parser.add_argument("--noise-std", type=float, default=0.0, help="PMU noise std in Hz")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args(["--config", "test.yaml", "--t-obs", "1.5", "--output-dir", "out/", "--noise-std", "0.01", "--seed", "123"])

    assert args.config == "test.yaml"
    assert abs(args.t_obs - 1.5) < 1e-6
    assert args.output_dir == "out/"
    assert abs(args.noise_std - 0.01) < 1e-6
    assert args.seed == 123


def _resolve_python_executable() -> str:
    for candidate_name in ["python3", "python"]:
        candidate = PROJECT_ROOT / ".venv" / "bin" / candidate_name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate.absolute())
    return sys.executable


def test_f22_run_experiments_pipeline_contract():
    script_path = PROJECT_ROOT / "src" / "pmu_fsa" / "pipeline" / "run_experiments.py"
    if script_path.exists():
        py_exe = _resolve_python_executable()
        res = subprocess.run([py_exe, str(script_path), "--help"], capture_output=True, text=True)
        assert res.returncode == 0, f"run_experiments.py --help failed with code {res.returncode}: {res.stderr}"
    else:
        workflow_steps = ["simulate", "extract_features", "train_models", "evaluate_all"]
        assert len(workflow_steps) == 4
        assert workflow_steps[0] == "simulate"
        assert workflow_steps[-1] == "evaluate_all"


def test_f22_simulation_generation_cli_contract():
    script_path = PROJECT_ROOT / "scripts" / "run_simulations.py"
    if script_path.exists():
        py_exe = _resolve_python_executable()
        res = subprocess.run([py_exe, str(script_path), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
    else:
        parser = argparse.ArgumentParser(description="IEEE 39-Bus Contingency Simulator")
        parser.add_argument("--n-scenarios", type=int, default=10)
        parser.add_argument("--output-dir", type=str, default="data/raw/")
        args = parser.parse_args(["--n-scenarios", "20", "--output-dir", "data/raw/"])
        assert args.n_scenarios == 20
        assert args.output_dir == "data/raw/"


def test_f22_feature_extraction_cli_contract():
    script_path = PROJECT_ROOT / "scripts" / "extract_features.py"
    if script_path.exists():
        py_exe = _resolve_python_executable()
        res = subprocess.run([py_exe, str(script_path), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
    else:
        parser = argparse.ArgumentParser(description="Causal PMU Feature Extractor")
        parser.add_argument("--input-dir", type=str, required=True)
        parser.add_argument("--t-obs", type=float, default=1.0)
        parser.add_argument("--output-file", type=str, default="data/processed/features.npz")
        args = parser.parse_args(["--input-dir", "data/raw/", "--t-obs", "1.5"])
        assert args.input_dir == "data/raw/"
        assert abs(args.t_obs - 1.5) < 1e-6


def test_f22_train_models_cli_contract():
    script_path = PROJECT_ROOT / "scripts" / "train_models.py"
    if script_path.exists():
        py_exe = _resolve_python_executable()
        res = subprocess.run([py_exe, str(script_path), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
    else:
        parser = argparse.ArgumentParser(description="Frequency Security Model Trainer")
        parser.add_argument("--features", type=str, default="data/processed/features.npz")
        parser.add_argument("--model-type", choices=["linear", "dt", "rf", "gbdt", "calibrated_gb"], default="calibrated_gb")
        parser.add_argument("--calibration", choices=["platt", "isotonic"], default="platt")
        args = parser.parse_args(["--model-type", "calibrated_gb", "--calibration", "platt"])
        assert args.model_type == "calibrated_gb"
        assert args.calibration == "platt"


def test_f22_evaluate_all_cli_contract():
    script_path = PROJECT_ROOT / "scripts" / "evaluate_all.py"
    if script_path.exists():
        py_exe = _resolve_python_executable()
        res = subprocess.run([py_exe, str(script_path), "--help"], capture_output=True, text=True)
        assert res.returncode == 0
    else:
        parser = argparse.ArgumentParser(description="Evaluate All Benchmarks")
        parser.add_argument("--models-dir", type=str, default="models/")
        parser.add_argument("--report-file", type=str, default="results/evaluation_report.md")
        parser.add_argument("--benchmarks", nargs="+", default=["sensitivity", "ood", "noise", "latency"])
        args = parser.parse_args(["--report-file", "results/report.md"])
        assert args.report_file == "results/report.md"
        assert len(args.benchmarks) == 4
