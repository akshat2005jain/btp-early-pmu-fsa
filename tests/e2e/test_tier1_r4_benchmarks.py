"""End-to-end Tier 1 tests for unsmoothed 50 Hz evaluation benchmarks, latency, interpretability, stats, and CLIs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from pmu_fsa.evaluation.interpretability import InterpretabilityAnalyzer
from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import WindowSensitivityBenchmark
from pmu_fsa.evaluation.statistical_tests import (
    paired_bootstrap_ci,
    paired_model_comparison_test,
)
from pmu_fsa.features.extractor import CANONICAL_FEATURE_NAMES
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.regressors import GradientBoostingNadirRegressor, LinearNadirRegressor
from tests.e2e.conftest import assert_strictly_causal
from tests.e2e.test_tier1_r2_features import extract_full_reference_feature_vector

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_python_executable() -> str:
    for candidate_name in ("python3", "python"):
        candidate = PROJECT_ROOT / ".venv" / "bin" / candidate_name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            return str(candidate.absolute())
    return sys.executable


def test_f18_window_sensitivity_held_out_unsmoothed_contract(synthetic_scenario_batch):
    bench = WindowSensitivityBenchmark(window_lengths=[0.5, 1.0, 1.5, 2.0], threshold=49.50)
    res = bench.run(synthetic_scenario_batch)

    assert res["columns"] == ["window_obs", "mae", "rmse", "r2", "f1", "ece"]
    assert res["n_train"] == 6
    assert res["n_val"] == 4
    assert res["n_test"] == 4
    assert res["n_ood_test"] == 6
    assert len(res["results"]) == 4

    for row in res["results"]:
        assert row["mae"] == row["mae_raw"]
        assert row["rmse"] == row["rmse_raw"]
        assert row["f1"] == row["f1_raw"]
        assert row["rmse"] >= row["mae"] - 1e-9
        assert 0.0 <= row["f1"] <= 1.0
        assert 0.0 <= row["ece"] <= 1.0

    assert bench.run([])["results"] == []


def test_f18_custom_window_grid_and_causality(sample_pmu_trajectory, synthetic_scenario_batch):
    custom_grid = [0.4, 0.8, 1.2]
    res = WindowSensitivityBenchmark(window_lengths=custom_grid, threshold=49.50).run(synthetic_scenario_batch)
    assert [r["window_obs"] for r in res["results"]] == custom_grid
    assert_strictly_causal(extract_full_reference_feature_vector, sample_pmu_trajectory, t_obs=0.5)


def test_f19_ood_generalization_unclamped_gaps_and_breakdown(synthetic_scenario_batch):
    bench = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.40, 0.50), threshold=49.50)
    report = bench.run(synthetic_scenario_batch)

    assert report["id_train_count"] == 6
    assert report["id_val_count"] == 4
    assert report["id_test_count"] == 4
    assert report["ood_scenario_count"] == 6
    assert report["distribution_shift_detected"] is True
    assert {"id_test", "ood_res", "ood_load", "ood_combined", "ood_all"}.issubset(report["regime_breakdown"].keys())

    gap = report["generalization_gap"]
    assert gap["delta_mae"] == pytest.approx(report["ood_metrics"]["mae"] - report["id_metrics"]["mae"], abs=1e-3)
    assert gap["delta_ece"] == pytest.approx(report["ood_metrics"]["ece"] - report["id_metrics"]["ece"], abs=1e-3)


def test_f20_measurement_noise_benchmark_unsmoothed_and_ols_resilience(synthetic_scenario_batch):
    noise_stds = [0.0, 0.005, 0.01, 0.02, 0.05]
    bench = MeasurementNoiseBenchmark(noise_levels=noise_stds, threshold=49.50, seed=42)
    res = bench.run(synthetic_scenario_batch)

    assert res["columns"] == ["noise_std_hz", "mae", "rmse", "ece"]
    assert [r["noise_std_hz"] for r in res["results"]] == noise_stds
    for row in res["results"]:
        assert row["mae"] == row["mae_raw"]
        assert row["rmse"] >= row["mae"] - 1e-9
        assert 0.0 <= row["ece"] <= 1.0

    comp = MeasurementNoiseBenchmark.compare_ols_vs_endpoint_rocof(noise_std=0.02, n_trials=120, seed=42)
    assert comp["noise_resilience_verified"] is True
    assert comp["variance_ratio"] < 0.25


def test_f21_six_stage_hardware_latency_profiler(real_pmu_trajectory, sample_classification_dataset):
    model = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=15)
    model.fit(sample_classification_dataset["X"], sample_classification_dataset["y_nadir"])

    lb = LatencyBenchmark()
    prof = lb.profile(
        classifier=model,
        scenarios=[real_pmu_trajectory],
        n_iter=12,
        warmup_reps=2,
        measure_simulation=True,
        n_sim_runs=2,
    )

    for stage_key in (
        "t_slice_ms",
        "t_feat_ms",
        "t_reg_ms",
        "t_clf_ms",
        "t_cal_ms",
        "t_infer_ms",
        "t_total_ms",
    ):
        assert prof[stage_key] > 0.0

    assert len(prof["stages"]) == 7
    assert prof["t_total_ms"] <= prof["t_total_p95_ms"] <= prof["t_total_p99_ms"] + 1e-9
    assert prof["sim_measured_on_hardware"] is True
    assert prof["sim_time_ms"] > 5.0
    assert prof["speedup_factor"] > 5.0
    assert prof["t_total_ms"] < 10.0

    reg = LinearNadirRegressor().fit(sample_classification_dataset["X"], sample_classification_dataset["y_nadir"])
    scaling = lb.profile_batch_scaling(model=reg, batch_sizes=[1, 10, 50], n_features=14, n_iter=10)
    for b_size in (1, 10, 50):
        assert scaling["batch_scaling"][b_size]["per_sample_ms"] < 2.0


def test_f21b_interpretability_and_paired_statistical_tests(synthetic_scenario_batch, sample_classification_dataset):
    X = pd.DataFrame(sample_classification_dataset["X"], columns=CANONICAL_FEATURE_NAMES)
    y_nadir = sample_classification_dataset["y_nadir"]

    analyzer = InterpretabilityAnalyzer(t_obs=1.0, threshold=49.50, f_nominal=50.0, seed=42)
    reg = GradientBoostingNadirRegressor(n_estimators=20, max_depth=3, random_state=42).fit(X.iloc[:36], y_nadir[:36])
    perm = analyzer.compute_permutation_importance(reg, X.iloc[36:], y_nadir[36:], task="regression", n_repeats=4)
    assert len(perm) == 14

    ablations = analyzer.run_feature_group_ablations(synthetic_scenario_batch)
    assert len(ablations["results"]) == 7

    ci = paired_bootstrap_ci(y_nadir, y_nadir + 0.04, metric="mae", n_boot=200, seed=42)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"]

    rng = np.random.default_rng(7)
    comp = paired_model_comparison_test(
        y_nadir,
        y_nadir + rng.normal(0.0, 0.01, size=len(y_nadir)),
        y_nadir + rng.normal(0.0, 0.25, size=len(y_nadir)),
        task="regression",
        n_boot=200,
        seed=7,
    )
    assert comp["p_value_wilcoxon"] < 0.01


def test_f22_cli_entrypoints_help_and_isolated_execution(tmp_path: Path):
    py_exe = _resolve_python_executable()
    scripts = [
        PROJECT_ROOT / "src" / "pmu_fsa" / "pipeline" / "run_experiments.py",
        PROJECT_ROOT / "scripts" / "run_simulations.py",
        PROJECT_ROOT / "scripts" / "extract_features.py",
        PROJECT_ROOT / "scripts" / "train_models.py",
        PROJECT_ROOT / "scripts" / "evaluate_all.py",
    ]
    for script in scripts:
        assert script.exists(), f"Expected production CLI script {script}"
        res = subprocess.run([py_exe, str(script), "--help"], capture_output=True, text=True)
        assert res.returncode == 0, f"{script.name} --help failed: {res.stderr}"

    report_file = tmp_path / "results" / "eval_report.md"
    res_iso = subprocess.run(
        [
            py_exe,
            str(PROJECT_ROOT / "scripts" / "evaluate_all.py"),
            "--data-dir",
            str(tmp_path / "empty_raw"),
            "--models-dir",
            str(tmp_path / "models"),
            "--tables-dir",
            str(tmp_path / "results" / "tables"),
            "--figures-dir",
            str(tmp_path / "results" / "figures"),
            "--output-dir",
            str(tmp_path / "results"),
            "--report-file",
            str(report_file),
            "--benchmarks",
            "sensitivity",
            "--quiet",
        ],
        capture_output=True,
        text=True,
    )
    assert res_iso.returncode == 0, res_iso.stderr
    assert report_file.exists()
