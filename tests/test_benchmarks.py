"""Tests for leak-free 50 Hz benchmarks, statistical tests, interpretability, and isolated CLIs."""

from __future__ import annotations

import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from pmu_fsa.evaluation.interpretability import InterpretabilityAnalyzer
from pmu_fsa.evaluation.latency import LatencyBenchmark
from pmu_fsa.evaluation.noise_stress import MeasurementNoiseBenchmark
from pmu_fsa.evaluation.ood_generalization import OODGeneralizationBenchmark
from pmu_fsa.evaluation.sensitivity import (
    WindowSensitivityBenchmark,
    split_scenarios_by_partition,
)
from pmu_fsa.evaluation.statistical_tests import (
    LearningCurveEvaluator,
    MultiSeedStabilityEvaluator,
    compare_all_models_statistical,
    paired_bootstrap_ci,
    paired_model_comparison_test,
)
from pmu_fsa.features.extractor import CANONICAL_FEATURE_NAMES, FeatureExtractor
from pmu_fsa.models.calibrated_gb import CalibratedGBDTClassifier
from pmu_fsa.models.regressors import GradientBoostingNadirRegressor
from pmu_fsa.pipeline.run_experiments import (
    ExperimentConfig,
    ExperimentPipeline,
    WORKFLOW_STEPS,
)
from pmu_fsa.simulation.scenario_generator import export_scenario_to_npz

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def generate_fast_50hz_pmu_scenario(
    scenario_id: str,
    partition: str,
    ood_type: str,
    renewable_penetration: float,
    load_scale: float,
    loss_mw: float,
    n_samples: int = 251,
    fps: int = 50,
    t_disturb: float = 1.0,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / float(fps)
    timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)

    f_nominal = 50.0
    t_post = np.maximum(0.0, timestamps - t_disturb)
    tau_nadir = 2.35  # t_nadir ~ 3.35s (> 1.0 + 2.0s so all windows up to 2.0s are pre-nadir causal)
    severity = (loss_mw / 600.0) * (1.0 + 0.70 * renewable_penetration) * load_scale
    drop_amp = 0.44 * severity + float(rng.uniform(-0.015, 0.015))
    surge = np.where(
        timestamps >= t_disturb,
        (t_post / tau_nadir) * np.exp(1.0 - t_post / tau_nadir),
        0.0,
    )
    f_sys = f_nominal - drop_amp * surge

    spatial_profile = np.linspace(-0.025, 0.025, 39) * severity
    bus_frequencies = f_sys[:, None] + np.outer(surge, spatial_profile) + rng.normal(0.0, 0.001, size=(n_samples, 39))
    bus_rocof = np.gradient(bus_frequencies, dt, axis=0)
    voltage_magnitudes = np.ones((n_samples, 39), dtype=float) - 0.02 * surge[:, None]
    voltage_angles = np.outer(t_post, np.linspace(-3.5, 3.5, 39) * severity)

    post_mask = timestamps >= t_disturb
    f_post = f_sys[post_mask]
    t_post_arr = timestamps[post_mask]
    nadir_idx = int(np.argmin(f_post))
    nadir_val = float(f_post[nadir_idx])
    t_nadir = float(t_post_arr[nadir_idx])
    is_secure = 1 if nadir_val >= 49.50 else 0

    return {
        "timestamps": timestamps,
        "bus_frequencies": bus_frequencies,
        "bus_rocof": bus_rocof,
        "voltage_magnitudes": voltage_magnitudes,
        "voltage_angles": voltage_angles,
        "pmu_sys_frequency": f_sys.copy(),
        "coi_frequency": f_sys.copy(),
        "metadata": {
            "scenario_id": scenario_id,
            "partition": partition,
            "ood_type": ood_type,
            "disturbed_gen": 30 + (seed % 10),
            "loss_fraction": round(min(1.0, loss_mw / 800.0), 2),
            "active_loss_mw": float(loss_mw),
            "renewable_penetration": float(renewable_penetration),
            "p_ren_requested": float(renewable_penetration),
            "p_ren_realized": float(renewable_penetration),
            "load_scale": float(load_scale),
            "t_disturb": float(t_disturb),
            "t_sim": float(timestamps[-1]),
            "fps": fps,
            "f_nominal": 50.0,
            "f_threshold_hz": 49.50,
            "ground_truth_nadir": round(nadir_val, 5),
            "ground_truth_t_nadir": round(t_nadir, 4),
            "bus_min_nadir": round(float(np.min(bus_frequencies[post_mask])), 5),
            "bus_min_t_nadir": round(t_nadir, 4),
            "is_secure": is_secure,
        },
    }


@pytest.fixture(scope="module")
def partitioned_50hz_scenario_batch() -> List[Dict[str, Any]]:
    specs = [
        # ID Train (8 scenarios: 4 secure, 4 insecure)
        ("scen_tr_01", "train", "id", 0.10, 0.90, 260.0),
        ("scen_tr_02", "train", "id", 0.15, 1.00, 320.0),
        ("scen_tr_03", "train", "id", 0.20, 0.95, 380.0),
        ("scen_tr_04", "train", "id", 0.25, 1.00, 420.0),
        ("scen_tr_05", "train", "id", 0.20, 1.05, 680.0),
        ("scen_tr_06", "train", "id", 0.25, 1.10, 750.0),
        ("scen_tr_07", "train", "id", 0.30, 1.00, 820.0),
        ("scen_tr_08", "train", "id", 0.30, 1.10, 900.0),
        # ID Validation (4 scenarios: 2 secure, 2 insecure)
        ("scen_va_01", "val", "id", 0.10, 1.00, 290.0),
        ("scen_va_02", "val", "id", 0.20, 0.90, 360.0),
        ("scen_va_03", "val", "id", 0.25, 1.05, 720.0),
        ("scen_va_04", "val", "id", 0.30, 1.10, 860.0),
        # ID Held-Out Test (6 scenarios: 3 secure, 3 insecure)
        ("scen_te_01", "test", "id", 0.10, 0.95, 280.0),
        ("scen_te_02", "test", "id", 0.20, 1.00, 350.0),
        ("scen_te_03", "test", "id", 0.25, 0.90, 410.0),
        ("scen_te_04", "test", "id", 0.20, 1.10, 700.0),
        ("scen_te_05", "test", "id", 0.30, 1.00, 790.0),
        ("scen_te_06", "test", "id", 0.30, 1.10, 910.0),
        # OOD Test (6 scenarios across ood_res, ood_load, ood_combined)
        ("scen_ood_01", "ood_test", "ood_res", 0.40, 1.00, 300.0),
        ("scen_ood_02", "ood_test", "ood_res", 0.50, 1.00, 780.0),
        ("scen_ood_03", "ood_test", "ood_load", 0.20, 0.80, 340.0),
        ("scen_ood_04", "ood_test", "ood_load", 0.25, 1.20, 760.0),
        ("scen_ood_05", "ood_test", "ood_combined", 0.45, 0.80, 330.0),
        ("scen_ood_06", "ood_test", "ood_combined", 0.50, 1.20, 850.0),
    ]
    return [
        generate_fast_50hz_pmu_scenario(
            scenario_id=sid,
            partition=part,
            ood_type=ood_t,
            renewable_penetration=pen,
            load_scale=ld,
            loss_mw=mw,
            seed=100 + idx,
        )
        for idx, (sid, part, ood_t, pen, ld, mw) in enumerate(specs)
    ]


class TestPartitionIsolationAndFixtures:

    def test_fixture_50hz_contract_and_speed(self, partitioned_50hz_scenario_batch):
        t0 = time.perf_counter()
        _ = [
            generate_fast_50hz_pmu_scenario(f"s_{i}", "train", "id", 0.20, 1.0, 500.0, seed=i)
            for i in range(12)
        ]
        assert (time.perf_counter() - t0) * 1000.0 < 120.0

        for s in partitioned_50hz_scenario_batch:
            assert 45.0 <= s["metadata"]["ground_truth_nadir"] <= 50.0
            assert s["metadata"]["f_nominal"] == 50.0
            assert s["metadata"]["f_threshold_hz"] == 49.50
            assert s["metadata"]["ground_truth_t_nadir"] > 3.05

    def test_split_scenarios_by_partition_strict_disjointness(self, partitioned_50hz_scenario_batch):
        tr, va, te, ood, proto = split_scenarios_by_partition(partitioned_50hz_scenario_batch)
        assert proto == "metadata_partitions"
        assert (len(tr), len(va), len(te), len(ood)) == (8, 4, 6, 6)

        id_tr = {s["metadata"]["scenario_id"] for s in tr}
        id_va = {s["metadata"]["scenario_id"] for s in va}
        id_te = {s["metadata"]["scenario_id"] for s in te}
        id_ood = {s["metadata"]["scenario_id"] for s in ood}
        assert id_tr.isdisjoint(id_va)
        assert id_tr.isdisjoint(id_te)
        assert id_tr.isdisjoint(id_ood)
        assert id_va.isdisjoint(id_te)
        assert id_te.isdisjoint(id_ood)

        # Fallback mode when no explicit "test" exists still isolates ood_test completely
        no_test_batch = [s for s in partitioned_50hz_scenario_batch if s["metadata"]["partition"] in ("train", "ood_test")]
        tr_fb, va_fb, te_fb, ood_fb, proto_fb = split_scenarios_by_partition(no_test_batch)
        assert "fallback" in proto_fb
        assert len(ood_fb) == 6
        fb_id_union = {s["metadata"]["scenario_id"] for s in tr_fb + va_fb + te_fb}
        assert fb_id_union.isdisjoint(id_ood)


class TestWindowSensitivityBenchmark:

    def test_held_out_and_ood_sweep_authentic_unsmoothed_metrics(self, partitioned_50hz_scenario_batch):
        bench = WindowSensitivityBenchmark(window_lengths=[2.0, 0.5, 1.5, 1.0])
        assert bench.window_lengths == [0.5, 1.0, 1.5, 2.0]
        assert bench.threshold == 49.50
        assert bench.f_nominal == 50.0

        res = bench.run(partitioned_50hz_scenario_batch)
        assert res["partition_protocol"] == "metadata_partitions"
        assert (res["n_train"], res["n_val"], res["n_test"], res["n_ood_test"]) == (8, 4, 6, 6)
        assert res["columns"] == ["window_obs", "mae", "rmse", "r2", "f1", "ece"]
        assert len(res["results"]) == 4
        assert len(res["ood_results"]) == 4

        lead_times = [r["t_lead"] for r in res["results"]]
        for i in range(len(lead_times) - 1):
            assert lead_times[i] > lead_times[i + 1] > 0.0

        for r in res["results"] + res["ood_results"]:
            assert r["n_causal"] == r["n_total"]
            assert r["rmse"] >= r["mae"] - 1e-6
            # Authentic unsmoothed reporting: displayed metric equals raw metric
            assert r["mae"] == r["mae_raw"]
            assert r["rmse"] == r["rmse_raw"]
            assert r["f1"] == r["f1_raw"]

        df_id = bench.to_dataframe(res, split="id_test")
        df_ood = bench.to_dataframe(res, split="ood_test")
        assert len(df_id) == 4 and len(df_ood) == 4
        assert "window_obs" in bench.to_markdown(res)

    def test_empty_and_invalid_windows(self):
        with pytest.raises(ValueError, match="strictly positive"):
            WindowSensitivityBenchmark(window_lengths=[0.5, 0.0])
        res = WindowSensitivityBenchmark().run([])
        assert res["results"] == [] and res["ood_results"] == []


class TestOODGeneralizationBenchmark:

    def test_held_out_id_vs_multi_regime_ood_and_unclamped_gap(self, partitioned_50hz_scenario_batch):
        bench = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.40, 0.50))
        res = bench.run(partitioned_50hz_scenario_batch)

        assert res["id_scenario_count"] == 18
        assert res["id_train_count"] == 8
        assert res["id_val_count"] == 4
        assert res["id_test_count"] == 6
        assert res["ood_scenario_count"] == 6
        assert (res["ood_res_count"], res["ood_load_count"], res["ood_combined_count"]) == (2, 2, 2)
        assert res["distribution_shift_detected"] is True

        rb = res["regime_breakdown"]
        for key in ("id_test", "ood_res", "ood_load", "ood_combined", "ood_all"):
            assert key in rb
            assert rb[key]["n_scenarios"] > 0

        expected_delta_mae = round(res["ood_metrics"]["mae"] - res["id_metrics"]["mae"], 4)
        assert res["delta_mae"] == pytest.approx(expected_delta_mae, abs=1e-6)
        assert res["generalization_gap"]["delta_mae"] == pytest.approx(expected_delta_mae, abs=1e-6)

        unc = res["uncertainty_elevation"]
        assert unc["mean_nadir_interval_hz_ood"] > 0.0
        assert unc["mean_nadir_interval_hz_id"] > 0.0

        df_reg = bench.to_regime_dataframe(res)
        df_sum = bench.to_dataframe(res)
        assert len(df_reg) == 5
        assert len(df_sum) >= 8
        assert "Generalization Gap" in bench.to_markdown(res)

    def test_ood_gap_never_artificially_floored_at_zero(self, partitioned_50hz_scenario_batch):
        # Pass a mock regressor whose predictions match OOD targets exactly while having 0.15 Hz error on ID test
        class _ExactNear493Regressor:
            n_features_in_ = 14

            def predict(self, X):
                return np.full(len(X), 49.40)

            def evaluate(self, X, y):
                err = np.abs(y - 49.40)
                return {
                    "mae": float(np.mean(err)),
                    "rmse": float(np.sqrt(np.mean(err ** 2))),
                    "bias": float(np.mean(49.40 - y)),
                    "median_ae": float(np.median(err)),
                    "p95_ae": float(np.percentile(err, 95)),
                    "r2": 0.0,
                }

        custom_batch = []
        for s in partitioned_50hz_scenario_batch:
            sc = dict(s)
            sc["metadata"] = dict(s["metadata"])
            if sc["metadata"]["partition"] == "test":
                sc["metadata"]["ground_truth_nadir"] = 49.80
                sc["metadata"]["is_secure"] = 1
            elif sc["metadata"]["partition"] == "ood_test":
                sc["metadata"]["ground_truth_nadir"] = 49.40
                sc["metadata"]["is_secure"] = 0
            custom_batch.append(sc)

        bench = OODGeneralizationBenchmark()
        res = bench.run(custom_batch, regressor=_ExactNear493Regressor())
        assert res["ood_metrics"]["mae"] < res["id_metrics"]["mae"]
        assert res["delta_mae"] < 0.0, "OOD benchmark clamped negative delta_mae!"

    def test_overlapping_and_identical_ranges(self, partitioned_50hz_scenario_batch):
        with pytest.raises(ValueError, match="overlaps"):
            OODGeneralizationBenchmark(id_range=(0.10, 0.35), ood_range=(0.30, 0.50))

        same_bench = OODGeneralizationBenchmark(id_range=(0.10, 0.30), ood_range=(0.10, 0.30))
        res_same = same_bench.run(partitioned_50hz_scenario_batch)
        assert res_same["delta_mae"] == 0.0
        assert res_same["distribution_shift_detected"] is False


class TestMeasurementNoiseBenchmark:

    def test_14_feature_noise_sweep_and_ols_resilience(self, partitioned_50hz_scenario_batch):
        levels = [0.0, 0.005, 0.01, 0.02, 0.05]
        bench = MeasurementNoiseBenchmark(noise_levels=[0.05, 0.0, 0.02, 0.005, 0.01])
        assert bench.noise_levels == levels

        res = bench.run(partitioned_50hz_scenario_batch)
        assert res["n_test"] == 6
        assert res["feature_names"] == CANONICAL_FEATURE_NAMES
        assert len(res["results"]) == 5

        assert res["results"][0]["prob_stability_mad"] == 0.0
        for r in res["results"]:
            assert r["n_features_used"] == 14
            assert r["mae"] == r["mae_raw"]
            assert r["rmse"] >= r["mae"] - 1e-6
            assert 0.0 <= r["ece"] <= 1.0

        cmp = MeasurementNoiseBenchmark.compare_ols_vs_endpoint_rocof(noise_std=0.03, n_trials=60, seed=99)
        assert cmp["noise_resilience_verified"] is True
        assert cmp["variance_ratio"] < 0.15

        with pytest.raises(ValueError, match="non-negative"):
            MeasurementNoiseBenchmark(noise_levels=[-0.01])


class TestLatencyInterpretabilityAndStatisticalTests:

    def test_6_stage_latency_profiler_and_real_ode_measurement(self, partitioned_50hz_scenario_batch):
        bench = LatencyBenchmark()
        prof_fast = bench.profile(
            scenarios=partitioned_50hz_scenario_batch[:4],
            n_iter=10,
            warmup_reps=2,
            sim_time_ms=120.0,
            measure_simulation=False,
        )
        expected_stages = {
            "stage_1_window_slicing",
            "stage_2_feature_extraction_14",
            "stage_3_regression_inference",
            "stage_4_classification_inference",
            "stage_5_probability_calibration",
            "stage_6_end_to_end_online",
            "ode_dynamic_simulation",
        }
        assert set(prof_fast["stages"].keys()) == expected_stages
        assert prof_fast["t_total_ms"] < 10.0
        expected_compliant = bool(prof_fast["t_total_ms"] < 10.0 and prof_fast["t_total_p99_ms"] < 10.0)
        assert prof_fast["latency_compliant_under_10ms"] is expected_compliant
        assert len(bench.to_stages_dataframe(prof_fast)) == 7
        assert "Computational Speedup" in bench.to_markdown(prof_fast)

        # Genuine hardware ODE simulation measurement (2 runs for fast test execution)
        prof_hw = bench.profile(
            n_iter=12,
            warmup_reps=2,
            measure_simulation=True,
            n_sim_runs=2,
        )
        assert prof_hw["sim_measured_on_hardware"] is True
        assert prof_hw["sim_time_ms"] > 5.0
        assert prof_hw["speedup_factor"] > 5.0

        batch_res = bench.profile_batch_scaling(batch_sizes=[1, 10, 50], n_iter=5)
        assert batch_res["batch_scaling"][50]["per_sample_ms"] < batch_res["batch_scaling"][1]["per_sample_ms"]

    def test_interpretability_and_7_feature_observability_ablations(self, partitioned_50hz_scenario_batch):
        analyzer = InterpretabilityAnalyzer(t_obs=1.0, threshold=49.50, f_nominal=50.0, seed=42)
        fe = FeatureExtractor(t_obs=1.0, f_nominal=50.0)
        X_df, y_nadir, y_sec, meta_df = fe.transform_batch(partitioned_50hz_scenario_batch, as_dataframe=True)

        reg = GradientBoostingNadirRegressor(n_estimators=25, max_depth=3, random_state=42).fit(X_df.iloc[:12], y_nadir[:12])
        clf = CalibratedGBDTClassifier(method="platt", threshold=49.50, n_estimators=25, random_state=42).fit(
            X_df.iloc[:12], y_nadir[:12], y_sec[:12]
        )

        perm_reg = analyzer.compute_permutation_importance(reg, X_df.iloc[12:18], y_nadir[12:18], task="regression", n_repeats=5)
        perm_clf = analyzer.compute_permutation_importance(clf, X_df.iloc[12:18], y_sec[12:18], task="classification", n_repeats=5)
        imp_df = analyzer.compute_impurity_importance(reg, feature_names=CANONICAL_FEATURE_NAMES)
        assert len(perm_reg) == 14 and len(perm_clf) == 14 and len(imp_df) == 14
        assert float(imp_df["impurity_importance"].sum()) == pytest.approx(1.0, abs=1e-6)

        corr = analyzer.compute_correlation_matrix(X_df, feature_names=CANONICAL_FEATURE_NAMES)
        assert corr["pearson_df"].shape == (14, 14)
        assert corr["spearman_df"].shape == (14, 14)
        assert corr["matrix_rank"] >= 8

        ablations = analyzer.run_feature_group_ablations(partitioned_50hz_scenario_batch)
        ab_names = [r["ablation"] for r in ablations["results"]]
        assert ab_names == [
            "all_14",
            "freq_only",
            "freq_plus_rocof",
            "no_spatial_angle",
            "spatial_angle_only",
            "partial_13bus_pmu",
            "fps_25_vs_50",
        ]
        for r in ablations["results"]:
            assert math.isfinite(r["id_mae"])
            assert math.isfinite(r["ood_mae"])

        stab = analyzer.compute_multiseed_rank_stability(
            X_df.iloc[:12], y_nadir[:12], X_df.iloc[12:18], y_nadir[12:18], seeds=(42, 43, 44)
        )
        assert len(stab["table"]) == 14
        assert -1.0 <= stab["mean_pairwise_spearman"] <= 1.0

        win_imp = analyzer.compute_importance_across_windows(partitioned_50hz_scenario_batch, windows=(0.5, 1.0))
        assert len(win_imp) == 28

        shifts = analyzer.compute_subgroup_feature_shifts(X_df, y_nadir, y_sec, meta_df)
        assert not shifts.empty

    def test_paired_bootstrap_wilcoxon_learning_curve_and_multiseed(self, partitioned_50hz_scenario_batch):
        rng = np.random.default_rng(42)
        y_true = np.linspace(48.8, 49.9, 30)
        pred_good = y_true + rng.normal(0.0, 0.02, size=30)
        pred_poor = y_true + rng.normal(0.0, 0.25, size=30)

        ci = paired_bootstrap_ci(y_true, pred_good, metric="mae", n_boot=400, seed=42)
        assert 0.0 <= ci["ci_low"] <= ci["estimate"] <= ci["ci_high"] < 0.05

        comp = paired_model_comparison_test(y_true, pred_good, pred_poor, task="regression", n_boot=400, seed=42)
        assert comp["delta_a_minus_b"] < 0.0
        assert comp["p_value_wilcoxon"] < 0.01
        assert comp["p_value_bootstrap"] < 0.01
        assert comp["significant_at_005"] is True

        comp_same = paired_model_comparison_test(y_true, pred_good, pred_good, task="regression", n_boot=100, seed=42)
        assert comp_same["delta_a_minus_b"] == 0.0
        assert comp_same["p_value_wilcoxon"] == 1.0

        df_all = compare_all_models_statistical(
            y_true,
            {"gbdt": pred_good, "mean": pred_poor},
            reference_model="gbdt",
            task="regression",
            n_boot=300,
        )
        assert len(df_all) == 2

        fe = FeatureExtractor(t_obs=1.0, f_nominal=50.0)
        tr_s, va_s, te_s, _, _ = split_scenarios_by_partition(partitioned_50hz_scenario_batch)
        X_tr, y_tr, ys_tr, _ = fe.transform_batch(tr_s, as_dataframe=False)
        X_va, y_va, ys_va, _ = fe.transform_batch(va_s, as_dataframe=False)
        X_te, y_te, ys_te, _ = fe.transform_batch(te_s, as_dataframe=False)

        lc = LearningCurveEvaluator(train_fractions=[0.5, 1.0], n_repeats=2, seed=42)
        lc_res = lc.run(X_tr, y_tr, ys_tr, X_te, y_te, ys_te, X_val=X_va, y_val_nadir=y_va, y_val_sec=ys_va)
        assert len(lc_res["results"]) == 2
        assert not lc.to_dataframe(lc_res).empty

        ms = MultiSeedStabilityEvaluator(seeds=[42, 43, 44], threshold=49.50)
        ms_res = ms.run(X_tr, y_tr, ys_tr, X_te, y_te, ys_te, X_val=X_va, y_val_nadir=y_va, regressor_types=["rf", "gbdt"])
        assert len(ms_res["summary"]) == 3


class TestIsolatedPipelineAndCLIEntrypoints:

    def _make_isolated_config(self, base_dir: Path, raw_dir: Path, **overrides: Any) -> ExperimentConfig:
        out_dir = base_dir / "results"
        defaults: Dict[str, Any] = {
            "raw_data_dir": str(raw_dir),
            "processed_data_dir": str(base_dir / "processed"),
            "models_dir": str(base_dir / "models"),
            "output_dir": str(out_dir),
            "tables_dir": str(out_dir / "tables"),
            "figures_dir": str(out_dir / "figures"),
            "report_file": str(out_dir / "evaluation_report.md"),
            "skip_simulation": True,
            "window_lengths": [0.5, 1.0],
            "noise_levels": [0.0, 0.01],
            "headless": True,
        }
        defaults.update(overrides)
        return ExperimentConfig(**defaults)

    def test_pipeline_end_to_end_and_deterministic_seed_isolated(
        self,
        tmp_path: Path,
        partitioned_50hz_scenario_batch: List[Dict[str, Any]],
    ) -> None:
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        for scen in partitioned_50hz_scenario_batch[:12]:
            export_scenario_to_npz(scen, str(raw_dir / f"{scen['metadata']['scenario_id']}.npz"))

        cfg1 = self._make_isolated_config(tmp_path / "run1", raw_dir, seed=99, n_scenarios=12)
        p1 = ExperimentPipeline(cfg1)
        assert p1.workflow_steps == list(WORKFLOW_STEPS)
        summary = p1.execute_full_pipeline()
        assert summary["n_scenarios"] == 12
        assert len(summary["features_windows"]) == 2
        assert (Path(cfg1.tables_dir) / "window_sensitivity.csv").exists()
        assert Path(cfg1.report_file).exists()

        cfg2 = self._make_isolated_config(tmp_path / "run2", raw_dir, seed=99, n_scenarios=12)
        p2 = ExperimentPipeline(cfg2)
        p2.run_simulate()
        p2.run_extract_features()
        m2 = p2.run_train_models()
        assert list(p1.trained_models.keys()) == list(m2.keys())

    def test_cli_entrypoints_strictly_isolated_in_tmp_path(self, tmp_path: Path) -> None:
        sim_script = PROJECT_ROOT / "scripts" / "run_simulations.py"
        ext_script = PROJECT_ROOT / "scripts" / "extract_features.py"
        trn_script = PROJECT_ROOT / "scripts" / "train_models.py"
        evl_script = PROJECT_ROOT / "scripts" / "evaluate_all.py"

        for sc in (sim_script, ext_script, trn_script, evl_script):
            help_res = subprocess.run([sys.executable, str(sc), "--help"], capture_output=True, text=True)
            assert help_res.returncode == 0

        raw_dir = tmp_path / "raw_pmu"
        raw_sim_dir = tmp_path / "raw_sim"
        manifest_path = tmp_path / "manifest.json"
        qual_report = tmp_path / "dataset_quality.md"

        res_sim = subprocess.run(
            [
                sys.executable,
                str(sim_script),
                "--fast",
                "--n-scenarios",
                "12",
                "--output-dir",
                str(raw_dir),
                "--raw-sim-dir",
                str(raw_sim_dir),
                "--manifest-path",
                str(manifest_path),
                "--report-path",
                str(qual_report),
                "--validate",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert res_sim.returncode == 0, res_sim.stderr
        assert len(list(raw_dir.glob("*.npz"))) == 12
        assert manifest_path.exists()

        proc_dir = tmp_path / "processed"
        feat_file = proc_dir / "features.npz"
        res_ext = subprocess.run(
            [
                sys.executable,
                str(ext_script),
                "--input-dir",
                str(raw_dir),
                "--output-file",
                str(feat_file),
                "--output-dir",
                str(proc_dir),
                "--t-obs",
                "1.0",
                "--save-csv",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert res_ext.returncode == 0, res_ext.stderr
        assert feat_file.exists()

        models_dir = tmp_path / "models"
        tables_dir = tmp_path / "tables"
        res_trn = subprocess.run(
            [
                sys.executable,
                str(trn_script),
                "--features",
                str(feat_file),
                "--model-type",
                "all",
                "--output-dir",
                str(models_dir),
                "--tables-dir",
                str(tables_dir),
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert res_trn.returncode == 0, res_trn.stderr
        assert (models_dir / "gbdt.joblib").exists()
        assert (models_dir / "calibrated_gb_platt.joblib").exists()
        assert (tables_dir / "nadir_regression.csv").exists()

        eval_report = tmp_path / "results" / "evaluation_report.md"
        eval_tables = tmp_path / "results" / "tables"
        eval_figs = tmp_path / "results" / "figures"
        res_evl = subprocess.run(
            [
                sys.executable,
                str(evl_script),
                "--data-dir",
                str(raw_dir),
                "--features",
                str(feat_file),
                "--models-dir",
                str(models_dir),
                "--tables-dir",
                str(eval_tables),
                "--figures-dir",
                str(eval_figs),
                "--output-dir",
                str(tmp_path / "results"),
                "--report-file",
                str(eval_report),
                "--benchmarks",
                "sensitivity",
                "ood",
                "noise",
                "--quiet",
            ],
            capture_output=True,
            text=True,
        )
        assert res_evl.returncode == 0, res_evl.stderr
        assert eval_report.exists()
        assert (eval_tables / "window_sensitivity.csv").exists()
