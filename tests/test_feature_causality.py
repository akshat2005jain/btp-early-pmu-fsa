"""Causality, PMU-only anti-leakage, and analytical accuracy tests for the 14 canonical features."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from pmu_fsa.features.extractor import (
    CANONICAL_FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    DESCRIPTIVE_FEATURE_NAMES,
    FEATURE_SPECIFICATIONS,
    FeatureExtractor,
    calc_initial_rocof,
    extract_feature_dataset,
    extract_feature_vector,
    extract_features,
    wrap_to_pi,
)
from pmu_fsa.features.window_slicer import (
    FORBIDDEN_WINDOW_CHANNEL_KEYS,
    CausalWindow,
    extract_causal_window,
    slice_scenario_batch,
)
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
from pmu_fsa.simulation.ieee39_data import AREA_PARTITION, NOMINAL_FREQ, SECURITY_THRESHOLD_HZ
from pmu_fsa.simulation.pmu_recorder import (
    CANONICAL_13BUS_PMU_PLACEMENT,
    PMUNoiseConfig,
    PMURecorder,
)


EXPECTED_14_FEATURES = [
    "rocof_init",
    "rocof_peak_neg",
    "rocof_end",
    "rocof_mean",
    "freq_drop_end",
    "freq_drop_max",
    "freq_dev_integral",
    "bus_drop_worst",
    "area_freq_spread_max",
    "bus_freq_std_mean",
    "angle_divergence_max",
    "tie_line_angle_excursion",
    "freq_curvature_init",
    "freq_curvature_mean",
]


@pytest.fixture(scope="module")
def raw_dynamic_trajectory() -> Dict[str, Any]:
    sim = DynamicSimulator(load_scaling=1.0)
    return sim.simulate(t_span=(0.0, 5.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)


@pytest.fixture(scope="module")
def base_dynamic_scenario(raw_dynamic_trajectory: Dict[str, Any]) -> Dict[str, Any]:
    recorder = PMURecorder(fps=50, f_nominal=50.0)
    stream = recorder.record_trajectory(raw_dynamic_trajectory, inject_noise=False)
    stream["metadata"]["scenario_id"] = "scen_test_base_001"
    stream["metadata"]["ground_truth_nadir"] = float(raw_dynamic_trajectory["metadata"]["ground_truth_nadir"])
    stream["metadata"]["ground_truth_t_nadir"] = float(raw_dynamic_trajectory["metadata"]["ground_truth_t_nadir"])
    stream["metadata"]["is_secure"] = int(stream["metadata"]["ground_truth_nadir"] >= SECURITY_THRESHOLD_HZ)
    return stream


@pytest.fixture
def canonical_constant_scenario() -> Dict[str, Any]:
    fps = 50
    t = np.arange(0.0, 5.0 + 1e-9, 1.0 / fps)
    n = len(t)
    return {
        "timestamps": t,
        "bus_frequencies": np.full((n, 39), 50.0),
        "bus_rocof": np.zeros((n, 39)),
        "voltage_magnitudes": np.ones((n, 39)),
        "voltage_angles": np.zeros((n, 39)),
        "pmu_sys_frequency": np.full(n, 50.0),
        "coi_frequency": np.full(n, 50.0),
        "metadata": {
            "scenario_id": "scen_synth_constant_50hz",
            "t_disturb": 1.0,
            "fps": fps,
            "f_nominal": 50.0,
            "ground_truth_nadir": 50.0,
            "ground_truth_t_nadir": 5.0,
            "is_secure": 1,
        },
    }


@pytest.fixture
def canonical_ramp_scenario() -> Dict[str, Any]:
    fps = 50
    t = np.arange(0.0, 5.0 + 1e-9, 1.0 / fps)
    n = len(t)
    t_0 = 1.0
    alpha = 0.50

    f = np.full(n, 50.0)
    mask = t >= t_0
    f[mask] = 50.0 - alpha * (t[mask] - t_0)

    rocof = np.zeros(n)
    rocof[mask] = -alpha

    return {
        "timestamps": t,
        "bus_frequencies": np.tile(f[:, None], (1, 39)),
        "bus_rocof": np.tile(rocof[:, None], (1, 39)),
        "voltage_magnitudes": np.ones((n, 39)),
        "voltage_angles": np.zeros((n, 39)),
        "pmu_sys_frequency": f.copy(),
        "coi_frequency": f.copy(),
        "metadata": {
            "scenario_id": "scen_synth_ramp_50hz",
            "t_disturb": t_0,
            "fps": fps,
            "alpha": alpha,
            "f_nominal": 50.0,
            "ground_truth_nadir": float(np.min(f)),
            "ground_truth_t_nadir": 5.0,
            "is_secure": 0,
        },
    }


def test_canonical_14_feature_schema_identity():
    assert NOMINAL_FREQ == 50.0
    assert CANONICAL_FEATURE_NAMES == EXPECTED_14_FEATURES
    assert DESCRIPTIVE_FEATURE_NAMES == EXPECTED_14_FEATURES
    assert CORE_FEATURE_NAMES == EXPECTED_14_FEATURES
    assert len(CANONICAL_FEATURE_NAMES) == 14
    assert len(set(CANONICAL_FEATURE_NAMES)) == 14
    assert set(FEATURE_SPECIFICATIONS.keys()) == set(EXPECTED_14_FEATURES)


class TestCausalityAndZeroLeakage:

    @pytest.mark.parametrize("t_obs", [0.5, 1.0, 1.5, 2.0])
    def test_causality_gaussian_and_nan_inf_future_mutations(
        self,
        base_dynamic_scenario: Dict[str, Any],
        t_obs: float,
    ) -> None:
        extractor = FeatureExtractor(f_nominal=50.0)
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=t_obs)
        assert list(clean_features.keys()) == EXPECTED_14_FEATURES

        corrupted = copy.deepcopy(base_dynamic_scenario)
        t_0 = corrupted["metadata"]["t_disturb"]
        ts = corrupted["timestamps"]
        cutoff_idx = int(np.searchsorted(ts, t_0 + t_obs + 1e-9, side="right"))

        corrupted["bus_frequencies"][cutoff_idx:, :] = np.nan
        corrupted["pmu_sys_frequency"][cutoff_idx:] = np.nan
        corrupted["coi_frequency"][cutoff_idx:] = np.nan
        corrupted["bus_rocof"][cutoff_idx:, :] = np.inf
        corrupted["voltage_angles"][cutoff_idx:, :] = -np.inf

        corrupted_features = extractor.extract_features_dict(corrupted, t_obs=t_obs)
        for k in EXPECTED_14_FEATURES:
            assert np.isfinite(corrupted_features[k])
            assert clean_features[k] == corrupted_features[k]

    def test_pinpoint_first_post_window_sample_unaccessed(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        t_obs = 1.0
        extractor = FeatureExtractor(f_nominal=50.0)
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=t_obs)

        corrupted = copy.deepcopy(base_dynamic_scenario)
        t_0 = corrupted["metadata"]["t_disturb"]
        cutoff_idx = int(np.searchsorted(corrupted["timestamps"], t_0 + t_obs + 1e-9, side="right"))
        corrupted["bus_frequencies"][cutoff_idx, :] = np.nan
        corrupted["coi_frequency"][cutoff_idx] = np.nan
        corrupted["bus_rocof"][cutoff_idx, :] = np.nan
        corrupted["voltage_angles"][cutoff_idx, :] = np.nan

        corrupted_features = extractor.extract_features_dict(corrupted, t_obs=t_obs)
        for k in EXPECTED_14_FEATURES:
            assert clean_features[k] == corrupted_features[k]

    def test_zero_internal_coi_frequency_leakage(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor(f_nominal=50.0)
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=1.0)

        poisoned_num = copy.deepcopy(base_dynamic_scenario)
        poisoned_num["coi_frequency"][:] = 123456.0
        feats_num = extractor.extract_features_dict(poisoned_num, t_obs=1.0)

        poisoned_nan = copy.deepcopy(base_dynamic_scenario)
        poisoned_nan["coi_frequency"][:] = np.nan
        feats_nan = extractor.extract_features_dict(poisoned_nan, t_obs=1.0)

        for k in EXPECTED_14_FEATURES:
            assert feats_num[k] == clean_features[k], f"Feature {k} leaked internal coi_frequency!"
            assert feats_nan[k] == clean_features[k], f"Feature {k} read NaN coi_frequency!"

    def test_missing_pmu_channels_fail_without_coi_fallback(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor(f_nominal=50.0)

        missing_bf = copy.deepcopy(base_dynamic_scenario)
        del missing_bf["bus_frequencies"]
        missing_bf["coi_frequency"][:] = 49.65
        with pytest.raises((KeyError, ValueError)):
            extractor.extract_features_dict(missing_bf, t_obs=1.0)

        w_no_bf = dict(extract_causal_window(base_dynamic_scenario, t_obs=1.0))
        del w_no_bf["bus_frequencies"]
        w_no_bf["coi_frequency"] = np.full(len(w_no_bf["timestamps"]), 49.65)
        with pytest.raises((KeyError, ValueError)):
            extract_features(w_no_bf, f_nominal=50.0)

        w_no_va = dict(extract_causal_window(base_dynamic_scenario, t_obs=1.0))
        del w_no_va["voltage_angles"]
        w_no_va["coi_frequency"] = np.full(len(w_no_va["timestamps"]), 49.65)
        with pytest.raises((KeyError, ValueError)):
            extract_features(w_no_va, f_nominal=50.0)

        with pytest.raises(ValueError, match="forbidden"):
            CausalWindow(
                timestamps=np.linspace(1.0, 2.0, 51),
                bus_frequencies=np.empty((51, 0)),
                bus_rocof=np.empty((51, 0)),
                voltage_magnitudes=np.ones((51, 39)),
                voltage_angles=np.zeros((51, 39)),
                coi_frequency=np.full(51, 49.5),
            )

    def test_noise_propagates_into_all_14_canonical_features(
        self,
        raw_dynamic_trajectory: Dict[str, Any],
    ) -> None:
        noise_cfg = PMUNoiseConfig(
            sigma_f=0.005,
            sigma_rocof=0.02,
            sigma_v=0.0005,
            sigma_theta_deg=0.05,
            seed=42,
        )
        rec = PMURecorder(fps=50, noise_config=noise_cfg, f_nominal=50.0)
        clean_stream = rec.record_trajectory(raw_dynamic_trajectory, inject_noise=False)
        noisy_stream = rec.record_trajectory(raw_dynamic_trajectory, inject_noise=True, seed_offset=3)

        extractor = FeatureExtractor(t_obs=1.0, f_nominal=50.0)
        f_clean = extractor.extract_features_dict(clean_stream)
        f_noisy = extractor.extract_features_dict(noisy_stream)

        for k in EXPECTED_14_FEATURES:
            delta = abs(f_noisy[k] - f_clean[k])
            assert delta > 0.0, f"PMU measurement noise failed to propagate into feature '{k}'"

    def test_forbidden_channels_and_ground_truth_stripped(
        self,
        raw_dynamic_trajectory: Dict[str, Any],
    ) -> None:
        w = extract_causal_window(raw_dynamic_trajectory, t_obs=1.0)
        for forbidden in FORBIDDEN_WINDOW_CHANNEL_KEYS:
            assert forbidden not in w

        df = FeatureExtractor(f_nominal=50.0).extract_scenario(raw_dynamic_trajectory, t_obs=1.0)
        assert list(df.columns) == EXPECTED_14_FEATURES


class TestWindowBoundaryAndGuardrails:

    @pytest.mark.parametrize(
        "t_obs, expected_count",
        [(0.5, 26), (1.0, 51), (1.5, 76), (2.0, 101)],
    )
    def test_exact_sample_counts_at_50fps(
        self,
        base_dynamic_scenario: Dict[str, Any],
        t_obs: float,
        expected_count: int,
    ) -> None:
        window = extract_causal_window(base_dynamic_scenario, t_obs=t_obs)
        assert window.n_samples == expected_count
        assert window.n_buses == 39
        assert window.timestamps[0] == pytest.approx(1.0, abs=1e-6)
        assert window.timestamps[-1] == pytest.approx(1.0 + t_obs, abs=1e-6)
        assert window.is_prior_to_nadir is True
        assert window.lead_time_s > 0.0

    def test_strict_nadir_and_require_pre_nadir_guardrails(self) -> None:
        scenario = {
            "timestamps": np.linspace(0.0, 5.0, 251),
            "bus_frequencies": np.full((251, 39), 50.0),
            "bus_rocof": np.zeros((251, 39)),
            "voltage_magnitudes": np.ones((251, 39)),
            "voltage_angles": np.zeros((251, 39)),
            "metadata": {
                "t_disturb": 1.0,
                "fps": 50,
                "f_nominal": 50.0,
                "ground_truth_nadir": 49.20,
                "ground_truth_t_nadir": 1.80,
            },
        }
        with pytest.warns(UserWarning, match="reaches or exceeds nadir time"):
            w_warn = extract_causal_window(scenario, t_obs=1.0, strict_nadir=False)
        assert w_warn.is_prior_to_nadir is False

        with pytest.raises(ValueError, match="Anti-leakage violation"):
            extract_causal_window(scenario, t_obs=1.0, strict_nadir=True)

        with pytest.raises(ValueError, match="Anti-leakage violation"):
            extract_causal_window(scenario, t_obs=1.0, require_pre_nadir=True)

        no_t_nadir = copy.deepcopy(scenario)
        del no_t_nadir["metadata"]["ground_truth_t_nadir"]
        with pytest.raises(ValueError, match="ground_truth_t_nadir"):
            extract_causal_window(no_t_nadir, t_obs=1.0, require_pre_nadir=True)

    @pytest.mark.parametrize("invalid_t_obs", [-1.0, 0.0])
    def test_invalid_t_obs_and_t0_raise_value_error(
        self,
        base_dynamic_scenario: Dict[str, Any],
        invalid_t_obs: float,
    ) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            extract_causal_window(base_dynamic_scenario, t_obs=invalid_t_obs)

        with pytest.raises(ValueError, match="outside trajectory bounds"):
            extract_causal_window(base_dynamic_scenario, t_obs=1.0, t_0=-2.0)

        with pytest.raises(ValueError, match="outside trajectory bounds"):
            extract_causal_window(base_dynamic_scenario, t_obs=1.0, t_0=99.0)


class TestFeatureAnalyticalCorrectness:

    def test_constant_50hz_all_zero(
        self,
        canonical_constant_scenario: Dict[str, Any],
    ) -> None:
        feat = FeatureExtractor(f_nominal=50.0).extract_features_dict(canonical_constant_scenario, t_obs=1.0)
        assert len(feat) == 14
        for k, val in feat.items():
            assert abs(val) < 1e-12, f"Expected 0.0 on flat 50 Hz stream for {k}, got {val}"

    def test_linear_ramp_50hz_analytical_exactness(
        self,
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        alpha = canonical_ramp_scenario["metadata"]["alpha"]
        t_obs = 1.0
        feat = FeatureExtractor(f_nominal=50.0).extract_features_dict(canonical_ramp_scenario, t_obs=t_obs)

        assert feat["rocof_init"] == pytest.approx(-alpha, abs=1e-10)
        assert feat["rocof_peak_neg"] == pytest.approx(-alpha, abs=1e-10)
        assert feat["rocof_end"] == pytest.approx(-alpha, abs=1e-10)
        assert feat["rocof_mean"] == pytest.approx(-alpha, abs=1e-10)
        assert feat["freq_drop_end"] == pytest.approx(-alpha * t_obs, abs=1e-10)
        assert feat["freq_drop_max"] == pytest.approx(-alpha * t_obs, abs=1e-10)
        assert feat["bus_drop_worst"] == pytest.approx(-alpha * t_obs, abs=1e-10)
        assert feat["freq_dev_integral"] == pytest.approx(-0.5 * alpha * (t_obs ** 2), abs=1e-10)
        assert feat["freq_curvature_init"] == pytest.approx(0.0, abs=1e-10)
        assert feat["freq_curvature_mean"] == pytest.approx(0.0, abs=1e-10)

    def test_quadratic_frequency_curvature_hz_per_s2_accuracy(self) -> None:
        fps = 50
        t = np.arange(0.0, 3.5 + 1e-9, 1.0 / fps)
        t_0 = 1.0
        r0 = -0.80
        curv_true = 0.60  # Hz/s^2 (d^2f/dt^2 = +0.60 Hz/s^2)

        f = np.full(len(t), 50.0)
        post = t >= t_0
        tau = t[post] - t_0
        f[post] = 50.0 + r0 * tau + 0.5 * curv_true * (tau ** 2)

        scen = {
            "timestamps": t,
            "bus_frequencies": np.tile(f[:, None], (1, 39)),
            "bus_rocof": np.zeros((len(t), 39)),
            "voltage_magnitudes": np.ones((len(t), 39)),
            "voltage_angles": np.zeros((len(t), 39)),
            "metadata": {"t_disturb": t_0, "fps": fps, "f_nominal": 50.0},
        }
        feat = FeatureExtractor(f_nominal=50.0).extract_features_dict(scen, t_obs=1.0)
        assert feat["freq_curvature_init"] == pytest.approx(curv_true, abs=1e-3)
        assert feat["freq_curvature_mean"] == pytest.approx(curv_true, abs=1e-3)

    def test_sinusoidal_interarea_dispersion_and_angle_wrapping(self) -> None:
        fps = 50
        t = np.arange(0.0, 2.5 + 1e-9, 1.0 / fps)
        n = len(t)
        omega = 2.0 * np.pi * 0.5
        amp = 0.20

        f_a1 = 50.0 + amp * np.sin(omega * t)
        f_a2 = 50.0 + amp * np.sin(omega * t - 2.0 * np.pi / 3.0)
        f_a3 = 50.0 + amp * np.sin(omega * t + 2.0 * np.pi / 3.0)

        bus_freq = np.zeros((n, 39))
        for b in AREA_PARTITION[1]:
            bus_freq[:, b - 1] = f_a1
        for b in AREA_PARTITION[2]:
            bus_freq[:, b - 1] = f_a2
        for b in AREA_PARTITION[3]:
            bus_freq[:, b - 1] = f_a3

        angles = np.full((n, 39), 179.0)
        idx_end = int(round(2.0 * fps))
        angles[idx_end, 3] = 179.0   # Bus 4 stays at +179 deg (0 deg shift)
        angles[idx_end, 4] = -179.0  # Bus 5 crosses +180 -> -179 deg (+2 deg shift)

        scen = {
            "timestamps": t,
            "bus_frequencies": bus_freq,
            "bus_rocof": np.zeros((n, 39)),
            "voltage_magnitudes": np.ones((n, 39)),
            "voltage_angles": angles,
            "metadata": {"t_disturb": 0.0, "fps": fps, "f_nominal": 50.0},
        }
        feat = FeatureExtractor(f_nominal=50.0).extract_features_dict(scen, t_obs=2.0)
        assert feat["area_freq_spread_max"] == pytest.approx(np.sqrt(3.0) * amp, rel=1e-2)
        assert feat["bus_freq_std_mean"] > 0.0
        assert feat["angle_divergence_max"] == pytest.approx(2.0, abs=1e-6)
        assert feat["tie_line_angle_excursion"] == pytest.approx(2.0, abs=1e-6)

        wrapped_deg = abs(np.degrees(float(wrap_to_pi(np.radians(179.0) - np.radians(-179.0)))))
        assert wrapped_deg == pytest.approx(2.0, abs=1e-8)

    def test_ols_initial_rocof_noise_resilience(self) -> None:
        fps = 50
        dt = 1.0 / fps
        k_points = int(round(0.10 * fps)) + 1
        t = np.arange(k_points) * dt
        alpha = 0.50
        clean_f = 50.0 - alpha * t

        rng = np.random.default_rng(42)
        ep_errors = []
        ols_errors = []
        for _ in range(400):
            noisy_f = clean_f + rng.normal(0.0, 0.01, size=k_points)
            ep_slope = (noisy_f[-1] - noisy_f[0]) / (t[-1] - t[0])
            ols_slope = calc_initial_rocof({"timestamps": t, "bus_frequencies": np.tile(noisy_f[:, None], (1, 39))}, delta_t_init=0.10)
            ep_errors.append((ep_slope + alpha) ** 2)
            ols_errors.append((ols_slope + alpha) ** 2)

        assert float(np.mean(ols_errors)) < float(np.mean(ep_errors))

    def test_partial_pmu_nan_columns_and_packet_dropout_imputation(
        self,
        base_dynamic_scenario: Dict[str, Any],
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        # 1. Partial PMU (26 all-NaN columns, 13 observed buses) succeeds under default ignore_unobserved
        partial_scen = copy.deepcopy(base_dynamic_scenario)
        obs_zero = {b - 1 for b in CANONICAL_13BUS_PMU_PLACEMENT}
        for c in range(39):
            if c not in obs_zero:
                partial_scen["bus_frequencies"][:, c] = np.nan
                partial_scen["voltage_angles"][:, c] = np.nan
        feats_partial = FeatureExtractor(f_nominal=50.0).extract_features_dict(partial_scen, t_obs=1.0)
        assert len(feats_partial) == 14
        assert all(np.isfinite(v) for v in feats_partial.values())

        # 2. Isolated packet dropout inside an observed column raises under ignore_unobserved, imputes under handle_nans=True
        dropout_scen = copy.deepcopy(canonical_ramp_scenario)
        idx_dist = int(np.searchsorted(dropout_scen["timestamps"], 1.0))
        dropout_scen["bus_frequencies"][idx_dist, :] = np.nan
        dropout_scen["bus_frequencies"][idx_dist + 5, 0] = np.nan

        with pytest.raises(ValueError, match="contains NaN or Inf"):
            FeatureExtractor(f_nominal=50.0, handle_nans=False).extract_features_dict(dropout_scen, t_obs=1.0)

        feats_imputed = FeatureExtractor(f_nominal=50.0, handle_nans=True).extract_features_dict(dropout_scen, t_obs=1.0)
        assert len(feats_imputed) == 14
        assert all(np.isfinite(v) for v in feats_imputed.values())


class TestOutputContractAndNumPyDataFrameAgreement:

    def test_exact_numpy_vs_dataframe_agreement_and_npz_compat(
        self,
        tmp_path: Path,
        base_dynamic_scenario: Dict[str, Any],
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        scenarios = [base_dynamic_scenario, canonical_ramp_scenario]
        extractor = FeatureExtractor(t_obs=1.0, f_nominal=50.0)

        X_df, y_df, ys_df, meta_df = extractor.transform_batch(scenarios, as_dataframe=True)
        X_np, y_np, ys_np, meta_list = extractor.transform_batch(scenarios, as_dataframe=False)

        assert isinstance(X_df, pd.DataFrame)
        assert X_df.shape == (2, 14)
        assert X_np.shape == (2, 14)
        assert list(X_df.columns) == EXPECTED_14_FEATURES
        assert float(np.max(np.abs(X_df.to_numpy() - X_np))) == 0.0
        np.testing.assert_array_equal(y_df, y_np)
        np.testing.assert_array_equal(ys_df, ys_np)
        assert len(meta_df) == len(meta_list) == 2

        vec = extract_feature_vector(extract_causal_window(base_dynamic_scenario, t_obs=1.0), f_nominal=50.0)
        np.testing.assert_allclose(vec, X_np[0], atol=1e-11)

        X_ds, _, _, ids = extract_feature_dataset(scenarios, t_obs=1.0, as_dataframe=True)
        pd.testing.assert_frame_equal(X_ds, X_df)
        assert len(ids) == 2

        windows = slice_scenario_batch(scenarios, t_obs=1.0)
        assert len(windows) == 2 and all(isinstance(w, CausalWindow) for w in windows)

        npz_path = tmp_path / "scen.npz"
        np.savez_compressed(
            npz_path,
            timestamps=base_dynamic_scenario["timestamps"],
            bus_frequencies=base_dynamic_scenario["bus_frequencies"],
            bus_rocof=base_dynamic_scenario["bus_rocof"],
            voltage_magnitudes=base_dynamic_scenario["voltage_magnitudes"],
            voltage_angles=base_dynamic_scenario["voltage_angles"],
            pmu_sys_frequency=base_dynamic_scenario["pmu_sys_frequency"],
            metadata=base_dynamic_scenario["metadata"],
        )
        f_from_path = extract_features(npz_path, t_obs=1.0, f_nominal=50.0)
        for idx_k, k in enumerate(EXPECTED_14_FEATURES):
            assert f_from_path[k] == pytest.approx(X_np[0, idx_k], abs=1e-11)

    def test_execution_latency_per_scenario(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor(f_nominal=50.0)
        for _ in range(5):
            _ = extractor.extract_features_dict(base_dynamic_scenario, t_obs=1.0)

        trial_latencies: List[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(50):
                _ = extractor.extract_features_dict(base_dynamic_scenario, t_obs=1.0)
            trial_latencies.append((time.perf_counter() - t0) / 50.0)

        assert min(trial_latencies) < 0.0035
