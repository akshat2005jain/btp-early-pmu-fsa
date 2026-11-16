"""Causality and analytical accuracy tests for PMU feature extraction."""

from __future__ import annotations

import copy
import math
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
from pmu_fsa.simulation.pmu_recorder import PMURecorder
from pmu_fsa.simulation.ieee39_data import AREA_PARTITION
from pmu_fsa.features.window_slicer import (
    CausalWindow,
    extract_causal_window,
    slice_scenario_batch,
)
from pmu_fsa.features.extractor import (
    CANONICAL_AREA_PARTITION,
    CORE_FEATURE_NAMES,
    DESCRIPTIVE_FEATURE_NAMES,
    FeatureExtractor,
    calc_angle_divergence,
    calc_coi_acceleration,
    calc_delta_f,
    calc_freq_integral,
    calc_initial_rocof,
    calc_interarea_dispersion,
    calc_max_rocof,
    calc_mean_rocof,
    calc_rocof_metrics,
    extract_feature_dataset,
    extract_feature_vector,
    extract_features,
    extract_features_from_scenario,
    wrap_to_pi,
)


@pytest.fixture(scope="module")
def base_dynamic_scenario() -> Dict[str, Any]:
    sim = DynamicSimulator(load_scaling=1.0)
    raw = sim.simulate(t_span=(0.0, 5.0), t_disturb=1.0, tripped_bus=35, p_loss_mw=650.0, fps=50)
    recorder = PMURecorder(fps=50)
    stream = recorder.record_trajectory(raw, inject_noise=False)
    stream["metadata"]["scenario_id"] = "scen_test_base_001"
    stream["metadata"]["ground_truth_nadir"] = float(raw["metadata"]["ground_truth_nadir"])
    stream["metadata"]["ground_truth_t_nadir"] = float(raw["metadata"]["ground_truth_t_nadir"])
    stream["metadata"]["is_secure"] = 0
    return stream


@pytest.fixture
def canonical_constant_scenario() -> Dict[str, Any]:
    fps = 50
    t = np.arange(0.0, 5.0 + 1e-9, 1.0 / fps)
    n = len(t)
    return {
        "timestamps": t,
        "bus_frequencies": np.full((n, 39), 60.0),
        "bus_rocof": np.zeros((n, 39)),
        "voltage_magnitudes": np.ones((n, 39)),
        "voltage_angles": np.zeros((n, 39)),
        "coi_frequency": np.full(n, 60.0),
        "metadata": {
            "scenario_id": "scen_synth_constant",
            "t_disturb": 1.0,
            "fps": fps,
            "f_nominal": 60.0,
            "ground_truth_nadir": 60.0,
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
    alpha = 0.5

    f = np.full(n, 60.0)
    mask = t >= t_0
    f[mask] = 60.0 - alpha * (t[mask] - t_0)

    rocof = np.zeros(n)
    rocof[mask] = -alpha

    return {
        "timestamps": t,
        "bus_frequencies": np.tile(f[:, None], (1, 39)),
        "bus_rocof": np.tile(rocof[:, None], (1, 39)),
        "voltage_magnitudes": np.ones((n, 39)),
        "voltage_angles": np.zeros((n, 39)),
        "coi_frequency": f.copy(),
        "metadata": {
            "scenario_id": "scen_synth_ramp",
            "t_disturb": t_0,
            "fps": fps,
            "alpha": alpha,
            "f_nominal": 60.0,
            "ground_truth_nadir": float(np.min(f)),
            "ground_truth_t_nadir": 5.0,
            "is_secure": 0,
        },
    }


class TestCausalityZeroLeakage:

    @pytest.mark.parametrize("t_obs", [0.5, 1.0, 1.5, 2.0])
    def test_causality_gaussian_noise_mutation(
        self,
        base_dynamic_scenario: Dict[str, Any],
        t_obs: float,
    ) -> None:
        extractor = FeatureExtractor()
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=t_obs)

        corrupted = copy.deepcopy(base_dynamic_scenario)
        t_0 = corrupted["metadata"]["t_disturb"]
        ts = corrupted["timestamps"]
        # Corrupt samples strictly after t_0 + t_obs
        cutoff_idx = int(np.searchsorted(ts, t_0 + t_obs + 1e-9, side="right"))

        rng = np.random.default_rng(2026)
        n_future = len(ts) - cutoff_idx
        corrupted["bus_frequencies"][cutoff_idx:, :] += rng.normal(0.0, 10.0, size=(n_future, 39))
        corrupted["coi_frequency"][cutoff_idx:] += rng.normal(0.0, 10.0, size=n_future)
        corrupted["bus_rocof"][cutoff_idx:, :] += rng.normal(0.0, 25.0, size=(n_future, 39))
        corrupted["voltage_angles"][cutoff_idx:, :] += rng.normal(0.0, np.pi, size=(n_future, 39))

        corrupted_features = extractor.extract_features_dict(corrupted, t_obs=t_obs)

        for k in clean_features:
            assert np.isclose(clean_features[k], corrupted_features[k], rtol=1e-12, atol=1e-12), (
                f"Feature '{k}' leaked future data under Gaussian noise! "
                f"clean={clean_features[k]}, mutated={corrupted_features[k]}"
            )

    @pytest.mark.parametrize("t_obs", [0.5, 1.0, 1.5])
    def test_causality_nan_inf_poisoning(
        self,
        base_dynamic_scenario: Dict[str, Any],
        t_obs: float,
    ) -> None:
        extractor = FeatureExtractor()
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=t_obs)

        corrupted = copy.deepcopy(base_dynamic_scenario)
        t_0 = corrupted["metadata"]["t_disturb"]
        ts = corrupted["timestamps"]
        cutoff_idx = int(np.searchsorted(ts, t_0 + t_obs + 1e-9, side="right"))

        corrupted["bus_frequencies"][cutoff_idx:, :] = np.nan
        corrupted["coi_frequency"][cutoff_idx:] = np.nan
        corrupted["bus_rocof"][cutoff_idx:, :] = np.inf
        corrupted["voltage_angles"][cutoff_idx:, :] = -np.inf

        corrupted_features = extractor.extract_features_dict(corrupted, t_obs=t_obs)

        for k in clean_features:
            assert not np.isnan(corrupted_features[k]), f"Feature '{k}' became NaN due to future leakage"
            assert not np.isinf(corrupted_features[k]), f"Feature '{k}' became Inf due to future leakage"
            assert np.isclose(clean_features[k], corrupted_features[k], rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("t_obs", [0.5, 1.0, 1.5, 2.0])
    def test_causality_signal_inversion(
        self,
        base_dynamic_scenario: Dict[str, Any],
        t_obs: float,
    ) -> None:
        extractor = FeatureExtractor()
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=t_obs)

        corrupted = copy.deepcopy(base_dynamic_scenario)
        t_0 = corrupted["metadata"]["t_disturb"]
        ts = corrupted["timestamps"]
        cutoff_idx = int(np.searchsorted(ts, t_0 + t_obs + 1e-9, side="right"))

        corrupted["bus_frequencies"][cutoff_idx:, :] = 120.0 - corrupted["bus_frequencies"][cutoff_idx:, :]
        corrupted["coi_frequency"][cutoff_idx:] = 120.0 - corrupted["coi_frequency"][cutoff_idx:]
        corrupted["bus_rocof"][cutoff_idx:, :] = -corrupted["bus_rocof"][cutoff_idx:, :]

        corrupted_features = extractor.extract_features_dict(corrupted, t_obs=t_obs)
        for k in clean_features:
            assert np.isclose(clean_features[k], corrupted_features[k], rtol=1e-12, atol=1e-12)

    def test_causality_secondary_collapse_injection(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        t_obs = 1.0
        extractor = FeatureExtractor()
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=t_obs)

        corrupted = copy.deepcopy(base_dynamic_scenario)
        t_0 = corrupted["metadata"]["t_disturb"]
        ts = corrupted["timestamps"]
        cutoff_idx = int(np.searchsorted(ts, t_0 + t_obs + 1e-9, side="right"))

        corrupted["bus_frequencies"][cutoff_idx:, :] = 40.0
        corrupted["coi_frequency"][cutoff_idx:] = 40.0
        corrupted["bus_rocof"][cutoff_idx:, :] = -30.0

        corrupted_features = extractor.extract_features_dict(corrupted, t_obs=t_obs)
        for k in clean_features:
            assert np.isclose(clean_features[k], corrupted_features[k], rtol=1e-12, atol=1e-12)

    def test_causality_ground_truth_metadata_isolation(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor()
        df = extractor.extract_scenario(base_dynamic_scenario, t_obs=1.0)

        feature_cols = df.columns.tolist() if hasattr(df, "columns") else list(df.keys())
        forbidden_cols = ["ground_truth_nadir", "ground_truth_t_nadir", "is_secure", "y_nadir", "y_secure"]
        for col in forbidden_cols:
            assert col not in feature_cols, f"Ground-truth target column '{col}' leaked into feature matrix!"


class TestWindowBoundaryAndClamping:

    @pytest.mark.parametrize(
        "t_obs, expected_count",
        [
            (0.5, 26),
            (1.0, 51),
            (1.5, 76),
            (2.0, 101),
        ],
    )
    def test_exact_sample_count_accessed(
        self,
        base_dynamic_scenario: Dict[str, Any],
        t_obs: float,
        expected_count: int,
    ) -> None:
        window = extract_causal_window(base_dynamic_scenario, t_obs=t_obs)
        assert len(window.timestamps) == expected_count, (
            f"Expected {expected_count} samples for t_obs={t_obs} at 50 fps, got {len(window.timestamps)}"
        )
        assert window.timestamps[0] == pytest.approx(base_dynamic_scenario["metadata"]["t_disturb"], abs=1e-6)
        assert window.timestamps[-1] <= base_dynamic_scenario["metadata"]["t_disturb"] + t_obs + 1e-6

    def test_pinpoint_cutoff_plus_one_unaccessed(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        t_obs = 1.0
        extractor = FeatureExtractor()
        clean_features = extractor.extract_features_dict(base_dynamic_scenario, t_obs=t_obs)

        corrupted = copy.deepcopy(base_dynamic_scenario)
        t_0 = corrupted["metadata"]["t_disturb"]
        ts = corrupted["timestamps"]
        cutoff_idx = int(np.searchsorted(ts, t_0 + t_obs + 1e-9, side="right"))

        # Poison only the first post-window sample at cutoff_idx
        corrupted["bus_frequencies"][cutoff_idx, :] = np.nan
        corrupted["coi_frequency"][cutoff_idx] = np.nan
        corrupted["bus_rocof"][cutoff_idx, :] = np.nan

        corrupted_features = extractor.extract_features_dict(corrupted, t_obs=t_obs)
        for k in clean_features:
            assert np.isclose(clean_features[k], corrupted_features[k], rtol=1e-12, atol=1e-12), (
                f"Sample at cutoff_idx ({cutoff_idx}) was read by feature '{k}'!"
            )

    def test_nadir_exceeded_guardrail_warning_and_strict_error(self) -> None:
        scenario = {
            "timestamps": np.linspace(0.0, 5.0, 251),
            "bus_frequencies": np.full((251, 39), 60.0),
            "bus_rocof": np.zeros((251, 39)),
            "voltage_magnitudes": np.ones((251, 39)),
            "voltage_angles": np.zeros((251, 39)),
            "coi_frequency": np.full(251, 60.0),
            "metadata": {
                "t_disturb": 1.0,
                "fps": 50,
                "ground_truth_nadir": 59.20,
                "ground_truth_t_nadir": 1.80,
            },
        }
        with pytest.warns(UserWarning, match="reaches or exceeds nadir time"):
            extract_causal_window(scenario, t_obs=1.0, strict_nadir=False)

        with pytest.raises(ValueError, match="Anti-leakage violation"):
            extract_causal_window(scenario, t_obs=1.0, strict_nadir=True)

    @pytest.mark.parametrize("invalid_t_obs", [-1.0, 0.0])
    def test_invalid_t_obs_raises_value_error(
        self,
        base_dynamic_scenario: Dict[str, Any],
        invalid_t_obs: float,
    ) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            extract_causal_window(base_dynamic_scenario, t_obs=invalid_t_obs)

    def test_invalid_t_disturb_raises_value_error(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError, match="outside trajectory bounds"):
            extract_causal_window(base_dynamic_scenario, t_obs=1.0, t_0=-5.0)

        with pytest.raises(ValueError, match="outside trajectory bounds"):
            extract_causal_window(base_dynamic_scenario, t_obs=1.0, t_0=100.0)

    def test_causal_window_dict_properties(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        w = extract_causal_window(base_dynamic_scenario, t_obs=1.0)

        assert isinstance(w, dict)
        assert "timestamps" in w
        assert "bus_frequencies" in w
        assert "bus_rocof" in w
        assert "voltage_magnitudes" in w
        assert "voltage_angles" in w
        assert "coi_frequency" in w
        assert "metadata" in w

        w_copy = copy.deepcopy(w)
        assert len(w_copy["timestamps"]) == len(w["timestamps"])

        assert isinstance(w.timestamps, np.ndarray)
        assert isinstance(w.bus_frequencies, np.ndarray)
        assert w.n_samples == 51
        assert w.n_buses == 39
        assert pytest.approx(w.duration, abs=1e-4) == 1.0
        assert w.fps == 50.0
        assert w.t_0 == 1.0
        assert w.t_obs == 1.0

    def test_batch_slicing_uniformity(
        self,
        base_dynamic_scenario: Dict[str, Any],
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        scenarios = [base_dynamic_scenario, canonical_ramp_scenario]
        windows = slice_scenario_batch(scenarios, t_obs=1.0)

        assert len(windows) == 2
        for w in windows:
            assert isinstance(w, CausalWindow)
            assert w.n_samples == 51
            assert pytest.approx(w.duration, abs=1e-4) == 1.0


class TestFeatureAnalyticalCorrectness:

    def test_constant_frequency_canonical(
        self,
        canonical_constant_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor()
        feat = extractor.extract_features_dict(canonical_constant_scenario, t_obs=1.0)

        assert abs(feat["rocof_init_coi"]) < 1e-12
        assert abs(feat["rocof_peak_neg_coi"]) < 1e-12
        assert abs(feat["rocof_mean_coi"]) < 1e-12
        assert abs(feat["delta_f_tobs_coi"]) < 1e-12
        assert abs(feat["delta_f_max_drop_coi"]) < 1e-12
        assert abs(feat["freq_integral_dev_coi"]) < 1e-12
        assert abs(feat["freq_integral_abs_coi"]) < 1e-12
        assert abs(feat["area_spread_max"]) < 1e-12
        assert abs(feat["bus_freq_var_mean"]) < 1e-12
        assert abs(feat["coi_acc_init"]) < 1e-12

    def test_linear_ramp_analytical_exactness(
        self,
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        alpha = canonical_ramp_scenario["metadata"]["alpha"]
        t_obs = 1.0
        extractor = FeatureExtractor()
        feat = extractor.extract_features_dict(canonical_ramp_scenario, t_obs=t_obs)

        assert feat["rocof_init_coi"] == pytest.approx(-alpha, abs=1e-10)
        assert feat["rocof_mean_coi"] == pytest.approx(-alpha, abs=1e-10)
        assert feat["rocof_peak_neg_coi"] == pytest.approx(-alpha, abs=1e-10)

        assert feat["delta_f_tobs_coi"] == pytest.approx(-alpha * t_obs, abs=1e-10)
        assert feat["delta_f_max_drop_coi"] == pytest.approx(-alpha * t_obs, abs=1e-10)

        expected_integral = -0.5 * alpha * (t_obs ** 2)
        assert feat["freq_integral_dev_coi"] == pytest.approx(expected_integral, abs=1e-10)
        assert feat["freq_integral_abs_coi"] == pytest.approx(abs(expected_integral), abs=1e-10)

    def test_sinusoidal_interarea_dispersion_analytical(self) -> None:
        fps = 50
        t = np.arange(0.0, 3.0 + 1e-9, 1.0 / fps)
        n = len(t)
        omega = 2.0 * np.pi * 0.5
        A = 0.20

        f_a1 = 60.0 + A * np.sin(omega * t)
        f_a2 = 60.0 + A * np.sin(omega * t - 2.0 * np.pi / 3.0)
        f_a3 = 60.0 + A * np.sin(omega * t + 2.0 * np.pi / 3.0)

        bus_freq = np.zeros((n, 39))
        for b in AREA_PARTITION[1]:
            bus_freq[:, b - 1] = f_a1
        for b in AREA_PARTITION[2]:
            bus_freq[:, b - 1] = f_a2
        for b in AREA_PARTITION[3]:
            bus_freq[:, b - 1] = f_a3

        scenario = {
            "timestamps": t,
            "bus_frequencies": bus_freq,
            "bus_rocof": np.zeros((n, 39)),
            "voltage_magnitudes": np.ones((n, 39)),
            "voltage_angles": np.zeros((n, 39)),
            "coi_frequency": np.full(n, 60.0),
            "metadata": {"t_disturb": 0.0, "fps": fps, "f_nominal": 60.0},
        }

        extractor = FeatureExtractor()
        feat = extractor.extract_features_dict(scenario, t_obs=2.0)

        # Balanced 3-phase oscillation across areas gives max spread = sqrt(3) * A
        expected_spread = np.sqrt(3.0) * A
        assert feat["area_spread_max"] == pytest.approx(expected_spread, rel=1e-2)
        assert feat["bus_freq_var_mean"] > 0.0

    def test_phase_angle_wrap_around_handling(self) -> None:
        # +179 deg vs -179 deg across +/-pi wraps to 2 deg, not 358 deg
        theta_a = np.radians(179.0)
        theta_b = np.radians(-179.0)

        wrapped_rad = float(wrap_to_pi(theta_a - theta_b))
        wrapped_deg = np.degrees(wrapped_rad)

        assert abs(wrapped_deg) == pytest.approx(2.0, abs=1e-8), (
            f"Wrap-around failure: expected 2.0 deg, got {abs(wrapped_deg):.2f} deg"
        )

    def test_ols_initial_rocof_noise_resilience(
        self,
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        alpha = canonical_ramp_scenario["metadata"]["alpha"]
        fps = 50
        dt = 1.0 / fps
        k_points = int(round(0.10 * fps)) + 1

        t = np.arange(k_points) * dt
        clean_f = 60.0 - alpha * t

        rng = np.random.default_rng(42)
        sigma = 0.01
        n_trials = 1000

        ep_errors = []
        ols_errors = []

        for _ in range(n_trials):
            noisy_f = clean_f + rng.normal(0.0, sigma, size=k_points)
            ep_slope = (noisy_f[-1] - noisy_f[0]) / (t[-1] - t[0])

            w_trial = {
                "timestamps": t,
                "coi_frequency": noisy_f,
            }
            ols_slope = calc_initial_rocof(w_trial, delta_t_init=0.1)

            ep_errors.append((ep_slope - (-alpha)) ** 2)
            ols_errors.append((ols_slope - (-alpha)) ** 2)

        ep_mse = float(np.mean(ep_errors))
        ols_mse = float(np.mean(ols_errors))

        assert ols_mse < ep_mse, f"OLS MSE ({ols_mse:.6f}) was not lower than Endpoint MSE ({ep_mse:.6f})"

    def test_missing_pmu_packet_causal_imputation(
        self,
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        corrupted = copy.deepcopy(canonical_ramp_scenario)
        corrupted["bus_frequencies"][15, :] = np.nan
        corrupted["coi_frequency"][15] = np.nan

        extractor = FeatureExtractor(handle_nans=True)
        feat = extractor.extract_features_dict(corrupted, t_obs=1.0)

        for k, v in feat.items():
            assert not np.isnan(v), f"Feature '{k}' contained NaN after causal imputation"

    def test_leading_nan_and_unobserved_channel_imputation(
        self,
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        corrupted_t0 = copy.deepcopy(canonical_ramp_scenario)
        t_dist = float(corrupted_t0["metadata"].get("t_disturb", 0.0))
        idx_dist = int(np.searchsorted(corrupted_t0["timestamps"], t_dist))
        corrupted_t0["coi_frequency"][idx_dist] = np.nan
        corrupted_t0["bus_frequencies"][idx_dist, :] = np.nan
        corrupted_t0["coi_frequency"][0] = np.nan
        corrupted_t0["bus_frequencies"][0, :] = np.nan

        extractor = FeatureExtractor(handle_nans=True)
        feats_t0 = extractor.extract_features_dict(corrupted_t0, t_obs=1.0)
        assert len(feats_t0) == 28, f"Expected 28 features, got {len(feats_t0)}"
        for k, v in feats_t0.items():
            assert not np.isnan(v), f"Feature '{k}' contained NaN after leading NaN imputation at t=0"
            assert not np.isinf(v), f"Feature '{k}' contained Inf after leading NaN imputation at t=0"

        scen_direct = {
            "timestamps": np.linspace(0.0, 2.0, 100),
            "bus_frequencies": np.full((100, 39), 60.0),
            "coi_frequency": np.full(100, 60.0),
            "bus_rocof": np.zeros((100, 39)),
            "voltage_magnitudes": np.ones((100, 39)),
            "voltage_angles": np.zeros((100, 39)),
            "metadata": {"t_disturb": 0.0, "f_nominal": 60.0, "ground_truth_nadir": 60.0, "is_secure": 1},
        }
        scen_direct["coi_frequency"][0] = np.nan
        scen_direct["bus_frequencies"][0, :] = np.nan
        feats_direct = extractor.extract_features_dict(scen_direct, t_obs=1.0)
        assert len(feats_direct) == 28
        for k, v in feats_direct.items():
            assert not np.isnan(v), f"Feature '{k}' contained NaN in direct t=0 test"
            assert not np.isinf(v), f"Feature '{k}' contained Inf in direct t=0 test"

        corrupted_bus0 = copy.deepcopy(canonical_ramp_scenario)
        corrupted_bus0["bus_frequencies"][:, 0] = np.nan

        feats_bus0 = extractor.extract_features_dict(corrupted_bus0, t_obs=1.0)
        assert len(feats_bus0) == 28
        for k, v in feats_bus0.items():
            assert not np.isnan(v), f"Feature '{k}' contained NaN after unobserved channel imputation"
            assert not np.isinf(v), f"Feature '{k}' contained Inf after unobserved channel imputation"

        corrupted_coi = copy.deepcopy(canonical_ramp_scenario)
        corrupted_coi["coi_frequency"][:] = np.nan

        feats_coi = extractor.extract_features_dict(corrupted_coi, t_obs=1.0)
        assert len(feats_coi) == 28
        for k, v in feats_coi.items():
            assert not np.isnan(v), f"Feature '{k}' contained NaN after all-NaN COI imputation"
            assert not np.isinf(v), f"Feature '{k}' contained Inf after all-NaN COI imputation"

        strict_extractor = FeatureExtractor(handle_nans=False)
        with pytest.raises(ValueError, match="contains NaN or Inf values"):
            _ = strict_extractor.extract_features_dict(corrupted_t0, t_obs=1.0)


class TestOutputContractAndTabularFormat:

    def test_feature_dataframe_structure(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor()
        df = extractor.extract_scenario(base_dynamic_scenario, t_obs=1.0)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.shape[1] >= 14

        core_cols = [
            "rocof_init_coi",
            "rocof_peak_neg_coi",
            "rocof_mean_coi",
            "delta_f_tobs_coi",
            "delta_f_max_drop_coi",
            "freq_integral_dev_coi",
            "freq_integral_abs_coi",
            "area_spread_max",
            "bus_freq_var_mean",
            "angle_div_max",
            "coi_acc_init",
        ]
        for col in core_cols:
            assert col in df.columns, f"Required feature column '{col}' missing from DataFrame"
            assert np.issubdtype(df[col].dtype, np.floating)

    def test_batch_extraction_consistency(
        self,
        base_dynamic_scenario: Dict[str, Any],
        canonical_ramp_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor()
        scenarios = [base_dynamic_scenario, canonical_ramp_scenario]

        X, y_nadir, y_secure, ids = extract_feature_dataset(scenarios, t_obs=1.0)

        assert len(X) == 2
        assert len(y_nadir) == 2
        assert len(y_secure) == 2
        assert len(ids) == 2

        df_single = extractor.extract_scenario(base_dynamic_scenario, t_obs=1.0)
        pd.testing.assert_frame_equal(X.iloc[[0]].reset_index(drop=True), df_single.reset_index(drop=True))

    def test_npz_file_input_compatibility(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_scenario.npz"
            np.savez(
                file_path,
                timestamps=base_dynamic_scenario["timestamps"],
                bus_frequencies=base_dynamic_scenario["bus_frequencies"],
                bus_rocof=base_dynamic_scenario["bus_rocof"],
                voltage_magnitudes=base_dynamic_scenario["voltage_magnitudes"],
                voltage_angles=base_dynamic_scenario["voltage_angles"],
                coi_frequency=base_dynamic_scenario["coi_frequency"],
                metadata=base_dynamic_scenario["metadata"],
            )

            w_path = extract_causal_window(str(file_path), t_obs=1.0)
            w_dict = extract_causal_window(base_dynamic_scenario, t_obs=1.0)

            np.testing.assert_allclose(w_path.timestamps, w_dict.timestamps)
            np.testing.assert_allclose(w_path.bus_frequencies, w_dict.bus_frequencies)
            np.testing.assert_allclose(w_path.coi_frequency, w_dict.coi_frequency)

    def test_execution_latency_per_scenario(
        self,
        base_dynamic_scenario: Dict[str, Any],
    ) -> None:
        extractor = FeatureExtractor()

        for _ in range(5):
            _ = extractor.extract_features_dict(base_dynamic_scenario, t_obs=1.0)

        n_trials = 5
        n_iters_per_trial = 50
        trial_latencies: List[float] = []

        for _ in range(n_trials):
            start = time.perf_counter()
            for _ in range(n_iters_per_trial):
                _ = extractor.extract_features_dict(base_dynamic_scenario, t_obs=1.0)
            trial_latencies.append((time.perf_counter() - start) / n_iters_per_trial)

        elapsed_per_sample = min(trial_latencies)

        assert elapsed_per_sample < 0.0035, (
            f"Extraction latency exceeded 3.5 ms budget: {elapsed_per_sample * 1000.0:.2f} ms/sample"
        )
