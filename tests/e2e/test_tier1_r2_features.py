"""End-to-end Tier 1 tests for 50 Hz causal PMU windowing and 14-column feature extraction."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List

import numpy as np
import pytest

from pmu_fsa.features.extractor import (
    CANONICAL_FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    DESCRIPTIVE_FEATURE_NAMES,
    FeatureExtractor,
    calc_angle_divergence,
    calc_coi_acceleration,
    calc_delta_f,
    calc_freq_integral,
    calc_initial_rocof,
    calc_interarea_dispersion,
    calc_rocof_metrics,
    extract_feature_vector,
    extract_features,
    wrap_to_pi,
)
from pmu_fsa.features.window_slicer import extract_causal_window
from pmu_fsa.simulation.pmu_recorder import PMUNoiseConfig, PMURecorder
from tests.e2e.conftest import (
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
    assert_strictly_causal,
    generate_synthetic_pmu_stream,
)

# Direct production bindings exported for Tier 2/3/4 cross-module reuse
ref_extract_causal_window = extract_causal_window
ref_calc_initial_rocof = calc_initial_rocof
ref_calc_rocof_metrics = calc_rocof_metrics
ref_calc_delta_f = calc_delta_f
ref_calc_freq_integral = calc_freq_integral
ref_calc_interarea_dispersion = calc_interarea_dispersion
ref_calc_angle_divergence = calc_angle_divergence
ref_calc_coi_acceleration = calc_coi_acceleration


def extract_full_reference_feature_vector(trajectory: Dict[str, Any], t_obs: float) -> List[float]:
    w = extract_causal_window(trajectory, t_obs=t_obs)
    vec = extract_feature_vector(w, f_nominal=50.0)
    return [float(v) for v in vec]


def test_f6_canonical_14_feature_schema_and_window_duration(sample_pmu_trajectory):
    assert len(CANONICAL_FEATURE_NAMES) == 14
    assert len(set(CANONICAL_FEATURE_NAMES)) == 14
    assert CANONICAL_FEATURE_NAMES == DESCRIPTIVE_FEATURE_NAMES == CORE_FEATURE_NAMES

    traj = sample_pmu_trajectory
    t_0 = traj["metadata"]["t_disturb"]
    for t_obs in (0.5, 1.0, 1.5, 2.0):
        w = extract_causal_window(traj, t_obs=t_obs)
        ts = w["timestamps"]
        assert ts[0] >= t_0 - 1e-6
        assert ts[-1] <= t_0 + t_obs + 1e-6
        assert abs((ts[-1] - ts[0]) - t_obs) <= 0.021


def test_f6_zero_future_and_zero_coi_leakage(real_pmu_trajectory):
    for t_obs in (0.5, 1.0, 1.5):
        assert_strictly_causal(extract_full_reference_feature_vector, real_pmu_trajectory, t_obs)

    extractor = FeatureExtractor(f_nominal=50.0)
    clean = extractor.extract_features_dict(real_pmu_trajectory, t_obs=1.0)
    poisoned = copy.deepcopy(real_pmu_trajectory)
    poisoned["coi_frequency"][:] = np.nan
    after_poison = extractor.extract_features_dict(poisoned, t_obs=1.0)
    for k in CANONICAL_FEATURE_NAMES:
        assert after_poison[k] == clean[k]


def test_f6_time_range_sample_counts_and_invalid_windows(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    fps = traj["metadata"]["fps"]
    lengths = []
    for t_obs in (0.5, 1.0, 1.5, 2.0):
        w = extract_causal_window(traj, t_obs=t_obs)
        expected = int(round(t_obs * fps)) + 1
        assert len(w["timestamps"]) == expected
        lengths.append(len(w["timestamps"]))

    assert lengths == sorted(lengths) and len(set(lengths)) == 4

    for bad_t in (-0.5, 0.0):
        with pytest.raises(ValueError, match="strictly positive"):
            extract_causal_window(traj, t_obs=bad_t)


def test_f7_ols_initial_rocof_50hz_ramp_and_noise_resilience(real_pmu_trajectory):
    fps = 50
    dt = 1.0 / fps
    true_slope = -1.75
    t_vals = np.arange(15, dtype=float) * dt
    f_vals = 50.0 + true_slope * t_vals
    w_ramp = {
        "timestamps": t_vals,
        "bus_frequencies": np.tile(f_vals[:, None], (1, 39)),
    }
    assert calc_initial_rocof(w_ramp, delta_t_init=0.10) == pytest.approx(true_slope, abs=1e-10)

    w_real = extract_causal_window(real_pmu_trajectory, t_obs=1.0)
    r_real = calc_initial_rocof(w_real, delta_t_init=0.10)
    assert -5.0 <= r_real < -0.02

    rng = np.random.default_rng(42)
    t_short = np.arange(6, dtype=float) * dt
    ols_errs = []
    ep_errs = []
    for _ in range(100):
        noisy_f = 50.0 + true_slope * t_short + rng.normal(0.0, 0.015, size=len(t_short))
        w_n = {"timestamps": t_short, "bus_frequencies": np.tile(noisy_f[:, None], (1, 39))}
        ols_errs.append(abs(calc_initial_rocof(w_n, delta_t_init=0.10) - true_slope))
        ep_errs.append(abs((noisy_f[-1] - noisy_f[0]) / 0.10 - true_slope))
    assert float(np.mean(ols_errs)) < float(np.mean(ep_errs))


def test_f7_epicenter_bus_and_disturbance_scaling(sample_pmu_trajectory):
    w = extract_causal_window(sample_pmu_trajectory, t_obs=0.5)
    slope_epi = calc_initial_rocof(w, delta_t_init=0.10, bus_idx=34)
    slope_dist = calc_initial_rocof(w, delta_t_init=0.10, bus_idx=29)
    assert slope_epi < 0.0
    assert abs(slope_epi) >= abs(slope_dist)

    s_small = generate_synthetic_pmu_stream(active_loss_mw=250.0, f_nadir=49.72)
    s_large = generate_synthetic_pmu_stream(active_loss_mw=900.0, f_nadir=49.05)
    r_small = calc_initial_rocof(extract_causal_window(s_small, t_obs=0.5))
    r_large = calc_initial_rocof(extract_causal_window(s_large, t_obs=0.5))
    assert abs(r_large) > abs(r_small)


def test_f8_causal_rocof_metrics_ordering(real_pmu_trajectory):
    for t_obs in (0.5, 1.0, 1.5):
        w = extract_causal_window(real_pmu_trajectory, t_obs=t_obs)
        pk_neg, r_end, mean_r = calc_rocof_metrics(w)
        assert pk_neg < 0.0
        assert pk_neg <= r_end + 1e-9
        assert pk_neg <= mean_r + 1e-9
        assert -15.0 <= pk_neg <= 0.0


def test_f9_frequency_drop_metrics_at_50hz(sample_pmu_trajectory, real_pmu_trajectory):
    drops = []
    for t_obs in (0.5, 1.0, 1.5):
        w = extract_causal_window(sample_pmu_trajectory, t_obs=t_obs)
        drop_end, drop_max = calc_delta_f(w, f_nominal=50.0)
        assert drop_end < 0.0
        assert drop_max <= drop_end + 1e-6
        drops.append(abs(drop_end))
    assert drops[0] < drops[1] < drops[2]

    feats_real = extract_features(extract_causal_window(real_pmu_trajectory, t_obs=1.0), f_nominal=50.0)
    assert feats_real["bus_drop_worst"] <= feats_real["freq_drop_max"] + 1e-9


def test_f10_trapezoidal_50hz_frequency_integral(sample_pmu_trajectory):
    t_vals = np.linspace(0.0, 1.0, 51)
    c = 0.80
    f_vals = 50.0 - c * t_vals
    w_lin = {"timestamps": t_vals, "bus_frequencies": np.tile(f_vals[:, None], (1, 39))}
    integ_dev, integ_abs = calc_freq_integral(w_lin, f_nominal=50.0)
    assert integ_dev == pytest.approx(-0.5 * c, abs=1e-10)
    assert integ_abs == pytest.approx(0.5 * c, abs=1e-10)

    integrals = []
    for t_obs in (0.5, 1.0, 1.5, 2.0):
        w = extract_causal_window(sample_pmu_trajectory, t_obs=t_obs)
        idv, iab = calc_freq_integral(w, f_nominal=50.0)
        assert idv < 0.0 and iab > 0.0
        integrals.append(abs(idv))
    assert integrals == sorted(integrals) and len(set(integrals)) == 4


def test_f11_interarea_dispersion_and_bus_std(real_pmu_trajectory):
    w = extract_causal_window(real_pmu_trajectory, t_obs=1.0)
    bf_end = w["bus_frequencies"][-1]
    for bus_list in (AREA_1_BUSES, AREA_2_BUSES, AREA_3_BUSES):
        mean_f = float(np.mean([bf_end[b - 1] for b in bus_list]))
        assert 45.0 <= mean_f <= 50.5

    spread_max, bus_std_mean = calc_interarea_dispersion(w)
    assert 0.0 < spread_max <= 2.0
    assert bus_std_mean > 0.0


def test_f12_wrap_safe_angle_divergence_in_degrees(real_pmu_trajectory):
    w = extract_causal_window(real_pmu_trajectory, t_obs=1.0)
    max_div_deg, tie_deg = calc_angle_divergence(w)
    assert 0.0 < max_div_deg < 180.0
    assert 0.0 < tie_deg <= max_div_deg + 1e-9

    # Circular wrap across +/-180 degrees (+179 deg -> -179 deg is 2.0 deg)
    angles = np.full((3, 39), 179.0)
    angles[-1, 4] = -179.0
    w_wrap = {"timestamps": np.array([0.0, 0.02, 0.04]), "voltage_angles": angles}
    div_w, tie_w = calc_angle_divergence(w_wrap)
    assert div_w == pytest.approx(2.0, abs=1e-6)
    assert tie_w == pytest.approx(2.0, abs=1e-6)
    assert abs(math.degrees(float(wrap_to_pi(math.radians(179.0) - math.radians(-179.0))))) == pytest.approx(2.0, abs=1e-6)


def test_f13_frequency_curvature_hz_per_s2_and_noise_propagation():
    fps = 50
    t = np.arange(0.0, 1.2 + 1e-9, 1.0 / fps)
    r0 = -0.75
    curv_true = 0.55  # d^2f/dt^2 in Hz/s^2
    f_quad = 50.0 + r0 * t + 0.5 * curv_true * (t ** 2)

    w_quad = {
        "timestamps": t,
        "bus_frequencies": np.tile(f_quad[:, None], (1, 39)),
        "metadata": {"fps": fps, "f_nominal": 50.0},
    }
    curv_init, curv_mean = calc_coi_acceleration(w_quad)
    assert curv_init == pytest.approx(curv_true, abs=1e-3)
    assert curv_mean == pytest.approx(curv_true, abs=1e-3)
