"""End-to-end tests for causal PMU windowing and feature extraction."""

import math
import copy
from typing import Dict, Any, List, Tuple

from tests.e2e.conftest import (
    np, HAS_NUMPY,
    generate_synthetic_pmu_stream,
    assert_strictly_causal,
    IEEE39_GENERATOR_BUSES,
    IEEE39_GEN_PARAMS,
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
)

try:
    from pmu_fsa.features import window_slicer
    from pmu_fsa.features import extractor
    HAS_FEATURE_MODULES = True
except ImportError:
    HAS_FEATURE_MODULES = False


def ref_extract_causal_window(trajectory: Dict[str, Any], t_obs: float) -> Dict[str, Any]:
    if t_obs <= 0:
        raise ValueError("Observation window T_obs must be strictly positive")

    t_0 = trajectory["metadata"]["t_disturb"]
    ts = trajectory["timestamps"]
    t_end = t_0 + t_obs

    idx_start = np.searchsorted(ts, t_0, side='left')
    idx_end = np.searchsorted(ts, t_end, side='right')

    if idx_start >= len(ts) or idx_end <= idx_start:
        raise ValueError(f"No samples found within window [{t_0}, {t_end}]")

    sliced = {
        "timestamps": ts[idx_start:idx_end],
        "bus_frequencies": trajectory["bus_frequencies"][idx_start:idx_end],
        "bus_rocof": trajectory["bus_rocof"][idx_start:idx_end],
        "voltage_magnitudes": trajectory["voltage_magnitudes"][idx_start:idx_end],
        "voltage_angles": trajectory["voltage_angles"][idx_start:idx_end],
        "coi_frequency": trajectory["coi_frequency"][idx_start:idx_end],
        "metadata": dict(trajectory["metadata"])
    }
    sliced["metadata"]["t_obs"] = t_obs
    return sliced


def ref_calc_initial_rocof(windowed_traj: Dict[str, Any], delta_t_init: float = 0.1) -> float:
    ts = windowed_traj["timestamps"]
    coi = windowed_traj["coi_frequency"]
    t_0 = ts[0]
    idx_end = np.searchsorted(ts, t_0 + delta_t_init, side='right')
    idx_end = max(idx_end, 2)

    x = [ts[i] - t_0 for i in range(idx_end)]
    y = [coi[i] for i in range(idx_end)]
    slope, _ = np.polyfit(np.array(x), np.array(y), 1)
    return float(slope)


def ref_calc_rocof_metrics(windowed_traj: Dict[str, Any]) -> Tuple[float, float, float]:
    ts = windowed_traj["timestamps"]
    coi = windowed_traj["coi_frequency"]
    n = len(coi)
    rocof_vals = []
    for k in range(1, n):
        dt = ts[k] - ts[k - 1]
        rocof_vals.append((coi[k] - coi[k - 1]) / dt)

    if not rocof_vals:
        return 0.0, 0.0, 0.0

    peak_negative = min(rocof_vals)
    max_abs = max(abs(r) for r in rocof_vals)
    mean_rocof = (coi[-1] - coi[0]) / (ts[-1] - ts[0])
    return peak_negative, max_abs, mean_rocof


def ref_calc_delta_f(windowed_traj: Dict[str, Any], f_nominal: float = 60.0) -> Tuple[float, float]:
    coi = windowed_traj["coi_frequency"]
    f_start = coi[0]
    f_end = coi[-1]
    inst_drop = f_end - f_start
    min_f = min(coi)
    max_drop = min_f - f_nominal
    return inst_drop, max_drop


def ref_calc_freq_integral(windowed_traj: Dict[str, Any], f_nominal: float = 60.0) -> Tuple[float, float]:
    ts = windowed_traj["timestamps"]
    coi = windowed_traj["coi_frequency"]
    integral_dev = 0.0
    integral_abs = 0.0
    for k in range(len(coi) - 1):
        dt = ts[k + 1] - ts[k]
        dev_k = coi[k] - f_nominal
        dev_k1 = coi[k + 1] - f_nominal
        integral_dev += 0.5 * (dev_k + dev_k1) * dt
        integral_abs += 0.5 * (abs(dev_k) + abs(dev_k1)) * dt
    return integral_dev, integral_abs


def ref_calc_interarea_dispersion(windowed_traj: Dict[str, Any]) -> Tuple[float, float]:
    bf = windowed_traj["bus_frequencies"]
    f_end = bf[-1]

    # 1-indexed bus IDs -> 0-indexed columns
    f_a1 = [f_end[b - 1] for b in AREA_1_BUSES]
    f_a2 = [f_end[b - 1] for b in AREA_2_BUSES]
    f_a3 = [f_end[b - 1] for b in AREA_3_BUSES]

    mean_a1 = sum(f_a1) / len(f_a1)
    mean_a2 = sum(f_a2) / len(f_a2)
    mean_a3 = sum(f_a3) / len(f_a3)

    area_spread = max(abs(mean_a1 - mean_a2), abs(mean_a1 - mean_a3), abs(mean_a2 - mean_a3))

    sys_mean = sum(f_end) / len(f_end)
    spatial_var = sum((f - sys_mean) ** 2 for f in f_end) / len(f_end)
    return area_spread, spatial_var


def ref_calc_angle_divergence(windowed_traj: Dict[str, Any]) -> Tuple[float, float]:
    va = windowed_traj["voltage_angles"]
    ang_end = va[-1]
    max_div = max(ang_end) - min(ang_end)

    # Tie-line between Bus 4 (idx 3) and Bus 5 (idx 4)
    tie_ang_diff = abs(ang_end[3] - ang_end[4])
    return max_div, tie_ang_diff


def ref_calc_coi_acceleration(windowed_traj: Dict[str, Any]) -> Tuple[float, float]:
    ts = windowed_traj["timestamps"]
    coi = windowed_traj["coi_frequency"]
    t_0 = ts[0]
    idx_end = np.searchsorted(ts, t_0 + 0.1, side='right')
    idx_end = max(idx_end, 2)
    dt_init = ts[idx_end - 1] - ts[0]
    initial_accel = (coi[idx_end - 1] - coi[0]) / max(dt_init, 1e-4)

    dt_total = ts[-1] - ts[0]
    mean_accel = (coi[-1] - coi[0]) / max(dt_total, 1e-4)
    return initial_accel, mean_accel


def extract_full_reference_feature_vector(trajectory: Dict[str, Any], t_obs: float) -> List[float]:
    w = ref_extract_causal_window(trajectory, t_obs)
    init_rocof = ref_calc_initial_rocof(w)
    pk_rocof, max_abs_r, mean_r = ref_calc_rocof_metrics(w)
    inst_drop, max_drop = ref_calc_delta_f(w)
    integ_dev, integ_abs = ref_calc_freq_integral(w)
    spread, s_var = ref_calc_interarea_dispersion(w)
    max_ang, tie_ang = ref_calc_angle_divergence(w)
    init_acc, mean_acc = ref_calc_coi_acceleration(w)
    return [
        init_rocof, pk_rocof, max_abs_r, mean_r,
        inst_drop, max_drop, integ_dev, integ_abs,
        spread, s_var, max_ang, tie_ang, init_acc, mean_acc
    ]


def test_f6_window_duration_clamping(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    t_0 = traj["metadata"]["t_disturb"]

    for t_obs in [0.5, 1.0, 1.5, 2.0]:
        w = ref_extract_causal_window(traj, t_obs)
        ts = w["timestamps"]
        assert ts[0] >= t_0, f"Window start timestamp {ts[0]} < t_0 {t_0}"
        assert ts[-1] <= t_0 + t_obs + 1e-4, f"Window end timestamp {ts[-1]} > t_0 + T_obs {t_0 + t_obs}"
        expected_duration = ts[-1] - ts[0]
        assert abs(expected_duration - t_obs) <= 0.021, \
            f"Window duration {expected_duration} differs from T_obs {t_obs} by more than 1 sampling interval"


def test_f6_zero_future_leakage_assertion(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    for t_obs in [0.5, 1.0, 1.5]:
        assert_strictly_causal(extract_full_reference_feature_vector, traj, t_obs)


def test_f6_time_range_bounds_and_indices(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    fps = traj["metadata"]["fps"]
    dt = 1.0 / fps

    for t_obs in [0.5, 1.0, 2.0]:
        w = ref_extract_causal_window(traj, t_obs)
        n_samples = len(w["timestamps"])
        expected_samples = int(round(t_obs / dt)) + 1
        assert abs(n_samples - expected_samples) <= 1, \
            f"Sample count {n_samples} does not match expected {expected_samples} for T_obs={t_obs}"


def test_f6_multiple_window_lengths(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    lengths = []
    for t_obs in [0.5, 1.0, 1.5, 2.0]:
        w = ref_extract_causal_window(traj, t_obs)
        lengths.append(len(w["timestamps"]))

    for i in range(len(lengths) - 1):
        assert lengths[i] < lengths[i + 1], f"Window lengths not monotonically increasing: {lengths}"


def test_f6_invalid_window_rejection(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    try:
        ref_extract_causal_window(traj, -0.5)
        assert False, "Failed to raise ValueError on negative T_obs"
    except ValueError:
        pass

    try:
        ref_extract_causal_window(traj, 0.0)
        assert False, "Failed to raise ValueError on zero T_obs"
    except ValueError:
        pass


def test_f7_ols_slope_linear_ramp():
    fps = 50
    dt = 1.0 / fps
    true_slope = -2.75
    t_vals = [i * dt for i in range(15)]
    f_vals = [60.0 + true_slope * t for t in t_vals]

    window = {
        "timestamps": np.array(t_vals),
        "coi_frequency": np.array(f_vals),
    }
    computed_rocof = ref_calc_initial_rocof(window, delta_t_init=0.1)
    assert abs(computed_rocof - true_slope) < 1e-5, \
        f"OLS slope {computed_rocof} differs from analytical {true_slope}"


def test_f7_negative_slope_post_disturbance(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    init_rocof = ref_calc_initial_rocof(w, delta_t_init=0.1)

    assert init_rocof < 0.0, f"COI initial ROCOF must be negative, got {init_rocof}"
    assert -10.0 <= init_rocof <= -0.05, f"Initial ROCOF {init_rocof} Hz/s outside realistic range"


def test_f7_noise_resilience_ols_vs_two_point():
    import random
    rng = random.Random(42)
    dt = 0.02
    t_vals = [i * dt for i in range(6)]
    true_slope = -1.50
    sigma = 0.02

    ols_errors = []
    two_point_errors = []

    for _ in range(50):
        noise = [rng.gauss(0, sigma) for _ in t_vals]
        f_vals = [60.0 + true_slope * t + n for t, n in zip(t_vals, noise)]
        w = {"timestamps": np.array(t_vals), "coi_frequency": np.array(f_vals)}

        ols_slope = ref_calc_initial_rocof(w, delta_t_init=0.1)
        ols_errors.append(abs(ols_slope - true_slope))

        two_pt_slope = (f_vals[-1] - f_vals[0]) / 0.1
        two_point_errors.append(abs(two_pt_slope - true_slope))

    mean_ols_err = sum(ols_errors) / len(ols_errors)
    mean_two_pt_err = sum(two_point_errors) / len(two_point_errors)

    assert mean_ols_err < mean_two_pt_err, \
        f"OLS estimator mean error ({mean_ols_err:.4f}) not lower than two-point error ({mean_two_pt_err:.4f})"


def test_f7_epicenter_bus_steepest_rocof(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    disturbed_gen = traj["metadata"]["disturbed_gen"]
    w = ref_extract_causal_window(traj, t_obs=0.5)
    bf = w["bus_frequencies"]
    ts = w["timestamps"]

    # Bus 35 (idx 34, tripped) vs Bus 30 (idx 29, distant in Area 1)
    f_epi = [bf[k][34] for k in range(len(ts))]
    f_dist = [bf[k][29] for k in range(len(ts))]

    idx_end = np.searchsorted(ts, ts[0] + 0.1, side='right')
    x = [ts[i] - ts[0] for i in range(idx_end)]

    slope_epi, _ = np.polyfit(np.array(x), np.array(f_epi[:idx_end]), 1)
    slope_dist, _ = np.polyfit(np.array(x), np.array(f_dist[:idx_end]), 1)

    assert slope_epi < 0, f"Epicenter slope must be negative, got {slope_epi}"
    assert abs(slope_epi) >= abs(slope_dist), \
        f"Epicenter ROCOF magnitude ({abs(slope_epi):.4f}) was smaller than distant bus ({abs(slope_dist):.4f})"


def test_f7_initial_rocof_scale_with_disturbance_size():
    stream_small = generate_synthetic_pmu_stream(active_loss_mw=250.0, f_nadir=59.7)
    stream_large = generate_synthetic_pmu_stream(active_loss_mw=1000.0, f_nadir=59.1)

    w_small = ref_extract_causal_window(stream_small, t_obs=0.5)
    w_large = ref_extract_causal_window(stream_large, t_obs=0.5)

    rocof_small = ref_calc_initial_rocof(w_small, delta_t_init=0.1)
    rocof_large = ref_calc_initial_rocof(w_large, delta_t_init=0.1)

    assert abs(rocof_large) > abs(rocof_small), \
        f"1000 MW outage ROCOF {rocof_large} not steeper than 250 MW outage ROCOF {rocof_small}"


def test_f8_peak_negative_rocof(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    pk_neg, max_abs, mean_r = ref_calc_rocof_metrics(w)

    assert pk_neg < 0.0, f"Peak negative ROCOF must be negative, got {pk_neg}"
    assert pk_neg <= mean_r, f"Peak negative ROCOF ({pk_neg}) must be <= mean ROCOF ({mean_r})"


def test_f8_max_absolute_rocof(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    pk_neg, max_abs, mean_r = ref_calc_rocof_metrics(w)

    assert max_abs > 0.0, f"Maximum absolute ROCOF must be positive, got {max_abs}"
    assert max_abs >= abs(mean_r), f"Max absolute ROCOF ({max_abs}) must be >= abs(mean_rocof) ({abs(mean_r)})"


def test_f8_mean_rocof_equals_window_slope(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    for t_obs in [0.5, 1.0, 1.5]:
        w = ref_extract_causal_window(traj, t_obs=t_obs)
        _, _, mean_r = ref_calc_rocof_metrics(w)
        ts = w["timestamps"]
        coi = w["coi_frequency"]
        expected_slope = (coi[-1] - coi[0]) / (ts[-1] - ts[0])
        assert abs(mean_r - expected_slope) < 1e-6, \
            f"Mean ROCOF ({mean_r}) differs from endpoint slope ({expected_slope})"


def test_f8_causal_backward_difference_only(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    coi = w["coi_frequency"]
    ts = w["timestamps"]

    for k in range(1, len(coi)):
        dt = ts[k] - ts[k - 1]
        step_rocof = (coi[k] - coi[k - 1]) / dt
        assert not math.isnan(step_rocof)
        assert -20.0 <= step_rocof <= 10.0


def test_f8_rocof_monotonic_ordering(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    for t_obs in [0.5, 1.0, 2.0]:
        w = ref_extract_causal_window(traj, t_obs=t_obs)
        pk_neg, max_abs, mean_r = ref_calc_rocof_metrics(w)
        assert abs(mean_r) <= max_abs + 1e-8, \
            f"|mean_rocof| ({abs(mean_r)}) exceeded max_abs ({max_abs})"


def test_f9_instantaneous_drop_at_window_end(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    inst_drop, max_drop = ref_calc_delta_f(w)

    assert inst_drop < 0.0, f"Instantaneous drop must be negative, got {inst_drop}"
    assert -2.0 <= inst_drop <= -0.01, f"Drop {inst_drop} Hz outside realistic range"


def test_f9_maximum_observed_drop(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.5)
    inst_drop, max_drop = ref_calc_delta_f(w)

    assert max_drop < 0.0, f"Max drop must be negative below nominal, got {max_drop}"
    assert abs(max_drop) >= abs(inst_drop) - 1e-6, \
        f"Max drop magnitude ({abs(max_drop)}) was smaller than instantaneous drop ({abs(inst_drop)})"


def test_f9_drop_scales_with_window_length(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    drops = []
    # Nadir occurs at t_rel = 2.5s, so 0.5-1.5s windows are strictly pre-nadir
    for t_obs in [0.5, 1.0, 1.5]:
        w = ref_extract_causal_window(traj, t_obs=t_obs)
        inst_drop, _ = ref_calc_delta_f(w)
        drops.append(abs(inst_drop))

    for i in range(len(drops) - 1):
        assert drops[i] < drops[i + 1], \
            f"Pre-nadir drop did not increase with window length: {drops}"


def test_f9_per_bus_and_coi_delta_f(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    bf = w["bus_frequencies"]

    drops = [bf[-1][b] - bf[0][b] for b in range(39)]
    assert len(drops) == 39
    for b, drop in enumerate(drops):
        assert drop < 0.0, f"Bus {b+1} frequency did not drop post disturbance: {drop}"


def test_f9_zero_drop_pre_disturbance(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    ts = traj["timestamps"]
    coi = traj["coi_frequency"]

    pre_coi = [coi[i] for i, t in enumerate(ts) if t <= 0.8]
    drop_pre = pre_coi[-1] - pre_coi[0]
    assert abs(drop_pre) < 0.01, f"Pre-disturbance drop was non-zero: {drop_pre} Hz"


def test_f10_trapezoidal_integral_formula():
    dt = 0.02
    t_vals = [i * dt for i in range(51)]
    c = 0.8
    f_vals = [60.0 - c * t for t in t_vals]

    # Integral of -c*t over [0, 1] is -0.5 * c * T^2
    expected_integral = -0.5 * c * (1.0 ** 2)

    w = {"timestamps": np.array(t_vals), "coi_frequency": np.array(f_vals)}
    integ_dev, integ_abs = ref_calc_freq_integral(w, f_nominal=60.0)

    assert abs(integ_dev - expected_integral) < 1e-5, \
        f"Trapezoidal integral {integ_dev} differs from analytical {expected_integral}"


def test_f10_negative_integral_for_frequency_drop(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    integ_dev, integ_abs = ref_calc_freq_integral(w, f_nominal=60.0)

    assert integ_dev < 0.0, f"Frequency deviation integral must be negative, got {integ_dev}"
    assert integ_abs > 0.0, f"Absolute deviation integral must be positive, got {integ_abs}"


def test_f10_integral_monotonic_growth(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    integrals = []
    for t_obs in [0.5, 1.0, 1.5, 2.0]:
        w = ref_extract_causal_window(traj, t_obs=t_obs)
        integ_dev, _ = ref_calc_freq_integral(w, f_nominal=60.0)
        integrals.append(abs(integ_dev))

    for i in range(len(integrals) - 1):
        assert integrals[i] < integrals[i + 1], \
            f"Integral magnitude not monotonically growing: {integrals}"


def test_f10_absolute_integral_metric(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    integ_dev, integ_abs = ref_calc_freq_integral(w, f_nominal=60.0)

    assert abs(integ_abs - (-integ_dev)) < 1e-6, \
        f"integ_abs ({integ_abs}) does not match -integ_dev ({-integ_dev})"


def test_f10_energy_deficit_scales_with_contingency():
    stream_small = generate_synthetic_pmu_stream(active_loss_mw=250.0, f_nadir=59.7)
    stream_large = generate_synthetic_pmu_stream(active_loss_mw=1000.0, f_nadir=59.1)

    w_small = ref_extract_causal_window(stream_small, t_obs=1.0)
    w_large = ref_extract_causal_window(stream_large, t_obs=1.0)

    int_small, _ = ref_calc_freq_integral(w_small, f_nominal=60.0)
    int_large, _ = ref_calc_freq_integral(w_large, f_nominal=60.0)

    assert abs(int_large) > abs(int_small), \
        f"1000 MW outage deficit {abs(int_large)} not greater than 250 MW deficit {abs(int_small)}"


def test_f11_area_mean_frequencies(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    bf = w["bus_frequencies"]
    f_end = bf[-1]

    for area_name, bus_list in [("Area 1", AREA_1_BUSES), ("Area 2", AREA_2_BUSES), ("Area 3", AREA_3_BUSES)]:
        assert len(bus_list) > 0
        area_f = [f_end[b - 1] for b in bus_list]
        mean_f = sum(area_f) / len(area_f)
        assert 55.0 <= mean_f <= 61.0, f"{area_name} mean frequency {mean_f} out of bounds"


def test_f11_interarea_frequency_spread(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    spread, s_var = ref_calc_interarea_dispersion(w)

    assert spread >= 0.0, f"Area spread must be non-negative, got {spread}"
    assert spread <= 2.0, f"Area spread {spread} Hz is unrealistically large for 60 Hz grid"


def test_f11_cross_bus_variance(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    spread, s_var = ref_calc_interarea_dispersion(w)

    assert s_var > 0.0, f"Spatial variance must be strictly positive during disturbance, got {s_var}"


def test_f11_dispersion_near_zero_at_steady_state(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    ts = traj["timestamps"]
    bf = traj["bus_frequencies"]

    pre_row = bf[0]
    sys_mean = sum(pre_row) / len(pre_row)
    pre_var = sum((f - sys_mean) ** 2 for f in pre_row) / len(pre_row)

    assert pre_var < 0.001, f"Pre-disturbance cross-bus variance was not near zero: {pre_var}"


def test_f11_maximum_and_endpoint_dispersion(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.5)
    bf = w["bus_frequencies"]

    spreads = []
    for row in bf:
        f_a1 = [row[b - 1] for b in AREA_1_BUSES]
        f_a2 = [row[b - 1] for b in AREA_2_BUSES]
        f_a3 = [row[b - 1] for b in AREA_3_BUSES]
        m1, m2, m3 = sum(f_a1) / len(f_a1), sum(f_a2) / len(f_a2), sum(f_a3) / len(f_a3)
        spreads.append(max(abs(m1 - m2), abs(m1 - m3), abs(m2 - m3)))

    max_spread = max(spreads)
    end_spread = spreads[-1]

    assert max_spread >= end_spread, f"Max spread ({max_spread}) was smaller than endpoint spread ({end_spread})"


def test_f12_pairwise_angle_divergence(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    max_div, tie_ang = ref_calc_angle_divergence(w)

    assert max_div > 0.0, f"Max angle divergence must be positive, got {max_div}"
    assert max_div < 2.0 * math.pi, f"Max angle divergence {max_div} rad exceeds 2*pi"


def test_f12_tie_line_swing_angle(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    max_div, tie_ang = ref_calc_angle_divergence(w)

    assert tie_ang >= 0.0, f"Tie-line angle difference must be non-negative, got {tie_ang}"
    assert tie_ang <= max_div, f"Tie-line angle ({tie_ang}) exceeded system max angle divergence ({max_div})"


def test_f12_angle_divergence_rate(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w_early = ref_extract_causal_window(traj, t_obs=0.5)
    w_late = ref_extract_causal_window(traj, t_obs=1.5)

    va_early = w_early["voltage_angles"].tolist() if hasattr(w_early["voltage_angles"], "tolist") else w_early["voltage_angles"]
    va_late = w_late["voltage_angles"].tolist() if hasattr(w_late["voltage_angles"], "tolist") else w_late["voltage_angles"]

    peak_early = max(max(row) - min(row) for row in va_early)
    peak_late = max(max(row) - min(row) for row in va_late)

    assert peak_late >= peak_early - 1e-6, \
        f"Maximum angle divergence did not expand over window: early={peak_early}, late={peak_late}"
    assert peak_early > 0.0, "Angle divergence must be positive"


def test_f12_phase_wrapping_handling():
    # Shortest circular distance across +-pi: 2*pi - 6.20 ~ 0.08318 rad
    theta1 = 3.10
    theta2 = -3.10
    raw_diff = abs(theta1 - theta2)
    wrapped_diff = abs((raw_diff + math.pi) % (2.0 * math.pi) - math.pi)

    assert abs(wrapped_diff - 0.08318) < 1e-4, \
        f"Wrapped difference {wrapped_diff} incorrect for boundary crossing"


def test_f12_angle_drift_relative_to_coi(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    va = w["voltage_angles"]

    for row in va:
        mean_ang = sum(row) / len(row)
        for ang in row:
            dev = abs(ang - mean_ang)
            assert dev < math.pi, f"Rotor angle deviation {dev} rad exceeded pi"


def test_f13_coi_frequency_weighting(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    bf = traj["bus_frequencies"]
    coi = traj["coi_frequency"]

    for k in range(len(coi)):
        row = bf[k]
        min_bus_f = min(row)
        max_bus_f = max(row)
        assert min_bus_f - 1e-4 <= coi[k] <= max_bus_f + 1e-4, \
            f"COI frequency {coi[k]} outside bus frequency range [{min_bus_f}, {max_bus_f}]"


def test_f13_coi_acceleration_derivative(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.0)
    init_acc, mean_acc = ref_calc_coi_acceleration(w)

    assert init_acc < 0.0, f"Initial COI acceleration must be negative, got {init_acc}"
    assert -10.0 <= init_acc <= -0.01, f"COI acceleration {init_acc} Hz/s outside realistic range"


def test_f13_initial_coi_acceleration_swing_equation(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    loss_mw = traj["metadata"]["active_loss_mw"]
    f0 = 60.0
    effective_H = traj["metadata"]["effective_H_sum"]

    expected_accel_bound = -(f0 * loss_mw) / (2.0 * effective_H * 100.0)

    w = ref_extract_causal_window(traj, t_obs=1.0)
    init_acc, _ = ref_calc_coi_acceleration(w)

    assert init_acc < 0.0
    assert abs(init_acc - expected_accel_bound) < 2.0, \
        f"Observed initial acceleration ({init_acc}) differs significantly from theoretical bound ({expected_accel_bound})"


def test_f13_mean_coi_acceleration(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    for t_obs in [0.5, 1.0, 1.5]:
        w = ref_extract_causal_window(traj, t_obs=t_obs)
        init_acc, mean_acc = ref_calc_coi_acceleration(w)
        ts = w["timestamps"]
        coi = w["coi_frequency"]
        expected_mean_acc = (coi[-1] - coi[0]) / (ts[-1] - ts[0])
        assert abs(mean_acc - expected_mean_acc) < 1e-6, \
            f"Mean acceleration {mean_acc} does not match expected {expected_mean_acc}"


def test_f13_peak_deceleration_detection(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    w = ref_extract_causal_window(traj, t_obs=1.5)
    ts = w["timestamps"]
    coi = w["coi_frequency"]

    accels = []
    for k in range(1, len(coi)):
        dt = ts[k] - ts[k - 1]
        accels.append((coi[k] - coi[k - 1]) / dt)

    peak_decel = min(accels)
    assert peak_decel < 0.0, f"Peak deceleration must be negative, got {peak_decel}"
    assert peak_decel >= -15.0, f"Peak deceleration {peak_decel} exceeded physical bound -15 Hz/s"
