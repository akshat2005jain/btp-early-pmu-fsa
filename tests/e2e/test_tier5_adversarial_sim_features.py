"""Stress and numerical edge-case E2E tests for 50 Hz simulation and 14-column feature extraction."""

from __future__ import annotations

import copy
import struct
import time

import numpy as np
import pytest

from pmu_fsa.features.extractor import (
    CANONICAL_FEATURE_NAMES,
    CORE_FEATURE_NAMES,
    DESCRIPTIVE_FEATURE_NAMES,
    _validate_and_clean_nans,
    calc_freq_integral,
    extract_features,
)
from pmu_fsa.features.window_slicer import CausalWindow, extract_causal_window
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator, compute_kron_reduction
from pmu_fsa.simulation.ieee39_data import NOMINAL_FREQ, get_ieee39_gen_data
from pmu_fsa.simulation.pmu_recorder import (
    PMUNoiseConfig,
    PMURecorder,
    apply_pmu_measurement_noise,
    compute_causal_rocof,
    compute_crc_ccitt,
    compute_fir_rocof,
    compute_tve,
)
from pmu_fsa.simulation.renewables import (
    DEFAULT_DISPLACED_BUSES,
    RenewablePenetrationConfig,
    apply_renewable_displacement,
    compute_effective_inertias,
    summarize_renewable_dispatch,
)


def test_t5_sim_kron_reduction_zero_and_heavy_load():
    assert NOMINAL_FREQ == 50.0
    Y_zero, K_zero, *_ = compute_kron_reduction(load_scaling=0.0, return_all=True)
    assert Y_zero.shape == (10, 10) and K_zero.shape == (39, 10)
    assert np.linalg.cond(Y_zero) < 35.0
    assert np.all(np.real(np.diag(Y_zero)) > 0.02)

    Y_heavy, *_ = compute_kron_reduction(load_scaling=2.0, return_all=True)
    assert Y_heavy.shape == (10, 10)
    assert np.all(np.isfinite(Y_heavy))


def test_t5_sim_zero_and_extreme_damping_stability():
    sim_zero = DynamicSimulator(load_scaling=1.0)
    sim_zero.D[:] = 0.0
    out_z = sim_zero.simulate(t_span=(0.0, 2.0), t_disturb=0.5, tripped_bus=35, p_loss_mw=650.0, fps=50)
    assert np.all(np.isfinite(out_z["bus_frequencies"]))
    assert (50.0 - 5.0) <= out_z["metadata"]["ground_truth_nadir"] <= 50.0

    sim_stiff = DynamicSimulator(load_scaling=1.0)
    sim_stiff.D[:] = 50.0
    out_s = sim_stiff.simulate(t_span=(0.0, 1.5), t_disturb=0.5, tripped_bus=35, p_loss_mw=650.0, fps=50)
    assert np.all(np.isfinite(out_s["bus_frequencies"]))
    assert out_s["metadata"]["ground_truth_nadir"] >= 49.60


def test_t5_sim_zero_and_ultra_low_inertia():
    sim = DynamicSimulator(load_scaling=1.0)
    sim.H[0] = 0.0
    y0 = np.concatenate([sim.delta0, np.zeros(sim.n_gen), sim.Pm0])
    with pytest.raises(ValueError, match="strictly positive"):
        sim.rhs(0.0, y0, 1.0, 650.0, 0)

    sim_low = DynamicSimulator(load_scaling=1.0)
    sim_low.H[:] = 0.2
    sim_low.H_tot = float(np.sum(sim_low.H))
    out = sim_low.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, p_loss_mw=650.0, fps=50)
    assert out["metadata"]["ground_truth_nadir"] < 46.0
    assert out["metadata"]["is_secure"] == 0
    assert np.all(np.isfinite(out["bus_frequencies"]))


def test_t5_sim_non_generator_bus_trip_warning_and_flat_identity():
    sim = DynamicSimulator()
    with pytest.warns(UserWarning, match="not a valid generator bus"):
        out = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=15, p_loss_mw=650.0, fps=50)
    assert out["metadata"]["ground_truth_nadir"] == pytest.approx(50.0, abs=1e-8)

    out_flat = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=None, p_loss_mw=0.0, fps=50)
    assert np.allclose(out_flat["bus_frequencies"], 50.0, atol=1e-9)
    assert out_flat["metadata"]["is_secure"] == 1


def test_t5_sim_overfrequency_post_end_disturbance_zero_voltage_and_solvers():
    sim = DynamicSimulator()
    out_over = sim.simulate(t_span=(0.0, 1.0), t_disturb=0.2, tripped_bus=35, p_loss_mw=-500.0, fps=50)
    assert float(np.max(out_over["bus_frequencies"])) > 50.03
    assert out_over["metadata"]["is_secure"] == 1

    out_late = sim.simulate(t_span=(0.0, 1.0), t_disturb=5.0, tripped_bus=35, p_loss_mw=650.0, fps=50)
    assert out_late["metadata"]["ground_truth_nadir"] == pytest.approx(50.0, abs=1e-8)

    sim_copy = copy.deepcopy(sim)
    sim_copy.K[0, :] = 0.0
    _, _, bus_freq, _ = sim_copy.reconstruct_bus_states(np.zeros((1, 10)), np.ones((1, 10)) * 0.01)
    assert np.isfinite(bus_freq[0, 0])

    out_rk = sim.simulate(t_span=(0.0, 1.5), t_disturb=0.2, tripped_bus=35, loss_fraction=1.0, fps=50, method="RK45")
    out_rad = sim.simulate(t_span=(0.0, 1.5), t_disturb=0.2, tripped_bus=35, loss_fraction=1.0, fps=50, method="Radau")
    assert abs(out_rk["metadata"]["ground_truth_nadir"] - out_rad["metadata"]["ground_truth_nadir"]) < 0.02


def test_t5_ren_linear_non_saturating_displacement_and_alpha_cap():
    gen_data = get_ieee39_gen_data(macro_intertie=False)
    assert len(DEFAULT_DISPLACED_BUSES) == 9

    h_vals = []
    realized = []
    for p_ren in (0.10, 0.25, 0.40, 0.45, 0.50):
        mod = apply_renewable_displacement(gen_data, RenewablePenetrationConfig(p_ren=p_ren))
        summary = summarize_renewable_dispatch(mod)
        assert summary["p_ren_realized"] == pytest.approx(p_ren, rel=0.01)
        realized.append(summary["p_ren_realized"])
        h_vals.append(summary["h_eff_total"])

    for i in range(len(h_vals) - 1):
        assert realized[i] < realized[i + 1]
        assert h_vals[i] > h_vals[i + 1]
    assert realized[-1] / realized[0] == pytest.approx(5.0, rel=1e-6)

    # Extreme p_ren=0.99 caps alpha at 0.95 so synchronous fleet never hits singular zero inertia
    mod_ext = apply_renewable_displacement(gen_data, RenewablePenetrationConfig(p_ren=0.99))
    sum_ext = summarize_renewable_dispatch(mod_ext)
    assert sum_ext["alpha"] == pytest.approx(0.95)
    assert sum_ext["h_eff_total"] > 40.0


def test_t5_ren_virtual_inertia_and_config_validation():
    gen_data = get_ieee39_gen_data(macro_intertie=False)
    cfg_no_vi = RenewablePenetrationConfig(p_ren=0.40, h_vir=0.0)
    cfg_with_vi = RenewablePenetrationConfig(p_ren=0.40, h_vir=5.0)

    h_no = sum(compute_effective_inertias(gen_data, cfg_no_vi).values())
    h_vi = sum(compute_effective_inertias(gen_data, cfg_with_vi).values())
    assert h_vi > h_no

    sum_vi = summarize_renewable_dispatch(apply_renewable_displacement(gen_data, cfg_with_vi))
    expected_bonus = 9 * sum_vi["alpha"] * 5.0
    assert (h_vi - h_no) == pytest.approx(expected_bonus, abs=0.5)

    for bad_p in (-0.1, 1.2):
        with pytest.raises(ValueError, match="p_ren"):
            RenewablePenetrationConfig(p_ren=bad_p)
    for bad_t in (0.0, -0.05):
        with pytest.raises(ValueError, match="t_inv"):
            RenewablePenetrationConfig(t_inv=bad_t)


def test_t5_pmu_c37_118_rates_truncated_frames_causal_rocof_and_tve():
    for valid_fps in (10, 25, 30, 50, 60, 100, 120):
        assert PMURecorder(fps=valid_fps).fps == valid_fps
    for invalid_fps in (-50, 0, 15, 45, 75, 200):
        with pytest.raises(ValueError, match="reporting rate must be"):
            PMURecorder(fps=invalid_fps)

    for short_len in (0, 8, 15, 17):
        with pytest.raises(ValueError, match="too short"):
            PMURecorder.parse_frame(bytes(short_len))

    header = struct.pack(">HHHIIH", 0xAA01, 18, 1, 1700000000, 0, 0)
    frame_18 = header + struct.pack(">H", compute_crc_ccitt(header))
    with pytest.raises(ValueError, match="Corrupted or truncated PMU frame"):
        PMURecorder.parse_frame(frame_18)

    assert compute_causal_rocof(np.array([50.0]), fps=50.0, window_length=5).shape == (1,)

    t_ramp = np.arange(0.0, 1.0, 0.02)
    f_ramp = 50.0 - 2.50 * t_ramp
    rocof = compute_fir_rocof(f_ramp, fps=50.0, window_length=5)
    assert np.allclose(rocof[4:], -2.50, atol=1e-12)

    tve = compute_tve(np.array([0.0, 1e-8, 1.0]), np.zeros(3), np.array([0.01, 1e-8, 1.01]), np.zeros(3))
    assert np.all(np.isfinite(tve))

    _, _, v_noisy, _ = apply_pmu_measurement_noise(
        np.full((100, 39), 50.0),
        np.zeros((100, 39)),
        np.full((100, 39), 0.02),
        np.zeros((100, 39)),
        PMUNoiseConfig(sigma_v=0.50, seed=123),
    )
    assert np.all(v_noisy >= 0.01)


def test_t5_slice_edge_cases_and_memory_isolation():
    scen = {
        "timestamps": np.linspace(0.0, 2.0, 101),
        "bus_frequencies": np.full((101, 39), 50.0),
        "bus_rocof": np.zeros((101, 39)),
        "voltage_magnitudes": np.ones((101, 39)),
        "voltage_angles": np.zeros((101, 39)),
        "coi_frequency": np.full(101, 50.0),
        "metadata": {"t_disturb": 1.0, "fps": 50},
    }
    with pytest.raises(ValueError, match="insufficient samples"):
        extract_causal_window(scen, t_obs=0.001)

    assert extract_causal_window(scen, t_obs=0.02).n_samples == 2
    assert extract_causal_window(scen, t_obs=0.10).n_samples == 6
    assert extract_causal_window(scen, t_obs=0.50).n_samples == 26
    assert extract_causal_window(scen, t_obs=1.00).n_samples == 51

    with pytest.warns(UserWarning, match="exceeds trajectory duration"):
        w_trunc = extract_causal_window(scen, t_obs=3.0)
    assert w_trunc.metadata["is_truncated"] is True

    w_mem = extract_causal_window(scen, t_obs=0.5)
    w_mem.bus_frequencies[0, 0] = 999.0
    assert scen["bus_frequencies"][50, 0] == 50.0

    scen_nadir = copy.deepcopy(scen)
    scen_nadir["timestamps"] = np.linspace(0.0, 5.0, 251)
    for k in ("bus_frequencies", "bus_rocof", "voltage_magnitudes", "voltage_angles"):
        scen_nadir[k] = np.resize(scen_nadir[k], (251, 39))
    scen_nadir["bus_frequencies"][:] = 50.0
    scen_nadir["coi_frequency"] = np.full(251, 50.0)
    scen_nadir["metadata"]["ground_truth_t_nadir"] = 2.2
    with pytest.raises(ValueError, match="Anti-leakage violation"):
        extract_causal_window(scen_nadir, t_obs=1.5, strict_nadir=True)
    with pytest.warns(UserWarning, match="reaches or exceeds nadir time"):
        w_clip = extract_causal_window(scen_nadir, t_obs=2.0, clip_to_nadir=True)
    assert w_clip.timestamps[-1] <= 2.2


def test_t5_feat_50hz_identities_wrap_around_nan_fallback_and_latency():
    assert CANONICAL_FEATURE_NAMES == CORE_FEATURE_NAMES == DESCRIPTIVE_FEATURE_NAMES
    assert len(CANONICAL_FEATURE_NAMES) == 14

    w_flat = CausalWindow(
        timestamps=np.linspace(1.0, 2.0, 51),
        bus_frequencies=np.full((51, 39), 50.0),
        bus_rocof=np.zeros((51, 39)),
        voltage_magnitudes=np.ones((51, 39)),
        voltage_angles=np.zeros((51, 39)),
        coi_frequency=np.full(51, 50.0),
        metadata={"t_disturb": 1.0, "fps": 50, "f_nominal": 50.0},
    )
    feats_flat = extract_features(w_flat, f_nominal=50.0)
    for k in CANONICAL_FEATURE_NAMES:
        assert abs(feats_flat[k]) < 1e-12

    t_ramp = np.arange(1.0, 2.0 + 0.01, 0.02)
    slope = -2.50
    f_ramp = 50.0 + slope * (t_ramp - 1.0)
    w_ramp = CausalWindow(
        timestamps=t_ramp,
        bus_frequencies=np.tile(f_ramp[:, None], (1, 39)),
        bus_rocof=np.full((len(t_ramp), 39), slope),
        voltage_magnitudes=np.ones((len(t_ramp), 39)),
        voltage_angles=np.zeros((len(t_ramp), 39)),
        coi_frequency=f_ramp,
        metadata={"t_disturb": 1.0, "fps": 50, "f_nominal": 50.0},
    )
    feats_ramp = extract_features(w_ramp, f_nominal=50.0)
    assert feats_ramp["rocof_init"] == pytest.approx(slope, abs=1e-6)
    assert feats_ramp["rocof_peak_neg"] == pytest.approx(slope, abs=1e-6)
    assert feats_ramp["rocof_mean"] == pytest.approx(slope, abs=1e-6)
    assert feats_ramp["freq_drop_end"] == pytest.approx(slope * 1.0, abs=1e-6)

    t_sym = np.linspace(1.0, 3.0, 101)
    f_sym = 50.0 + np.sin(2.0 * np.pi * (t_sym - 1.0))
    w_sym = CausalWindow(
        timestamps=t_sym,
        bus_frequencies=np.tile(f_sym[:, None], (1, 39)),
        bus_rocof=np.zeros((101, 39)),
        voltage_magnitudes=np.ones((101, 39)),
        voltage_angles=np.zeros((101, 39)),
        coi_frequency=f_sym,
        metadata={"t_disturb": 1.0, "fps": 50, "f_nominal": 50.0},
    )
    idv, iab = calc_freq_integral(w_sym, f_nominal=50.0)
    assert abs(idv) < 1e-12 and iab > 1.0

    # Circular wrap across +/-180 deg (+179.5 -> -179.5 is 1.0 deg)
    angles = np.full((2, 39), 179.5)
    angles[-1, 4] = -179.5
    angles[-1, 20:] = -179.5
    w_wrap = CausalWindow(
        timestamps=np.array([1.0, 1.02]),
        bus_frequencies=np.full((2, 39), 50.0),
        bus_rocof=np.zeros((2, 39)),
        voltage_magnitudes=np.ones((2, 39)),
        voltage_angles=angles,
        coi_frequency=np.array([50.0, 50.0]),
        metadata={"t_disturb": 1.0, "fps": 50, "f_nominal": 50.0},
    )
    feats_wrap = extract_features(w_wrap, f_nominal=50.0, angle_unit="deg")
    assert feats_wrap["angle_divergence_max"] == pytest.approx(1.0, abs=1e-6)
    assert feats_wrap["tie_line_angle_excursion"] == pytest.approx(1.0, abs=1e-6)

    cleaned = _validate_and_clean_nans(np.full((10, 39), np.nan), "bus_frequencies", policy="ffill", f_nominal=50.0)
    assert np.allclose(cleaned, 50.0)

    feats_empty_area = extract_features(w_flat, f_nominal=50.0, area_partition={1: [1, 2, 3], 2: [4, 5, 6], 3: []})
    assert feats_empty_area["area_freq_spread_max"] == 0.0

    _ = extract_features(w_flat, f_nominal=50.0)
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(50):
            _ = extract_features(w_flat, f_nominal=50.0)
        timings.append(((time.perf_counter() - t0) / 50.0) * 1000.0)
    assert min(timings) < 6.0
