"""Tests for the 50 Hz IEEE C37.118 PMU measurement recorder and causal ROCOF."""

from __future__ import annotations

import struct
import numpy as np
import pytest

from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
from pmu_fsa.simulation.pmu_recorder import (
    CANONICAL_13BUS_PMU_PLACEMENT,
    PMUNoiseConfig,
    PMURecorder,
    apply_pmu_measurement_noise,
    compute_causal_rocof,
    compute_crc_ccitt,
    compute_fir_rocof,
    compute_tve,
    estimate_pmu_system_frequency,
)


@pytest.fixture(scope="module")
def clean_simulation_trajectory():
    sim = DynamicSimulator(load_scaling=1.0)
    return sim.simulate(
        t_span=(0.0, 3.0),
        t_disturb=1.0,
        tripped_bus=35,
        loss_fraction=1.0,
        fps=100,
    )


def test_sampling_rates_25_50_60_fps_and_invalid_rejection(clean_simulation_trajectory):
    raw_traj = clean_simulation_trajectory

    for fps in (25, 50, 60):
        rec = PMURecorder(fps=fps, f_nominal=50.0)
        stream = rec.record_trajectory(raw_traj)
        ts = stream["timestamps"]
        diffs = np.diff(ts)
        expected_dt = 1.0 / float(fps)
        assert np.allclose(diffs, expected_dt, atol=1e-8)
        assert len(ts) == int(round(3.0 * fps)) + 1
        assert stream["metadata"]["fps"] == fps
        assert stream["bus_frequencies"].shape == (len(ts), 39)

    for bad_fps in (-50, 0, 15, 45, 200):
        with pytest.raises(ValueError, match="reporting rate must be"):
            PMURecorder(fps=bad_fps)


def test_soc_and_fracsec_timestamps(clean_simulation_trajectory):
    base_epoch = 1700000000
    for fps in (25, 50):
        rec = PMURecorder(fps=fps, base_epoch_seconds=base_epoch, f_nominal=50.0)
        stream = rec.record_trajectory(clean_simulation_trajectory)
        soc = stream["soc"]
        fracsec = stream["fracsec"]
        assert soc.dtype == np.uint32
        assert fracsec.dtype == np.uint32
        reconstructed = (soc.astype(np.float64) - base_epoch) + (fracsec.astype(np.float64) / 1_000_000.0)
        np.testing.assert_allclose(reconstructed, stream["timestamps"], atol=1e-6)


def test_seeded_noise_injection_across_f_v_theta_and_tve(clean_simulation_trajectory):
    raw_traj = clean_simulation_trajectory

    alias_cfg = PMUNoiseConfig(f_noise_std=0.004, seed=7)
    assert alias_cfg.sigma_f == pytest.approx(0.004)

    with pytest.raises(ValueError, match="non-negative"):
        PMUNoiseConfig(sigma_f=-0.001)

    noise_cfg = PMUNoiseConfig(
        sigma_f=0.005,
        sigma_rocof=0.02,
        sigma_v=0.0005,
        sigma_theta_deg=0.05,
        seed=123,
    )
    rec = PMURecorder(fps=50, noise_config=noise_cfg, f_nominal=50.0)
    clean_stream = rec.record_trajectory(raw_traj, inject_noise=False)
    noisy_a = rec.record_trajectory(raw_traj, inject_noise=True, seed_offset=0)
    noisy_a_repeat = rec.record_trajectory(raw_traj, inject_noise=True, seed_offset=0)
    noisy_b = rec.record_trajectory(raw_traj, inject_noise=True, seed_offset=1)

    np.testing.assert_array_equal(noisy_a["bus_frequencies"], noisy_a_repeat["bus_frequencies"])
    np.testing.assert_array_equal(noisy_a["voltage_magnitudes"], noisy_a_repeat["voltage_magnitudes"])
    np.testing.assert_array_equal(noisy_a["voltage_angles"], noisy_a_repeat["voltage_angles"])

    assert not np.allclose(noisy_a["bus_frequencies"], noisy_b["bus_frequencies"])
    assert not np.allclose(noisy_a["voltage_magnitudes"], noisy_b["voltage_magnitudes"])
    assert not np.allclose(noisy_a["voltage_angles"], noisy_b["voltage_angles"])

    f_err = noisy_a["bus_frequencies"] - clean_stream["bus_frequencies"]
    emp_sigma_f = float(np.std(f_err))
    assert emp_sigma_f == pytest.approx(0.005, rel=0.08)

    tve = compute_tve(
        clean_stream["voltage_magnitudes"],
        clean_stream["voltage_angles"],
        noisy_a["voltage_magnitudes"],
        noisy_a["voltage_angles"],
    )
    assert float(np.percentile(tve, 99)) < 0.010
    assert float(np.mean(tve)) < 0.005
    assert np.all((noisy_a["voltage_angles"] >= -180.0) & (noisy_a["voltage_angles"] < 180.0))


def test_wrap_safe_angle_interpolation_and_partial_pmu_masking(clean_simulation_trajectory):
    synth = {
        "timestamps": np.array([0.0, 0.04, 0.08]),
        "bus_frequencies": np.full((3, 39), 50.0),
        "voltage_magnitudes": np.ones((3, 39)),
        "voltage_angles": np.tile(np.array([[179.0], [-179.0], [-177.0]]), (1, 39)),
        "coi_frequency": np.array([999.0, 999.0, 999.0]),
        "metadata": {"f_nominal": 50.0},
    }
    rec = PMURecorder(fps=50, f_nominal=50.0)
    stream = rec.record_trajectory(synth)
    # Midpoint between +179 deg and -179 deg must wrap near +/-180 deg, never 0 deg
    mid_ang = float(stream["voltage_angles"][1, 0])
    assert abs(abs(mid_ang) - 180.0) < 1e-6
    # Wide-area PMU frequency must ignore poisoned raw coi_frequency=999.0
    assert np.allclose(stream["pmu_sys_frequency"], 50.0)
    assert np.allclose(stream["coi_frequency"], 50.0)

    partial = rec.record_trajectory(
        clean_simulation_trajectory,
        inject_noise=False,
        observed_buses=CANONICAL_13BUS_PMU_PLACEMENT,
    )
    obs_zero = {b - 1 for b in CANONICAL_13BUS_PMU_PLACEMENT}
    for col in range(39):
        if col in obs_zero:
            assert np.all(np.isfinite(partial["bus_frequencies"][:, col]))
        else:
            assert np.all(np.isnan(partial["bus_frequencies"][:, col]))
            assert np.all(np.isnan(partial["bus_rocof"][:, col]))
            assert np.all(np.isnan(partial["voltage_magnitudes"][:, col]))
            assert np.all(np.isnan(partial["voltage_angles"][:, col]))
    assert np.all(np.isfinite(partial["pmu_sys_frequency"]))
    assert len(partial["metadata"]["observed_buses"]) == 13


def test_strictly_causal_5point_rocof_50hz():
    fps = 50.0
    dt = 1.0 / fps
    t = np.arange(0.0, 4.0, dt)

    # 50 Hz ramp: exact slope recovery
    slope = -0.75
    f_ramp = 50.0 + slope * t
    rocof_ramp = compute_causal_rocof(f_ramp, fps=fps, window_length=5)
    assert rocof_ramp[0] == 0.0
    assert rocof_ramp[1] == pytest.approx(slope, abs=1e-12)
    assert np.allclose(rocof_ramp[1:], slope, atol=1e-12)

    # Strict causality: corrupting future samples after index k never alters rocof[:k+1]
    for cutoff in (1, 2, 4, 15, 50):
        f_mut = f_ramp.copy()
        f_mut[cutoff + 1 :] = np.nan
        rocof_mut = compute_causal_rocof(f_mut, fps=fps, window_length=5)
        np.testing.assert_array_equal(rocof_ramp[: cutoff + 1], rocof_mut[: cutoff + 1])

    # Noise resilience vs 2-point gradient on 50 Hz quadratic signal
    f_quad = 50.0 - 0.5 * t - 0.1 * (t ** 2)
    rocof_true = -0.5 - 0.2 * t
    rng = np.random.default_rng(999)
    f_noisy = f_quad + rng.normal(0.0, 0.01, size=len(t))

    raw_grad = np.gradient(f_noisy, dt)
    causal_ols = compute_fir_rocof(f_noisy, fps=fps, window_length=5)
    interior = slice(5, -5)
    rmse_raw = float(np.sqrt(np.mean((raw_grad[interior] - rocof_true[interior]) ** 2)))
    rmse_ols = float(np.sqrt(np.mean((causal_ols[interior] - rocof_true[interior]) ** 2)))
    assert rmse_ols < rmse_raw


def test_c37_118_2_frame_serialization_crc_and_error_handling(clean_simulation_trajectory):
    rec = PMURecorder(fps=50, id_code=42, f_nominal=50.0)
    pmu_stream = rec.record_trajectory(clean_simulation_trajectory)

    for frame_idx in (0, 10, 50, len(pmu_stream["timestamps"]) - 1):
        frame_bytes = rec.serialize_frame(frame_idx, pmu_stream)
        assert len(frame_bytes) == 338

        parsed = PMURecorder.parse_frame(frame_bytes, f_nominal=50.0)
        assert parsed["sync"] == 0xAA01
        assert parsed["frame_size"] == 338
        assert parsed["id_code"] == 42
        assert parsed["n_phasors"] == 39
        assert parsed["crc_valid"] is True
        assert parsed["soc"] == int(pmu_stream["soc"][frame_idx])
        assert parsed["fracsec"] == int(pmu_stream["fracsec"][frame_idx])
        assert parsed["freq_hz"] == pytest.approx(float(pmu_stream["pmu_sys_frequency"][frame_idx]), abs=1e-4)

        corrupted = bytearray(frame_bytes)
        corrupted[10] ^= 0xFF
        with pytest.raises(ValueError, match="CRC-CCITT mismatch"):
            PMURecorder.parse_frame(bytes(corrupted), f_nominal=50.0)

    for short_len in (0, 8, 17):
        with pytest.raises(ValueError, match="too short"):
            PMURecorder.parse_frame(bytes(short_len), f_nominal=50.0)

    header_only = struct.pack(">HHHIIH", 0xAA01, 18, 1, 1700000000, 0, 0)
    crc = compute_crc_ccitt(header_only)
    with pytest.raises(ValueError, match="Corrupted or truncated PMU frame"):
        PMURecorder.parse_frame(header_only + struct.pack(">H", crc), f_nominal=50.0)

    bad_sync_payload = struct.pack(">HHHIIHffff", 0xBB01, 34, 1, 1700000000, 0, 0, 1.0, 0.0, 0.0, 0.0)
    bad_crc = compute_crc_ccitt(bad_sync_payload)
    with pytest.raises(ValueError, match="Invalid SYNC word"):
        PMURecorder.parse_frame(bad_sync_payload + struct.pack(">H", bad_crc), f_nominal=50.0)


def test_apply_pmu_measurement_noise_and_system_freq_fallbacks():
    bf = np.full((20, 39), 50.0)
    br = np.zeros((20, 39))
    vm = np.full((20, 39), 0.005)
    va = np.zeros((20, 39))
    cfg = PMUNoiseConfig(sigma_f=0.005, sigma_v=0.10, sigma_theta_deg=0.05, seed=11)
    _, r_n, v_n, _ = apply_pmu_measurement_noise(bf, br, vm, va, noise_config=cfg, fps=50.0)
    assert np.all(v_n >= 0.01)
    assert np.all(np.isfinite(r_n))

    all_nan_bf = np.full((5, 39), np.nan)
    assert np.all(np.isnan(estimate_pmu_system_frequency(all_nan_bf)))
