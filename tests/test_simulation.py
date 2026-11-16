"""Tests for the IEEE 39-bus dynamic simulation engine."""

import numpy as np
import pytest

from pmu_fsa.simulation.ieee39_data import (
    BASE_MVA,
    NOMINAL_FREQ,
    N_BUSES,
    N_GENERATORS,
    GENERATOR_BUSES,
    AREA_PARTITION,
    BUS_TO_AREA,
    get_ieee39_bus_data,
    get_ieee39_branch_data,
    get_ieee39_gen_data,
)
from pmu_fsa.simulation.renewables import RenewablePenetrationConfig
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator, compute_kron_reduction


def test_steady_state_flat_run():
    sim = DynamicSimulator(load_scaling=1.0)
    res = sim.simulate(t_span=(0.0, 2.0), tripped_bus=None, fps=50)

    timestamps = res["timestamps"]
    bus_freq = res["bus_frequencies"]
    bus_rocof = res["bus_rocof"]
    coi_freq = res["coi_frequency"]

    assert len(timestamps) > 0
    assert bus_freq.shape == (len(timestamps), N_BUSES)

    max_df = np.max(np.abs(bus_freq - NOMINAL_FREQ))
    max_coi_df = np.max(np.abs(coi_freq - NOMINAL_FREQ))
    max_rocof = np.max(np.abs(bus_rocof))

    assert max_df < 1e-4, f"Bus frequency deviated in flat run: max |df| = {max_df} Hz"
    assert max_coi_df < 1e-4, f"COI frequency deviated in flat run: max |df_coi| = {max_coi_df} Hz"
    assert max_rocof < 1e-4, f"Bus ROCOF non-zero in flat run: max |rocof| = {max_rocof} Hz/s"


def test_theoretical_initial_rocof():
    sim = DynamicSimulator(load_scaling=1.0)
    t_disturb = 1.0
    p_loss_mw = 650.0
    p_loss_pu = p_loss_mw / BASE_MVA

    res = sim.simulate(
        t_span=(0.0, 5.0),
        t_disturb=t_disturb,
        tripped_bus=35,
        p_loss_mw=p_loss_mw,
        fps=50,
    )

    t = res["timestamps"]
    f_coi = res["coi_frequency"]

    idx0 = np.searchsorted(t, t_disturb)
    dt = t[idx0 + 1] - t[idx0]
    sim_initial_rocof = (f_coi[idx0 + 1] - f_coi[idx0]) / dt

    # Swing equation at t = t0+: df_COI/dt = -f0 * DeltaP / (2 * H_tot)
    theo_rocof = -NOMINAL_FREQ * p_loss_pu / (2.0 * sim.H_tot)

    rel_error = abs(sim_initial_rocof - theo_rocof) / abs(theo_rocof)

    assert rel_error < 0.02, (
        f"Initial ROCOF error exceeds 2% tolerance: "
        f"sim={sim_initial_rocof:.5f} Hz/s, theo={theo_rocof:.5f} Hz/s, rel_error={rel_error * 100:.2f}%"
    )


def test_renewable_penetration_inertia_reduction():
    t_disturb = 1.0
    p_loss_mw = 650.0

    cfg_low = RenewablePenetrationConfig(p_ren=0.10)
    sim_low = DynamicSimulator(load_scaling=1.0, renewable_config=cfg_low)
    res_low = sim_low.simulate(t_span=(0.0, 5.0), t_disturb=t_disturb, tripped_bus=35, p_loss_mw=p_loss_mw, fps=50)

    cfg_high = RenewablePenetrationConfig(p_ren=0.40)
    sim_high = DynamicSimulator(load_scaling=1.0, renewable_config=cfg_high)
    res_high = sim_high.simulate(t_span=(0.0, 5.0), t_disturb=t_disturb, tripped_bus=35, p_loss_mw=p_loss_mw, fps=50)

    assert sim_high.H_tot < sim_low.H_tot, (
        f"Expected higher penetration to reduce total inertia: "
        f"H_low={sim_low.H_tot}, H_high={sim_high.H_tot}"
    )

    t = res_low["timestamps"]
    idx0 = np.searchsorted(t, t_disturb)
    dt = t[1] - t[0]
    rocof_low = (res_low["coi_frequency"][idx0 + 1] - res_low["coi_frequency"][idx0]) / dt
    rocof_high = (res_high["coi_frequency"][idx0 + 1] - res_high["coi_frequency"][idx0]) / dt

    assert rocof_high < rocof_low, (
        f"Higher penetration should produce steeper (more negative) initial ROCOF: "
        f"rocof_low={rocof_low:.4f}, rocof_high={rocof_high:.4f}"
    )

    nadir_low = res_low["metadata"]["ground_truth_nadir"]
    nadir_high = res_high["metadata"]["ground_truth_nadir"]

    assert nadir_high < nadir_low, (
        f"Higher penetration should yield lower frequency nadir: "
        f"nadir_low={nadir_low:.4f} Hz, nadir_high={nadir_high:.4f} Hz"
    )


def test_dynamic_generator_trip_across_all_3_areas():
    # One representative generator per coherent area (G3, G6, G9)
    test_cases = [
        (1, 32, 650.0),
        (2, 35, 650.0),
        (3, 38, 830.0),
    ]

    sim = DynamicSimulator(load_scaling=1.0)
    for area_id, gen_bus, p_loss in test_cases:
        res = sim.simulate(
            t_span=(0.0, 5.0),
            t_disturb=1.0,
            tripped_bus=gen_bus,
            p_loss_mw=p_loss,
            fps=50,
        )

        bf = res["bus_frequencies"]
        assert not np.isnan(bf).any(), f"NaN in bus frequencies for trip at bus {gen_bus}"
        assert not np.isinf(bf).any(), f"Inf in bus frequencies for trip at bus {gen_bus}"

        nadir = res["metadata"]["ground_truth_nadir"]
        t_nadir = res["metadata"]["ground_truth_t_nadir"]

        assert nadir < NOMINAL_FREQ, f"Nadir {nadir} Hz should be below {NOMINAL_FREQ} Hz"
        assert t_nadir >= 1.0, f"Nadir time {t_nadir} s occurred before disturbance"

        area_freqs = res["area_frequencies"]
        assert area_id in area_freqs, f"Area {area_id} frequency missing"
        assert len(area_freqs[area_id]) == len(res["timestamps"])


def test_bus_voltage_and_analytical_frequency_reconstruction():
    sim = DynamicSimulator(load_scaling=1.0)
    res = sim.simulate(
        t_span=(0.0, 4.0),
        t_disturb=1.0,
        tripped_bus=35,
        p_loss_mw=650.0,
        fps=50,
    )

    v_mag = res["voltage_magnitudes"]
    v_ang = res["voltage_angles"]
    bus_freq = res["bus_frequencies"]

    assert np.all(v_mag >= 0.80), f"Voltage magnitude fell below 0.80 pu: min={np.min(v_mag)}"
    assert np.all(v_mag <= 1.25), f"Voltage magnitude exceeded 1.25 pu: max={np.max(v_mag)}"

    pre_mag = v_mag[0, :]
    assert np.all(pre_mag > 0.85)

    assert np.all(bus_freq >= NOMINAL_FREQ - 2.0), f"Frequency collapsed below {NOMINAL_FREQ - 2.0} Hz: min={np.min(bus_freq)}"
    assert np.all(bus_freq <= NOMINAL_FREQ + 1.0), f"Frequency exceeded {NOMINAL_FREQ + 1.0} Hz: max={np.max(bus_freq)}"
