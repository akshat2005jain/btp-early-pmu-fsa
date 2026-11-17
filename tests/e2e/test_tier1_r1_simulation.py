"""End-to-end Tier 1 tests for 50 Hz IEEE 39-bus simulation, renewables, PMU recording, and dataset validation."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pmu_fsa.features.extractor import CANONICAL_FEATURE_NAMES, FeatureExtractor
from pmu_fsa.simulation.dataset_validator import validate_dataset
from pmu_fsa.simulation.dynamic_engine import (
    DynamicSimulator,
    compute_kron_reduction,
    solve_power_flow,
)
from pmu_fsa.simulation.ieee39_data import (
    BASE_MVA,
    GENERATOR_BUSES,
    NOMINAL_FREQ,
    PHYSICAL_FREQ_BOUNDS,
    SECURITY_THRESHOLD_HZ,
    get_ieee39_bus_data,
    get_ieee39_gen_data,
)
from pmu_fsa.simulation.pmu_recorder import (
    CANONICAL_13BUS_PMU_PLACEMENT,
    PMUNoiseConfig,
    PMURecorder,
)
from pmu_fsa.simulation.renewables import (
    DEFAULT_DISPLACED_BUSES,
    RenewablePenetrationConfig,
    apply_renewable_displacement,
    summarize_renewable_dispatch,
)
from pmu_fsa.simulation.scenario_generator import (
    ScenarioGenerator,
    export_raw_and_pmu_scenario,
)
from tests.e2e.conftest import (
    IEEE39_GENERATOR_BUSES,
    assert_valid_pmu_trajectory,
)


def test_f1_bus_topology_and_50hz_nominal_standard(ieee39_topology):
    topo = ieee39_topology
    assert topo["n_buses"] == 39
    assert topo["base_mva"] == 100.0
    assert topo["nominal_freq"] == 50.0
    assert NOMINAL_FREQ == 50.0
    assert SECURITY_THRESHOLD_HZ == 49.50
    assert PHYSICAL_FREQ_BOUNDS == (45.0, 50.5)
    assert len(get_ieee39_bus_data()) == 39


def test_f1_generator_parameters_and_intertie_modes(ieee39_topology):
    gen_regional = {g["bus"]: g for g in get_ieee39_gen_data(macro_intertie=False)}
    gen_macro = {g["bus"]: g for g in get_ieee39_gen_data(macro_intertie=True)}
    assert len(gen_regional) == 10 and len(gen_macro) == 10

    assert gen_regional[39]["H"] == pytest.approx(42.0)
    assert gen_regional[39]["Xdp"] == pytest.approx(0.0400)
    assert gen_macro[39]["H"] == pytest.approx(500.0)
    assert gen_macro[39]["Xdp"] == pytest.approx(0.0060)

    for bus_id, gp in gen_regional.items():
        assert bus_id in GENERATOR_BUSES
        assert gp["P_mw"] > 0.0
        assert gp["S_mva"] > gp["P_mw"]
        assert gp["H"] >= 24.0
        assert gp["D"] >= 0.0
        assert gp["Xdp"] > 0.0
        assert gp["R"] == pytest.approx(0.05)
        assert gp["R_sys_pu"] == pytest.approx(gp["R"] * (BASE_MVA / gp["S_mva"]))
        assert gp["Tg"] == pytest.approx(4.20)


def test_f1_branch_connectivity_and_reactances(ieee39_topology):
    branches = ieee39_topology["branches"]
    assert len(branches) == 46

    adj = {i: set() for i in range(1, 40)}
    for from_b, to_b, _r, x, _b in branches:
        assert 1 <= from_b <= 39 and 1 <= to_b <= 39
        assert x > 0.0
        adj[from_b].add(to_b)
        adj[to_b].add(from_b)

    visited = set()
    queue = [1]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            queue.extend(nb for nb in adj[node] if nb not in visited)
    assert len(visited) == 39


def test_f1_three_area_partitioning(ieee39_topology):
    areas = ieee39_topology["area_partition"]
    area1, area2, area3 = set(areas["Area 1"]), set(areas["Area 2"]), set(areas["Area 3"])
    assert area1.isdisjoint(area2) and area1.isdisjoint(area3) and area2.isdisjoint(area3)
    assert (area1 | area2 | area3) == set(range(1, 40))

    branch_pairs = {(min(fb, tb), max(fb, tb)) for fb, tb, *_ in ieee39_topology["branches"]}
    assert {(4, 5), (1, 39), (16, 17)}.issubset(branch_pairs)


def test_f1_nr_power_flow_equilibrium_and_power_balance():
    V0, p_gen_mw, q_gen_mvar, diag = solve_power_flow(load_scaling=1.0)
    assert diag["converged"] is True
    assert diag["max_mismatch_pu"] < 1e-10
    assert V0.shape == (39,)
    assert len(p_gen_mw) == 10 and len(q_gen_mvar) == 10
    assert np.all((np.abs(V0) >= 0.90) & (np.abs(V0) <= 1.12))
    assert diag["total_gen_mw"] > diag["total_load_mw"]
    assert 20.0 <= diag["losses_mw"] <= 80.0


def test_f2_kron_admittance_and_50hz_steady_state_equilibrium():
    Y_red, K, E_mag0, delta0, Pm0 = compute_kron_reduction(load_scaling=1.0, return_all=True)
    assert Y_red.shape == (10, 10) and K.shape == (39, 10)
    assert len(E_mag0) == len(delta0) == len(Pm0) == 10

    sim = DynamicSimulator(load_scaling=1.0)
    assert sim.f_nominal == 50.0
    assert sim.omega_0 == pytest.approx(2.0 * math.pi * 50.0)

    out_flat = sim.simulate(t_span=(0.0, 3.0), t_disturb=1.0, tripped_bus=None, p_loss_mw=0.0, fps=50)
    assert float(np.max(np.abs(out_flat["bus_frequencies"] - 50.0))) < 1e-9
    assert float(np.max(np.abs(out_flat["generator_speeds"]))) < 1e-11
    assert out_flat["metadata"]["ground_truth_nadir"] == pytest.approx(50.0, abs=1e-9)
    assert out_flat["metadata"]["is_secure"] == 1


def test_f2_complete_generator_disconnection_and_partial_loss_scaling():
    sim = DynamicSimulator(load_scaling=1.0)
    out_full = sim.simulate(t_span=(0.0, 7.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    out_part = sim.simulate(t_span=(0.0, 7.0), t_disturb=1.0, tripped_bus=35, loss_fraction=0.5, fps=50)

    ts = out_full["timestamps"]
    post_mask = ts >= 1.0
    trip_idx = list(sim.gen_buses).index(35)

    assert float(np.max(np.abs(out_full["generator_electrical_powers"][post_mask, trip_idx]))) == 0.0
    assert float(np.max(np.abs(out_full["generator_powers"][post_mask, trip_idx]))) == 0.0
    assert out_full["metadata"]["effective_H_post"] == pytest.approx(sim.H_tot - sim.H[trip_idx])

    assert float(np.min(np.abs(out_part["generator_powers"][post_mask, trip_idx]))) > 1.0
    assert out_full["metadata"]["ground_truth_nadir"] < out_part["metadata"]["ground_truth_nadir"] < 50.0
    assert out_full["metadata"]["ground_truth_t_nadir"] < ts[-1]


def test_f2_tgov1_governor_and_initial_swing_rocof():
    sim_gov = DynamicSimulator(load_scaling=1.0)
    sim_nogov = DynamicSimulator(load_scaling=1.0)
    sim_nogov.R[:] = 1e9

    out_gov = sim_gov.simulate(t_span=(0.0, 8.0), t_disturb=1.0, tripped_bus=38, loss_fraction=1.0, fps=50)
    out_nogov = sim_nogov.simulate(t_span=(0.0, 8.0), t_disturb=1.0, tripped_bus=38, loss_fraction=1.0, fps=50)
    assert out_gov["metadata"]["ground_truth_nadir"] > out_nogov["metadata"]["ground_truth_nadir"] + 0.15

    loss_pu = out_gov["metadata"]["active_loss_pu"]
    h_post = out_gov["metadata"]["effective_H_post"]
    rocof_analytical = -50.0 * loss_pu / (2.0 * h_post)

    ts = out_gov["timestamps"]
    coi = out_gov["coi_frequency"]
    idx0 = int(np.searchsorted(ts, 1.0))
    obs_rocof = float((coi[idx0 + 1] - coi[idx0]) / (ts[idx0 + 1] - ts[idx0]))
    assert abs(obs_rocof - rocof_analytical) / abs(rocof_analytical) < 0.03


def test_f3_linear_renewable_displacement_without_saturation():
    assert len(DEFAULT_DISPLACED_BUSES) == 9
    assert 31 not in DEFAULT_DISPLACED_BUSES

    gen_base = get_ieee39_gen_data(macro_intertie=False)
    realized = []
    inertias = []
    for p_req in (0.10, 0.20, 0.30, 0.40, 0.50):
        cfg = RenewablePenetrationConfig(p_ren=p_req, h_vir=0.0)
        mod_gens = apply_renewable_displacement(gen_base, cfg)
        summary = summarize_renewable_dispatch(mod_gens)
        assert summary["p_ren_realized"] == pytest.approx(p_req, rel=0.01)
        realized.append(summary["p_ren_realized"])
        inertias.append(summary["h_eff_total"])

    for i in range(len(inertias) - 1):
        assert realized[i] < realized[i + 1]
        assert inertias[i] > inertias[i + 1]
    assert realized[-1] / realized[0] == pytest.approx(5.0, rel=1e-6)


def test_f3_renewable_penetration_deepens_50hz_nadir_and_virtual_inertia_mitigates():
    sim_low = DynamicSimulator(renewable_config=RenewablePenetrationConfig(p_ren=0.10))
    sim_high = DynamicSimulator(renewable_config=RenewablePenetrationConfig(p_ren=0.50, h_vir=0.0))
    sim_vi = DynamicSimulator(renewable_config=RenewablePenetrationConfig(p_ren=0.50, h_vir=5.0))

    out_low = sim_low.simulate(t_span=(0.0, 7.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    out_high = sim_high.simulate(t_span=(0.0, 7.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)
    out_vi = sim_vi.simulate(t_span=(0.0, 7.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=50)

    assert out_high["metadata"]["ground_truth_nadir"] < out_low["metadata"]["ground_truth_nadir"]
    assert out_vi["metadata"]["ground_truth_nadir"] > out_high["metadata"]["ground_truth_nadir"]


def test_f4_pmu_recorder_sampling_rates_soc_fracsec_noise_and_c37_118(real_pmu_trajectory):
    sim = DynamicSimulator(load_scaling=1.0)
    raw = sim.simulate(t_span=(0.0, 4.0), t_disturb=1.0, tripped_bus=35, loss_fraction=1.0, fps=100)

    for fps in (25, 50):
        noise_cfg = PMUNoiseConfig(sigma_f=0.005, sigma_v=0.0005, sigma_theta_deg=0.05, seed=42)
        rec = PMURecorder(fps=fps, noise_config=noise_cfg, f_nominal=50.0)
        s_clean = rec.record_trajectory(raw, inject_noise=False)
        s_noisy_1 = rec.record_trajectory(raw, inject_noise=True, seed_offset=7)
        s_noisy_2 = rec.record_trajectory(raw, inject_noise=True, seed_offset=7)
        s_noisy_3 = rec.record_trajectory(raw, inject_noise=True, seed_offset=8)

        assert len(s_clean["timestamps"]) == 4 * fps + 1
        assert s_clean["soc"].dtype == np.uint32
        assert s_clean["fracsec"].dtype == np.uint32
        np.testing.assert_array_equal(s_noisy_1["bus_frequencies"], s_noisy_2["bus_frequencies"])
        assert not np.allclose(s_noisy_1["bus_frequencies"], s_noisy_3["bus_frequencies"])
        assert not np.allclose(s_noisy_1["voltage_magnitudes"], s_clean["voltage_magnitudes"])
        assert not np.allclose(s_noisy_1["voltage_angles"], s_clean["voltage_angles"])

        frame_bytes = rec.serialize_frame(1, s_clean)
        parsed = PMURecorder.parse_frame(frame_bytes, f_nominal=50.0)
        assert parsed["crc_valid"] is True
        assert parsed["n_phasors"] == 39
        assert parsed["freq_hz"] == pytest.approx(float(s_clean["pmu_sys_frequency"][1]), abs=1e-4)

    s_part = PMURecorder(fps=50, f_nominal=50.0).record_trajectory(
        raw, inject_noise=False, observed_buses=CANONICAL_13BUS_PMU_PLACEMENT
    )
    obs_cols = ~np.all(np.isnan(s_part["bus_frequencies"]), axis=0)
    assert int(np.sum(obs_cols)) == 13

    assert_valid_pmu_trajectory(real_pmu_trajectory, n_buses=39, f_nominal=50.0)


def test_f5_scenario_batch_metadata_49_5hz_labeling_and_dataset_validator(
    tmp_path: Path,
    synthetic_scenario_batch,
):
    batch = synthetic_scenario_batch
    assert len(batch) == 20
    scenario_ids = [s["metadata"]["scenario_id"] for s in batch]
    assert len(scenario_ids) == len(set(scenario_ids))

    for scen in batch:
        meta = scen["metadata"]
        assert meta["disturbed_gen"] in IEEE39_GENERATOR_BUSES
        assert 45.0 <= meta["ground_truth_nadir"] <= 50.5
        expected_sec = 1 if meta["ground_truth_nadir"] >= 49.50 else 0
        assert meta["is_secure"] == expected_sec

    gen = ScenarioGenerator(fps=50, t_span=(0.0, 6.0), seed=55)
    pairs = gen.generate_scenario_batch(
        gen_buses=[30, 35],
        loss_fractions=[1.0, 0.40],
        penetrations=[0.10, 0.40],
        load_scales=[1.00, 1.20],
        max_scenarios=6,
        inject_noise=True,
        return_raw=True,
    )
    raw_sim_dir = tmp_path / "raw_sim"
    raw_pmu_dir = tmp_path / "raw_pmu"
    proc_dir = tmp_path / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    pmu_list = []
    for raw_t, pmu_t in pairs:
        export_raw_and_pmu_scenario(raw_t, pmu_t, raw_sim_dir, raw_pmu_dir)
        pmu_list.append(pmu_t)

    ext = FeatureExtractor(f_nominal=50.0)
    for w in (0.5, 1.0):
        X_df, yn, ys, meta_df = ext.transform_batch(pmu_list, t_obs=w, as_dataframe=True)
        np.savez_compressed(
            proc_dir / f"features_t{w:.1f}s.npz",
            X=X_df.to_numpy(dtype=np.float64),
            y_nadir=yn,
            y_secure=ys,
            scenario_ids=meta_df["scenario_id"].to_numpy(dtype=str),
            partitions=meta_df["partition"].to_numpy(dtype=str),
            ood_types=meta_df["ood_type"].to_numpy(dtype=str),
            feature_names=np.asarray(CANONICAL_FEATURE_NAMES, dtype=str),
        )

    val_res = validate_dataset(
        raw_pmu_dir=raw_pmu_dir,
        processed_dir=proc_dir,
        windows=(0.5, 1.0),
        f_nominal=50.0,
        threshold=49.50,
        raw_sim_dir=raw_sim_dir,
        expected_n_scenarios=6,
    )
    assert val_res["all_passed"] is True, f"Failed checks: {[k for k, v in val_res['checks'].items() if not v['passed']]}"
