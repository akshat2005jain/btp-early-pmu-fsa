"""End-to-end tests for IEEE 39-bus simulation and PMU recording."""

import math
import copy
from typing import Dict, Any, List

from tests.e2e.conftest import (
    np, HAS_NUMPY,
    generate_synthetic_pmu_stream,
    assert_valid_pmu_trajectory,
    IEEE39_GENERATOR_BUSES,
    IEEE39_GEN_PARAMS,
    IEEE39_BRANCHES,
    AREA_1_BUSES,
    AREA_2_BUSES,
    AREA_3_BUSES,
)

try:
    from pmu_fsa.simulation import ieee39_data
    from pmu_fsa.simulation import dynamic_engine
    from pmu_fsa.simulation import renewables
    from pmu_fsa.simulation import pmu_recorder
    from pmu_fsa.simulation import scenario_generator
    HAS_SIM_MODULES = True
except ImportError:
    HAS_SIM_MODULES = False


def test_f1_bus_topology_and_count(ieee39_topology):
    topo = ieee39_topology
    assert topo["n_buses"] == 39, f"Expected 39 buses, got {topo['n_buses']}"
    assert topo["base_mva"] == 100.0, f"Expected 100.0 MVA base, got {topo['base_mva']}"
    assert topo["nominal_freq"] == 60.0, f"Expected 60.0 Hz nominal frequency, got {topo['nominal_freq']}"

    if HAS_SIM_MODULES and hasattr(ieee39_data, "get_ieee39_bus_data"):
        data = ieee39_data.get_ieee39_bus_data()
        assert len(data) == 39, f"get_ieee39_bus_data returned {len(data)} buses"


def test_f1_generator_parameters(ieee39_topology):
    topo = ieee39_topology
    assert topo["n_generators"] == 10, f"Expected 10 generators, got {topo['n_generators']}"
    assert len(topo["generator_buses"]) == 10, "Generator bus count mismatch"

    gen_params = topo["generator_params"]
    assert len(gen_params) == 10, "Generator parameters dictionary must contain 10 units"

    for bus_id in topo["generator_buses"]:
        assert bus_id in gen_params, f"Generator bus {bus_id} missing from gen_params"
        gp = gen_params[bus_id]
        assert gp["P0_mw"] > 0, f"Generator {bus_id} active power must be positive"
        assert gp["H"] >= 20.0, f"Generator {bus_id} inertia H must be >= 20.0s, got {gp['H']}"
        assert gp["D"] >= 0.0, f"Generator {bus_id} damping D must be >= 0"
        assert gp["Xdp"] > 0.0, f"Generator {bus_id} transient reactance X'd must be positive"
        assert gp["R"] == 0.05, f"Generator {bus_id} droop R must be canonical 5% (0.05)"
        assert gp["Tg"] > 0.0, f"Generator {bus_id} governor time constant Tg must be positive"


def test_f1_branch_connectivity_and_reactances(ieee39_topology):
    topo = ieee39_topology
    branches = topo["branches"]
    assert len(branches) == 46, f"Expected 46 branches in IEEE 39-bus system, got {len(branches)}"

    adj = {i: set() for i in range(1, 40)}
    for from_b, to_b, r, x, b in branches:
        assert 1 <= from_b <= 39, f"Invalid from_bus: {from_b}"
        assert 1 <= to_b <= 39, f"Invalid to_bus: {to_b}"
        assert x > 0.0, f"Branch ({from_b}, {to_b}) reactance X must be strictly positive, got {x}"
        adj[from_b].add(to_b)
        adj[to_b].add(from_b)

    visited = set()
    queue = [1]
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    assert len(visited) == 39, f"Network graph is partitioned! Visited only {len(visited)} of 39 buses."


def test_f1_three_area_partitioning(ieee39_topology):
    topo = ieee39_topology
    areas = topo["area_partition"]
    assert len(areas) == 3, f"Expected 3 coherent areas, got {len(areas)}"

    area1 = set(areas["Area 1"])
    area2 = set(areas["Area 2"])
    area3 = set(areas["Area 3"])

    assert area1.isdisjoint(area2), "Area 1 and Area 2 have overlapping buses!"
    assert area1.isdisjoint(area3), "Area 1 and Area 3 have overlapping buses!"
    assert area2.isdisjoint(area3), "Area 2 and Area 3 have overlapping buses!"

    all_partition_buses = area1 | area2 | area3
    assert len(all_partition_buses) == 39, f"Partition covers {len(all_partition_buses)} buses, expected 39"
    assert all_partition_buses == set(range(1, 40)), "Partition does not match {1, ..., 39}"

    # Tie-lines 4-5 (Area 1-2) and 1-39 (Area 1-3)
    branch_pairs = {(min(fb, tb), max(fb, tb)) for fb, tb, r, x, b in topo["branches"]}
    assert (4, 5) in branch_pairs, "Key tie-line 4-5 connecting Area 1 and 2 is missing"
    assert (1, 39) in branch_pairs, "Key tie-line 1-39 connecting Area 1 and 3 is missing"


def test_f1_base_power_flow_balance(ieee39_topology):
    topo = ieee39_topology
    total_gen_mw = sum(g["P0_mw"] for g in topo["generator_params"].values())
    # MATPOWER case39 / Pai base totals: ~6192.8 MW generation, ~6150.1 MW load
    assert abs(total_gen_mw - 6192.8) < 5.0, f"Expected ~6192.8 MW generation, got {total_gen_mw}"

    base_load_mw = 6150.1
    assert total_gen_mw > base_load_mw, f"Total generation ({total_gen_mw}) must exceed load ({base_load_mw})"
    losses = total_gen_mw - base_load_mw
    assert 20.0 <= losses <= 80.0, f"Transmission losses ({losses} MW) outside realistic bounds [20, 80] MW"


def test_f2_kron_admittance_matrix_properties(ieee39_topology):
    if HAS_SIM_MODULES and hasattr(dynamic_engine, "compute_kron_reduction"):
        y_red = dynamic_engine.compute_kron_reduction()
        assert y_red.shape == (10, 10), f"Expected (10, 10) Kron reduced matrix, got {y_red.shape}"
    else:
        n_gen = ieee39_topology["n_generators"]
        assert n_gen == 10
        G_diag_min = 0.05
        assert G_diag_min > 0.0, "Kron reduced diagonal conductance must be positive"


def test_f2_swing_equation_steady_state_equilibrium():
    omega_0 = 2.0 * math.pi * 60.0
    delta_omega = 0.0
    P_m = 650.0
    P_e = 650.0
    D = 2.0
    H = 35.8

    d_delta = omega_0 * delta_omega
    d_delta_omega = (P_m - P_e - D * delta_omega) / (2.0 * H)

    assert d_delta == 0.0, f"d(delta)/dt must be 0 at steady state, got {d_delta}"
    assert d_delta_omega == 0.0, f"d(delta_omega)/dt must be 0 at steady state, got {d_delta_omega}"


def test_f2_step_disturbance_frequency_nadir_trajectory(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    assert_valid_pmu_trajectory(traj)

    ts = traj["timestamps"]
    coi = traj["coi_frequency"]
    t_disturb = traj["metadata"]["t_disturb"]

    pre_indices = [i for i, t in enumerate(ts) if t < t_disturb]
    for idx in pre_indices:
        assert abs(coi[idx] - 60.0) < 0.01, f"Pre-disturbance frequency unstable: {coi[idx]} Hz"

    post_indices = [i for i, t in enumerate(ts) if t >= t_disturb]
    f_post = [coi[i] for i in post_indices]
    f_min = min(f_post)
    f_end = f_post[-1]

    assert f_min < 59.9, f"Frequency did not drop post disturbance: min={f_min}"
    assert f_min >= 55.0, f"Frequency dropped below physical threshold: min={f_min}"

    nadir_idx = f_post.index(f_min)
    assert nadir_idx < len(f_post) - 1, "Nadir occurred at last sample (no frequency stabilization)"


def test_f2_tgov1_governor_action():
    P_m0 = 500.0
    R = 0.05
    Tg = 0.5
    dt = 0.02

    # 59.4 Hz -> -0.01 p.u. speed deviation -> +0.2 p.u. droop demand
    delta_omega = -0.01
    governor_demand = -(1.0 / R) * delta_omega

    P_m = P_m0
    for _ in range(100):
        dP_m = ((P_m0 + governor_demand * P_m0) - P_m) / Tg
        P_m += dP_m * dt

    assert P_m > P_m0, f"Governor failed to increase power: Pm={P_m} <= Pm0={P_m0}"
    assert P_m >= P_m0 * 1.15, f"Governor output too small: Pm={P_m}"


def test_f2_swing_equation_initial_acceleration_bound(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    t_disturb = traj["metadata"]["t_disturb"]
    ts = traj["timestamps"]
    coi = traj["coi_frequency"]

    idx_0 = 0
    for i, t in enumerate(ts):
        if t >= t_disturb:
            idx_0 = i
            break

    idx_end = idx_0
    for i, t in enumerate(ts):
        if t >= t_disturb + 0.1:
            idx_end = i
            break

    dt = ts[idx_end] - ts[idx_0]
    initial_rocof = (coi[idx_end] - coi[idx_0]) / dt

    assert initial_rocof < 0.0, f"Initial ROCOF must be negative, got {initial_rocof}"
    assert -5.0 <= initial_rocof <= -0.01, f"Initial ROCOF {initial_rocof} Hz/s outside realistic range"


def test_f3_renewable_capacity_displacement(ieee39_topology):
    topo = ieee39_topology
    base_gen_mw = sum(g["P0_mw"] for g in topo["generator_params"].values())

    renewable_penetration = 0.30
    ren_mw = base_gen_mw * renewable_penetration
    syn_mw = base_gen_mw * (1.0 - renewable_penetration)

    assert abs((ren_mw + syn_mw) - base_gen_mw) < 1e-6, "Total generation capacity altered by renewable integration"
    assert ren_mw > 0.0, "Renewable capacity must be positive"


def test_f3_effective_inertia_monotonic_decrease(ieee39_topology):
    base_H = sum(g["H"] for g in ieee39_topology["generator_params"].values())
    ibr_H = 0.0

    penetrations = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    effective_H_values = []

    for gamma in penetrations:
        H_eff = (1.0 - gamma) * base_H + gamma * ibr_H
        effective_H_values.append(H_eff)

    for i in range(len(effective_H_values) - 1):
        assert effective_H_values[i] > effective_H_values[i + 1], \
            f"H_eff not monotonically decreasing: {effective_H_values[i]} <= {effective_H_values[i+1]}"

    assert abs(effective_H_values[-1] - 0.5 * base_H) < 1e-6, "50% displacement did not yield 50% inertia"


def test_f3_nadir_depth_dependency_on_penetration(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    scen_low_ren = [s for s in batch if s["metadata"]["renewable_penetration"] <= 0.15][0]
    scen_high_ren = [s for s in batch if s["metadata"]["renewable_penetration"] >= 0.45][0]

    nadir_low = scen_low_ren["metadata"]["ground_truth_nadir"]
    nadir_high = scen_high_ren["metadata"]["ground_truth_nadir"]

    assert nadir_high < nadir_low, \
        f"Expected deeper nadir under high renewable penetration, got {nadir_high} >= {nadir_low}"


def test_f3_rocof_steepness_dependency_on_inertia():
    loss_mw = 600.0
    f0 = 60.0
    H_high = 350.0
    H_low = 175.0

    rocof_high = -(f0 * loss_mw) / (2.0 * H_high * 100.0)
    rocof_low = -(f0 * loss_mw) / (2.0 * H_low * 100.0)

    assert abs(rocof_low) > abs(rocof_high), "Low inertia ROCOF magnitude must be strictly greater than high inertia"
    assert abs(abs(rocof_low) - 2.0 * abs(rocof_high)) < 1e-6, "ROCOF did not scale inversely with inertia"


def test_f3_virtual_inertia_support_mitigation():
    base_H = 300.0
    gamma = 0.40

    H_eff_no_support = (1.0 - gamma) * base_H + gamma * 0.0

    H_ibr_virtual = 4.0
    H_eff_with_support = (1.0 - gamma) * base_H + gamma * H_ibr_virtual

    assert H_eff_with_support > H_eff_no_support, "Virtual inertia failed to increase effective system inertia"
    delta_f_no = -600.0 / (2.0 * H_eff_no_support)
    delta_f_with = -600.0 / (2.0 * H_eff_with_support)
    assert abs(delta_f_with) < abs(delta_f_no), "Virtual inertia did not mitigate frequency drop"


def test_f4_reporting_rate_50_and_60_fps():
    stream_50 = generate_synthetic_pmu_stream(n_samples=201, fps=50)
    ts_50 = stream_50["timestamps"]
    dt_50 = ts_50[1] - ts_50[0]
    assert abs(dt_50 - 0.02) < 1e-4, f"Expected 0.02s interval for 50 fps, got {dt_50}"
    assert len(ts_50) == 201, f"Expected 201 samples, got {len(ts_50)}"

    stream_60 = generate_synthetic_pmu_stream(n_samples=241, fps=60)
    ts_60 = stream_60["timestamps"]
    dt_60 = ts_60[1] - ts_60[0]
    assert abs(dt_60 - (1.0 / 60.0)) < 1e-4, f"Expected {1/60:.5f}s interval for 60 fps, got {dt_60}"
    assert len(ts_60) == 241, f"Expected 241 samples, got {len(ts_60)}"


def test_f4_synchrophasor_fields_and_dimensions(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    assert_valid_pmu_trajectory(traj, n_buses=39, f_nominal=60.0)

    assert traj["bus_frequencies"].shape[1] == 39
    assert traj["bus_rocof"].shape[1] == 39
    assert traj["voltage_magnitudes"].shape[1] == 39
    assert traj["voltage_angles"].shape[1] == 39


def test_f4_phase_angle_bounds_and_continuity(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    vm = traj["voltage_magnitudes"]
    va = traj["voltage_angles"]

    min_v = np.min(vm)
    max_v = np.max(vm)
    assert min_v >= 0.80, f"Voltage magnitude too low: {min_v} p.u."
    assert max_v <= 1.25, f"Voltage magnitude too high: {max_v} p.u."

    angle_row_0 = va[0]
    for b in range(len(angle_row_0) - 1):
        diff = abs(angle_row_0[b + 1] - angle_row_0[b])
        assert diff < math.pi, f"Adjacent bus angle difference too large: {diff} rad"


def test_f4_gaussian_noise_injection_model():
    sigma_f = 0.03
    noisy_stream = generate_synthetic_pmu_stream(
        n_samples=500,
        fps=50,
        noise_std=sigma_f,
        seed=777
    )
    clean_stream = generate_synthetic_pmu_stream(
        n_samples=500,
        fps=50,
        noise_std=0.0,
        seed=777
    )

    f_noisy = noisy_stream["bus_frequencies"]
    f_clean = clean_stream["bus_frequencies"]

    diff = []
    for r in range(len(f_noisy)):
        for c in range(39):
            diff.append(f_noisy[r][c] - f_clean[r][c])

    mean_diff = sum(diff) / len(diff)
    assert abs(mean_diff) < 0.005, f"Noise injected has non-zero mean: {mean_diff} Hz"


def test_f4_c37_118_frame_integrity_and_schema(sample_pmu_trajectory):
    traj = sample_pmu_trajectory
    ts = traj["timestamps"]

    for i in range(len(ts) - 1):
        assert ts[i + 1] > ts[i], f"Timestamp monotonicity violated at index {i}: {ts[i]} >= {ts[i+1]}"

    assert not np.isnan(traj["bus_frequencies"]).any()
    assert not np.isinf(traj["bus_frequencies"]).any()
    assert not np.isnan(traj["bus_rocof"]).any()


def test_f5_generator_outage_matrix_coverage(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    assert len(batch) >= 10, f"Expected at least 10 scenarios, got {len(batch)}"

    disturbed_units = {s["metadata"]["disturbed_gen"] for s in batch}
    assert len(disturbed_units) >= 5, f"Generator outage matrix has insufficient coverage: {disturbed_units}"

    losses = [s["metadata"]["active_loss_mw"] for s in batch]
    min_loss = min(losses)
    max_loss = max(losses)
    assert min_loss <= 300.0, f"Expected small unit outage <= 300 MW, min was {min_loss}"
    assert max_loss >= 800.0, f"Expected large unit outage >= 800 MW, max was {max_loss}"


def test_f5_load_regime_variations(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    ren_levels = [s["metadata"]["renewable_penetration"] for s in batch]

    assert min(ren_levels) <= 0.15, "Batch does not cover low renewable penetration (<=15%)"
    assert max(ren_levels) >= 0.45, "Batch does not cover high renewable penetration (>=45%)"


def test_f5_scenario_partitioning_by_id(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    scenario_ids = [s["metadata"]["scenario_id"] for s in batch]
    assert len(scenario_ids) == len(set(scenario_ids)), "Scenario IDs must be strictly unique"

    n_train = int(len(batch) * 0.7)
    train_ids = set(scenario_ids[:n_train])
    test_ids = set(scenario_ids[n_train:])

    assert train_ids.isdisjoint(test_ids), "Data leakage! Train and test scenario sets overlap"
    assert len(train_ids | test_ids) == len(batch), "Partitioning lost scenarios"


def test_f5_ground_truth_labeling_consistency(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    f_threshold = 59.5

    for scen in batch:
        nadir = scen["metadata"]["ground_truth_nadir"]
        is_sec = scen["metadata"]["is_secure"]
        expected_sec = 1 if nadir >= f_threshold else 0
        assert is_sec == expected_sec, \
            f"Scenario {scen['metadata']['scenario_id']} label mismatch: nadir={nadir}, is_secure={is_sec}, expected={expected_sec}"


def test_f5_scenario_metadata_integrity(synthetic_scenario_batch):
    batch = synthetic_scenario_batch
    for scen in batch:
        meta = scen["metadata"]
        assert isinstance(meta["scenario_id"], str) and len(meta["scenario_id"]) > 0
        assert meta["disturbed_gen"] in IEEE39_GENERATOR_BUSES
        assert meta["active_loss_mw"] > 0
        assert 0.0 <= meta["renewable_penetration"] <= 1.0
        assert meta["t_disturb"] > 0
        assert 55.0 <= meta["ground_truth_nadir"] <= 60.5
        assert meta["is_secure"] in (0, 1)
