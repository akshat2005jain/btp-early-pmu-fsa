"""Contingency scenario generation, dataset splitting, and NPZ serialization."""

import os
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Iterator
import numpy as np

from pmu_fsa.simulation.ieee39_data import GENERATOR_BUSES, GENERATOR_DATA, NOMINAL_FREQ, SECURITY_THRESHOLD_HZ
from pmu_fsa.simulation.renewables import RenewablePenetrationConfig
from pmu_fsa.simulation.dynamic_engine import DynamicSimulator
from pmu_fsa.simulation.pmu_recorder import PMURecorder, PMUNoiseConfig


GEN_CAPACITIES_MW: Dict[int, float] = {
    g["bus"]: float(g["P_mw"]) for g in GENERATOR_DATA
}


def assign_scenario_partition(
    renewable_penetration: float,
    load_scale: float,
    scenario_id: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> str:
    """Assign scenario to 'train', 'val', 'test', or 'ood_test' from operating point and ID hash."""
    is_ood = (renewable_penetration >= 0.35) or (load_scale < 0.88 or load_scale > 1.12)
    if is_ood:
        return "ood_test"

    hasher = hashlib.md5(scenario_id.encode("utf-8"))
    hash_val = int(hasher.hexdigest(), 16) % 10000 / 10000.0

    if hash_val < train_ratio:
        return "train"
    elif hash_val < (train_ratio + val_ratio):
        return "val"
    else:
        return "test"


def create_scenario_id(
    gen_bus: int,
    loss_mw: float,
    p_ren: float,
    load_scale: float,
    rep_idx: int = 0,
) -> str:
    """Format deterministic scenario ID from contingency and operating parameters."""
    return f"scen_gen{gen_bus}_loss{int(round(loss_mw))}_ren{int(round(p_ren*100))}_ld{int(round(load_scale*100))}_{rep_idx:03d}"


class ScenarioGenerator:
    """Generates post-contingency trajectories across generator outages and operating points."""

    def __init__(
        self,
        fps: int = 50,
        t_span: Tuple[float, float] = (0.0, 10.0),
        t_disturb: float = 1.0,
        security_threshold_hz: float = SECURITY_THRESHOLD_HZ,
        macro_intertie: bool = True,
        f_nominal: float = NOMINAL_FREQ,
    ):
        self.fps = fps
        self.t_span = t_span
        self.t_disturb = t_disturb
        self.security_threshold_hz = float(security_threshold_hz)
        self.macro_intertie = macro_intertie
        self.f_nominal = float(f_nominal)
        self.pmu_recorder = PMURecorder(fps=fps, f_nominal=self.f_nominal)

    def generate_single_scenario(
        self,
        tripped_bus: int,
        p_loss_mw: float,
        renewable_penetration: float,
        load_scale: float,
        rep_idx: int = 0,
        inject_noise: bool = False,
        noise_config: Optional[PMUNoiseConfig] = None,
    ) -> Dict[str, Any]:
        """Simulate one generator outage scenario and return the recorded PMU trajectory dict."""
        scenario_id = create_scenario_id(
            tripped_bus, p_loss_mw, renewable_penetration, load_scale, rep_idx
        )
        partition = assign_scenario_partition(
            renewable_penetration, load_scale, scenario_id
        )

        ren_cfg = RenewablePenetrationConfig(p_ren=renewable_penetration)
        sim = DynamicSimulator(
            load_scaling=load_scale,
            renewable_config=ren_cfg,
            macro_intertie=self.macro_intertie,
            f_nominal=self.f_nominal,
            security_threshold_hz=self.security_threshold_hz,
        )

        raw_traj = sim.simulate(
            t_span=self.t_span,
            t_disturb=self.t_disturb,
            tripped_bus=tripped_bus,
            p_loss_mw=p_loss_mw,
            fps=self.fps,
        )

        recorder = self.pmu_recorder
        if noise_config is not None:
            recorder = PMURecorder(fps=self.fps, noise_config=noise_config, f_nominal=self.f_nominal)

        pmu_traj = recorder.record_trajectory(raw_traj, inject_noise=inject_noise)

        pmu_traj["metadata"]["scenario_id"] = scenario_id
        pmu_traj["metadata"]["partition"] = partition
        pmu_traj["metadata"]["f_nominal"] = self.f_nominal
        pmu_traj["metadata"]["f_threshold_hz"] = self.security_threshold_hz
        pmu_traj["metadata"]["is_secure"] = 1 if pmu_traj["metadata"]["ground_truth_nadir"] >= self.security_threshold_hz else 0

        return pmu_traj

    def generate_scenario_batch(
        self,
        gen_buses: Optional[List[int]] = None,
        load_scales: Optional[List[float]] = None,
        penetrations: Optional[List[float]] = None,
        loss_fractions: Optional[List[float]] = None,
        max_scenarios: Optional[int] = None,
        inject_noise: bool = False,
    ) -> List[Dict[str, Any]]:
        """Generate scenarios over the grid of buses, loss fractions, renewables, and loads."""
        if gen_buses is None:
            gen_buses = list(GENERATOR_BUSES)
        if load_scales is None:
            load_scales = [0.80, 0.90, 1.00, 1.10, 1.20]
        if penetrations is None:
            penetrations = [0.10, 0.20, 0.30, 0.40, 0.50]
        if loss_fractions is None:
            loss_fractions = [1.0, 0.8, 0.5]

        scenarios = []
        count = 0

        for bus in gen_buses:
            nom_p = GEN_CAPACITIES_MW.get(bus, 500.0)
            for lf in loss_fractions:
                p_loss = nom_p * lf
                for p_ren in penetrations:
                    for ld in load_scales:
                        scen = self.generate_single_scenario(
                            tripped_bus=bus,
                            p_loss_mw=p_loss,
                            renewable_penetration=p_ren,
                            load_scale=ld,
                            rep_idx=count,
                            inject_noise=inject_noise,
                        )
                        scenarios.append(scen)
                        count += 1
                        if max_scenarios is not None and count >= max_scenarios:
                            return scenarios

        return scenarios


def export_scenario_to_npz(scenario: Dict[str, Any], filepath: str) -> None:
    """Save scenario trajectory to compressed .npz archive."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    np.savez_compressed(
        filepath,
        timestamps=scenario["timestamps"],
        bus_frequencies=scenario["bus_frequencies"],
        bus_rocof=scenario["bus_rocof"],
        voltage_magnitudes=scenario["voltage_magnitudes"],
        voltage_angles=scenario["voltage_angles"],
        coi_frequency=scenario["coi_frequency"],
        metadata=scenario["metadata"],
    )


def load_scenario_from_npz(filepath: str) -> Dict[str, Any]:
    """Load scenario trajectory from .npz file."""
    with np.load(filepath, allow_pickle=True) as data:
        return {
            "timestamps": data["timestamps"],
            "bus_frequencies": data["bus_frequencies"],
            "bus_rocof": data["bus_rocof"],
            "voltage_magnitudes": data["voltage_magnitudes"],
            "voltage_angles": data["voltage_angles"],
            "coi_frequency": data["coi_frequency"],
            "metadata": data["metadata"].item() if hasattr(data["metadata"], "item") else dict(data["metadata"]),
        }
