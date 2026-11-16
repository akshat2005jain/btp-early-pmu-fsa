"""Synchronous inertia displacement model for converter-interfaced renewable generation."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import copy
import math

from pmu_fsa.simulation.ieee39_data import GENERATOR_DATA, LOAD_DATA, GENERATOR_BUSES


@dataclass
class RenewablePenetrationConfig:
    """Parameters for renewable penetration and synchronous inertia displacement."""
    p_ren: float = 0.20
    displaced_buses: List[int] = field(default_factory=lambda: [32, 34, 36, 38])
    h_vir: float = 0.0  # virtual inertia constant (s)
    r_ren: float = 0.05  # synthetic droop (pu)
    t_inv: float = 0.02  # inverter current-loop time constant (s)

    def __post_init__(self):
        if not (0.0 <= self.p_ren <= 1.0):
            raise ValueError(f"p_ren must be in [0.0, 1.0], got {self.p_ren}")
        if self.t_inv <= 0:
            raise ValueError("t_inv must be positive")


def compute_effective_inertias(
    gen_data: List[Dict[str, Any]],
    config: Optional[RenewablePenetrationConfig] = None,
) -> Dict[int, float]:
    """Compute effective inertia H_eff = (1 - alpha)*H + alpha*H_vir per generator bus."""
    if config is None or config.p_ren <= 0.0:
        return {g["bus"]: float(g["H"]) for g in gen_data}

    displaced_set = set(config.displaced_buses)
    target_gens = [g for g in gen_data if g["bus"] in displaced_set]
    p_target_total = sum(g["P_mw"] for g in target_gens)

    p_load_total = sum(p for p, q in LOAD_DATA.values())
    p_ren_required = config.p_ren * p_load_total

    # Share of candidate synchronous capacity displaced to meet p_ren of total load
    if p_target_total > 0:
        alpha = min(1.0, max(0.0, float(p_ren_required / p_target_total)))
    else:
        alpha = min(1.0, max(0.0, float(config.p_ren)))

    effective_h = {}
    h_vir = max(0.0, float(config.h_vir))
    for g in gen_data:
        bus = g["bus"]
        h_orig = float(g["H"])
        if bus in displaced_set:
            h_eff = (1.0 - alpha) * h_orig + alpha * h_vir
        else:
            h_eff = h_orig
        if not math.isfinite(h_eff) or h_eff < 0.5:
            h_eff = 0.5  # floor to prevent singular swing dynamics
        effective_h[bus] = float(h_eff)

    return effective_h


def apply_renewable_displacement(
    gen_data: List[Dict[str, Any]],
    config: Optional[RenewablePenetrationConfig] = None,
) -> List[Dict[str, Any]]:
    """Return a copy of gen_data with inertia scaled for renewable displacement."""
    if config is None:
        config = RenewablePenetrationConfig(p_ren=0.0)

    effective_h = compute_effective_inertias(gen_data, config)
    modified_gens = copy.deepcopy(gen_data)
    for g in modified_gens:
        bus = g["bus"]
        g["H_orig"] = g["H"]
        g["H"] = effective_h[bus]
        g["renewable_fraction"] = 1.0 - (g["H"] / g["H_orig"]) if g["H_orig"] > 0 else 0.0
    return modified_gens
