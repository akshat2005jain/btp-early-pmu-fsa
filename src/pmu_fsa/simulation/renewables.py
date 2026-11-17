"""Co-located effective-response derating model for converter-interfaced renewables (Approach B).

This module models converter-interfaced generation co-located at non-slack synchronous
generator buses (Buses 30, 32-39) via effective rotational inertia, mechanical damping,
and primary governor participation derating by (1 - alpha) while preserving net terminal
bus active-power dispatch in the pre-disturbance AC power flow. Optional synthetic inertia
(h_vir) and fast converter droop (enable_droop, r_ren, t_inv) are supported when enabled.
Explicit network-level renewable bus placement, pre-contingency line-flow redistribution,
PLL dynamics, and converter current saturation are outside this model's scope.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import copy
import math

from pmu_fsa.simulation.ieee39_data import (
    BASE_MVA,
    GENERATOR_DATA,
    LOAD_DATA,
    GENERATOR_BUSES,
)

# All non-slack synchronous generator buses (total base active dispatch = 5620.0 MW)
DEFAULT_DISPLACED_BUSES: List[int] = [30, 32, 33, 34, 35, 36, 37, 38, 39]


@dataclass
class RenewablePenetrationConfig:
    """Configuration for co-located effective inertia/damping/droop derating (Approach B)."""
    p_ren: float = 0.20
    displaced_buses: List[int] = field(default_factory=lambda: list(DEFAULT_DISPLACED_BUSES))
    h_vir: float = 0.0
    enable_droop: bool = False
    r_ren: float = 0.05
    t_inv: float = 0.02

    def __post_init__(self) -> None:
        if not (0.0 <= self.p_ren <= 1.0):
            raise ValueError(f"p_ren must be in [0.0, 1.0], got {self.p_ren}")
        if self.r_ren <= 0.0:
            raise ValueError(f"r_ren must be positive, got {self.r_ren}")
        if self.t_inv <= 0.0:
            raise ValueError(f"t_inv must be positive, got {self.t_inv}")


def _compute_displacement_factors(
    gen_data: List[Dict[str, Any]],
    config: Optional[RenewablePenetrationConfig],
    load_scaling: float = 1.0,
) -> Dict[str, float]:
    """Compute alpha and realized renewable share for the requested operating point."""
    if config is None or config.p_ren <= 0.0:
        return {
            "p_ren_requested": 0.0 if config is None else float(config.p_ren),
            "p_ren_realized": 0.0,
            "alpha": 0.0,
            "displaced_mw": 0.0,
        }

    scale = float(load_scaling) if load_scaling > 0.0 else 1.0
    displaced_set = set(config.displaced_buses)
    target_gens = [g for g in gen_data if g["bus"] in displaced_set]

    p_load_base = sum(float(p) for p, _ in LOAD_DATA.values())
    p_ren_required = float(config.p_ren) * scale * p_load_base
    p_target_scaled = sum(float(g["P_mw"]) * scale for g in target_gens)
    total_gen_scaled = sum(float(g["P_mw"]) * scale for g in gen_data)

    if p_target_scaled > 0.0:
        alpha = min(0.95, max(0.0, float(p_ren_required / p_target_scaled)))
    else:
        alpha = 0.0

    displaced_mw = alpha * p_target_scaled if load_scaling > 0.0 else 0.0
    p_ren_realized = (alpha * p_target_scaled) / max(1e-6, total_gen_scaled)

    return {
        "p_ren_requested": float(config.p_ren),
        "p_ren_realized": float(p_ren_realized),
        "alpha": float(alpha),
        "displaced_mw": float(displaced_mw),
    }


def compute_effective_inertias(
    gen_data: List[Dict[str, Any]],
    config: Optional[RenewablePenetrationConfig] = None,
    load_scaling: float = 1.0,
) -> Dict[int, float]:
    """Compute effective inertia H_eff = (1 - alpha)*H + alpha*H_vir per generator bus."""
    if config is None or config.p_ren <= 0.0:
        return {int(g["bus"]): float(g["H"]) for g in gen_data}

    factors = _compute_displacement_factors(gen_data, config, load_scaling=load_scaling)
    alpha = factors["alpha"]
    displaced_set = set(config.displaced_buses)
    h_vir = max(0.0, float(config.h_vir))

    effective_h: Dict[int, float] = {}
    for g in gen_data:
        bus = int(g["bus"])
        h_orig = float(g["H"])
        if bus in displaced_set:
            h_eff = (1.0 - alpha) * h_orig + alpha * h_vir
        else:
            h_eff = h_orig
        if not math.isfinite(h_eff) or h_eff < 0.5:
            h_eff = 0.5
        effective_h[bus] = float(h_eff)

    return effective_h


def apply_renewable_displacement(
    gen_data: List[Dict[str, Any]],
    config: Optional[RenewablePenetrationConfig] = None,
    load_scaling: float = 1.0,
) -> List[Dict[str, Any]]:
    """Return a copy of gen_data with inertia and optional fast IBR droop scaled for displacement."""
    if config is None:
        config = RenewablePenetrationConfig(p_ren=0.0)

    factors = _compute_displacement_factors(gen_data, config, load_scaling=load_scaling)
    effective_h = compute_effective_inertias(gen_data, config, load_scaling=load_scaling)
    alpha = factors["alpha"]
    displaced_set = set(config.displaced_buses)

    modified_gens = copy.deepcopy(gen_data)
    for g in modified_gens:
        bus = int(g["bus"])
        h_orig = float(g["H"])
        alpha_bus = alpha if (config.p_ren > 0.0 and bus in displaced_set) else 0.0
        s_mva = float(g.get("S_mva", max(BASE_MVA, float(g["P_mw"]) / 0.85)))

        g["H_orig"] = h_orig
        g["H"] = effective_h[bus]
        g["alpha_displaced"] = float(alpha_bus)
        g["p_ren_requested"] = factors["p_ren_requested"]
        g["p_ren_realized"] = factors["p_ren_realized"]
        g["renewable_fraction"] = 1.0 - (g["H"] / h_orig) if h_orig > 0.0 else 0.0

        # Grid-following IBRs provide constant power by default (enable_droop=False);
        # when enable_droop=True, displaced capacity contributes fast synthetic droop with lag t_inv.
        g["enable_ren_droop"] = bool(config.enable_droop and alpha_bus > 0.0)
        if g["enable_ren_droop"]:
            g["inv_r_ren_sys_pu"] = float(alpha_bus * (s_mva / BASE_MVA) / config.r_ren)
        else:
            g["inv_r_ren_sys_pu"] = 0.0
        g["t_inv"] = float(config.t_inv)

    return modified_gens


def summarize_renewable_dispatch(
    modified_gens: List[Dict[str, Any]],
    load_scaling: float = 1.0,
) -> Dict[str, float]:
    """Summarize requested vs realized renewable penetration, displaced MW, and total inertia."""
    if not modified_gens:
        return {
            "p_ren_requested": 0.0,
            "p_ren_realized": 0.0,
            "displaced_mw": 0.0,
            "alpha": 0.0,
            "h_sync_total": 0.0,
            "h_eff_total": 0.0,
        }

    scale = float(load_scaling) if load_scaling > 0.0 else 0.0
    p_ren_requested = float(modified_gens[0].get("p_ren_requested", 0.0))
    p_ren_realized = float(modified_gens[0].get("p_ren_realized", 0.0))
    alpha = float(max(float(g.get("alpha_displaced", 0.0)) for g in modified_gens))
    displaced_mw = float(
        sum(float(g.get("alpha_displaced", 0.0)) * float(g["P_mw"]) * scale for g in modified_gens)
    )
    h_sync_total = float(sum(float(g.get("H_orig", g["H"])) for g in modified_gens))
    h_eff_total = float(sum(float(g["H"]) for g in modified_gens))

    return {
        "p_ren_requested": p_ren_requested,
        "p_ren_realized": p_ren_realized,
        "displaced_mw": displaced_mw,
        "alpha": alpha,
        "h_sync_total": h_sync_total,
        "h_eff_total": h_eff_total,
    }
