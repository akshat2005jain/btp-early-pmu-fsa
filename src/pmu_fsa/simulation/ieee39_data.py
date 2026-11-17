"""IEEE 39-bus (New England 10-machine) 50 Hz benchmark data."""

import math
from typing import Dict, List, Any, Tuple

BASE_MVA: float = 100.0
NOMINAL_FREQ: float = 50.0
OMEGA_0: float = 2.0 * math.pi * NOMINAL_FREQ
SECURITY_THRESHOLD_HZ: float = 49.50
PHYSICAL_FREQ_BOUNDS: Tuple[float, float] = (45.0, 50.5)
N_BUSES: int = 39
N_GENERATORS: int = 10
N_BRANCHES: int = 46

# Coherent 3-area bus partition (South, Central, North)
AREA_1_BUSES: List[int] = [1, 2, 3, 4, 18, 25, 30, 31, 32]
AREA_2_BUSES: List[int] = [5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 33, 34, 35, 36]
AREA_3_BUSES: List[int] = [9, 17, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 37, 38, 39]

AREA_PARTITION: Dict[int, List[int]] = {
    1: AREA_1_BUSES,
    2: AREA_2_BUSES,
    3: AREA_3_BUSES,
}

BUS_TO_AREA: Dict[int, int] = {}
for _area_id, _buses in AREA_PARTITION.items():
    for _b in _buses:
        BUS_TO_AREA[_b] = _area_id

GENERATOR_BUSES: List[int] = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]


def compute_system_droop_pu(
    r_machine_pu: float,
    s_mva: float,
    base_mva: float = BASE_MVA,
) -> float:
    """Convert machine-base droop R into system-base droop R_sys_pu = R * (S_base / S_mva)."""
    if s_mva <= 0.0:
        raise ValueError(f"Generator MVA rating s_mva must be positive, got {s_mva}")
    return float(r_machine_pu) * (float(base_mva) / float(s_mva))


BUS39_REGIONAL_PARAMS: Dict[str, Any] = {
    "type": "Regional/Thermal",
    "H": 42.0,
    "Xdp": 0.0400,
    "Xd": 0.1300,
    "Xq": 0.1200,
}

BUS39_MACRO_INTERTIE_PARAMS: Dict[str, Any] = {
    "type": "Intertie/Macro",
    "H": 500.0,
    "Xdp": 0.0060,
    "Xd": 0.0200,
    "Xq": 0.0190,
}

# Machine parameters: H, D, X on 100 MVA system base; R is 5% droop on machine rating S_mva,
# R_sys_pu = R * (BASE_MVA / S_mva) is droop on the 100 MVA system base, and Tg (s) is the
# effective first-order governor + steam-reheat turbine time constant.
# Default Bus 39 represents an islanded regional plant (H = 42.0 s, Xdp = 0.0400 pu).
GENERATOR_DATA: List[Dict[str, Any]] = [
    {
        "unit": 1, "bus": 30, "pai_gen": 10, "name": "G1", "type": "Thermal",
        "S_mva": 350.0, "P_mw": 250.0, "Q_mvar": 144.9, "V_pu": 1.0475,
        "H": 42.0, "D": 2.0, "Xdp": 0.0310, "Xd": 0.1000, "Xq": 0.0690,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 350.0), "Tg": 4.20,
    },
    {
        "unit": 2, "bus": 31, "pai_gen": 2, "name": "G2", "type": "Nuclear/Slack",
        "S_mva": 750.0, "P_mw": 572.8, "Q_mvar": 398.7, "V_pu": 0.9820,
        "H": 30.3, "D": 2.0, "Xdp": 0.0697, "Xd": 0.2950, "Xq": 0.2820,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 750.0), "Tg": 4.20,
    },
    {
        "unit": 3, "bus": 32, "pai_gen": 3, "name": "G3", "type": "Nuclear",
        "S_mva": 800.0, "P_mw": 650.0, "Q_mvar": 205.4, "V_pu": 0.9831,
        "H": 35.8, "D": 2.0, "Xdp": 0.0531, "Xd": 0.2495, "Xq": 0.2370,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 800.0), "Tg": 4.20,
    },
    {
        "unit": 4, "bus": 33, "pai_gen": 4, "name": "G4", "type": "Fossil/Thermal",
        "S_mva": 750.0, "P_mw": 632.0, "Q_mvar": 108.9, "V_pu": 0.9972,
        "H": 28.6, "D": 2.0, "Xdp": 0.0436, "Xd": 0.2620, "Xq": 0.2580,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 750.0), "Tg": 4.20,
    },
    {
        "unit": 5, "bus": 34, "pai_gen": 5, "name": "G5", "type": "Fossil/Thermal",
        "S_mva": 650.0, "P_mw": 508.0, "Q_mvar": 166.9, "V_pu": 1.0123,
        "H": 26.0, "D": 2.0, "Xdp": 0.1320, "Xd": 0.6700, "Xq": 0.6200,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 650.0), "Tg": 4.20,
    },
    {
        "unit": 6, "bus": 35, "pai_gen": 6, "name": "G6", "type": "Nuclear",
        "S_mva": 800.0, "P_mw": 650.0, "Q_mvar": 211.2, "V_pu": 1.0493,
        "H": 34.8, "D": 2.0, "Xdp": 0.0500, "Xd": 0.2540, "Xq": 0.2410,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 800.0), "Tg": 4.20,
    },
    {
        "unit": 7, "bus": 36, "pai_gen": 7, "name": "G7", "type": "Fossil/Thermal",
        "S_mva": 700.0, "P_mw": 560.0, "Q_mvar": 100.4, "V_pu": 1.0635,
        "H": 26.4, "D": 2.0, "Xdp": 0.0490, "Xd": 0.2950, "Xq": 0.2920,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 700.0), "Tg": 4.20,
    },
    {
        "unit": 8, "bus": 37, "pai_gen": 8, "name": "G8", "type": "Nuclear",
        "S_mva": 650.0, "P_mw": 540.0, "Q_mvar": 0.6, "V_pu": 1.0278,
        "H": 24.3, "D": 2.0, "Xdp": 0.0570, "Xd": 0.2900, "Xq": 0.2800,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 650.0), "Tg": 4.20,
    },
    {
        "unit": 9, "bus": 38, "pai_gen": 9, "name": "G9", "type": "Nuclear",
        "S_mva": 1000.0, "P_mw": 830.0, "Q_mvar": 22.7, "V_pu": 1.0265,
        "H": 34.5, "D": 2.0, "Xdp": 0.0570, "Xd": 0.2106, "Xq": 0.2050,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 1000.0), "Tg": 4.20,
    },
    {
        "unit": 10, "bus": 39, "pai_gen": 1, "name": "G10", "type": "Regional/Thermal",
        "S_mva": 1200.0, "P_mw": 1000.0, "Q_mvar": 87.9, "V_pu": 1.0300,
        "H": 42.0, "D": 2.0, "Xdp": 0.0400, "Xd": 0.1300, "Xq": 0.1200,
        "R": 0.05, "R_sys_pu": compute_system_droop_pu(0.05, 1200.0), "Tg": 4.20,
    },
]

# Bus loads: bus_id -> (P_mw, Q_mvar)
LOAD_DATA: Dict[int, Tuple[float, float]] = {
    3: (322.0, 2.4),
    4: (500.0, 184.0),
    7: (233.8, 84.0),
    8: (522.0, 176.0),
    12: (8.5, 88.0),
    15: (320.0, 153.0),
    16: (329.0, 32.3),
    18: (158.0, 30.0),
    20: (680.0, 103.0),
    21: (274.0, 115.0),
    23: (247.5, 84.6),
    24: (308.6, -92.2),
    25: (224.0, 47.2),
    26: (139.0, 17.0),
    27: (281.0, 75.5),
    28: (206.0, 27.6),
    29: (283.5, 26.9),
    31: (9.2, 4.6),
    39: (1104.0, 250.0),
}

BRANCH_DATA: List[Dict[str, Any]] = [
    # Transmission lines
    {"from": 1, "to": 2, "r": 0.0035, "x": 0.0411, "b": 0.6987, "tap": 1.0},
    {"from": 1, "to": 39, "r": 0.0010, "x": 0.0250, "b": 0.7500, "tap": 1.0},
    {"from": 2, "to": 3, "r": 0.0013, "x": 0.0151, "b": 0.2572, "tap": 1.0},
    {"from": 2, "to": 25, "r": 0.0070, "x": 0.0086, "b": 0.1460, "tap": 1.0},
    {"from": 3, "to": 4, "r": 0.0013, "x": 0.0213, "b": 0.2214, "tap": 1.0},
    {"from": 3, "to": 18, "r": 0.0011, "x": 0.0133, "b": 0.2138, "tap": 1.0},
    {"from": 4, "to": 5, "r": 0.0008, "x": 0.0128, "b": 0.1342, "tap": 1.0},
    {"from": 4, "to": 14, "r": 0.0008, "x": 0.0129, "b": 0.1382, "tap": 1.0},
    {"from": 5, "to": 6, "r": 0.0002, "x": 0.0026, "b": 0.0434, "tap": 1.0},
    {"from": 5, "to": 8, "r": 0.0008, "x": 0.0112, "b": 0.1476, "tap": 1.0},
    {"from": 6, "to": 7, "r": 0.0006, "x": 0.0092, "b": 0.1130, "tap": 1.0},
    {"from": 6, "to": 11, "r": 0.0007, "x": 0.0082, "b": 0.1389, "tap": 1.0},
    {"from": 7, "to": 8, "r": 0.0004, "x": 0.0046, "b": 0.0780, "tap": 1.0},
    {"from": 8, "to": 9, "r": 0.0023, "x": 0.0363, "b": 0.3804, "tap": 1.0},
    {"from": 9, "to": 39, "r": 0.0010, "x": 0.0250, "b": 1.2000, "tap": 1.0},
    {"from": 10, "to": 11, "r": 0.0004, "x": 0.0043, "b": 0.0729, "tap": 1.0},
    {"from": 10, "to": 13, "r": 0.0004, "x": 0.0043, "b": 0.0729, "tap": 1.0},
    {"from": 13, "to": 14, "r": 0.0009, "x": 0.0101, "b": 0.1723, "tap": 1.0},
    {"from": 14, "to": 15, "r": 0.0018, "x": 0.0217, "b": 0.3660, "tap": 1.0},
    {"from": 15, "to": 16, "r": 0.0009, "x": 0.0094, "b": 0.1710, "tap": 1.0},
    {"from": 16, "to": 17, "r": 0.0007, "x": 0.0089, "b": 0.1342, "tap": 1.0},
    {"from": 16, "to": 19, "r": 0.0016, "x": 0.0195, "b": 0.3040, "tap": 1.0},
    {"from": 16, "to": 21, "r": 0.0008, "x": 0.0135, "b": 0.2548, "tap": 1.0},
    {"from": 16, "to": 24, "r": 0.0003, "x": 0.0059, "b": 0.0680, "tap": 1.0},
    {"from": 17, "to": 18, "r": 0.0007, "x": 0.0082, "b": 0.1319, "tap": 1.0},
    {"from": 17, "to": 27, "r": 0.0013, "x": 0.0173, "b": 0.3216, "tap": 1.0},
    {"from": 21, "to": 22, "r": 0.0008, "x": 0.0140, "b": 0.2565, "tap": 1.0},
    {"from": 22, "to": 23, "r": 0.0006, "x": 0.0096, "b": 0.1846, "tap": 1.0},
    {"from": 23, "to": 24, "r": 0.0022, "x": 0.0350, "b": 0.3610, "tap": 1.0},
    {"from": 25, "to": 26, "r": 0.0032, "x": 0.0323, "b": 0.5130, "tap": 1.0},
    {"from": 26, "to": 27, "r": 0.0014, "x": 0.0147, "b": 0.2396, "tap": 1.0},
    {"from": 26, "to": 28, "r": 0.0043, "x": 0.0474, "b": 0.7802, "tap": 1.0},
    {"from": 26, "to": 29, "r": 0.0057, "x": 0.0625, "b": 1.0290, "tap": 1.0},
    {"from": 28, "to": 29, "r": 0.0014, "x": 0.0151, "b": 0.2490, "tap": 1.0},
    # Transformers
    {"from": 2, "to": 30, "r": 0.0000, "x": 0.0181, "b": 0.0000, "tap": 1.025},
    {"from": 31, "to": 6, "r": 0.0000, "x": 0.0250, "b": 0.0000, "tap": 0.900},
    {"from": 10, "to": 32, "r": 0.0000, "x": 0.0200, "b": 0.0000, "tap": 1.070},
    {"from": 12, "to": 11, "r": 0.0016, "x": 0.0435, "b": 0.0000, "tap": 1.006},
    {"from": 12, "to": 13, "r": 0.0016, "x": 0.0435, "b": 0.0000, "tap": 1.006},
    {"from": 19, "to": 20, "r": 0.0007, "x": 0.0138, "b": 0.0000, "tap": 1.060},
    {"from": 19, "to": 33, "r": 0.0007, "x": 0.0142, "b": 0.0000, "tap": 1.070},
    {"from": 20, "to": 34, "r": 0.0009, "x": 0.0180, "b": 0.0000, "tap": 1.009},
    {"from": 22, "to": 35, "r": 0.0000, "x": 0.0143, "b": 0.0000, "tap": 1.025},
    {"from": 23, "to": 36, "r": 0.0005, "x": 0.0272, "b": 0.0000, "tap": 1.000},
    {"from": 25, "to": 37, "r": 0.0006, "x": 0.0232, "b": 0.0000, "tap": 1.025},
    {"from": 29, "to": 38, "r": 0.0008, "x": 0.0156, "b": 0.0000, "tap": 1.025},
]


def get_ieee39_bus_data() -> List[Dict[str, Any]]:
    """Return bus data records for all 39 buses."""
    gen_by_bus = {g["bus"]: g for g in GENERATOR_DATA}
    buses = []
    for bus_id in range(1, N_BUSES + 1):
        area = BUS_TO_AREA.get(bus_id, 1)
        p_load, q_load = LOAD_DATA.get(bus_id, (0.0, 0.0))
        is_gen = bus_id in gen_by_bus

        # MATPOWER bus types: 1=PQ, 2=PV, 3=slack
        if bus_id == 31:
            b_type = 3
        elif is_gen:
            b_type = 2
        else:
            b_type = 1

        v_mag = gen_by_bus[bus_id]["V_pu"] if is_gen else 1.0
        v_ang = 0.0
        p_gen = gen_by_bus[bus_id]["P_mw"] if is_gen else 0.0
        q_gen = gen_by_bus[bus_id]["Q_mvar"] if is_gen else 0.0

        buses.append({
            "bus_id": bus_id,
            "bus_type": b_type,
            "area": area,
            "base_kv": 345.0 if bus_id not in [12, 20] else 138.0,
            "V_pu": v_mag,
            "theta_deg": v_ang,
            "P_load_mw": p_load,
            "Q_load_mvar": q_load,
            "P_gen_mw": p_gen,
            "Q_gen_mvar": q_gen,
        })
    return buses


def get_ieee39_branch_data() -> List[Dict[str, Any]]:
    """Return all 46 branches (lines and transformers)."""
    return [dict(b) for b in BRANCH_DATA]


def get_ieee39_gen_data(macro_intertie: bool = False) -> List[Dict[str, Any]]:
    """Return dynamic parameter dicts for the 10 generators."""
    gens = [dict(g) for g in GENERATOR_DATA]
    for g in gens:
        if g["bus"] == 39:
            params = BUS39_MACRO_INTERTIE_PARAMS if macro_intertie else BUS39_REGIONAL_PARAMS
            g.update(params)
        g["R_sys_pu"] = compute_system_droop_pu(g["R"], g["S_mva"], BASE_MVA)
    return gens


def get_total_system_inertia(macro_intertie: bool = False) -> float:
    """Return total synchronous inertia H_tot = sum(H_i) on 100 MVA base."""
    return float(sum(g["H"] for g in get_ieee39_gen_data(macro_intertie=macro_intertie)))
