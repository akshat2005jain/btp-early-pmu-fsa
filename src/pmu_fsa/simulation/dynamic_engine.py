"""AC power-flow-initialized Kron-reduced dynamic simulator for the IEEE 39-bus system."""

from typing import Dict, List, Any, Optional, Tuple, Union
import math
import warnings
import numpy as np
from scipy.integrate import solve_ivp

from pmu_fsa.simulation.ieee39_data import (
    BASE_MVA,
    NOMINAL_FREQ,
    OMEGA_0,
    SECURITY_THRESHOLD_HZ,
    N_BUSES,
    N_GENERATORS,
    GENERATOR_BUSES,
    GENERATOR_DATA,
    LOAD_DATA,
    BRANCH_DATA,
    AREA_PARTITION,
    BUS_TO_AREA,
    compute_system_droop_pu,
    get_ieee39_bus_data,
    get_ieee39_branch_data,
    get_ieee39_gen_data,
)
from pmu_fsa.simulation.renewables import (
    RenewablePenetrationConfig,
    apply_renewable_displacement,
    summarize_renewable_dispatch,
)


def build_ybus(branches: List[Dict[str, Any]], n_buses: int = N_BUSES) -> np.ndarray:
    """Build the nodal admittance matrix Y_bus (1-based branch bus IDs -> 0-based indices)."""
    y_bus = np.zeros((n_buses, n_buses), dtype=complex)
    for br in branches:
        from_idx = br["from"] - 1
        to_idx = br["to"] - 1
        r = float(br["r"])
        x = float(br["x"])
        b = float(br.get("b", 0.0))
        tap = float(br.get("tap", 1.0))
        if tap <= 0.0:
            tap = 1.0

        z_s = r + 1j * x
        if abs(z_s) < 1e-12:
            continue
        y_s = 1.0 / z_s
        y_c = 1j * (b / 2.0)

        # Off-nominal tap ratio on the 'from' side
        y_bus[from_idx, from_idx] += (y_s + y_c) / (tap ** 2)
        y_bus[to_idx, to_idx] += (y_s + y_c)
        y_bus[from_idx, to_idx] -= y_s / tap
        y_bus[to_idx, from_idx] -= y_s / tap

    return y_bus


def solve_power_flow(
    load_scaling: float = 1.0,
    gen_data: Optional[List[Dict[str, Any]]] = None,
    branches: Optional[List[Dict[str, Any]]] = None,
    loads: Optional[Dict[int, Tuple[float, float]]] = None,
    max_iter: int = 25,
    tol: float = 1e-11,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Solve Newton-Raphson AC power flow on the 39-bus Y_bus.

    Bus 31 (or the first generator bus if 31 is absent) acts as slack with fixed V_pu and
    theta = 0 rad; remaining generator buses are PV buses with scheduled P_mw * load_scaling
    and fixed V_pu; non-generator buses are PQ buses with (P_L, Q_L) * load_scaling.
    """
    if gen_data is None:
        gen_data = get_ieee39_gen_data(macro_intertie=False)
    if branches is None:
        branches = get_ieee39_branch_data()
    if loads is None:
        loads = dict(LOAD_DATA)

    n_buses = N_BUSES
    Y_bus = build_ybus(branches, n_buses)
    G = Y_bus.real
    B = Y_bus.imag

    gen_by_bus: Dict[int, Dict[str, Any]] = {int(g["bus"]) - 1: g for g in gen_data}
    gen_bus_indices = [int(g["bus"]) - 1 for g in gen_data]

    slack_idx = 30 if 30 in gen_by_bus else gen_bus_indices[0]
    non_slack = sorted(b for b in range(n_buses) if b != slack_idx)
    pq_indices = sorted(b for b in range(n_buses) if b not in gen_by_bus)

    V_init = np.ones(n_buses, dtype=float)
    for b_idx, g in gen_by_bus.items():
        V_init[b_idx] = float(g["V_pu"])

    V = V_init.copy()
    theta = np.zeros(n_buses, dtype=float)

    p_load_pu = np.zeros(n_buses, dtype=float)
    q_load_pu = np.zeros(n_buses, dtype=float)
    for bus_id, (p_mw, q_mvar) in loads.items():
        b_idx = int(bus_id) - 1
        p_load_pu[b_idx] = (float(p_mw) * float(load_scaling)) / BASE_MVA
        q_load_pu[b_idx] = (float(q_mvar) * float(load_scaling)) / BASE_MVA

    p_spec = -p_load_pu.copy()
    q_spec = -q_load_pu.copy()
    for b_idx, g in gen_by_bus.items():
        p_spec[b_idx] += (float(g["P_mw"]) * float(load_scaling)) / BASE_MVA

    converged = False
    max_mismatch = float("inf")
    it_used = 0
    n_ns = len(non_slack)

    for it in range(1, max_iter + 1):
        it_used = it
        Vc = V * np.exp(1j * theta)
        S_calc = Vc * np.conj(Y_bus @ Vc)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        dP = p_spec[non_slack] - P_calc[non_slack]
        dQ = q_spec[pq_indices] - Q_calc[pq_indices]
        mismatch = np.concatenate([dP, dQ])
        max_mismatch = float(np.max(np.abs(mismatch))) if mismatch.size > 0 else 0.0

        if not np.isfinite(max_mismatch):
            break
        if max_mismatch < tol:
            converged = True
            break

        ang = theta[:, None] - theta[None, :]
        sin_a = np.sin(ang)
        cos_a = np.cos(ang)
        V_safe = np.maximum(V, 1e-4)

        J11 = V[:, None] * V[None, :] * (G * sin_a - B * cos_a)
        np.fill_diagonal(J11, -Q_calc - np.diag(B) * (V ** 2))

        J12 = V[:, None] * (G * cos_a + B * sin_a)
        np.fill_diagonal(J12, (P_calc / V_safe) + np.diag(G) * V)

        J21 = -V[:, None] * V[None, :] * (G * cos_a + B * sin_a)
        np.fill_diagonal(J21, P_calc - np.diag(G) * (V ** 2))

        J22 = V[:, None] * (G * sin_a - B * cos_a)
        np.fill_diagonal(J22, (Q_calc / V_safe) - np.diag(B) * V)

        J = np.block([
            [J11[np.ix_(non_slack, non_slack)], J12[np.ix_(non_slack, pq_indices)]],
            [J21[np.ix_(pq_indices, non_slack)], J22[np.ix_(pq_indices, pq_indices)]],
        ])

        try:
            dx = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            break

        if not np.all(np.isfinite(dx)):
            break

        step_scale = 1.0
        max_step = float(np.max(np.abs(dx))) if dx.size > 0 else 0.0
        if max_step > 1.0:
            step_scale = 1.0 / max_step

        theta[non_slack] += step_scale * dx[:n_ns]
        V[pq_indices] += step_scale * dx[n_ns:]
        if np.any(V < 0.25) or np.any(V > 2.0):
            break

    if converged:
        V0 = V * np.exp(1j * theta)
        S_net = V0 * np.conj(Y_bus @ V0)
        P_gen_solved_mw = (S_net.real[gen_bus_indices] + p_load_pu[gen_bus_indices]) * BASE_MVA
        Q_gen_solved_mvar = (S_net.imag[gen_bus_indices] + q_load_pu[gen_bus_indices]) * BASE_MVA
    else:
        # Fallback for extreme non-solvable synthetic stress probes (e.g. 5x load collapse)
        V0 = V_init.astype(complex)
        P_gen_solved_mw = np.array([float(g["P_mw"]) * float(load_scaling) for g in gen_data], dtype=float)
        Q_gen_solved_mvar = np.array([float(g["Q_mvar"]) * float(load_scaling) for g in gen_data], dtype=float)

    total_gen_mw = float(np.sum(P_gen_solved_mw))
    total_load_mw = float(np.sum(p_load_pu) * BASE_MVA)
    diagnostics: Dict[str, Any] = {
        "converged": bool(converged),
        "max_mismatch_pu": float(max_mismatch) if math.isfinite(max_mismatch) else float("inf"),
        "iterations": int(it_used),
        "total_gen_mw": total_gen_mw,
        "total_load_mw": total_load_mw,
        "losses_mw": total_gen_mw - total_load_mw,
    }
    return V0, P_gen_solved_mw, Q_gen_solved_mvar, diagnostics


def _kron_reduce_from_admittances(
    Y_NN_no_gen: np.ndarray,
    gen_bus_indices: List[int],
    y_g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Schur-complement Kron matrices (Y_red, K) for generator admittances y_g."""
    m = len(gen_bus_indices)
    n_buses = Y_NN_no_gen.shape[0]

    Y_NN = Y_NN_no_gen.copy()
    Y_GN = np.zeros((m, n_buses), dtype=complex)
    for i, b_idx in enumerate(gen_bus_indices):
        if abs(y_g[i]) > 0.0:
            Y_NN[b_idx, b_idx] += y_g[i]
            Y_GN[i, b_idx] = -y_g[i]

    Y_NG = Y_GN.T
    Y_GG = np.diag(y_g)

    inv_Y_NN = np.linalg.inv(Y_NN)
    K = -inv_Y_NN @ Y_NG
    Y_red = Y_GG - Y_GN @ inv_Y_NN @ Y_NG
    return Y_red, K


def compute_kron_reduction(
    load_scaling: float = 1.0,
    gen_data: Optional[List[Dict[str, Any]]] = None,
    branches: Optional[List[Dict[str, Any]]] = None,
    loads: Optional[Dict[int, Tuple[float, float]]] = None,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute AC-power-flow-initialized Kron-reduced Y_red (and K, E_mag0, delta0, Pm0)."""
    if gen_data is None:
        gen_data = get_ieee39_gen_data(macro_intertie=False)
    if branches is None:
        branches = get_ieee39_branch_data()
    if loads is None:
        loads = dict(LOAD_DATA)

    m = len(gen_data)
    n_buses = N_BUSES
    gen_bus_indices = [int(g["bus"]) - 1 for g in gen_data]

    V0, P_gen_solved_mw, Q_gen_solved_mvar, _ = solve_power_flow(
        load_scaling=load_scaling,
        gen_data=gen_data,
        branches=branches,
        loads=loads,
    )

    y_load = np.zeros(n_buses, dtype=complex)
    for bus_id, (p_mw, q_mvar) in loads.items():
        b_idx = int(bus_id) - 1
        v_sq = abs(V0[b_idx]) ** 2 if abs(V0[b_idx]) > 0.1 else 1.0
        p_pu = (float(p_mw) * float(load_scaling)) / BASE_MVA
        q_pu = (float(q_mvar) * float(load_scaling)) / BASE_MVA
        y_load[b_idx] = (p_pu - 1j * q_pu) / v_sq

    xdp = np.array([float(g["Xdp"]) for g in gen_data], dtype=float)
    y_g = 1.0 / (1j * xdp)

    Y_bus = build_ybus(branches, n_buses)
    Y_NN_no_gen = Y_bus + np.diag(y_load)
    Y_red, K = _kron_reduce_from_admittances(Y_NN_no_gen, gen_bus_indices, y_g)

    E_mag0 = np.zeros(m, dtype=float)
    delta0 = np.zeros(m, dtype=float)
    for i in range(m):
        b_idx = gen_bus_indices[i]
        p_g0 = P_gen_solved_mw[i] / BASE_MVA
        q_g0 = Q_gen_solved_mvar[i] / BASE_MVA
        V_term = V0[b_idx]
        I_g0 = (p_g0 - 1j * q_g0) / np.conj(V_term)
        E_prime = V_term + 1j * xdp[i] * I_g0
        E_mag0[i] = abs(E_prime)
        delta0[i] = np.angle(E_prime)

    E_vec0 = E_mag0 * np.exp(1j * delta0)
    I_G0 = Y_red @ E_vec0
    Pm0 = np.real(E_vec0 * np.conj(I_G0))

    if return_all:
        return Y_red, K, E_mag0, delta0, Pm0
    return Y_red


class DynamicSimulator:
    """10-machine Kron-reduced dynamic simulator with AC power flow initialization and droop governors."""

    def __init__(
        self,
        load_scaling: float = 1.0,
        renewable_config: Optional[RenewablePenetrationConfig] = None,
        macro_intertie: bool = False,
        f_nominal: float = NOMINAL_FREQ,
        security_threshold_hz: float = SECURITY_THRESHOLD_HZ,
    ):
        self.load_scaling = float(load_scaling)
        self.renewable_config = renewable_config or RenewablePenetrationConfig(p_ren=0.0)
        self.macro_intertie = bool(macro_intertie)
        self.f_nominal = float(f_nominal)
        self.omega_0 = 2.0 * math.pi * self.f_nominal
        self.security_threshold_hz = float(security_threshold_hz)

        base_gens = get_ieee39_gen_data(macro_intertie=self.macro_intertie)
        self.gen_data = apply_renewable_displacement(
            base_gens,
            self.renewable_config,
            load_scaling=self.load_scaling,
        )
        self.renewable_summary = summarize_renewable_dispatch(
            self.gen_data,
            load_scaling=self.load_scaling,
        )
        self.branches = get_ieee39_branch_data()
        self.loads = dict(LOAD_DATA)

        self.n_gen = len(self.gen_data)
        self.n_bus = N_BUSES
        self.gen_buses = np.array([int(g["bus"]) for g in self.gen_data], dtype=int)
        self._gen_bus_indices = [int(b) - 1 for b in self.gen_buses]

        (
            self.V0,
            self.P_gen_solved_mw,
            self.Q_gen_solved_mvar,
            self.pf_diagnostics,
        ) = solve_power_flow(
            load_scaling=self.load_scaling,
            gen_data=self.gen_data,
            branches=self.branches,
            loads=self.loads,
        )

        self.Y_bus = build_ybus(self.branches, self.n_bus)
        self._p_load_pu = np.zeros(self.n_bus, dtype=float)
        self._q_load_pu = np.zeros(self.n_bus, dtype=float)
        for bus_id, (p_mw, q_mvar) in self.loads.items():
            b_idx = int(bus_id) - 1
            self._p_load_pu[b_idx] = (float(p_mw) * self.load_scaling) / BASE_MVA
            self._q_load_pu[b_idx] = (float(q_mvar) * self.load_scaling) / BASE_MVA

        v0_sq = np.where(np.abs(self.V0) > 0.1, np.abs(self.V0) ** 2, 1.0)
        self._y_load_pre = (self._p_load_pu - 1j * self._q_load_pu) / v0_sq
        self._Y_NN_no_gen_pre = self.Y_bus + np.diag(self._y_load_pre)

        self.xdp = np.array([float(g["Xdp"]) for g in self.gen_data], dtype=float)
        self.y_g = 1.0 / (1j * self.xdp)
        self.Y_red, self.K = _kron_reduce_from_admittances(
            self._Y_NN_no_gen_pre,
            self._gen_bus_indices,
            self.y_g,
        )

        self.E_mag0 = np.zeros(self.n_gen, dtype=float)
        self.delta0 = np.zeros(self.n_gen, dtype=float)
        for i, b_idx in enumerate(self._gen_bus_indices):
            p_g0 = self.P_gen_solved_mw[i] / BASE_MVA
            q_g0 = self.Q_gen_solved_mvar[i] / BASE_MVA
            V_term = self.V0[b_idx]
            I_g0 = (p_g0 - 1j * q_g0) / np.conj(V_term)
            E_prime = V_term + 1j * self.xdp[i] * I_g0
            self.E_mag0[i] = abs(E_prime)
            self.delta0[i] = np.angle(E_prime)

        self._E_vec0 = self.E_mag0 * np.exp(1j * self.delta0)
        I_G0 = self.Y_red @ self._E_vec0
        self.Pm0 = np.real(self._E_vec0 * np.conj(I_G0))
        self.max_pm0_mismatch_mw = float(
            np.max(np.abs(self.Pm0 * BASE_MVA - self.P_gen_solved_mw))
        )

        self.H = np.array([float(g["H"]) for g in self.gen_data], dtype=float)
        self.sync_fraction = np.array(
            [1.0 - float(g.get("alpha_displaced", 0.0)) for g in self.gen_data],
            dtype=float,
        )
        self.D = np.array([float(g["D"]) for g in self.gen_data], dtype=float) * self.sync_fraction
        self.S_mva = np.array(
            [float(g.get("S_mva", BASE_MVA)) for g in self.gen_data],
            dtype=float,
        )
        self.R = np.array([float(g["R"]) for g in self.gen_data], dtype=float)
        self.R_sys_pu = np.array(
            [
                float(g.get("R_sys_pu", compute_system_droop_pu(g["R"], g.get("S_mva", BASE_MVA))))
                for g in self.gen_data
            ],
            dtype=float,
        )
        self.inv_R = self.sync_fraction / np.maximum(self.R_sys_pu, 1e-6)
        self.Tg = np.array([float(g["Tg"]) for g in self.gen_data], dtype=float)

        self.inv_R_ren = np.array(
            [float(g.get("inv_r_ren_sys_pu", 0.0)) for g in self.gen_data],
            dtype=float,
        )
        self.t_inv = np.array(
            [float(g.get("t_inv", 0.02)) for g in self.gen_data],
            dtype=float,
        )
        self.has_ren_droop = bool(np.any(self.inv_R_ren > 0.0))
        self.H_tot = float(np.sum(self.H))

    def _build_post_disturbance_kron(
        self,
        trip_gen_idx: int,
        rem: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute post-contingency (Y_red_post, K_post) with voltage-compensated active load."""
        y_g_post = self.y_g.copy()
        if rem <= 1e-6:
            y_g_post[trip_gen_idx] = 0.0 + 0.0j
        else:
            y_g_post[trip_gen_idx] = rem * self.y_g[trip_gen_idx]

        v0_sq = np.where(np.abs(self.V0) > 0.1, np.abs(self.V0) ** 2, 1.0)
        b_load = -self._q_load_pu / v0_sq
        g_load = self._p_load_pu / v0_sq

        E_act = self._E_vec0.copy()
        if rem <= 1e-6:
            E_act[trip_gen_idx] = 0.0 + 0.0j

        # Calibrate post-contingency shunt conductance at t0+ so constant-power active loads
        # above 0.80 pu do not artificially disappear under reactive voltage sag.
        if self.pf_diagnostics.get("converged", False) and np.any(self._p_load_pu > 0.0):
            for _ in range(6):
                Y_NN_iter = self.Y_bus + np.diag(g_load + 1j * b_load)
                _, K_iter = _kron_reduce_from_admittances(
                    Y_NN_iter, self._gen_bus_indices, y_g_post
                )
                V_post0 = K_iter @ E_act
                v_post_sq = np.maximum(np.abs(V_post0) ** 2, 0.65)
                g_load = self._p_load_pu / v_post_sq

        Y_NN_no_gen_post = self.Y_bus + np.diag(g_load + 1j * b_load)
        Y_red_post, K_post = _kron_reduce_from_admittances(
            Y_NN_no_gen_post,
            self._gen_bus_indices,
            y_g_post,
        )
        if rem <= 1e-6:
            Y_red_post[trip_gen_idx, :] = 0.0 + 0.0j
            Y_red_post[:, trip_gen_idx] = 0.0 + 0.0j
            K_post[:, trip_gen_idx] = 0.0 + 0.0j
        return Y_red_post, K_post

    def compute_electrical_power(
        self,
        delta: np.ndarray,
        E_mag: np.ndarray,
        Y_red: Optional[np.ndarray] = None,
        disconnected_idx: int = -1,
    ) -> np.ndarray:
        """Return electrical air-gap power Pe = Re(E' * conj(Y_red * E')) for each machine."""
        Y_mat = self.Y_red if Y_red is None else Y_red
        E_vec = E_mag * np.exp(1j * delta)
        if disconnected_idx >= 0:
            E_vec = E_vec.copy()
            E_vec[disconnected_idx] = 0.0 + 0.0j
        I_G = Y_mat @ E_vec
        Pe = np.real(E_vec * np.conj(I_G))
        if disconnected_idx >= 0:
            Pe[disconnected_idx] = 0.0
        return Pe

    def _compute_current_inv_r(self) -> np.ndarray:
        """Compute current system-base droop gain 1/R_sys_pu honoring any runtime self.R edits."""
        r_sys = np.array(
            [compute_system_droop_pu(self.R[i], self.S_mva[i]) for i in range(self.n_gen)],
            dtype=float,
        )
        return self.sync_fraction / np.maximum(r_sys, 1e-6)

    def _rhs_segment(
        self,
        state: np.ndarray,
        Y_red_seg: np.ndarray,
        H_seg: np.ndarray,
        D_seg: np.ndarray,
        Pm0_seg: np.ndarray,
        inv_R_seg: np.ndarray,
        inv_R_ren_seg: np.ndarray,
        disconnected_idx: int = -1,
        surplus_pu: float = 0.0,
        surplus_idx: int = -1,
    ) -> np.ndarray:
        """Evaluate ODE right-hand side for a fixed topology segment."""
        m = self.n_gen
        delta = state[0:m]
        domega = state[m : 2 * m]
        Pm = state[2 * m : 3 * m]
        P_ren = state[3 * m : 4 * m] if self.has_ren_droop else None

        Pe = self.compute_electrical_power(
            delta,
            self.E_mag0,
            Y_red=Y_red_seg,
            disconnected_idx=disconnected_idx,
        )

        p_mech_total = Pm if P_ren is None else (Pm + P_ren)
        p_accel = p_mech_total - Pe - D_seg * domega
        if surplus_idx >= 0 and surplus_pu != 0.0:
            p_accel[surplus_idx] += surplus_pu

        d_delta = self.omega_0 * domega
        d_domega = np.zeros(m, dtype=float)
        d_Pm = np.zeros(m, dtype=float)

        active = np.ones(m, dtype=bool)
        if disconnected_idx >= 0:
            active[disconnected_idx] = False
            d_delta[disconnected_idx] = 0.0

        if np.any(H_seg[active] <= 0.0):
            raise ValueError("Generator inertia H must be strictly positive (H > 0)")

        d_domega[active] = p_accel[active] / (2.0 * H_seg[active])
        d_Pm[active] = (Pm0_seg[active] - inv_R_seg[active] * domega[active] - Pm[active]) / self.Tg[active]

        if P_ren is not None:
            d_P_ren = np.zeros(m, dtype=float)
            d_P_ren[active] = (
                -inv_R_ren_seg[active] * domega[active] - P_ren[active]
            ) / np.maximum(self.t_inv[active], 1e-4)
            return np.concatenate([d_delta, d_domega, d_Pm, d_P_ren])

        return np.concatenate([d_delta, d_domega, d_Pm])

    def rhs(
        self,
        t: float,
        state: np.ndarray,
        t_disturb: float,
        p_loss_mw: float,
        trip_gen_idx: int,
        loss_fraction: Optional[float] = None,
    ) -> np.ndarray:
        """Evaluate ODE derivatives for 3rd-order state [delta (m), domega (m), Pm (m)]."""
        if np.any(self.H <= 0.0):
            raise ValueError("Generator inertia H must be strictly positive (H > 0)")

        inv_R_base = self._compute_current_inv_r()
        if t < t_disturb or trip_gen_idx < 0:
            return self._rhs_segment(
                state,
                Y_red_seg=self.Y_red,
                H_seg=self.H,
                D_seg=self.D,
                Pm0_seg=self.Pm0,
                inv_R_seg=inv_R_base,
                inv_R_ren_seg=self.inv_R_ren,
            )

        if loss_fraction is None and p_loss_mw < 0.0:
            return self._rhs_segment(
                state,
                Y_red_seg=self.Y_red,
                H_seg=self.H,
                D_seg=self.D,
                Pm0_seg=self.Pm0,
                inv_R_seg=inv_R_base,
                inv_R_ren_seg=self.inv_R_ren,
                surplus_pu=-float(p_loss_mw) / BASE_MVA,
                surplus_idx=trip_gen_idx,
            )

        if loss_fraction is None:
            p_base = max(1e-6, float(self.P_gen_solved_mw[trip_gen_idx]))
            lf = min(1.0, max(0.0, float(p_loss_mw) / p_base))
        else:
            lf = min(1.0, max(0.0, float(loss_fraction)))

        rem = max(0.0, 1.0 - lf)
        Y_red_post, _ = self._build_post_disturbance_kron(trip_gen_idx, rem)
        H_post = self.H.copy()
        D_post = self.D.copy()
        Pm0_post = self.Pm0.copy()
        inv_R_post = inv_R_base.copy()
        inv_R_ren_post = self.inv_R_ren.copy()

        if rem <= 1e-6:
            H_post[trip_gen_idx] = 0.0
            D_post[trip_gen_idx] = 0.0
            Pm0_post[trip_gen_idx] = 0.0
            inv_R_post[trip_gen_idx] = 0.0
            inv_R_ren_post[trip_gen_idx] = 0.0
            state_eff = state.copy()
            state_eff[self.n_gen + trip_gen_idx] = 0.0
            state_eff[2 * self.n_gen + trip_gen_idx] = 0.0
            if self.has_ren_droop and len(state_eff) > 3 * self.n_gen + trip_gen_idx:
                state_eff[3 * self.n_gen + trip_gen_idx] = 0.0
            return self._rhs_segment(
                state_eff,
                Y_red_seg=Y_red_post,
                H_seg=H_post,
                D_seg=D_post,
                Pm0_seg=Pm0_post,
                inv_R_seg=inv_R_post,
                inv_R_ren_seg=inv_R_ren_post,
                disconnected_idx=trip_gen_idx,
            )

        H_post[trip_gen_idx] *= rem
        D_post[trip_gen_idx] *= rem
        Pm0_post[trip_gen_idx] *= rem
        inv_R_post[trip_gen_idx] *= rem
        inv_R_ren_post[trip_gen_idx] *= rem
        return self._rhs_segment(
            state,
            Y_red_seg=Y_red_post,
            H_seg=H_post,
            D_seg=D_post,
            Pm0_seg=Pm0_post,
            inv_R_seg=inv_R_post,
            inv_R_ren_seg=inv_R_ren_post,
        )

    def reconstruct_bus_states(
        self,
        delta_traj: np.ndarray,
        domega_traj: np.ndarray,
        dt: float = 1.0 / 50.0,
        K_pre: Optional[np.ndarray] = None,
        K_post: Optional[np.ndarray] = None,
        post_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct 39-bus voltage magnitudes, angles (deg), frequencies (Hz), and ROCOF (Hz/s)."""
        n_steps = delta_traj.shape[0]
        K_0 = self.K if K_pre is None else K_pre
        K_1 = K_0 if K_post is None else K_post

        E_prime = self.E_mag0[None, :] * np.exp(1j * delta_traj)
        dE_prime = 1j * self.omega_0 * domega_traj * E_prime

        if post_mask is None:
            V_bus = E_prime @ K_0.T
            dV_bus = dE_prime @ K_0.T
        else:
            V_bus = np.zeros((n_steps, self.n_bus), dtype=complex)
            dV_bus = np.zeros((n_steps, self.n_bus), dtype=complex)
            pre_mask = ~post_mask
            if np.any(pre_mask):
                V_bus[pre_mask] = E_prime[pre_mask] @ K_0.T
                dV_bus[pre_mask] = dE_prime[pre_mask] @ K_0.T
            if np.any(post_mask):
                V_bus[post_mask] = E_prime[post_mask] @ K_1.T
                dV_bus[post_mask] = dE_prime[post_mask] @ K_1.T

        V_mag = np.abs(V_bus)
        V_angle_deg = np.degrees(np.angle(V_bus))

        v_denom_safe = np.maximum(V_mag, 1e-6)
        V_bus_safe = np.where(V_mag < 1e-6, v_denom_safe * (1.0 + 0.0j), V_bus)
        omega_dev_rad_s = np.imag(dV_bus / V_bus_safe)
        bus_freq = self.f_nominal + (omega_dev_rad_s / (2.0 * math.pi))

        dt_safe = float(dt) if dt > 0.0 else (1.0 / 50.0)
        bus_rocof = np.zeros_like(bus_freq)
        if post_mask is None:
            if n_steps > 1:
                bus_rocof = np.gradient(bus_freq, dt_safe, axis=0)
        else:
            pre_mask = ~post_mask
            if int(np.sum(pre_mask)) > 1:
                bus_rocof[pre_mask] = np.gradient(bus_freq[pre_mask], dt_safe, axis=0)
            if int(np.sum(post_mask)) > 1:
                bus_rocof[post_mask] = np.gradient(bus_freq[post_mask], dt_safe, axis=0)

        return V_mag, V_angle_deg, bus_freq, bus_rocof

    def compute_coi_metrics(
        self,
        domega_traj: np.ndarray,
        bus_freq: np.ndarray,
        dt: float = 1.0 / 50.0,
        H_post: Optional[np.ndarray] = None,
        post_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """Compute system COI frequency (Hz), COI ROCOF (Hz/s), and per-area COI frequencies."""
        n_steps = domega_traj.shape[0]
        h_pre = self.H
        h_pre_tot = max(1e-8, float(np.sum(h_pre)))
        w_pre = h_pre / h_pre_tot

        h_1 = h_pre if H_post is None else H_post
        h_1_tot = max(1e-8, float(np.sum(h_1)))
        w_post = h_1 / h_1_tot

        machine_f = self.f_nominal * (1.0 + domega_traj)
        f_coi = np.zeros(n_steps, dtype=float)

        if post_mask is None:
            f_coi = np.sum(machine_f * w_pre[None, :], axis=1)
        else:
            pre_mask = ~post_mask
            if np.any(pre_mask):
                f_coi[pre_mask] = np.sum(machine_f[pre_mask] * w_pre[None, :], axis=1)
            if np.any(post_mask):
                f_coi[post_mask] = np.sum(machine_f[post_mask] * w_post[None, :], axis=1)

        dt_safe = float(dt) if dt > 0.0 else (1.0 / 50.0)
        rocof_coi = np.zeros(n_steps, dtype=float)
        if post_mask is None:
            if n_steps > 1:
                rocof_coi = np.gradient(f_coi, dt_safe)
        else:
            pre_mask = ~post_mask
            if int(np.sum(pre_mask)) > 1:
                rocof_coi[pre_mask] = np.gradient(f_coi[pre_mask], dt_safe)
            if int(np.sum(post_mask)) > 1:
                rocof_coi[post_mask] = np.gradient(f_coi[post_mask], dt_safe)

        area_coi: Dict[int, np.ndarray] = {}
        for area_id in [1, 2, 3]:
            area_bus_set = set(AREA_PARTITION[area_id])
            gen_idx = [i for i, b in enumerate(self.gen_buses) if int(b) in area_bus_set]
            if not gen_idx:
                area_coi[area_id] = f_coi.copy()
                continue

            f_area = np.zeros(n_steps, dtype=float)
            h_area_pre = h_pre[gen_idx]
            sum_pre = float(np.sum(h_area_pre))
            w_area_pre = (h_area_pre / sum_pre) if sum_pre > 0.0 else np.full(len(gen_idx), 1.0 / len(gen_idx))

            h_area_post = h_1[gen_idx]
            sum_post = float(np.sum(h_area_post))
            w_area_post = (h_area_post / sum_post) if sum_post > 0.0 else None

            if post_mask is None:
                f_area = np.sum(machine_f[:, gen_idx] * w_area_pre[None, :], axis=1)
            else:
                pre_mask = ~post_mask
                if np.any(pre_mask):
                    f_area[pre_mask] = np.sum(
                        machine_f[pre_mask][:, gen_idx] * w_area_pre[None, :], axis=1
                    )
                if np.any(post_mask):
                    if w_area_post is not None:
                        f_area[post_mask] = np.sum(
                            machine_f[post_mask][:, gen_idx] * w_area_post[None, :], axis=1
                        )
                    else:
                        f_area[post_mask] = f_coi[post_mask]
            area_coi[area_id] = f_area

        return f_coi, rocof_coi, area_coi

    def simulate(
        self,
        t_span: Tuple[float, float] = (0.0, 10.0),
        t_disturb: float = 1.0,
        tripped_bus: Optional[int] = 35,
        p_loss_mw: float = 650.0,
        fps: int = 50,
        method: str = "RK45",
        rtol: float = 1e-6,
        atol: float = 1e-8,
        loss_fraction: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Integrate pre- and post-contingency segments exactly across t_disturb."""
        if np.any(self.H <= 0.0):
            raise ValueError("Generator inertia H must be strictly positive (H > 0)")

        m = self.n_gen
        dt = 1.0 / float(fps)
        n_steps = int(round((float(t_span[1]) - float(t_span[0])) * float(fps))) + 1
        t_eval = np.linspace(float(t_span[0]), float(t_span[0]) + (n_steps - 1) * dt, n_steps)
        t_eval = np.clip(t_eval, float(t_span[0]), float(t_span[1]))

        trip_gen_idx = -1
        if tripped_bus is not None:
            matches = [i for i, b in enumerate(self.gen_buses) if int(b) == int(tripped_bus)]
            if matches:
                trip_gen_idx = matches[0]
            else:
                warnings.warn(
                    f"Contingency bus {tripped_bus} is not a valid generator bus in {self.gen_buses}. "
                    "No generator trip disturbance will occur.",
                    UserWarning,
                    stacklevel=2,
                )

        inv_R_base = self._compute_current_inv_r()
        effective_H_pre = float(np.sum(self.H))
        self.H_tot = effective_H_pre

        has_valid_window_disturb = (
            trip_gen_idx >= 0 and float(t_disturb) <= float(t_span[1])
        )

        surplus_pu = 0.0
        surplus_idx = -1
        disconnected_idx = -1
        rem = 1.0

        if not has_valid_window_disturb:
            lf_actual = 0.0
            actual_loss_mw = float(p_loss_mw) if (tripped_bus is not None and trip_gen_idx < 0) else 0.0
            event_type = "no_disturbance"
        elif loss_fraction is None and float(p_loss_mw) < 0.0:
            lf_actual = 0.0
            actual_loss_mw = float(p_loss_mw)
            surplus_pu = -float(p_loss_mw) / BASE_MVA
            surplus_idx = trip_gen_idx
            event_type = "generation_surplus"
        else:
            p_gen_unit_mw = max(1e-6, float(self.P_gen_solved_mw[trip_gen_idx]))
            if loss_fraction is None:
                lf_actual = min(1.0, max(0.0, float(p_loss_mw) / p_gen_unit_mw))
            else:
                lf_actual = min(1.0, max(0.0, float(loss_fraction)))
            actual_loss_mw = lf_actual * p_gen_unit_mw
            rem = max(0.0, 1.0 - lf_actual)
            if lf_actual <= 1e-12:
                event_type = "no_disturbance"
            elif rem <= 1e-6:
                rem = 0.0
                lf_actual = 1.0
                actual_loss_mw = p_gen_unit_mw
                disconnected_idx = trip_gen_idx
                event_type = "complete_generator_trip"
            else:
                event_type = "partial_generation_loss"

        if event_type in ("complete_generator_trip", "partial_generation_loss"):
            Y_red_post, K_post = self._build_post_disturbance_kron(trip_gen_idx, rem)
            H_post = self.H.copy()
            D_post = self.D.copy()
            Pm0_post = self.Pm0.copy()
            inv_R_post = inv_R_base.copy()
            inv_R_ren_post = self.inv_R_ren.copy()
            if rem == 0.0:
                H_post[trip_gen_idx] = 0.0
                D_post[trip_gen_idx] = 0.0
                Pm0_post[trip_gen_idx] = 0.0
                inv_R_post[trip_gen_idx] = 0.0
                inv_R_ren_post[trip_gen_idx] = 0.0
            else:
                H_post[trip_gen_idx] *= rem
                D_post[trip_gen_idx] *= rem
                Pm0_post[trip_gen_idx] *= rem
                inv_R_post[trip_gen_idx] *= rem
                inv_R_ren_post[trip_gen_idx] *= rem
        else:
            Y_red_post = self.Y_red
            K_post = self.K
            H_post = self.H.copy()
            D_post = self.D.copy()
            Pm0_post = self.Pm0.copy()
            inv_R_post = inv_R_base.copy()
            inv_R_ren_post = self.inv_R_ren.copy()

        effective_H_post = float(np.sum(H_post))

        y0_pre = (
            np.concatenate([self.delta0, np.zeros(m), self.Pm0, np.zeros(m)])
            if self.has_ren_droop
            else np.concatenate([self.delta0, np.zeros(m), self.Pm0])
        )

        t0_start = float(t_span[0])
        t_end = float(t_span[1])
        t_dist = float(t_disturb)

        if event_type == "no_disturbance" or t_dist > t_end:
            sol = solve_ivp(
                fun=lambda t, y: self._rhs_segment(
                    y,
                    Y_red_seg=self.Y_red,
                    H_seg=self.H,
                    D_seg=self.D,
                    Pm0_seg=self.Pm0,
                    inv_R_seg=inv_R_base,
                    inv_R_ren_seg=self.inv_R_ren,
                ),
                t_span=(t0_start, t_end),
                y0=y0_pre,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
            )
            t_traj = sol.t
            y_traj = sol.y
            post_mask = np.zeros(len(t_traj), dtype=bool)
        elif t_dist <= t0_start:
            y0_post = y0_pre.copy()
            if event_type == "complete_generator_trip":
                y0_post[m + trip_gen_idx] = 0.0
                y0_post[2 * m + trip_gen_idx] = 0.0
            elif event_type == "partial_generation_loss":
                y0_post[2 * m + trip_gen_idx] *= rem

            sol_post = solve_ivp(
                fun=lambda t, y: self._rhs_segment(
                    y,
                    Y_red_seg=Y_red_post,
                    H_seg=H_post,
                    D_seg=D_post,
                    Pm0_seg=Pm0_post,
                    inv_R_seg=inv_R_post,
                    inv_R_ren_seg=inv_R_ren_post,
                    disconnected_idx=disconnected_idx,
                    surplus_pu=surplus_pu,
                    surplus_idx=surplus_idx,
                ),
                t_span=(t0_start, t_end),
                y0=y0_post,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
            )
            t_traj = sol_post.t
            y_traj = sol_post.y
            post_mask = np.ones(len(t_traj), dtype=bool)
        else:
            pre_eval_mask = t_eval < (t_dist - 1e-12)
            t_eval_pre = t_eval[pre_eval_mask]
            t_eval_seg1 = np.append(t_eval_pre, t_dist)

            sol_pre = solve_ivp(
                fun=lambda t, y: self._rhs_segment(
                    y,
                    Y_red_seg=self.Y_red,
                    H_seg=self.H,
                    D_seg=self.D,
                    Pm0_seg=self.Pm0,
                    inv_R_seg=inv_R_base,
                    inv_R_ren_seg=self.inv_R_ren,
                ),
                t_span=(t0_start, t_dist),
                y0=y0_pre,
                t_eval=t_eval_seg1,
                method=method,
                rtol=rtol,
                atol=atol,
            )
            y_at_disturb = sol_pre.y[:, -1].copy()
            if event_type == "complete_generator_trip":
                y_at_disturb[m + trip_gen_idx] = 0.0
                y_at_disturb[2 * m + trip_gen_idx] = 0.0
                if self.has_ren_droop:
                    y_at_disturb[3 * m + trip_gen_idx] = 0.0
            elif event_type == "partial_generation_loss":
                y_at_disturb[2 * m + trip_gen_idx] *= rem
                if self.has_ren_droop:
                    y_at_disturb[3 * m + trip_gen_idx] *= rem

            t_eval_post = t_eval[~pre_eval_mask].copy()
            if len(t_eval_post) > 0:
                t_eval_post[0] = max(t_dist, t_eval_post[0])
                t_eval_post[-1] = min(t_end, t_eval_post[-1])
                sol_post = solve_ivp(
                    fun=lambda t, y: self._rhs_segment(
                        y,
                        Y_red_seg=Y_red_post,
                        H_seg=H_post,
                        D_seg=D_post,
                        Pm0_seg=Pm0_post,
                        inv_R_seg=inv_R_post,
                        inv_R_ren_seg=inv_R_ren_post,
                        disconnected_idx=disconnected_idx,
                        surplus_pu=surplus_pu,
                        surplus_idx=surplus_idx,
                    ),
                    t_span=(t_dist, t_end),
                    y0=y_at_disturb,
                    t_eval=t_eval_post,
                    method=method,
                    rtol=rtol,
                    atol=atol,
                )
                t_traj = np.concatenate([sol_pre.t[:-1], sol_post.t])
                y_traj = np.hstack([sol_pre.y[:, :-1], sol_post.y])
                post_mask = np.concatenate([
                    np.zeros(len(sol_pre.t) - 1, dtype=bool),
                    np.ones(len(sol_post.t), dtype=bool),
                ])
            else:
                t_traj = sol_pre.t[:-1]
                y_traj = sol_pre.y[:, :-1]
                post_mask = np.zeros(len(t_traj), dtype=bool)

        delta_traj = y_traj[0:m, :].T
        domega_traj = y_traj[m : 2 * m, :].T
        Pm_traj = y_traj[2 * m : 3 * m, :].T

        Pe_traj = np.zeros_like(Pm_traj)
        pre_mask = ~post_mask
        if np.any(pre_mask):
            E_pre = self.E_mag0[None, :] * np.exp(1j * delta_traj[pre_mask])
            Pe_traj[pre_mask] = np.real(E_pre * np.conj(E_pre @ self.Y_red.T))
        if np.any(post_mask):
            E_post = self.E_mag0[None, :] * np.exp(1j * delta_traj[post_mask])
            if disconnected_idx >= 0:
                E_post[:, disconnected_idx] = 0.0 + 0.0j
            Pe_traj[post_mask] = np.real(E_post * np.conj(E_post @ Y_red_post.T))
            if disconnected_idx >= 0:
                Pe_traj[post_mask, disconnected_idx] = 0.0

        V_mag, V_angle_deg, bus_freq, bus_rocof = self.reconstruct_bus_states(
            delta_traj,
            domega_traj,
            dt=dt,
            K_pre=self.K,
            K_post=K_post,
            post_mask=post_mask if np.any(post_mask) else None,
        )

        f_coi, rocof_coi, area_coi = self.compute_coi_metrics(
            domega_traj,
            bus_freq,
            dt=dt,
            H_post=H_post,
            post_mask=post_mask if np.any(post_mask) else None,
        )

        eval_post_mask = t_traj >= (t_dist - 1e-12)
        if np.any(eval_post_mask):
            post_coi = f_coi[eval_post_mask]
            post_bus = bus_freq[eval_post_mask, :]
            post_t = t_traj[eval_post_mask]

            coi_min_idx = int(np.argmin(post_coi))
            ground_truth_nadir = float(post_coi[coi_min_idx])
            ground_truth_t_nadir = float(post_t[coi_min_idx])

            bus_t_idx, bus_col_idx = np.unravel_index(int(np.argmin(post_bus)), post_bus.shape)
            bus_min_nadir = float(post_bus[bus_t_idx, bus_col_idx])
            bus_min_t_nadir = float(post_t[bus_t_idx])
            nadir_bus_id = int(bus_col_idx + 1)
        else:
            coi_min_idx = int(np.argmin(f_coi))
            ground_truth_nadir = float(f_coi[coi_min_idx])
            ground_truth_t_nadir = float(t_traj[coi_min_idx])

            bus_t_idx, bus_col_idx = np.unravel_index(int(np.argmin(bus_freq)), bus_freq.shape)
            bus_min_nadir = float(bus_freq[bus_t_idx, bus_col_idx])
            bus_min_t_nadir = float(t_traj[bus_t_idx])
            nadir_bus_id = int(bus_col_idx + 1)

        if event_type == "no_disturbance" and abs(ground_truth_nadir - self.f_nominal) < 1e-9:
            ground_truth_nadir = float(self.f_nominal)
            bus_min_nadir = float(self.f_nominal)

        is_secure = int(ground_truth_nadir >= self.security_threshold_hz)
        active_loss_val = float(actual_loss_mw) if tripped_bus is not None else 0.0

        metadata: Dict[str, Any] = {
            "disturbed_gen": int(tripped_bus) if tripped_bus is not None else 0,
            "event_type": event_type,
            "loss_fraction": float(lf_actual),
            "active_loss_mw": active_loss_val,
            "active_loss_pu": active_loss_val / BASE_MVA,
            "renewable_penetration": float(self.renewable_config.p_ren),
            "p_ren_requested": float(self.renewable_summary["p_ren_requested"]),
            "p_ren_realized": float(self.renewable_summary["p_ren_realized"]),
            "load_scale": float(self.load_scaling),
            "macro_intertie": bool(self.macro_intertie),
            "power_flow_converged": bool(self.pf_diagnostics["converged"]),
            "power_flow_max_mismatch_pu": float(self.pf_diagnostics["max_mismatch_pu"]),
            "pre_fault_total_gen_mw": float(self.pf_diagnostics["total_gen_mw"]),
            "pre_fault_total_load_mw": float(self.pf_diagnostics["total_load_mw"]),
            "t_disturb": t_dist,
            "t_sim": t_end - t0_start,
            "fps": int(fps),
            "f_nominal": float(self.f_nominal),
            "f_threshold_hz": float(self.security_threshold_hz),
            "effective_H_sum": effective_H_pre,
            "effective_H_pre": effective_H_pre,
            "effective_H_post": effective_H_post,
            "ground_truth_nadir": ground_truth_nadir,
            "ground_truth_t_nadir": ground_truth_t_nadir,
            "bus_min_nadir": bus_min_nadir,
            "bus_min_t_nadir": bus_min_t_nadir,
            "nadir_bus_id": nadir_bus_id,
            "is_secure": is_secure,
        }

        return {
            "timestamps": t_traj,
            "bus_frequencies": bus_freq,
            "bus_rocof": bus_rocof,
            "voltage_magnitudes": V_mag,
            "voltage_angles": V_angle_deg,
            "coi_frequency": f_coi,
            "coi_rocof": rocof_coi,
            "area_frequencies": area_coi,
            "generator_deltas": delta_traj,
            "generator_speeds": domega_traj,
            "generator_powers": Pm_traj,
            "generator_electrical_powers": Pe_traj,
            "metadata": metadata,
        }
