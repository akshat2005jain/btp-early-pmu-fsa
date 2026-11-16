"""Kron-reduced swing and governor ODE simulator for the IEEE 39-bus system."""

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
    get_ieee39_bus_data,
    get_ieee39_branch_data,
    get_ieee39_gen_data,
)
from pmu_fsa.simulation.renewables import RenewablePenetrationConfig, apply_renewable_displacement


def build_ybus(branches: List[Dict[str, Any]], n_buses: int = N_BUSES) -> np.ndarray:
    """Build the nodal admittance matrix Y_bus (1-based branch bus IDs -> 0-based indices)."""
    y_bus = np.zeros((n_buses, n_buses), dtype=complex)
    for br in branches:
        from_idx = br["from"] - 1
        to_idx = br["to"] - 1
        r = br["r"]
        x = br["x"]
        b = br.get("b", 0.0)
        tap = br.get("tap", 1.0)
        if tap <= 0.0:
            tap = 1.0

        z_s = r + 1j * x
        if abs(z_s) < 1e-12:
            continue
        y_s = 1.0 / z_s
        y_c = 1j * (b / 2.0)

        # Off-nominal tap ratio is on the 'from' side
        y_bus[from_idx, from_idx] += (y_s + y_c) / (tap ** 2)
        y_bus[to_idx, to_idx] += (y_s + y_c)
        y_bus[from_idx, to_idx] -= y_s / tap
        y_bus[to_idx, from_idx] -= y_s / tap

    return y_bus


def compute_kron_reduction(
    load_scaling: float = 1.0,
    gen_data: Optional[List[Dict[str, Any]]] = None,
    branches: Optional[List[Dict[str, Any]]] = None,
    loads: Optional[Dict[int, Tuple[float, float]]] = None,
    return_all: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute Kron-reduced admittance Y_red (and K, E_mag0, delta0, Pm0 when return_all=True)."""
    if gen_data is None:
        gen_data = get_ieee39_gen_data()
    if branches is None:
        branches = get_ieee39_branch_data()
    if loads is None:
        loads = dict(LOAD_DATA)

    m = len(gen_data)
    N = N_BUSES

    # 1-based bus IDs -> 0-based indices
    gen_bus_indices = [g["bus"] - 1 for g in gen_data]

    V0 = np.ones(N, dtype=complex)
    for g in gen_data:
        b_idx = g["bus"] - 1
        V0[b_idx] = g["V_pu"] + 0.0j

    # Constant-impedance shunt load equivalent: y_L = (P_L - j*Q_L) / |V_0|^2
    y_load = np.zeros(N, dtype=complex)
    for bus_id, (p_mw, q_mvar) in loads.items():
        b_idx = bus_id - 1
        v_sq = abs(V0[b_idx]) ** 2 if abs(V0[b_idx]) > 0.1 else 1.0
        p_pu = (p_mw * load_scaling) / BASE_MVA
        q_pu = (q_mvar * load_scaling) / BASE_MVA
        y_load[b_idx] = (p_pu - 1j * q_pu) / v_sq

    xdp = np.array([g["Xdp"] for g in gen_data], dtype=float)
    y_g = 1.0 / (1j * xdp)

    Y_bus = build_ybus(branches, N)

    Y_NN = Y_bus + np.diag(y_load)
    for i, b_idx in enumerate(gen_bus_indices):
        Y_NN[b_idx, b_idx] += y_g[i]

    Y_GN = np.zeros((m, N), dtype=complex)
    for i, b_idx in enumerate(gen_bus_indices):
        Y_GN[i, b_idx] = -y_g[i]

    Y_NG = Y_GN.T
    Y_GG = np.diag(y_g)

    # Schur complement over network buses: Y_red = Y_GG - Y_GN * Y_NN^-1 * Y_NG, V_bus = K * E'
    inv_Y_NN = np.linalg.inv(Y_NN)
    K = -inv_Y_NN @ Y_NG
    Y_red = Y_GG - Y_GN @ inv_Y_NN @ Y_NG

    # Internal EMF behind X'd from pre-fault terminal injection: E' = V_t + j*X'd*I_g
    E_mag0 = np.zeros(m, dtype=float)
    delta0 = np.zeros(m, dtype=float)

    for i, g in enumerate(gen_data):
        b_idx = gen_bus_indices[i]
        p_g0 = g["P_mw"] / BASE_MVA
        q_g0 = g["Q_mvar"] / BASE_MVA
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
    """10-machine Kron-reduced dynamic simulator with first-order droop governors."""

    def __init__(
        self,
        load_scaling: float = 1.0,
        renewable_config: Optional[RenewablePenetrationConfig] = None,
        macro_intertie: bool = True,
        f_nominal: float = NOMINAL_FREQ,
        security_threshold_hz: float = SECURITY_THRESHOLD_HZ,
    ):
        self.load_scaling = load_scaling
        self.renewable_config = renewable_config or RenewablePenetrationConfig(p_ren=0.0)
        self.macro_intertie = macro_intertie
        self.f_nominal = float(f_nominal)
        self.omega_0 = 2.0 * math.pi * self.f_nominal
        self.security_threshold_hz = float(security_threshold_hz)

        base_gens = get_ieee39_gen_data()
        if not macro_intertie:
            for g in base_gens:
                if g["bus"] == 39:
                    g["H"] = 42.0

        self.gen_data = apply_renewable_displacement(base_gens, self.renewable_config)
        self.branches = get_ieee39_branch_data()

        (
            self.Y_red,
            self.K,
            self.E_mag0,
            self.delta0,
            self.Pm0,
        ) = compute_kron_reduction(
            load_scaling=self.load_scaling,
            gen_data=self.gen_data,
            branches=self.branches,
            return_all=True,
        )

        self.H = np.array([g["H"] for g in self.gen_data], dtype=float)
        self.D = np.array([g["D"] for g in self.gen_data], dtype=float)
        self.R = np.array([g["R"] for g in self.gen_data], dtype=float)
        self.Tg = np.array([g["Tg"] for g in self.gen_data], dtype=float)
        self.gen_buses = np.array([g["bus"] for g in self.gen_data], dtype=int)
        self.n_gen = len(self.gen_data)
        self.n_bus = N_BUSES
        self.H_tot = float(np.sum(self.H))

    def compute_electrical_power(self, delta: np.ndarray, E_mag: np.ndarray) -> np.ndarray:
        """Return electrical air-gap power Pe = Re(E' * conj(Y_red * E')) for each machine."""
        E_vec = E_mag * np.exp(1j * delta)
        I_G = self.Y_red @ E_vec
        return np.real(E_vec * np.conj(I_G))

    def rhs(
        self,
        t: float,
        state: np.ndarray,
        t_disturb: float,
        p_loss_mw: float,
        trip_gen_idx: int,
    ) -> np.ndarray:
        """Evaluate ODE derivatives for 3rd-order state [delta (m), domega (m), Pm (m)]."""
        m = self.n_gen
        delta = state[0:m]
        domega = state[m : 2 * m]
        Pm = state[2 * m : 3 * m]

        Pe = self.compute_electrical_power(delta, self.E_mag0)

        p_accel = Pm - Pe - self.D * domega
        if t >= t_disturb and trip_gen_idx >= 0:
            p_loss_pu = p_loss_mw / BASE_MVA
            p_accel[trip_gen_idx] -= p_loss_pu

        # 3rd-order classical swing + simplified TGOV1 droop governor
        d_delta = self.omega_0 * domega

        if np.any(self.H <= 0.0):
            raise ValueError("Generator inertia H must be strictly positive (H > 0)")
        H_safe = np.maximum(self.H, 1e-4)
        d_domega = p_accel / (2.0 * H_safe)

        d_Pm = (self.Pm0 - (domega / self.R) - Pm) / self.Tg

        return np.concatenate([d_delta, d_domega, d_Pm])

    def reconstruct_bus_states(
        self,
        delta_traj: np.ndarray,
        domega_traj: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Reconstruct 39-bus voltage magnitudes, angles (deg), frequencies (Hz), and unscaled ROCOF."""
        n_steps = delta_traj.shape[0]
        m = self.n_gen
        N = self.n_bus

        V_mag = np.zeros((n_steps, N), dtype=float)
        V_angle_deg = np.zeros((n_steps, N), dtype=float)
        bus_freq = np.zeros((n_steps, N), dtype=float)

        for step in range(n_steps):
            delta = delta_traj[step]
            domega = domega_traj[step]

            E_prime = self.E_mag0 * np.exp(1j * delta)
            V_bus = self.K @ E_prime
            v_abs = np.abs(V_bus)
            v_abs_safe = np.where(v_abs < 1e-4, 1e-4, v_abs)

            V_mag[step] = v_abs
            V_angle_deg[step] = np.degrees(np.angle(V_bus))

            # Analytical bus frequency: f_k = f_0 + Im((dV_k/dt) / V_k) / (2*pi), dE'/dt = j*omega_0*domega*E'
            dE_prime = 1j * self.omega_0 * domega * E_prime
            dV_bus = self.K @ dE_prime

            v_abs_bus = np.abs(V_bus)
            v_denom_safe = np.maximum(v_abs_bus, 1e-6)
            V_bus_safe = np.where(v_abs_bus < 1e-6, v_denom_safe * (1.0 + 0.0j), V_bus)
            omega_dev_rad_s = np.imag(dV_bus / V_bus_safe)
            bus_freq[step] = self.f_nominal + (omega_dev_rad_s / (2.0 * math.pi))

        bus_rocof = np.zeros_like(bus_freq)
        if n_steps > 1:
            bus_rocof = np.gradient(bus_freq, axis=0)

        return V_mag, V_angle_deg, bus_freq, bus_rocof

    def compute_coi_metrics(
        self,
        domega_traj: np.ndarray,
        bus_freq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
        """Compute global COI frequency, unscaled COI ROCOF, and per-area COI frequencies."""
        n_steps = domega_traj.shape[0]
        H_weights = self.H / self.H_tot

        machine_f = self.f_nominal * (1.0 + domega_traj)
        f_coi = np.sum(machine_f * H_weights[None, :], axis=1)

        rocof_coi = np.zeros(n_steps, dtype=float)
        if n_steps > 1:
            rocof_coi = np.gradient(f_coi)

        area_coi = {}
        for area_id in [1, 2, 3]:
            area_bus_set = set(AREA_PARTITION[area_id])
            gen_indices_in_area = [
                i for i, b in enumerate(self.gen_buses) if b in area_bus_set
            ]
            if gen_indices_in_area:
                h_area = self.H[gen_indices_in_area]
                h_area_tot = np.sum(h_area)
                weights = h_area / h_area_tot
                area_coi[area_id] = np.sum(
                    machine_f[:, gen_indices_in_area] * weights[None, :], axis=1
                )
            else:
                area_coi[area_id] = f_coi.copy()

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
    ) -> Dict[str, Any]:
        """Integrate post-contingency trajectory over t_span and return bus and machine channels."""
        m = self.n_gen
        dt = 1.0 / fps
        t_eval = np.arange(t_span[0], t_span[1] + 0.5 * dt, dt)

        y0 = np.concatenate([self.delta0, np.zeros(m), self.Pm0])

        if np.any(self.H <= 0.0):
            raise ValueError("Generator inertia H must be strictly positive (H > 0)")

        trip_gen_idx = -1
        if tripped_bus is not None:
            matches = [i for i, b in enumerate(self.gen_buses) if b == tripped_bus]
            if matches:
                trip_gen_idx = matches[0]
            else:
                warnings.warn(
                    f"Contingency bus {tripped_bus} is not a valid generator bus in {self.gen_buses}. "
                    "No generator trip disturbance will occur.",
                    UserWarning,
                    stacklevel=2,
                )

        sol = solve_ivp(
            fun=lambda t, y: self.rhs(t, y, t_disturb, p_loss_mw, trip_gen_idx),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        t_steps = len(sol.t)
        delta_traj = sol.y[0:m, :].T
        domega_traj = sol.y[m : 2 * m, :].T
        Pm_traj = sol.y[2 * m : 3 * m, :].T

        V_mag, V_angle_deg, bus_freq, bus_rocof = self.reconstruct_bus_states(
            delta_traj, domega_traj
        )

        f_coi, rocof_coi, area_coi = self.compute_coi_metrics(domega_traj, bus_freq)

        # Scale finite-difference ROCOF by sample interval dt
        if t_steps > 1:
            bus_rocof = np.gradient(bus_freq, dt, axis=0)
            rocof_coi = np.gradient(f_coi, dt)

        post_disturb_mask = sol.t >= t_disturb
        if np.any(post_disturb_mask):
            post_f = bus_freq[post_disturb_mask, :]
            post_t = sol.t[post_disturb_mask]
            ground_truth_nadir = float(np.min(post_f))
            nadir_indices = np.unravel_index(np.argmin(post_f), post_f.shape)
            ground_truth_t_nadir = float(post_t[nadir_indices[0]])
        else:
            ground_truth_nadir = float(np.min(bus_freq))
            ground_truth_t_nadir = float(sol.t[np.argmin(bus_freq)])

        is_secure = 1 if ground_truth_nadir >= self.security_threshold_hz else 0

        metadata = {
            "disturbed_gen": tripped_bus if tripped_bus is not None else 0,
            "active_loss_mw": p_loss_mw if tripped_bus is not None else 0.0,
            "renewable_penetration": self.renewable_config.p_ren,
            "load_scale": self.load_scaling,
            "t_disturb": t_disturb,
            "t_sim": t_span[1] - t_span[0],
            "fps": fps,
            "f_nominal": self.f_nominal,
            "f_threshold_hz": self.security_threshold_hz,
            "effective_H_sum": self.H_tot,
            "ground_truth_nadir": ground_truth_nadir,
            "ground_truth_t_nadir": ground_truth_t_nadir,
            "is_secure": is_secure,
        }

        return {
            "timestamps": sol.t,
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
            "metadata": metadata,
        }
