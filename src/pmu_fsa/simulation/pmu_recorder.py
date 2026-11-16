"""IEEE C37.118 synchrophasor resampling, noise injection, and binary frame serialization."""

import struct
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np


def compute_crc_ccitt(data: bytes) -> int:
    """Compute CRC-CCITT checksum (poly 0x1021, init 0xFFFF) over input bytes."""
    crc = 0xFFFF
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass
class PMUNoiseConfig:
    """Gaussian noise standard deviations for synthetic PMU channels."""
    sigma_f: float = 0.01  # Hz
    sigma_rocof: float = 0.02  # Hz/s
    sigma_v: float = 0.0005  # pu
    sigma_theta_deg: float = 0.05  # deg
    seed: Optional[int] = 42


def apply_pmu_measurement_noise(
    bus_frequencies: np.ndarray,
    bus_rocof: np.ndarray,
    voltage_magnitudes: np.ndarray,
    voltage_angles: np.ndarray,
    noise_config: Optional[PMUNoiseConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add zero-mean Gaussian noise to bus frequency, ROCOF, voltage magnitude, and angle arrays."""
    if noise_config is None:
        noise_config = PMUNoiseConfig()

    rng = np.random.default_rng(noise_config.seed)

    f_noisy = bus_frequencies + rng.normal(0.0, noise_config.sigma_f, size=bus_frequencies.shape)
    rocof_noisy = bus_rocof + rng.normal(0.0, noise_config.sigma_rocof, size=bus_rocof.shape)
    v_noisy = voltage_magnitudes + rng.normal(0.0, noise_config.sigma_v, size=voltage_magnitudes.shape)
    v_noisy = np.maximum(0.01, v_noisy)

    theta_noisy = voltage_angles + rng.normal(0.0, noise_config.sigma_theta_deg, size=voltage_angles.shape)

    return f_noisy, rocof_noisy, v_noisy, theta_noisy


def compute_tve(
    v_mag_true: np.ndarray,
    v_ang_deg_true: np.ndarray,
    v_mag_meas: np.ndarray,
    v_ang_deg_meas: np.ndarray,
) -> np.ndarray:
    """Compute IEEE C37.118.1 Total Vector Error |V_meas - V_true| / |V_true| as a fraction."""
    ang_rad_true = np.radians(v_ang_deg_true)
    ang_rad_meas = np.radians(v_ang_deg_meas)

    v_true_complex = v_mag_true * np.exp(1j * ang_rad_true)
    v_meas_complex = v_mag_meas * np.exp(1j * ang_rad_meas)

    diff = np.abs(v_meas_complex - v_true_complex)
    mag_true = np.maximum(1e-6, np.abs(v_true_complex))
    return diff / mag_true


def compute_fir_rocof(
    frequencies: np.ndarray,
    fps: float = 50.0,
    window_length: int = 5,
) -> np.ndarray:
    """Estimate ROCOF via a 5-point Savitzky-Golay FIR differentiator (or np.gradient for short windows)."""
    dt = 1.0 / fps
    n_samples = frequencies.shape[0]
    rocof = np.zeros_like(frequencies)

    if n_samples <= 1:
        return rocof
    elif window_length >= 5 and n_samples >= 5:
        # 5-point OLS differentiator weights: [-2, -1, 0, 1, 2] / (10 * dt)
        c = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) / (10.0 * dt)
        if frequencies.ndim == 1:
            for n in range(2, n_samples - 2):
                rocof[n] = np.sum(c * frequencies[n - 2 : n + 3])
            rocof[0] = (frequencies[1] - frequencies[0]) / dt
            rocof[1] = (frequencies[2] - frequencies[0]) / (2.0 * dt)
            rocof[-2] = (frequencies[-1] - frequencies[-3]) / (2.0 * dt)
            rocof[-1] = (frequencies[-1] - frequencies[-2]) / dt
        else:
            for n in range(2, n_samples - 2):
                rocof[n, :] = np.tensordot(c, frequencies[n - 2 : n + 3, :], axes=(0, 0))
            rocof[0, :] = (frequencies[1, :] - frequencies[0, :]) / dt
            rocof[1, :] = (frequencies[2, :] - frequencies[0, :]) / (2.0 * dt)
            rocof[-2, :] = (frequencies[-1, :] - frequencies[-3, :]) / (2.0 * dt)
            rocof[-1, :] = (frequencies[-1, :] - frequencies[-2, :]) / dt
    else:
        rocof = np.gradient(frequencies, dt, axis=0)

    return rocof


class PMURecorder:
    """Resamples dynamic trajectories into IEEE C37.118 synchrophasor streams and binary frames."""

    def __init__(
        self,
        fps: int = 50,
        noise_config: Optional[PMUNoiseConfig] = None,
        base_epoch_seconds: int = 1700000000,
        id_code: int = 1,
        f_nominal: float = 50.0,
    ):
        SUPPORTED_FPS = (10, 25, 30, 50, 60, 100, 120)
        if fps not in SUPPORTED_FPS:
            raise ValueError(f"PMU reporting rate must be one of {SUPPORTED_FPS} frames/s, got {fps}")
        self.fps = fps
        self.dt = 1.0 / fps
        self.noise_config = noise_config
        self.base_epoch_seconds = base_epoch_seconds
        self.id_code = id_code
        self.f_nominal = float(f_nominal)

    def record_trajectory(
        self,
        raw_trajectory: Dict[str, Any],
        inject_noise: bool = False,
    ) -> Dict[str, Any]:
        """Resample trajectory onto uniform reporting timestamps and optionally add measurement noise."""
        orig_t = raw_trajectory["timestamps"]
        t_start = orig_t[0]
        t_end = orig_t[-1]

        uniform_timestamps = np.arange(t_start, t_end + 0.5 * self.dt, self.dt)
        n_samples = len(uniform_timestamps)
        n_buses = raw_trajectory["bus_frequencies"].shape[1]

        bus_freq = np.zeros((n_samples, n_buses))
        bus_rocof = np.zeros((n_samples, n_buses))
        v_mag = np.zeros((n_samples, n_buses))
        v_ang = np.zeros((n_samples, n_buses))

        for b in range(n_buses):
            bus_freq[:, b] = np.interp(uniform_timestamps, orig_t, raw_trajectory["bus_frequencies"][:, b])
            bus_rocof[:, b] = np.interp(uniform_timestamps, orig_t, raw_trajectory["bus_rocof"][:, b])
            v_mag[:, b] = np.interp(uniform_timestamps, orig_t, raw_trajectory["voltage_magnitudes"][:, b])
            v_ang[:, b] = np.interp(uniform_timestamps, orig_t, raw_trajectory["voltage_angles"][:, b])

        coi_freq = np.interp(uniform_timestamps, orig_t, raw_trajectory["coi_frequency"])

        if inject_noise and self.noise_config is not None:
            bus_freq, bus_rocof, v_mag, v_ang = apply_pmu_measurement_noise(
                bus_freq, bus_rocof, v_mag, v_ang, self.noise_config
            )
            # Recompute ROCOF from noisy frequency via 5-point FIR differentiator
            bus_rocof = compute_fir_rocof(bus_freq, fps=self.fps, window_length=5)

        time_base = 1000000
        soc_array = np.zeros(n_samples, dtype=np.uint32)
        fracsec_array = np.zeros(n_samples, dtype=np.uint32)

        for i, t in enumerate(uniform_timestamps):
            total_sec = self.base_epoch_seconds + t
            soc = int(np.floor(total_sec))
            fraction = float(total_sec - soc)
            frac_count = int(np.round(fraction * time_base))
            soc_array[i] = soc
            fracsec_array[i] = frac_count & 0x00FFFFFF  # lower 24 bits = fractional count; top 8 = time quality

        metadata = dict(raw_trajectory.get("metadata", {}))
        metadata["fps"] = self.fps
        metadata["is_noisy"] = inject_noise

        return {
            "timestamps": uniform_timestamps,
            "soc": soc_array,
            "fracsec": fracsec_array,
            "bus_frequencies": bus_freq,
            "bus_rocof": bus_rocof,
            "voltage_magnitudes": v_mag,
            "voltage_angles": v_ang,
            "coi_frequency": coi_freq,
            "metadata": metadata,
        }

    def serialize_frame(
        self,
        frame_idx: int,
        pmu_stream: Dict[str, Any],
        id_code: Optional[int] = None,
    ) -> bytes:
        """Pack a single timestamp into an IEEE C37.118.2 binary data frame."""
        pmu_id = id_code if id_code is not None else self.id_code
        soc = int(pmu_stream["soc"][frame_idx])
        fracsec = int(pmu_stream["fracsec"][frame_idx])
        stat = 0x0000

        freq_hz = float(pmu_stream["coi_frequency"][frame_idx])
        f_nom = float(pmu_stream.get("metadata", {}).get("f_nominal", self.f_nominal))
        freq_dev = freq_hz - f_nom
        dt = pmu_stream["timestamps"][1] - pmu_stream["timestamps"][0] if len(pmu_stream["timestamps"]) > 1 else 1.0 / self.fps
        dfreq = float(pmu_stream["bus_rocof"][frame_idx, 0])

        # C37.118.2 data frame header: SYNC=0xAA01, FRAMESIZE (patched below), IDCODE, SOC, FRACSEC, STAT
        header = struct.pack(">HHHIIH", 0xAA01, 0, pmu_id, soc, fracsec, stat)

        phasor_bytes = bytearray()
        n_buses = pmu_stream["voltage_magnitudes"].shape[1]
        for b in range(n_buses):
            v_m = float(pmu_stream["voltage_magnitudes"][frame_idx, b])
            v_a_rad = float(np.radians(pmu_stream["voltage_angles"][frame_idx, b]))
            phasor_bytes.extend(struct.pack(">ff", v_m, v_a_rad))

        freq_bytes = struct.pack(">ff", freq_dev, dfreq)

        frame_payload = bytearray(header) + phasor_bytes + freq_bytes
        total_frame_size = len(frame_payload) + 2

        struct.pack_into(">H", frame_payload, 2, total_frame_size)

        chk = compute_crc_ccitt(bytes(frame_payload))
        frame_bytes = bytes(frame_payload) + struct.pack(">H", chk)

        return frame_bytes

    @staticmethod
    def parse_frame(frame_bytes: bytes, f_nominal: float = 50.0) -> Dict[str, Any]:
        """Unpack and CRC-verify an IEEE C37.118.2 binary data frame."""
        if len(frame_bytes) < 18:
            raise ValueError(f"Frame length {len(frame_bytes)} too short for C37.118 header and CRC")

        payload = frame_bytes[:-2]
        if len(payload) < 24:
            raise ValueError("Corrupted or truncated PMU frame")

        try:
            received_chk = struct.unpack(">H", frame_bytes[-2:])[0]
            calculated_chk = compute_crc_ccitt(payload)
            if received_chk != calculated_chk:
                raise ValueError(f"CRC-CCITT mismatch: received 0x{received_chk:04X}, computed 0x{calculated_chk:04X}")

            sync, frame_size, id_code, soc, fracsec, stat = struct.unpack(">HHHIIH", payload[:16])
            if sync != 0xAA01:
                raise ValueError(f"Invalid SYNC word: 0x{sync:04X}, expected 0xAA01")

            offset = 16
            n_phasors = (len(payload) - 16 - 8) // 8
            phasors = []
            for _ in range(n_phasors):
                mag, ang = struct.unpack(">ff", payload[offset : offset + 8])
                phasors.append((mag, np.degrees(ang)))
                offset += 8

            if len(payload) < offset + 8:
                raise ValueError("Corrupted or truncated PMU frame")

            freq_dev, dfreq = struct.unpack(">ff", payload[offset : offset + 8])
            freq_hz = float(f_nominal) + freq_dev
        except struct.error as e:
            raise ValueError("Corrupted or truncated PMU frame") from e

        return {
            "sync": sync,
            "frame_size": frame_size,
            "id_code": id_code,
            "soc": soc,
            "fracsec": fracsec,
            "stat": stat,
            "n_phasors": n_phasors,
            "phasors": phasors,
            "freq_hz": freq_hz,
            "rocof": dfreq,
            "crc_valid": True,
        }
