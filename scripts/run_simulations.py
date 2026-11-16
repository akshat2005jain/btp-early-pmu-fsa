#!/usr/bin/env python3
"""Generate IEEE 39-bus contingency trajectory .npz files."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from pmu_fsa.simulation.scenario_generator import (
    ScenarioGenerator,
    export_scenario_to_npz,
)
from pmu_fsa.simulation.pmu_recorder import PMUNoiseConfig

logger = logging.getLogger("run_simulations")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="IEEE 39-Bus Contingency Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-scenarios", type=int, default=10, help="Number of scenarios to generate")
    parser.add_argument("--output-dir", type=str, default="data/raw/", help="Target directory for .npz scenarios")
    parser.add_argument("--fps", type=int, default=50, choices=[50, 60], help="PMU frame rate (fps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--t-disturb", type=float, default=1.0, help="Disturbance onset time (s)")
    parser.add_argument("--t-max", type=float, default=10.0, help="Total simulation duration (s)")
    parser.add_argument("--gen-buses", type=int, nargs="+", default=None, help="Generator buses to trip")
    parser.add_argument("--penetrations", type=float, nargs="+", default=None, help="Renewable penetrations")
    parser.add_argument("--load-scales", type=float, nargs="+", default=None, help="Load scaling factors")
    parser.add_argument("--loss-fractions", type=float, nargs="+", default=None, help="Generator loss fractions (e.g. 1.0 0.65 0.35)")
    parser.add_argument("--clean", action="store_true", help="Remove existing .npz files in output-dir before generating")
    parser.add_argument("--inject-noise", action="store_true", help="Inject PMU noise")
    parser.add_argument("--noise-std", type=float, default=0.0, help="PMU noise std dev (Hz)")
    parser.add_argument("--f-nominal", type=float, default=50.0, help="Nominal grid frequency (Hz)")
    parser.add_argument("--threshold", type=float, default=49.50, help="Frequency security threshold (Hz)")
    parser.add_argument("--no-macro-intertie", action="store_true", help="Use localized Bus 39 inertia (H=42s) for islanded multi-area dynamics")
    parser.add_argument("--fast", action="store_true", help="Use fast synthetic mode for quick unit tests")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    return parser.parse_args(args)


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir = Path(parsed.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if parsed.clean:
        for old_fp in out_dir.glob("*.npz"):
            old_fp.unlink()

    np.random.seed(parsed.seed)

    if parsed.fast:
        logger.info("Generating %d fast synthetic scenarios...", parsed.n_scenarios)
        dt = 1.0 / float(parsed.fps)
        n_samples = int(parsed.t_max * parsed.fps)
        timestamps = np.linspace(0.0, (n_samples - 1) * dt, n_samples)
        for i in range(parsed.n_scenarios):
            pen = 0.15 + (i * 0.05) % 0.40
            f_drop = 0.50 * (1.0 - np.exp(-np.maximum(0.0, timestamps - parsed.t_disturb) / 1.5))
            coi = parsed.f_nominal - f_drop
            bf = np.tile(coi[:, np.newaxis], (1, 39)) + np.random.normal(0, 0.01, size=(n_samples, 39))
            rocof = np.gradient(bf, dt, axis=0)
            scen_id = f"scen_fast_{i:03d}"
            nadir_val = float(np.min(coi))
            scen = {
                "timestamps": timestamps,
                "bus_frequencies": bf,
                "bus_rocof": rocof,
                "voltage_magnitudes": np.ones((n_samples, 39)),
                "voltage_angles": np.zeros((n_samples, 39)),
                "coi_frequency": coi,
                "metadata": {
                    "scenario_id": scen_id,
                    "t_disturb": parsed.t_disturb,
                    "t_sim": parsed.t_max,
                    "fps": parsed.fps,
                    "f_nominal": parsed.f_nominal,
                    "f_threshold_hz": parsed.threshold,
                    "renewable_penetration": pen,
                    "load_scale": 1.0,
                    "ground_truth_nadir": round(nadir_val, 4),
                    "ground_truth_t_nadir": 3.5,
                    "is_secure": 1 if nadir_val >= parsed.threshold else 0,
                    "partition": "ood_test" if pen >= 0.35 else "train",
                },
            }
            export_scenario_to_npz(scen, str(out_dir / f"{scen_id}.npz"))
        logger.info("Successfully generated %d synthetic scenarios in %s", parsed.n_scenarios, out_dir)
        return 0

    noise_cfg = None
    if parsed.inject_noise or parsed.noise_std > 0.0:
        noise_cfg = PMUNoiseConfig(f_noise_std=parsed.noise_std or 0.005)

    logger.info("Generating %d scenarios at %d fps (f0=%.1f Hz)...", parsed.n_scenarios, parsed.fps, parsed.f_nominal)
    generator = ScenarioGenerator(
        fps=parsed.fps,
        t_span=(0.0, parsed.t_max),
        t_disturb=parsed.t_disturb,
        security_threshold_hz=parsed.threshold,
        macro_intertie=(not parsed.no_macro_intertie),
        f_nominal=parsed.f_nominal,
    )

    scenarios = generator.generate_scenario_batch(
        gen_buses=parsed.gen_buses,
        load_scales=parsed.load_scales,
        penetrations=parsed.penetrations,
        loss_fractions=parsed.loss_fractions,
        max_scenarios=parsed.n_scenarios,
        inject_noise=(parsed.inject_noise or parsed.noise_std > 0.0),
    )

    for scen in scenarios:
        scen_id = scen["metadata"]["scenario_id"]
        filepath = out_dir / f"{scen_id}.npz"
        export_scenario_to_npz(scen, str(filepath))

    logger.info("Successfully generated and saved %d scenarios into %s", len(scenarios), out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
