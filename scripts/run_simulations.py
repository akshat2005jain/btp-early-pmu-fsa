#!/usr/bin/env python3
"""Generate IEEE 39-bus raw simulation and observable PMU trajectory .npz files."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from pmu_fsa.simulation.dataset_validator import (
    validate_dataset,
    write_dataset_quality_report,
)
from pmu_fsa.simulation.ieee39_data import BASE_MVA
from pmu_fsa.simulation.pmu_recorder import PMUNoiseConfig
from pmu_fsa.simulation.scenario_generator import (
    DEFAULT_LOAD_SCALES,
    DEFAULT_LOSS_FRACTIONS,
    DEFAULT_PENETRATIONS,
    ScenarioGenerator,
    assign_scenario_partition,
    export_raw_and_pmu_scenario,
    get_software_versions,
)

logger = logging.getLogger("run_simulations")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="IEEE 39-Bus Contingency Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n-scenarios", type=int, default=10, help="Number of scenarios to generate (750 for full grid)")
    parser.add_argument("--output-dir", type=str, default="data/raw/", help="Target directory for observable PMU .npz scenarios")
    parser.add_argument(
        "--raw-sim-dir",
        type=str,
        default=None,
        help="Target directory for internal ODE simulator .npz trajectories (defaults to data/raw_sim/ when output-dir is data/raw/)",
    )
    parser.add_argument("--fps", type=int, default=50, choices=[25, 50, 60], help="PMU reporting rate (fps)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--t-disturb", type=float, default=1.0, help="Disturbance onset time (s)")
    parser.add_argument("--t-max", type=float, default=10.0, help="Total simulation duration (s)")
    parser.add_argument("--gen-buses", type=int, nargs="+", default=None, help="Generator buses to trip")
    parser.add_argument(
        "--penetrations",
        type=float,
        nargs="+",
        default=list(DEFAULT_PENETRATIONS),
        help="Renewable penetration levels",
    )
    parser.add_argument(
        "--load-scales",
        type=float,
        nargs="+",
        default=list(DEFAULT_LOAD_SCALES),
        help="System load scaling factors",
    )
    parser.add_argument(
        "--loss-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_LOSS_FRACTIONS),
        help="Generator outage fractions (e.g. 1.0 0.70 0.40)",
    )
    parser.add_argument("--observed-buses", type=int, nargs="+", default=None, help="Optional 1-based observed PMU bus subset")
    parser.add_argument("--clean", action="store_true", help="Remove existing .npz files in target directories before generating")
    parser.add_argument("--inject-noise", action="store_true", help="Inject synthetic PMU measurement noise")
    parser.add_argument("--noise-std", type=float, default=0.0, help="PMU frequency noise std dev sigma_f (Hz)")
    parser.add_argument("--f-nominal", type=float, default=50.0, help="Nominal grid frequency (Hz)")
    parser.add_argument("--threshold", type=float, default=49.50, help="Frequency security threshold (Hz)")
    parser.add_argument("--macro-intertie", action="store_true", help="Opt in to 500s macro-grid intertie inertia on Bus 39")
    parser.add_argument("--no-macro-intertie", action="store_true", help="Use regional Bus 39 inertia H=42s (default behavior)")
    parser.add_argument("--validate", action="store_true", help="Run Phase-5 dataset validation and write manifest + quality report")
    parser.add_argument("--processed-dir", type=str, default=None, help="Optional processed feature directory for validation")
    parser.add_argument("--manifest-path", type=str, default=None, help="Override output path for manifest.json")
    parser.add_argument("--report-path", type=str, default=None, help="Override output path for dataset_quality_report.md")
    parser.add_argument("--fast", action="store_true", help="Use fast synthetic trajectories for CLI smoke tests")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet logging mode")
    return parser.parse_args(args)


def _resolve_sim_and_report_paths(
    output_dir_arg: str,
    raw_sim_arg: Optional[str],
    manifest_arg: Optional[str],
    report_arg: Optional[str],
):
    out_dir = Path(output_dir_arg)
    norm_out = str(output_dir_arg).rstrip("/")
    is_default_data = norm_out in ("data/raw", str(PROJECT_ROOT / "data" / "raw"))

    if raw_sim_arg is not None:
        raw_sim_dir = Path(raw_sim_arg)
    elif is_default_data:
        raw_sim_dir = out_dir.parent / "raw_sim"
    else:
        raw_sim_dir = out_dir.parent / f"{out_dir.name}_sim"

    if manifest_arg is not None:
        manifest_path = Path(manifest_arg)
    elif is_default_data:
        manifest_path = Path("data/manifest.json")
    else:
        manifest_path = out_dir.parent / "manifest.json"

    if report_arg is not None:
        report_path = Path(report_arg)
    elif is_default_data:
        report_path = Path("results/dataset_quality_report.md")
    else:
        report_path = out_dir.parent / "dataset_quality_report.md"

    return out_dir, raw_sim_dir, manifest_path, report_path


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    out_dir, raw_sim_dir, manifest_path, report_path = _resolve_sim_and_report_paths(
        parsed.output_dir,
        parsed.raw_sim_dir,
        parsed.manifest_path,
        parsed.report_path,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_sim_dir.mkdir(parents=True, exist_ok=True)

    if parsed.clean:
        for d in (out_dir, raw_sim_dir):
            for old_fp in d.glob("*.npz"):
                old_fp.unlink()

    rng = np.random.default_rng(parsed.seed)
    np.random.seed(parsed.seed)
    use_macro_intertie = bool(parsed.macro_intertie and not parsed.no_macro_intertie)
    inject_noise = bool(parsed.inject_noise or parsed.noise_std > 0.0)
    sigma_f = float(parsed.noise_std) if parsed.noise_std > 0.0 else 0.005

    if parsed.fast:
        logger.info("Generating %d fast synthetic scenarios...", parsed.n_scenarios)
        dt = 1.0 / float(parsed.fps)
        timestamps = np.arange(0.0, parsed.t_max + 0.5 * dt, dt)
        n_samples = len(timestamps)
        sw_versions = get_software_versions()

        for i in range(parsed.n_scenarios):
            pen = [0.10, 0.20, 0.30, 0.40, 0.50][i % 5]
            ld = [0.90, 1.00, 1.10, 0.80, 1.20][(i // 2) % 5]
            lf = [0.40, 1.00, 0.70][i % 3]
            gen_bus = 30 + (i % 10)
            loss_mw = 600.0 * lf
            scen_id = f"scen_fast_{i:03d}"
            part, ood_type = assign_scenario_partition(pen, ld, scen_id, return_ood_type=True)

            t_post = np.maximum(0.0, timestamps - parsed.t_disturb)
            amp = 0.25 + 0.55 * lf + 0.35 * pen
            f_drop = amp * (t_post / 1.8) * np.exp(1.0 - t_post / 1.8) * (timestamps >= parsed.t_disturb)
            coi = parsed.f_nominal - f_drop
            bf_true = np.tile(coi[:, np.newaxis], (1, 39))
            bf_meas = bf_true + (rng.normal(0.0, sigma_f, size=(n_samples, 39)) if inject_noise else 0.0)
            rocof = np.gradient(bf_meas, dt, axis=0)
            pmu_sys = np.mean(bf_meas, axis=1)

            post_mask = timestamps >= parsed.t_disturb
            coi_post = coi[post_mask]
            t_post_arr = timestamps[post_mask]
            nadir_idx = int(np.argmin(coi_post))
            nadir_val = float(coi_post[nadir_idx])
            t_nadir_val = float(t_post_arr[nadir_idx])

            meta = {
                "seed": int(parsed.seed + i),
                "scenario_id": scen_id,
                "partition": part,
                "ood_type": ood_type,
                "disturbed_gen": gen_bus,
                "loss_fraction": lf,
                "active_loss_mw": loss_mw,
                "active_loss_pu": loss_mw / BASE_MVA,
                "renewable_penetration": pen,
                "p_ren_requested": pen,
                "p_ren_realized": pen,
                "load_scale": ld,
                "t_disturb": parsed.t_disturb,
                "t_sim": parsed.t_max,
                "ground_truth_nadir": round(nadir_val, 6),
                "ground_truth_t_nadir": round(t_nadir_val, 4),
                "bus_min_nadir": round(float(np.min(bf_true[post_mask])), 6),
                "bus_min_t_nadir": round(t_nadir_val, 4),
                "is_secure": int(nadir_val >= parsed.threshold),
                "fps": parsed.fps,
                "f_nominal": parsed.f_nominal,
                "f_threshold_hz": parsed.threshold,
                "macro_intertie": use_macro_intertie,
                "is_noisy": inject_noise,
                "noise_config": {"inject_noise": inject_noise, "sigma_f": sigma_f if inject_noise else 0.0},
                "software_versions": sw_versions,
            }
            raw_traj = {
                "timestamps": timestamps,
                "generator_deltas": np.zeros((n_samples, 10)),
                "generator_speeds": np.zeros((n_samples, 10)),
                "generator_powers": np.zeros((n_samples, 10)),
                "coi_frequency": coi,
                "coi_rocof": np.gradient(coi, dt),
                "bus_frequencies": bf_true,
                "voltage_magnitudes": np.ones((n_samples, 39)),
                "voltage_angles": np.zeros((n_samples, 39)),
                "metadata": dict(meta),
            }
            pmu_traj = {
                "timestamps": timestamps,
                "bus_frequencies": bf_meas,
                "bus_rocof": rocof,
                "voltage_magnitudes": np.ones((n_samples, 39)),
                "voltage_angles": np.zeros((n_samples, 39)),
                "pmu_sys_frequency": pmu_sys,
                "metadata": dict(meta),
            }
            export_raw_and_pmu_scenario(raw_traj, pmu_traj, raw_sim_dir, out_dir)

        logger.info("Generated %d synthetic scenarios in %s (raw_sim: %s)", parsed.n_scenarios, out_dir, raw_sim_dir)
    else:
        noise_cfg = PMUNoiseConfig(sigma_f=sigma_f, seed=parsed.seed) if inject_noise else None

        logger.info(
            "Generating %d scenarios at %d fps (f0=%.1f Hz, macro_intertie=%s, inject_noise=%s)...",
            parsed.n_scenarios,
            parsed.fps,
            parsed.f_nominal,
            use_macro_intertie,
            inject_noise,
        )
        generator = ScenarioGenerator(
            fps=parsed.fps,
            t_span=(0.0, parsed.t_max),
            t_disturb=parsed.t_disturb,
            security_threshold_hz=parsed.threshold,
            macro_intertie=use_macro_intertie,
            f_nominal=parsed.f_nominal,
            noise_config=noise_cfg,
            seed=parsed.seed,
        )

        pairs = generator.generate_scenario_batch(
            gen_buses=parsed.gen_buses,
            load_scales=parsed.load_scales,
            penetrations=parsed.penetrations,
            loss_fractions=parsed.loss_fractions,
            max_scenarios=parsed.n_scenarios,
            inject_noise=inject_noise,
            noise_config=noise_cfg,
            seed=parsed.seed,
            observed_buses=parsed.observed_buses,
            return_raw=True,
        )

        for raw_traj, pmu_traj in pairs:
            export_raw_and_pmu_scenario(raw_traj, pmu_traj, raw_sim_dir, out_dir)

        logger.info(
            "Saved %d PMU scenarios to %s and %d raw ODE trajectories to %s",
            len(pairs),
            out_dir,
            len(pairs),
            raw_sim_dir,
        )

    if parsed.validate:
        logger.info("Running Phase-5 dataset validation...")
        val_report = validate_dataset(
            raw_pmu_dir=out_dir,
            processed_dir=parsed.processed_dir,
            f_nominal=parsed.f_nominal,
            threshold=parsed.threshold,
            raw_sim_dir=raw_sim_dir,
            expected_n_scenarios=parsed.n_scenarios,
        )
        md_written, man_written = write_dataset_quality_report(
            val_report,
            md_path=report_path,
            manifest_path=manifest_path,
        )
        logger.info(
            "Validation complete (all_passed=%s). Manifest: %s | Report: %s",
            val_report["all_passed"],
            man_written,
            md_written,
        )
        if not val_report["all_passed"]:
            failed = [k for k, v in val_report["checks"].items() if not v.get("passed")]
            logger.error("Dataset quality checks failed: %s", failed)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
