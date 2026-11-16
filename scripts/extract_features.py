#!/usr/bin/env python3
"""Extract causal PMU features from raw scenario .npz files."""

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
import pandas as pd

from pmu_fsa.simulation.scenario_generator import load_scenario_from_npz
from pmu_fsa.features.extractor import FeatureExtractor, DESCRIPTIVE_FEATURE_NAMES

logger = logging.getLogger("extract_features")


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Causal PMU Feature Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with raw .npz scenarios")
    parser.add_argument("--t-obs", type=float, default=1.0, help="Observation window length in seconds")
    parser.add_argument("--output-file", type=str, default="data/processed/features.npz", help="Output .npz path")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save features")
    parser.add_argument("--all-windows", action="store_true", help="Extract for windows [0.5, 1.0, 1.5, 2.0]s")
    parser.add_argument("--f-nominal", type=float, default=50.0, help="Nominal power system frequency (Hz)")
    parser.add_argument("--save-csv", action="store_true", help="Also export CSV representation")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    return parser.parse_args(args)


def extract_for_window(extractor: FeatureExtractor, scenarios: list, t_obs: float, output_path: Path, save_csv: bool = False):
    X, y_nadir, y_secure, meta_df = extractor.transform_batch(scenarios, t_obs=t_obs, as_dataframe=True)

    feat_names = list(X.columns) if hasattr(X, "columns") else DESCRIPTIVE_FEATURE_NAMES
    scen_ids = meta_df["scenario_id"].values if hasattr(meta_df, "columns") else [m.get("scenario_id", f"s_{i}") for i, m in enumerate(meta_df)]
    parts = meta_df["partition"].values if hasattr(meta_df, "columns") else [m.get("partition", "train") for m in meta_df]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=X.values if hasattr(X, "values") else X,
        y_nadir=y_nadir,
        y_secure=y_secure,
        feature_names=feat_names,
        scenario_ids=scen_ids,
        partitions=parts,
    )
    logger.info("Saved feature dataset (N=%d, D=%d) to %s", len(scenarios), X.shape[1], output_path)

    if save_csv and hasattr(X, "to_csv"):
        csv_path = output_path.with_suffix(".csv")
        df_combined = X.copy()
        df_combined["ground_truth_nadir"] = y_nadir
        df_combined["is_secure"] = y_secure
        if hasattr(meta_df, "columns"):
            for c in ["scenario_id", "partition", "renewable_penetration", "load_scale"]:
                if c in meta_df.columns:
                    df_combined[c] = meta_df[c]
        df_combined.to_csv(csv_path, index=False)
        logger.info("Saved tabular CSV to %s", csv_path)


def main(args=None) -> int:
    parsed = parse_args(args)
    logging.basicConfig(
        level=logging.WARNING if parsed.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    in_dir = Path(parsed.input_dir)
    if not in_dir.exists():
        logger.error("Input directory does not exist: %s", in_dir)
        return 1

    scenario_files = sorted(list(in_dir.glob("*.npz")))
    if not scenario_files:
        logger.error("No .npz scenario files found in %s", in_dir)
        return 1

    logger.info("Loading %d scenarios from %s...", len(scenario_files), in_dir)
    scenarios = [load_scenario_from_npz(str(fp)) for fp in scenario_files]

    out_file = Path(parsed.output_file)
    if parsed.output_dir is not None:
        out_dir = Path(parsed.output_dir)
    else:
        out_dir = out_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor = FeatureExtractor(f_nominal=parsed.f_nominal)

    windows = [0.5, 1.0, 1.5, 2.0] if parsed.all_windows else [parsed.t_obs]
    for w in windows:
        if parsed.all_windows:
            w_out_file = out_dir / f"features_t{w:.1f}s.npz"
        else:
            w_out_file = out_file if parsed.output_dir is None else out_dir / out_file.name
        extract_for_window(extractor, scenarios, w, w_out_file, save_csv=parsed.save_csv)
        if parsed.all_windows and abs(w - parsed.t_obs) < 1e-6:
            primary_out = out_file if parsed.output_dir is None else out_dir / out_file.name
            if primary_out.resolve() != w_out_file.resolve():
                extract_for_window(extractor, scenarios, w, primary_out, save_csv=parsed.save_csv)

    return 0


if __name__ == "__main__":
    sys.exit(main())
