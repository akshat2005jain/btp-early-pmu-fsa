#!/usr/bin/env python3
"""Standalone CLI runner for the 50 Hz PMU-FSA E2E test suite."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import inspect
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TIER_MODULE_MAP = {
    "1": [
        "tests/e2e/test_tier1_r1_simulation.py",
        "tests/e2e/test_tier1_r2_features.py",
        "tests/e2e/test_tier1_r3_models.py",
        "tests/e2e/test_tier1_r4_benchmarks.py",
    ],
    "2": [
        "tests/e2e/test_tier2_boundaries.py",
    ],
    "3": [
        "tests/e2e/test_tier3_combinations.py",
    ],
    "4": [
        "tests/e2e/test_tier4_scenarios.py",
    ],
    "5": [
        "tests/e2e/test_tier5_adversarial_sim_features.py",
        "tests/e2e/test_tier5_adversarial_models_eval.py",
    ],
}


def _unwrap_fixture(fn: Any) -> Any:
    if hasattr(fn, "_get_wrapped_function"):
        return fn._get_wrapped_function()
    if hasattr(fn, "__wrapped__"):
        return fn.__wrapped__
    if hasattr(fn, "_fixture_function"):
        return fn._fixture_function
    return fn


def load_module_from_file(filepath: Path):
    mod_name = filepath.stem
    spec = importlib.util.spec_from_file_location(mod_name, str(filepath))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module spec from {filepath}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def run_tests_via_introspection(test_files: List[Path], verbose: bool = False) -> Dict[str, Any]:
    total_start = time.time()
    results = []
    n_passed = 0
    n_failed = 0
    n_skipped = 0

    conftest_path = PROJECT_ROOT / "tests" / "e2e" / "conftest.py"
    fixtures = {}
    if conftest_path.exists():
        c_mod = load_module_from_file(conftest_path)
        for name, obj in inspect.getmembers(c_mod):
            if callable(obj) and (
                hasattr(obj, "_pytestfixturefunction")
                or name.startswith("sample_")
                or name.startswith("ieee39_")
                or name.startswith("synthetic_")
                or name.startswith("real_")
            ):
                fixtures[name] = obj

    for t_file in test_files:
        module_name = t_file.name
        if verbose:
            print(f"\n--- Running tests in {module_name} ---")
        try:
            mod = load_module_from_file(t_file)
        except Exception as e:
            results.append({
                "module": module_name,
                "test": "module_import",
                "status": "FAILED",
                "error": str(e),
                "duration": 0.0,
            })
            n_failed += 1
            continue

        test_funcs = []
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) and name.startswith("test_"):
                test_funcs.append((name, obj, None))
            elif inspect.isclass(obj) and (name.startswith("Test") or name.endswith("Test")):
                cls_inst = obj()
                for m_name, m_obj in inspect.getmembers(cls_inst):
                    if inspect.ismethod(m_obj) and m_name.startswith("test_"):
                        test_funcs.append((f"{name}::{m_name}", m_obj, cls_inst))

        for test_name, func, _ in test_funcs:
            t0 = time.time()
            sig = inspect.signature(func)
            kwargs = {}
            skip_test = False
            skip_reason = ""
            temp_dirs: List[Path] = []

            try:
                for p_name in sig.parameters.keys():
                    if p_name == "tmp_path":
                        td = Path(tempfile.mkdtemp(prefix="pmu_fsa_e2e_"))
                        temp_dirs.append(td)
                        kwargs[p_name] = td
                    elif p_name in fixtures:
                        raw_fn = _unwrap_fixture(fixtures[p_name])
                        try:
                            kwargs[p_name] = raw_fn()
                        except BaseException as fe:
                            skip_test = True
                            skip_reason = f"Fixture {p_name} error: {fe}"
                            break
                    elif hasattr(mod, p_name) and callable(getattr(mod, p_name)):
                        raw_mod_fn = _unwrap_fixture(getattr(mod, p_name))
                        kwargs[p_name] = raw_mod_fn()

                if skip_test:
                    duration = time.time() - t0
                    results.append({
                        "module": module_name,
                        "test": test_name,
                        "status": "SKIPPED",
                        "error": skip_reason,
                        "duration": duration,
                    })
                    n_skipped += 1
                    if verbose:
                        print(f"  [SKIPPED] {test_name}: {skip_reason}")
                    continue

                try:
                    func(**kwargs)
                    duration = time.time() - t0
                    results.append({
                        "module": module_name,
                        "test": test_name,
                        "status": "PASSED",
                        "error": None,
                        "duration": duration,
                    })
                    n_passed += 1
                    if verbose:
                        print(f"  [PASSED]  {test_name} ({duration:.4f}s)")
                except Exception as ex:
                    duration = time.time() - t0
                    results.append({
                        "module": module_name,
                        "test": test_name,
                        "status": "FAILED",
                        "error": f"{type(ex).__name__}: {ex}",
                        "duration": duration,
                    })
                    n_failed += 1
                    if verbose:
                        print(f"  [FAILED]  {test_name} ({duration:.4f}s): {ex}")
            finally:
                for td in temp_dirs:
                    shutil.rmtree(td, ignore_errors=True)

    total_duration = time.time() - total_start
    return {
        "results": results,
        "total": len(results),
        "passed": n_passed,
        "failed": n_failed,
        "skipped": n_skipped,
        "duration": total_duration,
    }


def run_tests_via_pytest(test_files: List[Path], verbose: bool = False) -> int:
    import pytest
    args = [str(f) for f in test_files]
    args.append("-v" if verbose else "-q")
    return int(pytest.main(args))


def format_markdown_report(summary: Dict[str, Any], tier: str) -> str:
    total = summary["total"]
    passed = summary["passed"]
    failed = summary["failed"]
    skipped = summary["skipped"]
    duration = summary["duration"]
    pass_rate = (passed / total * 100.0) if total > 0 else 0.0

    lines = [
        "# E2E Test Suite Execution Report (50.0 Hz Canonical Standard)",
        "",
        f"- **Timestamp**: `{datetime.now(timezone.utc).isoformat()}`",
        f"- **Tier**: `Tier {tier}`",
        f"- **Total Tests**: `{total}`",
        f"- **Passed**: `{passed}`",
        f"- **Failed**: `{failed}`",
        f"- **Skipped**: `{skipped}`",
        f"- **Pass Rate**: `{pass_rate:.1f}%`",
        f"- **Execution Time**: `{duration:.3f}s`",
        "",
        "## Summary Results Table",
        "",
        "| Module | Test Case | Status | Duration (s) | Notes |",
        "|---|---|:---:|:---:|---|",
    ]

    for item in summary["results"]:
        status_icon = "PASSED" if item["status"] == "PASSED" else ("FAILED" if item["status"] == "FAILED" else "SKIPPED")
        notes = item["error"] if item["error"] else "-"
        notes = notes.replace("\n", " ").replace("|", "\\|")[:80]
        lines.append(f"| `{item['module']}` | `{item['test']}` | {status_icon} | {item['duration']:.4f} | {notes} |")

    lines.append("")
    if failed == 0 and total > 0:
        lines.append("## Status: ALL TESTS PASSED")
    else:
        lines.append(f"## Status: {failed} TEST(S) FAILED")
    lines.append("")
    return "\n".join(lines)


def _maybe_reexec_with_venv():
    """Re-exec under .venv/bin/python if currently running on system Python."""
    if os.environ.get("PMU_FSA_VENV_REEXEC") == "1":
        return

    venv_dir = PROJECT_ROOT / ".venv"
    if not venv_dir.is_dir():
        return

    current_exe = Path(sys.executable)
    try:
        if current_exe.relative_to(venv_dir):
            return
    except ValueError:
        pass

    try:
        if Path(sys.prefix).resolve() == venv_dir.resolve():
            return
    except Exception:
        pass

    venv_python = None
    for candidate_name in ("python3", "python"):
        candidate = venv_dir / "bin" / candidate_name
        if candidate.exists() and os.access(str(candidate), os.X_OK):
            venv_python = candidate
            break

    if venv_python is None:
        return

    env = os.environ.copy()
    env["PMU_FSA_VENV_REEXEC"] = "1"
    pythonpath = env.get("PYTHONPATH", "")
    parts = [p for p in pythonpath.split(os.pathsep) if p]
    for path_entry in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
        if path_entry not in parts:
            parts.insert(0, path_entry)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    venv_exe_str = str(venv_python.absolute())
    args = [venv_exe_str, str(Path(__file__).resolve())] + sys.argv[1:]
    try:
        os.execve(venv_exe_str, args, env)
    except Exception:
        sys.executable = venv_exe_str


def main():
    _maybe_reexec_with_venv()
    parser = argparse.ArgumentParser(
        description="Run E2E test suite for PMU-Based Early Frequency-Security Assessment (50.0 Hz)"
    )
    parser.add_argument(
        "--tier",
        choices=["1", "2", "3", "4", "5", "all"],
        default="1",
        help="Test tier to execute (default: 1)",
    )
    parser.add_argument(
        "--report",
        "--report-path",
        dest="report",
        type=str,
        default=None,
        help="Markdown output report path",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose execution output")
    args = parser.parse_args()

    if args.tier == "all":
        candidate_paths = []
        for paths in TIER_MODULE_MAP.values():
            candidate_paths.extend(paths)
    else:
        candidate_paths = TIER_MODULE_MAP[args.tier]

    test_files = []
    for p in candidate_paths:
        full_path = PROJECT_ROOT / p
        if full_path.exists():
            test_files.append(full_path)

    if not test_files:
        print(f"No existing test files found for Tier {args.tier}. Looked for: {candidate_paths}")
        sys.exit(1)

    print("============================================================")
    print(f"PMU-FSA 50 Hz E2E Test Runner: Tier {args.tier.upper()}")
    print(f"Found {len(test_files)} test file(s) to execute.")
    print("============================================================")

    summary = run_tests_via_introspection(test_files, verbose=args.verbose)
    md_report = format_markdown_report(summary, args.tier)

    print("\n" + "=" * 60)
    print(
        f"Results: {summary['passed']}/{summary['total']} passed "
        f"({summary['failed']} failed, {summary['skipped']} skipped) in {summary['duration']:.2f}s"
    )
    print("=" * 60)

    if summary["failed"] > 0:
        print("\nFailed Tests:")
        for r in summary["results"]:
            if r["status"] == "FAILED":
                print(f"  [FAILED] {r['module']}::{r['test']}: {r['error']}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(md_report, encoding="utf-8")
        print(f"Markdown report written to: {report_path.resolve()}")

    if summary["failed"] > 0 or summary["skipped"] > 0 or summary["total"] == 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
