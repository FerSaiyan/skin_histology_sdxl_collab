#!/usr/bin/env python
"""
Integrated orchestrator for the MVP multi-physics simulation pipeline.

Chains currently implemented classifier-free components:

  • GA skeleton       — scripts/ga/ga_optimiser.py (mock or JSON-fitness mode)
  • MCX scaffolding   — scripts/simulation/mcx_build_volume.py
                        + scripts/simulation/mcx_batch_runner.py
  • Thermal baseline  — scripts/simulation/thermal_build_model.py
                        + scripts/simulation/thermal_solve.py
                        + scripts/simulation/thermal_visualise.py

Subprocess-based — each component is called via its existing CLI.

Usage examples
--------------

Minimal smoke run (tiny label volume, GA mock, MCX dry-run):
    python -c "
    import numpy as np
    vol = np.array([[[0,0,0,0],[0,1,1,0],[0,1,2,0],[0,0,0,0]]], dtype=np.int32)
    np.save('/tmp/mvp_smoke_labels.npy', vol)
    "
    python scripts/simulation/run_mvp_multiphysics_pipeline.py \
        --label-volume /tmp/mvp_smoke_labels.npy \
        --output-dir /tmp/mvp_smoke_run

Full pipeline, skipping GA:
    python scripts/simulation/run_mvp_multiphysics_pipeline.py \
        --label-volume /tmp/mvp_smoke_labels.npy \
        --output-dir /tmp/mvp_thermal_only \
        --skip-ga \
        --mcx-run

Fail-fast mode:
    python scripts/simulation/run_mvp_multiphysics_pipeline.py \
        --label-volume /tmp/mvp_smoke_labels.npy \
        --output-dir /tmp/mvp_failfast \
        --fail-fast

With external GA fitness data:
    python scripts/simulation/run_mvp_multiphysics_pipeline.py \
        --label-volume /tmp/mvp_smoke_labels.npy \
        --output-dir /tmp/mvp_with_data \
        --ga-fitness-json /path/to/fitness_data.json

Keeping defaults for smoke: GA mock, MCX dry-run, conservative thermal.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STEP_ORDER = ["ga", "mcx_build", "mcx_batch", "thermal_build", "thermal_solve", "thermal_visualise"]


def _step_name(step_id: str) -> str:
    """Human-readable step label."""
    labels = {
        "ga": "GA Optimiser",
        "mcx_build": "MCX Build Volume",
        "mcx_batch": "MCX Batch Runner",
        "thermal_build": "Thermal Build Model",
        "thermal_solve": "Thermal Solve (Pennes)",
        "thermal_visualise": "Thermal Visualise",
    }
    return labels.get(step_id, step_id)


def _run_cmd(
    cmd: List[str],
    step_id: str,
    timeout: int = 600,
) -> Dict[str, Any]:
    """Run a subprocess command and return a structured result dict.

    Returns:
        Dict with keys: command, returncode, status, stdout_snippet,
        stderr_snippet, stdout_full, stderr_full, error.
    """
    print(f"\n  [{step_id}] Running: {' '.join(cmd)}")
    sys.stdout.flush()

    result: Dict[str, Any] = {
        "command": cmd,
        "returncode": None,
        "status": "pending",
        "stdout_snippet": None,
        "stderr_snippet": None,
        "stdout_full": None,
        "stderr_full": None,
        "error": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        result["returncode"] = proc.returncode
        result["stdout_full"] = proc.stdout
        result["stderr_full"] = proc.stderr

        # Store snippets (first 20 lines)
        stdout_lines = proc.stdout.splitlines()
        stderr_lines = proc.stderr.splitlines()
        result["stdout_snippet"] = stdout_lines[:20]
        result["stderr_snippet"] = stderr_lines[:10]

        if proc.returncode == 0:
            result["status"] = "completed"
            print(f"  [{step_id}] OK")
        else:
            result["status"] = "failed"
            print(f"  [{step_id}] FAILED (rc={proc.returncode})")
            # Show tail of stderr
            for line in stderr_lines[-5:]:
                print(f"         stderr: {line}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = f"Timed out after {timeout}s"
        print(f"  [{step_id}] TIMEOUT ({timeout}s)")
    except FileNotFoundError as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [{step_id}] ERROR: {e}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  [{step_id}] ERROR: {e}")

    return result


def _skipped_result(reason: str) -> Dict[str, Any]:
    """Standard skipped-step result payload for manifest consistency."""
    return {
        "command": [],
        "returncode": None,
        "status": "skipped",
        "error": reason,
        "stdout_snippet": None,
        "stderr_snippet": None,
        "stdout_full": None,
        "stderr_full": None,
    }


def _artifacts_exist(output_dir: Path, step_id: str) -> List[str]:
    """Return list of existing artifact paths for a given step."""
    artifacts: List[str] = []

    if step_id == "ga":
        for fname in ("best_genome.json", "ga_history.csv", "population_final.json"):
            p = output_dir / "ga" / fname
            if p.exists():
                artifacts.append(str(p))

    elif step_id == "mcx_build":
        for fname in ("mcx_volume.npy", "mcx_media_table.json", "mcx_config.json", "mcx_build_manifest.json"):
            p = output_dir / "mcx" / fname
            if p.exists():
                artifacts.append(str(p))

    elif step_id == "mcx_batch":
        manifest_p = output_dir / "mcx" / "batch" / "mcx_batch_manifest.json"
        if manifest_p.exists():
            artifacts.append(str(manifest_p))

    elif step_id == "thermal_build":
        for fname in ("thermal_model.npz", "thermal_model_manifest.json"):
            p = output_dir / "thermal" / fname
            if p.exists():
                artifacts.append(str(p))

    elif step_id == "thermal_solve":
        for fname in ("temperature_final.npy", "temperature_timeseries_summary.json", "thermal_solve_manifest.json"):
            p = output_dir / "thermal" / "solve" / fname
            if p.exists():
                artifacts.append(str(p))

    elif step_id == "thermal_visualise":
        vis_dir = output_dir / "thermal" / "vis"
        for fname in ("thermal_visualise_summary.json",):
            p = vis_dir / fname
            if p.exists():
                artifacts.append(str(p))
        # Also collect any PNG slices
        for png in sorted(vis_dir.glob("slice_*.png")):
            artifacts.append(str(png))

    return artifacts


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------


def run_ga(
    repo_root: Path,
    output_dir: Path,
    ga_mock: bool,
    ga_fitness_json: Optional[str],
    ga_generations: int = 5,
    ga_population_size: int = 8,
    ga_mutation_rate: float = 0.2,
    ga_mutation_strength: float = 0.1,
    ga_elite_fraction: float = 0.1,
    ga_tournament_size: int = 3,
    ga_seed: int = 42,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Run GA optimiser with configurable parameters."""
    ga_dir = output_dir / "ga"
    ga_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "ga" / "ga_optimiser.py"),
        "--generations", str(ga_generations),
        "--population-size", str(ga_population_size),
        "--output-dir", str(ga_dir),
        "--mutation-rate", str(ga_mutation_rate),
        "--mutation-strength", str(ga_mutation_strength),
        "--elite-fraction", str(ga_elite_fraction),
        "--tournament-size", str(ga_tournament_size),
        "--seed", str(ga_seed),
    ]

    if ga_fitness_json:
        cmd.extend(["--fitness-json", ga_fitness_json])
        print("  [ga] Using fitness JSON (real data mode)")
    else:
        cmd.append("--mock")
        print("  [ga] Using mock mode (synthetic metrics)")

    return _run_cmd(cmd, step_id="ga", timeout=timeout)


def run_mcx_build(
    repo_root: Path,
    output_dir: Path,
    label_volume: str,
    nphoton: Optional[int] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Build MCX artifacts from label volume."""
    mcx_dir = output_dir / "mcx"
    mcx_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "simulation" / "mcx_build_volume.py"),
        "--volume", label_volume,
        "--output-dir", str(mcx_dir),
    ]

    if nphoton is not None:
        cmd.extend(["--nphoton", str(nphoton)])

    return _run_cmd(cmd, step_id="mcx_build", timeout=timeout)


def run_mcx_batch(
    repo_root: Path,
    output_dir: Path,
    mcx_run: bool,
    mcx_binary: str = "mcx",
    timeout: int = 600,
) -> Dict[str, Any]:
    """Run MCX batch runner (dry-run by default, real if --mcx-run)."""
    batch_dir = output_dir / "mcx" / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "mcx" / "mcx_config.json"
    if not config_path.exists():
        return _skipped_result(
            f"MCX config not found at {config_path}. Run mcx_build_volume first."
        )

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "simulation" / "mcx_batch_runner.py"),
        str(config_path),
        "--output-dir", str(batch_dir),
        "--mcx-binary", mcx_binary,
    ]

    if not mcx_run:
        cmd.append("--dry-run")
        print("  [mcx_batch] Dry-run mode (pass --mcx-run to execute)")

    return _run_cmd(cmd, step_id="mcx_batch", timeout=timeout)


def run_thermal_build(
    repo_root: Path,
    output_dir: Path,
    label_volume: str,
    timeout: int = 120,
) -> Dict[str, Any]:
    """Build thermal model from label volume."""
    thermal_dir = output_dir / "thermal"
    thermal_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "simulation" / "thermal_build_model.py"),
        "--label-volume", label_volume,
        "--output-dir", str(thermal_dir),
    ]

    return _run_cmd(cmd, step_id="thermal_build", timeout=timeout)


def run_thermal_solve(
    repo_root: Path,
    output_dir: Path,
    thermal_dt: float = 0.01,
    thermal_num_steps: int = 50,
    thermal_source_mode: str = "spherical",
    thermal_source_radius_vox: float = 2.0,
    thermal_source_power: float = 5e5,
    thermal_source_center: Optional[Tuple[float, float, float]] = None,
    timeout: int = 300,
) -> Dict[str, Any]:
    """Run Pennes bioheat solver with configurable parameters."""
    solve_dir = output_dir / "thermal" / "solve"
    solve_dir.mkdir(parents=True, exist_ok=True)

    model_npz = output_dir / "thermal" / "thermal_model.npz"
    if not model_npz.exists():
        return _skipped_result(
            f"Thermal model not found at {model_npz}. Run thermal_build_model first."
        )

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "simulation" / "thermal_solve.py"),
        "--model-npz", str(model_npz),
        "--output-dir", str(solve_dir),
        "--dt", str(thermal_dt),
        "--num-steps", str(thermal_num_steps),
        "--source-mode", thermal_source_mode,
        "--source-radius-vox", str(thermal_source_radius_vox),
        "--source-power", str(thermal_source_power),
        "--checkpoint-interval", "10",
    ]

    if thermal_source_center is not None:
        cmd.extend([
            "--source-center",
            str(thermal_source_center[0]),
            str(thermal_source_center[1]),
            str(thermal_source_center[2]),
        ])

    return _run_cmd(cmd, step_id="thermal_solve", timeout=timeout)


def run_thermal_visualise(
    repo_root: Path,
    output_dir: Path,
    timeout: int = 60,
) -> Dict[str, Any]:
    """Visualise solver output — stats JSON + optional slice PNG."""
    vis_dir = output_dir / "thermal" / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    temp_npy = output_dir / "thermal" / "solve" / "temperature_final.npy"
    if not temp_npy.exists():
        return _skipped_result(
            f"Temperature array not found at {temp_npy}. Run thermal_solve first."
        )

    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "simulation" / "thermal_visualise.py"),
        "--temperature-npy", str(temp_npy),
        "--output-dir", str(vis_dir),
        "--slice-axis", "0",
        "--slice-index", "0",
    ]

    return _run_cmd(cmd, step_id="thermal_visualise", timeout=timeout)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(
    args: argparse.Namespace,
    steps: Dict[str, Dict[str, Any]],
    label_volume_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Assemble the full run manifest."""
    manifest: Dict[str, Any] = {
        "pipeline": "mvp_multiphysics",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_args": {
            "label_volume": label_volume_path,
            "output_dir": output_dir,
            "ga_mock": args.ga_mock,
            "ga_fitness_json": args.ga_fitness_json,
            "ga_generations": args.ga_generations,
            "ga_population_size": args.ga_population_size,
            "ga_mutation_rate": args.ga_mutation_rate,
            "ga_mutation_strength": args.ga_mutation_strength,
            "ga_elite_fraction": args.ga_elite_fraction,
            "ga_tournament_size": args.ga_tournament_size,
            "ga_seed": args.ga_seed,
            "skip_ga": args.skip_ga,
            "skip_mcx": args.skip_mcx,
            "skip_thermal": args.skip_thermal,
            "mcx_run": args.mcx_run,
            "mcx_binary": args.mcx_binary,
            "mcx_photons": args.mcx_photons,
            "thermal_dt": args.thermal_dt,
            "thermal_num_steps": args.thermal_num_steps,
            "thermal_source_mode": args.thermal_source_mode,
            "thermal_source_radius_vox": args.thermal_source_radius_vox,
            "thermal_source_power": args.thermal_source_power,
            "thermal_source_center": args.thermal_source_center,
            "fail_fast": args.fail_fast,
        },
        "steps": {},
    }

    for step_id in _STEP_ORDER:
        if step_id not in steps:
            continue
        result = steps[step_id]
        entry: Dict[str, Any] = {
            "step_name": _step_name(step_id),
            "status": result.get("status", "unknown"),
            "returncode": result.get("returncode"),
            "command": result.get("command", []),
            "error": result.get("error"),
        }

        # Include stdout/stderr snippets if present (trim to 5 lines for readability)
        stdout_snip = result.get("stdout_snippet")
        if stdout_snip:
            entry["stdout_summary"] = stdout_snip[:5]
        stderr_snip = result.get("stderr_snippet")
        if stderr_snip:
            entry["stderr_summary"] = stderr_snip[:5]

        # Collect artifacts
        out_path = Path(output_dir)
        artifacts = _artifacts_exist(out_path, step_id)
        if artifacts:
            entry["artifacts"] = artifacts

        manifest["steps"][step_id] = entry

    # Overall summary
    statuses = [v.get("status", "unknown") for v in steps.values()]
    manifest["summary"] = {
        "total_steps": len(steps),
        "completed": sum(1 for s in statuses if s == "completed"),
        "failed": sum(1 for s in statuses if s == "failed"),
        "skipped": sum(1 for s in statuses if s == "skipped"),
        "errors": sum(1 for s in statuses if s == "error"),
        "has_errors": any(s in ("failed", "timeout", "error") for s in statuses),
    }

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Run the MVP multi-physics simulation pipeline end-to-end.\n\n"
            "Chains GA optimisation + MCX light transport scaffolding + thermal "
            "(Pennes bioheat) simulation on a shared label volume.\n\n"
            "All components are classifier-free and use existing CLI interfaces."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n\n"
            "  # Minimal smoke run (GA mock, MCX dry-run, conservative thermal)\n"
            "  python scripts/simulation/run_mvp_multiphysics_pipeline.py \\\n"
            "      --label-volume /tmp/mvp_smoke_labels.npy \\\n"
            "      --output-dir /tmp/mvp_smoke_run\n\n"
            "  # Thermal only (skip GA and MCX)\n"
            "  python scripts/simulation/run_mvp_multiphysics_pipeline.py \\\n"
            "      --label-volume /tmp/mvp_smoke_labels.npy \\\n"
            "      --output-dir /tmp/mvp_thermal_only \\\n"
            "      --skip-ga --skip-mcx\n\n"
            "  # Full pipeline with real MCX execution (requires MCX binary)\n"
            "  python scripts/simulation/run_mvp_multiphysics_pipeline.py \\\n"
            "      --label-volume /tmp/mvp_smoke_labels.npy \\\n"
            "      --output-dir /tmp/mvp_full \\\n"
            "      --mcx-run\n\n"
            "  # Fail-fast (stop on first error)\n"
            "  python scripts/simulation/run_mvp_multiphysics_pipeline.py \\\n"
            "      --label-volume /tmp/mvp_smoke_labels.npy \\\n"
            "      --output-dir /tmp/mvp_ff \\\n"
            "      --fail-fast\n"
        ),
    )
    ap.add_argument(
        "--label-volume",
        required=True,
        help="Path to label volume (.npy or NIfTI .nii/.nii.gz). "
        "Labels should be integer-coded (0=background, 1=healthy, 2=lesion).",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Root output directory. Subdirectories will be created: "
        "ga/, mcx/, thermal/, manifests/.",
    )
    # GA mode flags
    ga_group = ap.add_mutually_exclusive_group()
    ga_group.add_argument(
        "--ga-mock",
        action="store_true",
        default=False,
        help="Run GA with synthetic mock metrics (default, lightweight for smoke "
        "testing). Mutually exclusive with --ga-fitness-json.",
    )
    ga_group.add_argument(
        "--ga-fitness-json",
        type=str,
        default=None,
        help="Path to JSON file with fitness data for GA. "
        "When provided, GA uses real data instead of mock metrics. "
        "Expected format: list of genome+metric records (see ga_optimiser.py --help).",
    )
    # GA loop parameters
    ap.add_argument(
        "--ga-generations",
        type=int,
        default=5,
        help="Number of GA generations (default: 5). Must be >= 1.",
    )
    ap.add_argument(
        "--ga-population-size",
        type=int,
        default=8,
        help="GA population size per generation (default: 8). Must be >= 2.",
    )
    ap.add_argument(
        "--ga-mutation-rate",
        type=float,
        default=0.2,
        help="GA per-parameter mutation probability (default: 0.2). "
        "Expected range [0.0, 1.0].",
    )
    ap.add_argument(
        "--ga-mutation-strength",
        type=float,
        default=0.1,
        help="GA mutation strength (stddev in normalised space, default: 0.1). "
        "Expected range [0.0, 1.0].",
    )
    ap.add_argument(
        "--ga-elite-fraction",
        type=float,
        default=0.1,
        help="GA fraction of top individuals preserved (default: 0.1). "
        "Expected range [0.0, 1.0].",
    )
    ap.add_argument(
        "--ga-tournament-size",
        type=int,
        default=3,
        help="GA tournament selection size (default: 3). Must be >= 2.",
    )
    ap.add_argument(
        "--ga-seed",
        type=int,
        default=42,
        help="GA random seed (default: 42).",
    )
    # Step toggles
    ap.add_argument(
        "--skip-ga",
        action="store_true",
        help="Skip the GA optimisation step.",
    )
    ap.add_argument(
        "--skip-mcx",
        action="store_true",
        help="Skip the MCX build and batch runner steps.",
    )
    ap.add_argument(
        "--skip-thermal",
        action="store_true",
        help="Skip the thermal model build, solve, and visualise steps.",
    )
    # MCX execution
    ap.add_argument(
        "--mcx-run",
        action="store_true",
        help="Actually execute MCX simulations (requires MCX binary). "
        "By default, the batch runner runs in dry-run mode (no execution).",
    )
    ap.add_argument(
        "--mcx-binary",
        type=str,
        default="mcx",
        help="Path to the MCX binary (default: 'mcx'). Used when --mcx-run is set.",
    )
    ap.add_argument(
        "--mcx-photons",
        type=int,
        default=None,
        help="Number of photons for MCX simulation. If not provided, "
        "the default in mcx_build_volume.py (10,000,000) is used.",
    )
    # Thermal solver parameters
    ap.add_argument(
        "--thermal-dt",
        type=float,
        default=0.01,
        help="Thermal solver time step in seconds (default: 0.01).",
    )
    ap.add_argument(
        "--thermal-num-steps",
        type=int,
        default=50,
        help="Number of thermal solver time steps (default: 50).",
    )
    ap.add_argument(
        "--thermal-source-mode",
        type=str,
        default="spherical",
        choices=["none", "spherical"],
        help="Thermal heat source mode (default: spherical).",
    )
    ap.add_argument(
        "--thermal-source-radius-vox",
        type=float,
        default=2.0,
        help="Thermal source sphere radius in voxels (default: 2.0).",
    )
    ap.add_argument(
        "--thermal-source-power",
        type=float,
        default=5e5,
        help="Thermal source volumetric power density in W/m³ (default: 5e5).",
    )
    ap.add_argument(
        "--thermal-source-center",
        type=float,
        nargs=3,
        default=None,
        metavar=("CZ", "CY", "CX"),
        help="Thermal source center (Z Y X) in voxel indices. Defaults to volume center.",
    )
    # Error handling
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first step failure. By default, independent later "
        "steps continue after a failure.",
    )
    # Override defaults
    ap.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Per-step timeout in seconds (default: 600). Must be > 0.",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def _validate_args(args: argparse.Namespace) -> None:
    """Validate runtime argument ranges, raising SystemExit on invalid values."""
    errors: List[str] = []

    # GA parameters
    if args.ga_generations < 1:
        errors.append(f"--ga-generations must be >= 1, got {args.ga_generations}")
    if args.ga_population_size < 2:
        errors.append(f"--ga-population-size must be >= 2, got {args.ga_population_size}")
    if not (0.0 <= args.ga_mutation_rate <= 1.0):
        errors.append(
            f"--ga-mutation-rate must be in [0.0, 1.0], got {args.ga_mutation_rate}"
        )
    if not (0.0 <= args.ga_mutation_strength <= 1.0):
        errors.append(
            f"--ga-mutation-strength must be in [0.0, 1.0], "
            f"got {args.ga_mutation_strength}"
        )
    if not (0.0 <= args.ga_elite_fraction <= 1.0):
        errors.append(
            f"--ga-elite-fraction must be in [0.0, 1.0], "
            f"got {args.ga_elite_fraction}"
        )
    if args.ga_tournament_size < 2:
        errors.append(
            f"--ga-tournament-size must be >= 2, got {args.ga_tournament_size}"
        )

    # MCX parameters
    if args.mcx_photons is not None and args.mcx_photons <= 0:
        errors.append(
            f"--mcx-photons must be > 0, got {args.mcx_photons}"
        )

    # Thermal parameters
    if args.thermal_dt <= 0:
        errors.append(f"--thermal-dt must be > 0, got {args.thermal_dt}")
    if args.thermal_num_steps < 0:
        errors.append(
            f"--thermal-num-steps must be >= 0, got {args.thermal_num_steps}"
        )
    if args.thermal_source_radius_vox < 0:
        errors.append(
            f"--thermal-source-radius-vox must be >= 0, "
            f"got {args.thermal_source_radius_vox}"
        )
    if args.thermal_source_power <= 0:
        errors.append(
            f"--thermal-source-power must be > 0, got {args.thermal_source_power}"
        )

    # Timeout
    if args.timeout <= 0:
        errors.append(f"--timeout must be > 0, got {args.timeout}")

    if errors:
        print("ERROR: Invalid argument(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Manifest schema validation
# ---------------------------------------------------------------------------


def _validate_manifest(manifest: Dict[str, Any]) -> None:
    """Validate the manifest structure before writing.

    Raises ValueError if required keys are missing or types are wrong.
    """
    for key in ("pipeline", "timestamp_utc", "input_args", "steps", "summary"):
        if key not in manifest:
            raise ValueError(f"Manifest missing required key: {key}")

    summary = manifest["summary"]
    for key in ("total_steps", "completed", "failed", "skipped", "errors",
                 "has_errors"):
        if key not in summary:
            raise ValueError(f"Manifest summary missing required key: {key}")

    for step_id, entry in manifest["steps"].items():
        for key in ("step_name", "status", "returncode", "command", "error"):
            if key not in entry:
                raise ValueError(
                    f"Manifest step '{step_id}' missing required key: {key}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Default GA mode to mock unless explicit fitness JSON is provided.
    if not args.ga_fitness_json:
        args.ga_mock = True

    # Validate argument ranges before launching anything.
    _validate_args(args)

    # --- Resolve paths ---
    label_volume_path = str(Path(args.label_volume).resolve())
    output_dir = Path(args.output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[2]

    # Verify label volume exists
    if not Path(label_volume_path).exists():
        print(f"ERROR: label volume not found: {label_volume_path}", file=sys.stderr)
        return 1

    # --- Create output subdirs ---
    for sub in ("ga", "mcx", "thermal", "manifests"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    print(f"=== MVP Multi-Physics Pipeline ===")
    print(f"  Label volume: {label_volume_path}")
    print(f"  Output dir  : {output_dir}")
    print(f"  Fail-fast   : {args.fail_fast}")
    print(f"  Skip GA     : {args.skip_ga}")
    print(f"  Skip MCX    : {args.skip_mcx}")
    print(f"  Skip thermal: {args.skip_thermal}")
    print()

    # --- Steps registry ---
    steps: Dict[str, Dict[str, Any]] = {}
    failed = False

    # ---- Step 1: GA ----
    if not args.skip_ga and not failed:
        steps["ga"] = run_ga(
            repo_root=repo_root,
            output_dir=output_dir,
            ga_mock=args.ga_mock,
            ga_fitness_json=args.ga_fitness_json,
            ga_generations=args.ga_generations,
            ga_population_size=args.ga_population_size,
            ga_mutation_rate=args.ga_mutation_rate,
            ga_mutation_strength=args.ga_mutation_strength,
            ga_elite_fraction=args.ga_elite_fraction,
            ga_tournament_size=args.ga_tournament_size,
            ga_seed=args.ga_seed,
            timeout=args.timeout,
        )
        if steps["ga"]["status"] not in ("completed",):
            if args.fail_fast:
                print("\n  [FAIL-FAST] GA step failed. Stopping.")
                failed = True

    # ---- Step 2: MCX Build ----
    if not args.skip_mcx and not failed:
        steps["mcx_build"] = run_mcx_build(
            repo_root=repo_root,
            output_dir=output_dir,
            label_volume=label_volume_path,
            nphoton=args.mcx_photons,
            timeout=args.timeout,
        )
        if steps["mcx_build"]["status"] not in ("completed",):
            if args.fail_fast:
                print("\n  [FAIL-FAST] MCX build step failed. Stopping.")
                failed = True

    # ---- Step 3: MCX Batch Runner (depends on MCX build) ----
    if not args.skip_mcx and not failed:
        # Check if mcx_build succeeded before running batch
        mcx_build_ok = (
            steps.get("mcx_build", {}).get("status") == "completed"
        )
        # If mcx_build was skipped (no result), still try batch
        if "mcx_build" not in steps or mcx_build_ok:
            steps["mcx_batch"] = run_mcx_batch(
                repo_root=repo_root,
                output_dir=output_dir,
                mcx_run=args.mcx_run,
                mcx_binary=args.mcx_binary,
                timeout=args.timeout,
            )
            if steps["mcx_batch"]["status"] not in ("completed", "skipped"):
                if args.fail_fast:
                    print("\n  [FAIL-FAST] MCX batch step failed. Stopping.")
                    failed = True
        else:
            steps["mcx_batch"] = _skipped_result(
                "Skipped because mcx_build did not complete successfully."
            )

    # ---- Step 4: Thermal Build ----
    if not args.skip_thermal and not failed:
        steps["thermal_build"] = run_thermal_build(
            repo_root=repo_root,
            output_dir=output_dir,
            label_volume=label_volume_path,
            timeout=args.timeout,
        )
        if steps["thermal_build"]["status"] not in ("completed",):
            if args.fail_fast:
                print("\n  [FAIL-FAST] Thermal build step failed. Stopping.")
                failed = True

    # ---- Step 5: Thermal Solve (depends on Thermal Build) ----
    if not args.skip_thermal and not failed:
        thermal_build_ok = (
            steps.get("thermal_build", {}).get("status") == "completed"
        )
        if "thermal_build" not in steps or thermal_build_ok:
            steps["thermal_solve"] = run_thermal_solve(
                repo_root=repo_root,
                output_dir=output_dir,
                thermal_dt=args.thermal_dt,
                thermal_num_steps=args.thermal_num_steps,
                thermal_source_mode=args.thermal_source_mode,
                thermal_source_radius_vox=args.thermal_source_radius_vox,
                thermal_source_power=args.thermal_source_power,
                thermal_source_center=args.thermal_source_center,
                timeout=args.timeout,
            )
            if steps["thermal_solve"]["status"] not in ("completed", "skipped"):
                if args.fail_fast:
                    print("\n  [FAIL-FAST] Thermal solve step failed. Stopping.")
                    failed = True
        else:
            steps["thermal_solve"] = _skipped_result(
                "Skipped because thermal_build did not complete successfully."
            )

    # ---- Step 6: Thermal Visualise (depends on Thermal Solve) ----
    if not args.skip_thermal and not failed:
        thermal_solve_ok = (
            steps.get("thermal_solve", {}).get("status") == "completed"
        )
        if "thermal_solve" not in steps or thermal_solve_ok:
            steps["thermal_visualise"] = run_thermal_visualise(
                repo_root=repo_root,
                output_dir=output_dir,
                timeout=args.timeout,
            )
        else:
            steps["thermal_visualise"] = _skipped_result(
                "Skipped because thermal_solve did not complete successfully."
            )

    # ---- Write manifest ----
    manifest = build_manifest(args, steps, label_volume_path, str(output_dir))
    # Validate manifest structure before writing
    try:
        _validate_manifest(manifest)
    except ValueError as e:
        print(f"  [WARN] Manifest validation: {e}", file=sys.stderr)
    manifest_path = output_dir / "manifests" / "mvp_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")

    # ---- Summary ----
    summary = manifest["summary"]
    print(f"\n=== Pipeline Summary ===")
    print(f"  Total steps : {summary['total_steps']}")
    print(f"  Completed   : {summary['completed']}")
    print(f"  Failed      : {summary['failed']}")
    print(f"  Skipped     : {summary['skipped']}")
    print(f"  Errors      : {summary['errors']}")

    if summary["has_errors"]:
        print("\n  Failed steps:")
        for step_id, entry in manifest["steps"].items():
            if entry["status"] in ("failed", "timeout", "error"):
                err_msg = entry.get("error", "see stderr snippet")
                print(f"    - {step_id}: {err_msg}")
        return 1
    else:
        print("\n  All steps completed successfully.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
