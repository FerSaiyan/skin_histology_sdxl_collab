#!/usr/bin/env python
"""
Config-driven execution layer for the MVP multi-physics simulation pipeline.

Reads a YAML configuration (see ``configs/mvp_multiphysics_example.yaml``)
and invokes ``run_mvp_multiphysics_pipeline.py`` with the mapped CLI arguments.

Usage::

    # Minimal smoke test with the example config
    python scripts/simulation/run_mvp_multiphysics_from_config.py \\
        --config configs/mvp_multiphysics_example.yaml --dry-run

    # Real run with CLI overrides
    python scripts/simulation/run_mvp_multiphysics_from_config.py \\
        --config configs/mvp_multiphysics_example.yaml \\
        --label-volume /path/to/volume.npy \\
        --output-dir /tmp/my_run

Dependencies: PyYAML (included in repo requirements.txt).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# YAML import with friendly error
# ---------------------------------------------------------------------------
try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required but not installed.\n"
        "  Install with:  pip install PyYAML\n"
        "  Or:            pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = ["label_volume", "output_dir"]


def load_config(path: str) -> Dict[str, Any]:
    """Load and validate a YAML config file.

    Returns the parsed config dict.

    Raises SystemExit on missing file, invalid YAML, or missing required keys.
    """
    p = Path(path)
    if not p.exists():
        print(f"ERROR: Config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(p, "r") as f:
        try:
            config: Dict[str, Any] = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"ERROR: Failed to parse YAML config: {e}", file=sys.stderr)
            sys.exit(1)

    if not isinstance(config, dict):
        print(
            "ERROR: Config must be a YAML mapping (top-level dict).",
            file=sys.stderr,
        )
        sys.exit(1)

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in config]
    if missing:
        print(
            f"ERROR: Config missing required field(s): {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    return config


# ---------------------------------------------------------------------------
# Config → CLI args mapper
# ---------------------------------------------------------------------------


def _resolve_path(value: str, config_dir: Path) -> str:
    """Resolve a path string; if relative, anchor to config directory."""
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((config_dir / p).resolve())


def _get_overrides(parsed_args: argparse.Namespace) -> Dict[str, str]:
    """Gather CLI overrides that were explicitly set."""
    overrides: Dict[str, str] = {}
    for key, attr in [("label_volume", "label_volume"), ("output_dir", "output_dir")]:
        val = getattr(parsed_args, attr, None)
        if val is not None:
            overrides[key] = val
    if getattr(parsed_args, "fail_fast", False):
        overrides["fail_fast"] = "true"
    return overrides


def build_pipeline_command(
    config: Dict[str, Any],
    overrides: Dict[str, str],
    repo_root: Path,
    config_dir: Path,
) -> List[str]:
    """Build the CLI argument list for ``run_mvp_multiphysics_pipeline.py``.

    Args:
        config: Parsed YAML config dict.
        overrides: CLI-overridden values (label_volume, output_dir, fail_fast).
        repo_root: Repository root path (for resolving the pipeline script).

    Returns:
        List of command-line tokens suitable for ``subprocess.run``.
    """
    pipeline_script = str(
        repo_root / "scripts" / "simulation" / "run_mvp_multiphysics_pipeline.py"
    )
    cmd = [sys.executable, pipeline_script]

    # --- Required paths (support CLI override) ---
    # CLI override values are relative to CWD (not config_dir); config values
    # are relative to config_dir.  The downstream pipeline does its own
    # Path(…).resolve() so we pass the value through as-is for overrides.
    if "label_volume" in overrides:
        label_volume = str(overrides["label_volume"])
    else:
        label_volume = _resolve_path(str(config["label_volume"]), config_dir)
    cmd.extend(["--label-volume", label_volume])

    if "output_dir" in overrides:
        output_dir = str(overrides["output_dir"])
    else:
        output_dir = _resolve_path(str(config["output_dir"]), config_dir)
    cmd.extend(["--output-dir", output_dir])

    # --- Optical GA options ---
    oga = config.get("optical_ga", {})
    if not oga.get("enabled", True):
        cmd.append("--skip-optical-ga")
    else:
        # Forward and fitness mode
        forward_mode = oga.get("forward_mode", "surrogate")
        if forward_mode not in ("surrogate", "realistic"):
            print(
                f"ERROR: optical_ga.forward_mode must be 'surrogate' or 'realistic', "
                f"got {forward_mode}",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd.extend(["--optical-ga-forward-mode", forward_mode])

        fitness_mode = oga.get("fitness_mode", "lab")
        if fitness_mode not in ("lab", "ita", "ita_no_a"):
            print(
                f"ERROR: optical_ga.fitness_mode must be 'lab', 'ita', or 'ita_no_a', "
                f"got {fitness_mode}",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd.extend(["--optical-ga-fitness-mode", fitness_mode])

        # Target colour
        target_L = oga.get("target_L")
        if target_L is not None:
            cmd.extend(["--optical-ga-target-L", str(target_L)])
        target_a = oga.get("target_a")
        if target_a is not None:
            cmd.extend(["--optical-ga-target-a", str(target_a)])
        target_b = oga.get("target_b")
        if target_b is not None:
            cmd.extend(["--optical-ga-target-b", str(target_b)])

        # GA loop parameters (optional)
        oga_params: List[tuple] = [
            ("generations", "--optical-ga-generations", lambda v: v >= 1, ">= 1"),
            ("population_size", "--optical-ga-population-size", lambda v: v >= 2, ">= 2"),
            ("mutation_rate", "--optical-ga-mutation-rate",
             lambda v: 0.0 <= v <= 1.0, "[0.0, 1.0]"),
            ("mutation_strength", "--optical-ga-mutation-strength",
             lambda v: 0.0 <= v <= 1.0, "[0.0, 1.0]"),
            ("elite_fraction", "--optical-ga-elite-fraction",
             lambda v: 0.0 <= v <= 1.0, "[0.0, 1.0]"),
            ("tournament_size", "--optical-ga-tournament-size",
             lambda v: v >= 2, ">= 2"),
            ("seed", "--optical-ga-seed", None, None),
        ]
        for _key, _flag, _validator, _range_hint in oga_params:
            _val = oga.get(_key)
            if _val is not None:
                if _validator is not None and not _validator(_val):
                    print(
                        f"ERROR: optical_ga.{_key} must be {_range_hint}, got {_val}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                cmd.extend([_flag, str(_val)])

        # Dermal chromophores toggle
        if oga.get("use_dermal_chromophores", False):
            cmd.append("--optical-ga-use-dermal-chromophores")

    # --- MCX options ---
    mcx = config.get("mcx", {})
    if not mcx.get("enabled", True):
        cmd.append("--skip-mcx")
    else:
        mcx_mode = mcx.get("mode", "dry-run")
        if mcx_mode not in ("dry-run", "run"):
            print(
                f"ERROR: mcx.mode must be 'dry-run' or 'run', got {mcx_mode}",
                file=sys.stderr,
            )
            sys.exit(1)
        if mcx_mode == "run":
            cmd.append("--mcx-run")

        if bool(mcx.get("render_absorption_video", False)):
            cmd.append("--mcx-render-absorption-video")
        if bool(mcx.get("render_reflectance_spectrum", False)):
            cmd.append("--mcx-render-reflectance-spectrum")

    # --- Thermal options ---
    thermal = config.get("thermal", {})
    if not thermal.get("enabled", True):
        cmd.append("--skip-thermal")
    else:
        # Thermal solver parameters
        dt = thermal.get("dt")
        if dt is not None:
            try:
                dt_val = float(dt)
                if dt_val <= 0:
                    print(
                        f"ERROR: thermal.dt must be > 0, got {dt}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except (TypeError, ValueError):
                print(
                    f"ERROR: thermal.dt must be a number > 0, got {dt}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cmd.extend(["--thermal-dt", str(dt)])

        num_steps = thermal.get("num_steps")
        if num_steps is not None:
            try:
                ns_val = int(num_steps)
                if ns_val < 0:
                    print(
                        f"ERROR: thermal.num_steps must be >= 0, got {num_steps}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except (TypeError, ValueError):
                print(
                    f"ERROR: thermal.num_steps must be an integer >= 0, "
                    f"got {num_steps}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cmd.extend(["--thermal-num-steps", str(ns_val)])

        # Heat source parameters
        source = thermal.get("source", {})
        source_mode = source.get("mode")
        if source_mode is not None:
            if source_mode not in ("none", "spherical"):
                print(
                    f"ERROR: thermal.source.mode must be 'none' or 'spherical', got {source_mode}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cmd.extend(["--thermal-source-mode", source_mode])

        radius_vox = source.get("radius_vox")
        if radius_vox is not None:
            try:
                rv = float(radius_vox)
                if rv < 0:
                    print(
                        f"ERROR: thermal.source.radius_vox must be >= 0, "
                        f"got {radius_vox}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except (TypeError, ValueError):
                print(
                    f"ERROR: thermal.source.radius_vox must be a number >= 0, "
                    f"got {radius_vox}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cmd.extend(["--thermal-source-radius-vox", str(radius_vox)])

        power = source.get("power")
        if power is not None:
            try:
                pw = float(power)
                if pw <= 0:
                    print(
                        f"ERROR: thermal.source.power must be > 0, "
                        f"got {power}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            except (TypeError, ValueError):
                print(
                    f"ERROR: thermal.source.power must be a number > 0, "
                    f"got {power}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cmd.extend(["--thermal-source-power", str(pw)])

        center = source.get("center")
        if center is not None:
            if not isinstance(center, (list, tuple)) or len(center) != 3:
                print(
                    f"ERROR: thermal.source.center must be a list of 3 floats, got {center}",
                    file=sys.stderr,
                )
                sys.exit(1)
            cmd.extend(["--thermal-source-center", str(center[0]), str(center[1]), str(center[2])])

    # --- MCX photons count validation ---
    mcx_photons = mcx.get("photons")
    if mcx_photons is not None:
        try:
            pv = int(mcx_photons)
            if pv <= 0:
                print(
                    f"ERROR: mcx.photons must be > 0, got {mcx_photons}",
                    file=sys.stderr,
                )
                sys.exit(1)
        except (TypeError, ValueError):
            print(
                f"ERROR: mcx.photons must be an integer > 0, got {mcx_photons}",
                file=sys.stderr,
            )
            sys.exit(1)
        cmd.extend(["--mcx-photons", str(mcx_photons)])

    # --- MCX binary path (threaded to the pipeline) ---
    mcx_binary = mcx.get("mcx_binary")
    if mcx_binary:
        cmd.extend(["--mcx-binary", mcx_binary])

    # --- Error handling ---
    if config.get("fail_fast", False) or "fail_fast" in overrides:
        cmd.append("--fail-fast")

    # --- Timeout ---
    timeout = config.get("timeout")
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])

    return cmd


# ---------------------------------------------------------------------------
# Resolved config snapshot builder
# ---------------------------------------------------------------------------


def build_resolved_config(
    config: Dict[str, Any],
    overrides: Dict[str, str],
    config_dir: Path,
) -> Dict[str, Any]:
    """Build a resolved config snapshot (with overrides applied) for JSON output.

    This is the authoritative representation of what was actually run.
    """
    resolved = dict(config)  # shallow copy
    for key, val in overrides.items():
        if key == "fail_fast":
            resolved["fail_fast"] = True
        else:
            resolved[key] = val

    # Normalize path-like fields — only for paths that came from the config
    # file, not CLI overrides (CLI overrides are relative to CWD; the downstream
    # pipeline does its own Path(…).resolve()).
    for path_key in ("label_volume", "output_dir"):
        if path_key not in overrides and path_key in resolved and resolved[path_key] is not None:
            resolved[path_key] = _resolve_path(str(resolved[path_key]), config_dir)
    # No fitness_json to resolve for optical_ga (it uses target Lab directly)
    return resolved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Config-driven execution layer for the MVP multi-physics pipeline.\n\n"
            "Reads a YAML config and invokes run_mvp_multiphysics_pipeline.py "
            "with correctly mapped CLI arguments.\n"
            "Supports optional CLI overrides for label_volume, output_dir, "
            "and fail_fast."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n\n"
            "  # Dry-run to inspect resolved command\n"
            "  python scripts/simulation/run_mvp_multiphysics_from_config.py \\\n"
            "      --config configs/mvp_multiphysics_example.yaml --dry-run\n\n"
            "  # Full run with config defaults\n"
            "  python scripts/simulation/run_mvp_multiphysics_from_config.py \\\n"
            "      --config configs/mvp_multiphysics_example.yaml\n\n"
            "  # Override label volume and output dir\n"
            "  python scripts/simulation/run_mvp_multiphysics_from_config.py \\\n"
            "      --config configs/mvp_multiphysics_example.yaml \\\n"
            "      --label-volume /tmp/my_volume.npy \\\n"
            "      --output-dir /tmp/my_run\n\n"
            "  # Fail-fast override\n"
            "  python scripts/simulation/run_mvp_multiphysics_from_config.py \\\n"
            "      --config configs/mvp_multiphysics_example.yaml --fail-fast\n"
        ),
    )
    ap.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )
    ap.add_argument(
        "--label-volume",
        default=None,
        help="Override label_volume from config.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from config.",
    )
    ap.add_argument(
        "--fail-fast",
        action="store_true",
        help="Override fail_fast from config (stop on first error).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved command + config snapshot; do not execute.",
    )
    return ap.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # --- Load config ---
    config_path = Path(args.config)
    config_dir = config_path.parent
    config = load_config(str(config_path))

    # --- Gather CLI overrides ---
    overrides = _get_overrides(args)

    # --- Resolve paths ---
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_script = repo_root / "scripts" / "simulation" / "run_mvp_multiphysics_pipeline.py"

    if not pipeline_script.exists():
        print(
            f"ERROR: Pipeline script not found: {pipeline_script}",
            file=sys.stderr,
        )
        return 1

    # --- Build resolved command ---
    cmd = build_pipeline_command(config, overrides, repo_root, config_dir)

    # --- Build resolved config snapshot ---
    resolved_config = build_resolved_config(config, overrides, config_dir)

    # --- Dry-run: print and exit ---
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN — No commands executed")
        print("=" * 60)
        print()
        print("Resolved command:")
        print(f"  {' '.join(cmd)}")
        print()
        print("Resolved config snapshot:")
        print(json.dumps(resolved_config, indent=2))
        print()
        print(f"Config source: {config_path.resolve()}")
        return 0

    # --- Write config snapshot to output manifests dir ---
    resolved_output = overrides.get("output_dir", config.get("output_dir", ""))
    if resolved_output:
        resolved_output_path = Path(_resolve_path(str(resolved_output), config_dir))
        manifests_dir = resolved_output_path / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "config_source": str(config_path.resolve()),
            "resolved_config": resolved_config,
            "command": cmd,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "dry_run": False,
        }
        snapshot_path = manifests_dir / "mvp_config_snapshot.json"
        snapshot_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"Wrote config snapshot: {snapshot_path}")

    # --- Execute pipeline ---
    print(f"\nExecuting pipeline via: {' '.join(cmd)}\n")
    sys.stdout.flush()

    t0 = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - t0

    if proc.returncode == 0:
        print(f"\nPipeline completed in {elapsed:.1f}s (return code {proc.returncode}).")
    else:
        print(
            f"\nPipeline FAILED in {elapsed:.1f}s (return code {proc.returncode}).",
            file=sys.stderr,
        )

    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
