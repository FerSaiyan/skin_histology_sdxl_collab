#!/usr/bin/env python
"""
Batch runner for MCX simulations (MCX v2025.10 compatible).

Reads one or more ``mcx_config.json`` files (or a glob pattern) and either:
  - Dry-run (``--dry-run``): writes commands to a manifest JSON without
    executing anything.
  - Run mode: executes each MCX command if ``--mcx-binary`` is provided and
    exists, capturing return code, stdout, and stderr.

Command construction (MCX v2025.10):
  The runner changes to the per-job output directory before executing MCX so
  that output files (e.g., ``skin_mcx_sim.mc2``) land there.  The config is
  passed via ``-f config.json``.  Output format and type are set in the JSON
  ``Session`` block (OutputFormat/OutputType).

Note on backward compatibility:
  The old ``--input-format jds / --load-config / --dump-output`` flags were
  never part of any MCX release and have been removed.  If you need pre-v2025
  support, either provide pointers to older binaries or re-enable legacy flag
  construction by overriding ``_build_mcx_command`` at the call site.

Output:
  - ``mcx_batch_manifest.json`` — per-job status, command, return code,
    stdout/stderr summary.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _gather_configs(paths: List[str]) -> List[Path]:
    """Resolve a mix of file paths and glob patterns to existing config files."""
    config_files: List[Path] = []
    for p in paths:
        # Check if it's a glob
        candidates = sorted(globmod.glob(p))
        if candidates:
            config_files.extend(Path(c) for c in candidates)
        else:
            # Treat as literal path
            pp = Path(p)
            if pp.exists():
                config_files.append(pp)
            else:
                print(
                    f"  [WARN] No match for: {p}",
                    file=sys.stderr,
                )
    return config_files


def _build_mcx_command(
    config_path: Path,
    mcx_binary: str,
    output_dir: Path,
) -> List[str]:
    """Build an MCX v2025.10 command line from a config file.

    Constructs a command of the form::

        cd <output_dir> && mcx_binary -f /path/to/config.json

    MCX writes output files (``<Session.ID>.mc2`` etc.) to the current
    working directory.  The caller is responsible for setting ``cwd`` in
    ``subprocess.run`` (see ``main()``).

    If the config needs row-major volume data (C order), pass ``-a 1``
    here.  By default, the volume byte layout from numpy C-order tobytes
    with shape (Z,Y,X) matches MCX's column-major interpretation with
    Dim=[X,Y,Z], so ``-a`` is omitted.
    """
    cmd = [
        mcx_binary,
        "-f",
        str(config_path.resolve()),
    ]
    return cmd


def _summarise_output(
    stdout: str,
    stderr: str,
    max_lines: int = 30,
) -> Dict[str, Any]:
    """Build a summary dict from captured stdout/stderr."""
    stdout_lines = stdout.splitlines()
    stderr_lines = stderr.splitlines()

    summary: Dict[str, Any] = {
        "stdout_lines": len(stdout_lines),
        "stderr_lines": len(stderr_lines),
        "stdout_head": stdout_lines[:max_lines],
        "stderr_head": stderr_lines[:max_lines],
        "stdout_tail": stdout_lines[-max_lines:] if len(stdout_lines) > max_lines else [],
        "stderr_tail": stderr_lines[-max_lines:] if len(stderr_lines) > max_lines else [],
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Batch runner for MCX simulations.",
    )
    ap.add_argument(
        "configs",
        nargs="+",
        help="One or more mcx_config.json paths or glob patterns.",
    )
    ap.add_argument(
        "--mcx-binary",
        default="mcx",
        help="Path to the MCX binary. Used in run mode (default: 'mcx').",
    )
    ap.add_argument(
        "--output-dir",
        default="./mcx_batch_output",
        help="Base output directory for MCX simulation results "
        "(default: ./mcx_batch_output).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate commands; do not execute them.",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-job timeout in seconds (default: 3600). Must be > 0.",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    # --- Validate numeric args ---
    if args.timeout <= 0:
        raise SystemExit(
            f"ERROR: --timeout must be > 0, got {args.timeout}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Gather config files ---
    config_files = _gather_configs(args.configs)
    if not config_files:
        raise SystemExit("No valid mcx_config.json files found. Exiting.")

    print(f"Found {len(config_files)} config file(s).")

    # --- Check MCX binary in run mode ---
    if not args.dry_run:
        resolved_mcx = shutil.which(args.mcx_binary)
        if resolved_mcx is None:
            mcx_path = Path(args.mcx_binary)
            if not mcx_path.exists():
                raise SystemExit(
                    f"MCX binary not found at '{args.mcx_binary}'. "
                    "Use --dry-run to generate commands only, or set --mcx-binary."
                )
            if not os.access(str(mcx_path), os.X_OK):
                raise SystemExit(
                    f"MCX binary at '{args.mcx_binary}' is not executable."
                )
            resolved_mcx = str(mcx_path.resolve())
        args.mcx_binary = resolved_mcx

    # --- Build and process jobs ---
    jobs: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for i, cfg_path in enumerate(config_files):
        sim_out = output_dir / f"sim_{i:04d}"
        sim_out.mkdir(parents=True, exist_ok=True)

        cmd = _build_mcx_command(cfg_path, args.mcx_binary, sim_out)

        job: Dict[str, Any] = {
            "job_id": i,
            "config_path": str(cfg_path.resolve()),
            "command": cmd,
            "output_dir": str(sim_out.resolve()),
            "status": "pending",
            "return_code": None,
            "summary": None,
            "error": None,
        }

        if args.dry_run:
            job["status"] = "dry_run"
            print(f"\n  [{i:04d}] (dry-run) {cfg_path.name}")
            print(f"         Working dir: {sim_out.resolve()}")
            print(f"         Command: {' '.join(cmd)}")
        else:
            print(f"\n  [{i:04d}] Running: {cfg_path.name} ...", end=" ")
            sys.stdout.flush()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                    cwd=str(sim_out.resolve()),
                )
                job["return_code"] = proc.returncode
                job["summary"] = _summarise_output(proc.stdout, proc.stderr)
                if proc.returncode == 0:
                    job["status"] = "completed"
                    print("OK")
                else:
                    job["status"] = "failed"
                    print(f"FAILED (rc={proc.returncode})")
                    # Print tail of stderr for context
                    err_tail = proc.stderr.strip().splitlines()[-5:]
                    for line in err_tail:
                        print(f"         stderr: {line}")
            except subprocess.TimeoutExpired:
                job["status"] = "timeout"
                job["error"] = f"Timed out after {args.timeout}s"
                print(f"TIMEOUT ({args.timeout}s)")
            except FileNotFoundError:
                job["status"] = "error"
                job["error"] = f"MCX binary not found: {args.mcx_binary}"
                print(f"ERROR: {job['error']}")
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)
                print(f"ERROR: {e}")

        jobs.append(job)

    # --- Write batch manifest ---
    manifest: Dict[str, Any] = {
        "timestamp": timestamp,
        "dry_run": args.dry_run,
        "mcx_binary": args.mcx_binary,
        "num_jobs": len(jobs),
        "jobs": jobs,
    }

    manifest_path = output_dir / "mcx_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote batch manifest: {manifest_path}")

    # --- Summary ---
    if args.dry_run:
        print(f"\n=== Dry-run complete. {len(jobs)} command(s) generated. ===")
    else:
        completed = sum(1 for j in jobs if j["status"] == "completed")
        failed = sum(1 for j in jobs if j["status"] == "failed")
        timeout = sum(1 for j in jobs if j["status"] == "timeout")
        error = sum(1 for j in jobs if j["status"] == "error")
        print(f"\n=== Batch Summary: {completed} OK, {failed} failed, "
              f"{timeout} timeout, {error} error ===")


if __name__ == "__main__":
    main()
