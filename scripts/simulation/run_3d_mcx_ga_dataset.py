#!/usr/bin/env python
"""
Dataset-scale 3D MCX + GA runner.

Pipeline per volume:
  1) label_to_optical_priors.py (class-aware priors from present labels)
  2) run_optical_ga_from_labels.py (target Lab optimization)
  3) mcx_build_volume.py
  4) mcx_batch_runner.py (dry-run or real run)

Input can be either:
  - pre-materialized volumes manifest (`--volumes-manifest`), or
  - tiles_for_simulation label tiles materialized on the fly.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _run(cmd: List[str], timeout: int) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "status": "completed" if proc.returncode == 0 else "failed",
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout.splitlines()[-30:],
            "stderr_tail": proc.stderr.splitlines()[-30:],
            "command": cmd,
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "returncode": None, "command": cmd}
    except Exception as e:
        return {"status": "error", "returncode": None, "command": cmd, "error": str(e)}


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run 3D MCX + GA across a volume dataset.")
    ap.add_argument("--output-dir", required=True, help="Output directory for run artifacts.")

    # Data source
    ap.add_argument(
        "--volumes-manifest",
        default="",
        help="Existing volumes manifest.json from materialize_3d_mcx_volumes_from_tiles.py.",
    )
    ap.add_argument(
        "--tiles-root",
        default="data/artifacts/tiles_for_simulation",
        help="tiles_for_simulation root for on-the-fly materialization.",
    )
    ap.add_argument(
        "--materialize-depth",
        type=int,
        default=16,
        help="Depth used when materializing from tiles.",
    )

    # Size control
    ap.add_argument("--run-all", action="store_true", help="Run all available volumes.")
    ap.add_argument("--num-volumes", type=int, default=0, help="Limit number of volumes (0 = all).")
    ap.add_argument("--seed", type=int, default=42, help="Sampling seed.")

    # GA target + budget
    ap.add_argument("--target-L", type=float, default=60.0)
    ap.add_argument("--target-a", type=float, default=10.0)
    ap.add_argument("--target-b", type=float, default=15.0)
    ap.add_argument("--ga-generations", type=int, default=5)
    ap.add_argument("--ga-population-size", type=int, default=12)
    ap.add_argument("--ga-seed", type=int, default=42)
    ap.add_argument("--ga-forward-mode", choices=["surrogate", "realistic"], default="surrogate")

    # MCX execution
    ap.add_argument("--mcx-run", action="store_true", help="Execute MCX (otherwise dry-run).")
    ap.add_argument("--mcx-binary", default="mcx")
    ap.add_argument("--mcx-photons", type=int, default=1000000)
    ap.add_argument("--render-absorption-video", action="store_true")
    ap.add_argument("--render-reflectance-spectrum", action="store_true")

    ap.add_argument("--timeout", type=int, default=900)
    return ap


def _materialize_if_needed(args: argparse.Namespace, out_root: Path) -> Path:
    if args.volumes_manifest:
        p = Path(args.volumes_manifest).resolve()
        if not p.exists():
            raise SystemExit(f"Volumes manifest not found: {p}")
        return p

    mat_out = out_root / "materialized_volumes"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parents[0] / "materialize_3d_mcx_volumes_from_tiles.py"),
        "--tiles-root",
        str(Path(args.tiles_root).resolve()),
        "--output-dir",
        str(mat_out),
        "--depth",
        str(args.materialize_depth),
        "--max-volumes",
        "0" if args.run_all else str(args.num_volumes),
        "--seed",
        str(args.seed),
    ]
    res = _run(cmd, args.timeout)
    if res["status"] != "completed":
        raise SystemExit(f"Materialization failed: {res}")
    return mat_out / "manifest.json"


def main() -> int:
    args = _build_cli().parse_args()
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = _materialize_if_needed(args, out_root)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: List[Dict[str, Any]] = list(data.get("records", []))
    if not records:
        raise SystemExit("No volumes found in manifest")

    if not args.run_all and args.num_volumes > 0:
        records = records[: int(args.num_volumes)]

    run_records: List[Dict[str, Any]] = []
    repo_root = Path(__file__).resolve().parents[2]

    for rec in records:
        tile_id = rec["tile_id"]
        vol_path = Path(rec["volume_npy"]).resolve()
        vout = out_root / "runs" / tile_id
        vout.mkdir(parents=True, exist_ok=True)

        priors_json = vout / "optical_priors.json"
        ga_out = vout / "optical_ga"
        mcx_out = vout / "mcx"
        batch_out = vout / "mcx_batch"

        cmd_priors = [
            sys.executable,
            str(repo_root / "scripts" / "optical_ga" / "label_to_optical_priors.py"),
            "--label-npy",
            str(vol_path),
            "--output-priors-json",
            str(priors_json),
        ]
        res_priors = _run(cmd_priors, args.timeout)

        cmd_ga = [
            sys.executable,
            str(repo_root / "scripts" / "optical_ga" / "run_optical_ga_from_labels.py"),
            "--label-npy",
            str(vol_path),
            "--priors-json",
            str(priors_json),
            "--output-dir",
            str(ga_out),
            "--target-L",
            str(args.target_L),
            "--target-a",
            str(args.target_a),
            "--target-b",
            str(args.target_b),
            "--generations",
            str(args.ga_generations),
            "--population-size",
            str(args.ga_population_size),
            "--seed",
            str(args.ga_seed),
            "--forward-mode",
            str(args.ga_forward_mode),
        ]
        res_ga = _run(cmd_ga, args.timeout)

        cmd_mcx_build = [
            sys.executable,
            str(repo_root / "scripts" / "simulation" / "mcx_build_volume.py"),
            "--volume",
            str(vol_path),
            "--output-dir",
            str(mcx_out),
            "--nphoton",
            str(args.mcx_photons),
            "--enforce-air-top",
            "--auto-flip-z-to-air-top",
        ]
        res_build = _run(cmd_mcx_build, args.timeout)

        cfg = mcx_out / "mcx_config.json"
        cmd_batch = [
            sys.executable,
            str(repo_root / "scripts" / "simulation" / "mcx_batch_runner.py"),
            str(cfg),
            "--output-dir",
            str(batch_out),
            "--mcx-binary",
            str(args.mcx_binary),
            "--timeout",
            str(args.timeout),
        ]
        if not args.mcx_run:
            cmd_batch.append("--dry-run")
        if args.render_absorption_video:
            cmd_batch.append("--render-absorption-video")
        if args.render_reflectance_spectrum:
            cmd_batch.append("--render-reflectance-spectrum")

        res_batch = _run(cmd_batch, args.timeout)

        run_records.append(
            {
                "tile_id": tile_id,
                "volume_npy": str(vol_path),
                "present_class_ids": rec.get("present_class_ids", []),
                "steps": {
                    "label_to_optical_priors": res_priors,
                    "run_optical_ga_from_labels": res_ga,
                    "mcx_build_volume": res_build,
                    "mcx_batch_runner": res_batch,
                },
            }
        )

    out_manifest = {
        "input_manifest": str(manifest_path),
        "num_runs": len(run_records),
        "run_all": bool(args.run_all),
        "num_volumes_requested": int(args.num_volumes),
        "ga_target_lab": [float(args.target_L), float(args.target_a), float(args.target_b)],
        "records": run_records,
    }
    mpath = out_root / "dataset_run_manifest.json"
    mpath.write_text(json.dumps(out_manifest, indent=2), encoding="utf-8")
    print(f"Wrote dataset run manifest: {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
