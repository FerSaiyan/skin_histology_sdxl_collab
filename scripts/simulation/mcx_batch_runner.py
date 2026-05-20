#!/usr/bin/env python
"""
Batch runner for MCX simulations (MCX v2025.10 compatible).

Reads one or more `mcx_config.json` files (or glob patterns), then either:
  - dry-run (`--dry-run`): emits commands + manifest only
  - run mode: executes MCX and captures stdout/stderr

Optional post-processing:
  - `--render-absorption-video`: writes per-job depth-sweep MP4 from `.mc2`
  - `--render-reflectance-spectrum`: writes `reflectance_spectrum.png` from
    absorbed fraction parsed in MCX stdout across wavelength-tagged jobs
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _gather_configs(paths: List[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        candidates = sorted(globmod.glob(p))
        if candidates:
            out.extend(Path(c) for c in candidates)
        else:
            pp = Path(p)
            if pp.exists():
                out.append(pp)
            else:
                print(f"  [WARN] No match for: {p}", file=sys.stderr)
    return out


def _build_mcx_command(config_path: Path, mcx_binary: str) -> List[str]:
    return [mcx_binary, "-f", str(config_path.resolve())]


def _summarise_output(stdout: str, stderr: str, max_lines: int = 30) -> Dict[str, Any]:
    so = stdout.splitlines()
    se = stderr.splitlines()
    return {
        "stdout_lines": len(so),
        "stderr_lines": len(se),
        "stdout_head": so[:max_lines],
        "stderr_head": se[:max_lines],
        "stdout_tail": so[-max_lines:] if len(so) > max_lines else [],
        "stderr_tail": se[-max_lines:] if len(se) > max_lines else [],
    }


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _parse_absorbed_fraction(stdout: str) -> Optional[float]:
    clean = _strip_ansi(stdout)
    m = re.search(r"absorbed:\s*([\d.]+)%", clean)
    if not m:
        return None
    return float(m.group(1)) / 100.0


def _guess_wavelength_nm(config_path: Path, session_id: str, regex: str) -> Optional[float]:
    target = f"{session_id} {config_path.name}"
    m = re.search(regex, target)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _load_config_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _render_absorption_video(
    config: Dict[str, Any],
    job_dir: Path,
    out_path: Path,
    fps: int,
    frame_axis: str,
) -> Optional[str]:
    try:
        import cv2
    except ImportError:
        return "opencv-python not available; skipping video"

    session_id = str(config.get("Session", {}).get("ID", "skin_mcx_sim"))
    mc2_path = job_dir / f"{session_id}.mc2"
    if not mc2_path.exists():
        return f"mc2 not found: {mc2_path.name}"

    dim = config.get("Domain", {}).get("Dim", None)
    if not isinstance(dim, list) or len(dim) != 3:
        return "Domain.Dim missing in config"

    x, y, z = int(dim[0]), int(dim[1]), int(dim[2])
    raw = np.fromfile(str(mc2_path), dtype=np.float32)
    expect = x * y * z
    if raw.size < expect:
        return f"mc2 too small: {raw.size} < expected {expect}"

    arr = raw[:expect].reshape((z, y, x))

    if frame_axis == "z":
        slices = [arr[i, :, :] for i in range(arr.shape[0])]
    elif frame_axis == "y":
        slices = [arr[:, i, :] for i in range(arr.shape[1])]
    else:
        slices = [arr[:, :, i] for i in range(arr.shape[2])]

    h, w = slices[0].shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, fps)),
        (int(w), int(h)),
        True,
    )

    for sl in slices:
        smin = float(np.min(sl))
        smax = float(np.max(sl))
        if smax > smin:
            norm = ((sl - smin) / (smax - smin) * 255.0).astype(np.uint8)
        else:
            norm = np.zeros_like(sl, dtype=np.uint8)
        frame = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        writer.write(frame)
    writer.release()
    return None


def _render_reflectance_plot(
    points: List[Dict[str, Any]],
    out_png: Path,
) -> Optional[str]:
    if len(points) < 2:
        return "need at least 2 wavelength points"
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not available"

    pts = [p for p in points if p.get("wavelength_nm") is not None and p.get("reflectance_est") is not None]
    if len(pts) < 2:
        return "insufficient valid wavelength/reflectance points"

    pts = sorted(pts, key=lambda p: float(p["wavelength_nm"]))
    xs = [float(p["wavelength_nm"]) for p in pts]
    ys = [float(p["reflectance_est"]) for p in pts]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(xs, ys, marker="o", linewidth=1.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Estimated Reflectance")
    ax.set_title("MCX Reflectance Spectrum (estimated from absorption)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150)
    plt.close(fig)
    return None


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch runner for MCX simulations.")
    ap.add_argument("configs", nargs="+", help="One or more mcx_config.json paths or glob patterns.")
    ap.add_argument("--mcx-binary", default="mcx", help="Path to MCX binary (default: mcx).")
    ap.add_argument("--output-dir", default="./mcx_batch_output", help="Base output directory.")
    ap.add_argument("--dry-run", action="store_true", help="Generate commands only; do not execute.")
    ap.add_argument("--timeout", type=int, default=3600, help="Per-job timeout in seconds.")
    ap.add_argument("--render-absorption-video", action="store_true", help="Render per-job MP4 from mc2 depth slices.")
    ap.add_argument("--video-fps", type=int, default=8, help="Video FPS when rendering absorption MP4.")
    ap.add_argument("--video-axis", choices=["z", "y", "x"], default="z", help="Slice axis for absorption video.")
    ap.add_argument(
        "--render-reflectance-spectrum",
        action="store_true",
        help="Render reflectance_spectrum.png from absorbed fraction across wavelength-tagged jobs.",
    )
    ap.add_argument(
        "--wavelength-regex",
        default=r"(\d+(?:\.\d+)?)nm",
        help="Regex used to parse wavelength from Session.ID/config filename; group(1) must be numeric.",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    if args.timeout <= 0:
        raise SystemExit(f"ERROR: --timeout must be > 0, got {args.timeout}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_files = _gather_configs(args.configs)
    if not config_files:
        raise SystemExit("No valid mcx_config.json files found. Exiting.")

    print(f"Found {len(config_files)} config file(s).")

    if not args.dry_run:
        resolved_mcx = shutil.which(args.mcx_binary)
        if resolved_mcx is None:
            mcx_path = Path(args.mcx_binary)
            if not mcx_path.exists():
                raise SystemExit(
                    f"MCX binary not found at '{args.mcx_binary}'. Use --dry-run or set --mcx-binary."
                )
            if not os.access(str(mcx_path), os.X_OK):
                raise SystemExit(f"MCX binary at '{args.mcx_binary}' is not executable.")
            resolved_mcx = str(mcx_path.resolve())
        args.mcx_binary = resolved_mcx

    jobs: List[Dict[str, Any]] = []
    spectrum_points: List[Dict[str, Any]] = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for i, cfg_path in enumerate(config_files):
        sim_out = output_dir / f"sim_{i:04d}"
        sim_out.mkdir(parents=True, exist_ok=True)

        config_json = _load_config_json(cfg_path)
        session_id = str(config_json.get("Session", {}).get("ID", cfg_path.stem))
        cmd = _build_mcx_command(cfg_path, args.mcx_binary)

        job: Dict[str, Any] = {
            "job_id": i,
            "config_path": str(cfg_path.resolve()),
            "session_id": session_id,
            "command": cmd,
            "output_dir": str(sim_out.resolve()),
            "status": "pending",
            "return_code": None,
            "summary": None,
            "error": None,
            "absorbed_fraction": None,
            "reflectance_est": None,
            "wavelength_nm": None,
            "absorption_video": None,
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
                    err_tail = proc.stderr.strip().splitlines()[-5:]
                    for line in err_tail:
                        print(f"         stderr: {line}")

                absorbed = _parse_absorbed_fraction(proc.stdout)
                if absorbed is not None:
                    job["absorbed_fraction"] = float(absorbed)
                    job["reflectance_est"] = float(max(0.0, min(1.0, 1.0 - absorbed)))
                wl = _guess_wavelength_nm(cfg_path, session_id, args.wavelength_regex)
                if wl is not None:
                    job["wavelength_nm"] = float(wl)

                if args.render_absorption_video and proc.returncode == 0:
                    video_path = sim_out / f"{session_id}_absorption.mp4"
                    err = _render_absorption_video(
                        config=config_json,
                        job_dir=sim_out,
                        out_path=video_path,
                        fps=int(args.video_fps),
                        frame_axis=str(args.video_axis),
                    )
                    if err is None:
                        job["absorption_video"] = str(video_path.resolve())
                    else:
                        job["absorption_video"] = f"skip: {err}"

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

        if job.get("wavelength_nm") is not None and job.get("reflectance_est") is not None:
            spectrum_points.append(
                {
                    "job_id": job["job_id"],
                    "wavelength_nm": job["wavelength_nm"],
                    "reflectance_est": job["reflectance_est"],
                    "config_path": job["config_path"],
                }
            )

        jobs.append(job)

    manifest: Dict[str, Any] = {
        "timestamp": timestamp,
        "dry_run": args.dry_run,
        "mcx_binary": args.mcx_binary,
        "num_jobs": len(jobs),
        "jobs": jobs,
    }

    if args.render_reflectance_spectrum and not args.dry_run:
        spectrum_png = output_dir / "reflectance_spectrum.png"
        err = _render_reflectance_plot(spectrum_points, spectrum_png)
        manifest["reflectance_spectrum"] = {
            "points": spectrum_points,
            "png": str(spectrum_png.resolve()) if err is None else None,
            "status": "ok" if err is None else f"skip: {err}",
        }

    manifest_path = output_dir / "mcx_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote batch manifest: {manifest_path}")

    if args.dry_run:
        print(f"\n=== Dry-run complete. {len(jobs)} command(s) generated. ===")
    else:
        completed = sum(1 for j in jobs if j["status"] == "completed")
        failed = sum(1 for j in jobs if j["status"] == "failed")
        timeout = sum(1 for j in jobs if j["status"] == "timeout")
        error = sum(1 for j in jobs if j["status"] == "error")
        print(f"\n=== Batch Summary: {completed} OK, {failed} failed, {timeout} timeout, {error} error ===")


if __name__ == "__main__":
    main()
