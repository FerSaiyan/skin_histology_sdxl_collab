#!/usr/bin/env python
"""
Explicit finite-difference solver for the Pennes bioheat equation on a 3D
thermal model grid.

Equation (Pennes, 1948):
    rho * c * dT/dt = div(k * grad T) + wb * rho_b * c_b * (T_b - T)
                      + Q_met + Q_ext

  where:
    rho      — density              [kg/m³]
    c        — specific heat        [J/(kg·K)]
    k        — thermal conductivity  [W/(m·K)]
    wb       — blood perfusion rate  [1/s]
    rho_b    — blood density         [kg/m³]  (1060)
    c_b      — blood specific heat   [J/(kg·K)] (3617)
    T_b      — arterial blood temp   [°C]
    Q_met    — metabolic heat        [W/m³]
    Q_ext    — external heat source  [W/m³]

Boundary condition: Dirichlet (fixed at initial temperature) on all faces.

Input:
  --model-npz  from :py:mod:`thermal_build_model` (contains rho, c, k, wb,
                     qmet, label_volume arrays).

Source modes:
  - ``none``       : no external heating.
  - ``spherical``  : uniform power density inside a sphere of radius
                     ``--source-radius-vox`` centered at ``--source-center``.
                     *Coordinates are (z, y, x) in voxel indices, matching
                     numpy array ordering (axis 0 = Z, axis 1 = Y, axis 2 = X).*

Outputs (in ``--output-dir``):
  - ``temperature_final.npy``          — final 3D temperature array [°C]
  - ``temperature_timeseries_summary.json`` — min/max/mean per checkpoint step
  - ``thermal_solve_manifest.json``    — full provenance + parameters
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BLOOD_RHO = 1060.0  # kg/m³
BLOOD_C = 3617.0  # J/(kg·K)

_DEFAULT_DT = 0.01  # seconds
_DEFAULT_NUM_STEPS = 100
_DEFAULT_INITIAL_TEMP = 37.0  # °C
_DEFAULT_BLOOD_TEMP = 37.0  # °C
_CFL_FACTOR = 0.5  # safety factor for explicit stability


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_thermal_model(model_npz: str) -> Dict[str, np.ndarray]:
    """Load thermal model arrays from a .npz produced by thermal_build_model."""
    p = Path(model_npz)
    if not p.exists():
        raise SystemExit(f"Model .npz not found: {model_npz}")

    data = np.load(str(p))
    required = ["rho", "c", "k", "wb", "qmet", "label_volume"]
    missing = [k for k in required if k not in data]
    if missing:
        raise SystemExit(
            f"Model .npz missing required arrays: {missing}. "
            f"Found: {list(data.keys())}"
        )

    return {k: data[k] for k in required}


# ---------------------------------------------------------------------------
# Spherical source
# ---------------------------------------------------------------------------


def _build_spherical_source(
    shape: Tuple[int, int, int],
    center: Tuple[float, float, float],
    radius_vox: float,
    power_w_m3: float,
) -> np.ndarray:
    """Create a 3D heat source array with uniform power inside a sphere.

    Args:
        shape: (Z, Y, X) grid dimensions.
        center: (cz, cy, cx) in voxel index coordinates (floats allowed).
        radius_vox: sphere radius in voxel units.
        power_w_m3: volumetric power density inside the sphere [W/m³].

    Returns:
        Float64 array of the same shape, zero outside the sphere.
    """
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0], dtype=np.float64),
        np.arange(shape[1], dtype=np.float64),
        np.arange(shape[2], dtype=np.float64),
        indexing="ij",
    )
    dist = np.sqrt(
        (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    )
    source = np.where(dist <= radius_vox, power_w_m3, 0.0)
    return source


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


def _compute_stable_dt(
    rho: np.ndarray,
    c: np.ndarray,
    k: np.ndarray,
    wb: np.ndarray,
    voxel_size_mm: float,
) -> float:
    """Estimate the maximum stable time step using Fourier-number criterion.

    For the 3D explicit scheme:
        dt_stable ≤ (rho * c) / (6 * k / h² + wb * rho_b * c_b)

    where h = voxel_size_mm * 1e-3 (converted to metres).
    Returns the minimum across all voxels, divided by the CFL safety factor.
    """
    h = voxel_size_mm * 1e-3  # metres
    h2 = h * h

    # Avoid division by zero in air (k ≈ 0, rho small)
    denom = 6.0 * k / h2 + wb * BLOOD_RHO * BLOOD_C
    # Mask voxels with negligible thermal activity
    active = denom > 1e-12
    if not active.any():
        return 1.0  # no active tissue — arbitrary large

    dt_candidates = np.full_like(rho, fill_value=np.inf, dtype=np.float64)
    dt_candidates[active] = (rho[active] * c[active]) / denom[active]

    dt_min = float(dt_candidates.min())
    # Guard against infinities from perfect insulators
    if not np.isfinite(dt_min):
        finite = dt_candidates[np.isfinite(dt_candidates)]
        dt_min = float(finite.min()) if len(finite) > 0 else 1.0

    return dt_min * _CFL_FACTOR


def _apply_boundary_conditions(T: np.ndarray, T_fixed: float) -> np.ndarray:
    """Enforce Dirichlet BC: fixed temperature on all faces.

    Modifies the array in-place (and returns it for convenience).
    """
    T[0, :, :] = T_fixed
    T[-1, :, :] = T_fixed
    T[:, 0, :] = T_fixed
    T[:, -1, :] = T_fixed
    T[:, :, 0] = T_fixed
    T[:, :, -1] = T_fixed
    return T


def solve_pennes(
    rho: np.ndarray,
    c: np.ndarray,
    k: np.ndarray,
    wb: np.ndarray,
    qmet: np.ndarray,
    source: np.ndarray,
    dt: float,
    num_steps: int,
    initial_temp: float,
    blood_temp: float,
    voxel_size_mm: float = 1.0,
    checkpoint_interval: int = 10,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Explicit finite-difference Pennes solver.

    Args:
        rho, c, k, wb, qmet: property arrays (all same shape ZxYxX).
        source: external heat source array [W/m³].
        dt: time step [s].
        num_steps: number of time steps.
        initial_temp: initial and boundary temperature [°C].
        blood_temp: arterial blood temperature [°C].
        voxel_size_mm: isotropic voxel side length [mm].
        checkpoint_interval: log summary every N steps.

    Returns:
        (T_final, timeseries) where timeseries is a list of dicts with
        step, time, min, max, mean.
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if num_steps < 0:
        raise ValueError(f"num_steps must be >= 0, got {num_steps}")
    if voxel_size_mm <= 0:
        raise ValueError(f"voxel_size_mm must be > 0, got {voxel_size_mm}")
    if checkpoint_interval <= 0:
        raise ValueError(
            f"checkpoint_interval must be > 0, got {checkpoint_interval}"
        )

    shape = rho.shape
    for name, arr in {
        "c": c,
        "k": k,
        "wb": wb,
        "qmet": qmet,
        "source": source,
    }.items():
        if arr.shape != shape:
            raise ValueError(
                f"Array shape mismatch: {name} has shape {arr.shape}, expected {shape}"
            )

    h = voxel_size_mm * 1e-3  # metres

    # --- Stability check ---
    stable_dt = _compute_stable_dt(rho, c, k, wb, voxel_size_mm)
    if dt > stable_dt:
        warnings.warn(
            f"  [WARN] Requested dt={dt:.6e}s exceeds stable estimate "
            f"dt_max≈{stable_dt:.6e}s.  Solution may become unstable."
        )

    # --- Initialise temperature ---
    T = np.full(shape, initial_temp, dtype=np.float64)
    _apply_boundary_conditions(T, initial_temp)

    # Pre-compute scaled properties for efficiency
    # dT/dt term coefficient: dt / (rho * c)
    inv_rho_c = dt / (rho * c + 1e-30)  # avoid division by zero in air

    wb_scaled = wb * BLOOD_RHO * BLOOD_C  # perfusion coupling coefficient
    # Perfusion contribution per dt
    perf_factor = dt * wb_scaled / (rho * c + 1e-30)

    # Source + metabolic summed
    total_source = source + qmet
    src_factor = inv_rho_c * total_source  # pre-multiply by dt/(rho*c)

    # Store timeseries summary
    timeseries: List[Dict[str, Any]] = []
    h2 = 1.0 / (h * h)

    # Helper arrays for diffusion
    kc = k.copy()  # conductivity array (will not mutate)

    for step in range(num_steps):
        # --- Diffusion: div(k * grad T) ---
        # Central differences with harmonic-average-like interface conductivity
        # We use arithmetic mean of adjacent k values for interface conductivity.

        # Z-gradient
        k_z_l = (kc[:-2, 1:-1, 1:-1] + kc[1:-1, 1:-1, 1:-1]) * 0.5
        k_z_r = (kc[1:-1, 1:-1, 1:-1] + kc[2:, 1:-1, 1:-1]) * 0.5
        dT_z = k_z_l * (T[1:-1, 1:-1, 1:-1] - T[:-2, 1:-1, 1:-1]) + k_z_r * (
            T[2:, 1:-1, 1:-1] - T[1:-1, 1:-1, 1:-1]
        )

        # Y-gradient
        k_y_l = (kc[1:-1, :-2, 1:-1] + kc[1:-1, 1:-1, 1:-1]) * 0.5
        k_y_r = (kc[1:-1, 1:-1, 1:-1] + kc[1:-1, 2:, 1:-1]) * 0.5
        dT_y = k_y_l * (T[1:-1, 1:-1, 1:-1] - T[1:-1, :-2, 1:-1]) + k_y_r * (
            T[1:-1, 2:, 1:-1] - T[1:-1, 1:-1, 1:-1]
        )

        # X-gradient
        k_x_l = (kc[1:-1, 1:-1, :-2] + kc[1:-1, 1:-1, 1:-1]) * 0.5
        k_x_r = (kc[1:-1, 1:-1, 1:-1] + kc[1:-1, 1:-1, 2:]) * 0.5
        dT_x = k_x_l * (T[1:-1, 1:-1, 1:-1] - T[1:-1, 1:-1, :-2]) + k_x_r * (
            T[1:-1, 1:-1, 2:] - T[1:-1, 1:-1, 1:-1]
        )

        laplacian = (dT_z + dT_y + dT_x) * h2

        # --- Pennes update (interior only) ---
        T_int = T[1:-1, 1:-1, 1:-1]

        # Diffusion contribution
        diffusion_term = inv_rho_c[1:-1, 1:-1, 1:-1] * laplacian

        # Perfusion contribution
        perfusion_term = perf_factor[1:-1, 1:-1, 1:-1] * (blood_temp - T_int)

        # Source + metabolic contribution
        src_term = src_factor[1:-1, 1:-1, 1:-1]

        T_new_int = T_int + diffusion_term + perfusion_term + src_term

        # Update interior
        T[1:-1, 1:-1, 1:-1] = T_new_int

        # Re-apply boundary conditions
        _apply_boundary_conditions(T, initial_temp)

        # --- NaN/Inf check ---
        if not np.isfinite(T).all():
            bad_count = int(np.logical_not(np.isfinite(T)).sum())
            raise SystemExit(
                f"NaN or Inf detected at step {step + 1} ({bad_count} voxels). "
                "Solution diverged.  Reduce dt, check property values, "
                "or verify voxel_size_mm."
            )

        # --- Checkpoint summary ---
        if (step + 1) % checkpoint_interval == 0 or step == num_steps - 1:
            elapsed_time = (step + 1) * dt
            timeseries.append(
                {
                    "step": step + 1,
                    "time_s": round(elapsed_time, 6),
                    "min_degC": float(round(T.min(), 4)),
                    "max_degC": float(round(T.max(), 4)),
                    "mean_degC": float(round(T.mean(), 4)),
                }
            )
            print(
                f"  Step {step + 1:6d} / {num_steps}  "
                f"t={elapsed_time:.4f}s  "
                f"T [{T.min():.2f}, {T.max():.2f}]  mean={T.mean():.2f} °C"
            )

    return T, timeseries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Solve the Pennes bioheat equation on a 3D thermal model grid "
            "using explicit finite differences."
        ),
    )
    ap.add_argument(
        "--model-npz",
        required=True,
        help="Path to thermal_model.npz from thermal_build_model.py.",
    )
    ap.add_argument(
        "--output-dir",
        default="./thermal_solve_output",
        help="Output directory (default: ./thermal_solve_output).",
    )
    ap.add_argument(
        "--dt",
        type=float,
        default=_DEFAULT_DT,
        help=f"Time step in seconds (default: {_DEFAULT_DT}).",
    )
    ap.add_argument(
        "--num-steps",
        type=int,
        default=_DEFAULT_NUM_STEPS,
        help=f"Number of time steps (default: {_DEFAULT_NUM_STEPS}).",
    )
    ap.add_argument(
        "--initial-temp",
        type=float,
        default=_DEFAULT_INITIAL_TEMP,
        help=f"Initial and boundary temperature in °C (default: {_DEFAULT_INITIAL_TEMP}).",
    )
    ap.add_argument(
        "--blood-temp",
        type=float,
        default=_DEFAULT_BLOOD_TEMP,
        help=f"Arterial blood temperature in °C (default: {_DEFAULT_BLOOD_TEMP}).",
    )
    ap.add_argument(
        "--source-mode",
        default="none",
        choices=["none", "spherical"],
        help="External heat source mode (default: none).",
    )
    ap.add_argument(
        "--source-center",
        type=float,
        nargs=3,
        default=None,
        metavar=("CZ", "CY", "CX"),
        help="Source center (Z Y X) in voxel indices for spherical source. "
        "Axis ordering matches numpy array: axis 0 = Z, 1 = Y, 2 = X. "
        "(default: volume center).",
    )
    ap.add_argument(
        "--source-radius-vox",
        type=float,
        default=3.0,
        help="Sphere radius in voxels for spherical source (default: 3.0).",
    )
    ap.add_argument(
        "--source-power",
        type=float,
        default=1e6,
        help="Volumetric power density in W/m³ inside the source "
        "(default: 1e6 ≈ 1 MW/m³).",
    )
    ap.add_argument(
        "--voxel-size-mm",
        type=float,
        default=1.0,
        help="Isotropic voxel side length in mm (default: 1.0).",
    )
    ap.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Log temperature summary every N steps (default: 10).",
    )
    return ap


def main() -> None:
    ap = _build_cli()
    args = ap.parse_args()

    if args.dt <= 0:
        raise SystemExit(f"--dt must be > 0, got {args.dt}")
    if args.num_steps < 0:
        raise SystemExit(f"--num-steps must be >= 0, got {args.num_steps}")
    if args.voxel_size_mm <= 0:
        raise SystemExit(
            f"--voxel-size-mm must be > 0, got {args.voxel_size_mm}"
        )
    if args.checkpoint_interval <= 0:
        raise SystemExit(
            f"--checkpoint-interval must be > 0, got {args.checkpoint_interval}"
        )
    if args.source_radius_vox < 0:
        raise SystemExit(
            f"--source-radius-vox must be >= 0, got {args.source_radius_vox}"
        )
    if args.source_power <= 0:
        raise SystemExit(
            f"--source-power must be > 0, got {args.source_power}"
        )
    if args.source_mode == "spherical" and args.source_radius_vox == 0:
        print(
            "  [INFO] Spherical source with radius=0 produces no external heating."
        )

    # --- Load model ---
    print(f"Loading thermal model: {args.model_npz}")
    model = _load_thermal_model(args.model_npz)
    shape = model["rho"].shape
    print(f"  Shape: {shape}")

    # --- Build source ---
    if args.source_mode == "none":
        source = np.zeros(shape, dtype=np.float64)
        print("  Source mode: none")
    elif args.source_mode == "spherical":
        center = args.source_center
        if center is None:
            # Default to volume centre
            center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
            print(
                f"  Source center defaulting to volume center: "
                f"(Z={center[0]}, Y={center[1]}, X={center[2]})"
            )
        else:
            print(
                f"  Source center (Z,Y,X): ({center[0]}, {center[1]}, {center[2]})"
            )
        source = _build_spherical_source(
            shape=shape,
            center=center,
            radius_vox=args.source_radius_vox,
            power_w_m3=args.source_power,
        )
        within = int((source > 0).sum())
        print(
            f"  Spherical source: radius={args.source_radius_vox} vox, "
            f"power={args.source_power:.4e} W/m³, "
            f"{within} voxels active"
        )
    else:
        raise SystemExit(f"Unknown source mode: {args.source_mode}")

    # --- Solve ---
    print(
        f"\nSolving Pennes bioheat equation: "
        f"dt={args.dt}s, steps={args.num_steps}, "
        f"total_time={args.dt * args.num_steps:.2f}s"
    )
    t0 = time.time()
    T_final, timeseries = solve_pennes(
        rho=model["rho"],
        c=model["c"],
        k=model["k"],
        wb=model["wb"],
        qmet=model["qmet"],
        source=source,
        dt=args.dt,
        num_steps=args.num_steps,
        initial_temp=args.initial_temp,
        blood_temp=args.blood_temp,
        voxel_size_mm=args.voxel_size_mm,
        checkpoint_interval=args.checkpoint_interval,
    )
    wall_time = time.time() - t0
    print(f"\n  Solver finished in {wall_time:.2f}s (wall clock)")

    # --- Write outputs ---
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Final temperature
    temp_out = out / "temperature_final.npy"
    np.save(str(temp_out), T_final)
    print(f"  Wrote: {temp_out}  shape={T_final.shape}")

    # Timeseries summary
    ts_out = out / "temperature_timeseries_summary.json"
    ts_out.write_text(json.dumps(timeseries, indent=2), encoding="utf-8")
    print(f"  Wrote: {ts_out}")

    # Manifest
    manifest: Dict[str, Any] = {
        "source_model_npz": str(Path(args.model_npz).resolve()),
        "output_dir": str(out.resolve()),
        "solver": "explicit_fd_pennes_3d",
        "parameters": {
            "dt_s": args.dt,
            "num_steps": args.num_steps,
            "total_time_s": args.dt * args.num_steps,
            "initial_temp_degC": args.initial_temp,
            "blood_temp_degC": args.blood_temp,
            "source_mode": args.source_mode,
            "source_center_z_y_x": list(args.source_center)
            if args.source_center
            else None,
            "source_radius_vox": args.source_radius_vox,
            "source_power_W_m3": args.source_power,
            "voxel_size_mm": args.voxel_size_mm,
            "checkpoint_interval": args.checkpoint_interval,
        },
        "volume_shape": list(shape),
        "wall_clock_s": round(wall_time, 3),
        "temperature_range_degC": [
            float(round(float(T_final.min()), 4)),
            float(round(float(T_final.max()), 4)),
        ],
        "temperature_mean_degC": float(round(float(T_final.mean()), 4)),
        "artifacts": {
            "temperature_final.npy": str(temp_out.resolve()),
            "temperature_timeseries_summary.json": str(ts_out.resolve()),
        },
    }
    manifest_out = out / "thermal_solve_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  Wrote: {manifest_out}")

    # --- Final summary ---
    print(f"\n=== Thermal Solve Complete ===")
    print(f"  Final T range: [{T_final.min():.4f}, {T_final.max():.4f}] °C")
    print(f"  Final T mean : {T_final.mean():.4f} °C")
    print(f"  Steps        : {args.num_steps}")
    print(f"  Wall time    : {wall_time:.2f}s")
    print(f"  Artifacts in : {out.resolve()}")


if __name__ == "__main__":
    main()
