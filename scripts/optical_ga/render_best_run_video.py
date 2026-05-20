#!/usr/bin/env python
"""
Render a best-run video for optical GA results (MCX or PyXOpto branch).

Reads a run folder containing per-generation history (ga_history.csv) and
optional best_genome.json / spectra files.  Produces per-frame PNGs and,
if ffmpeg is available, an MP4 video.

Each frame shows:
- Generation index (title)
- Target colour swatch (from best_genome.json target Lab)
- Current best colour swatch (from that generation's best Lab)
- ΔE between target and current best
- Convergence curve (best/mean fitness over generations)
- Optional reflectance spectrum subplot (if spectra directory found)

Usage::

    python scripts/optical_ga/render_best_run_video.py \\
        --run-dir outputs/optical_ga/my_run \\
        --output-dir outputs/optical_ga/my_run/video

    # Smoke test with synthetic data:
    python scripts/optical_ga/render_best_run_video.py \\
        --run-dir /tmp/opt_smoke --output-dir /tmp/opt_vid \\
        --fps 2 --keep-frames
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure repo root is on sys.path for internal imports
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ---------------------------------------------------------------------------
# Matplotlib import (delayed until render to allow headless backend)
# ---------------------------------------------------------------------------

_HAS_MPL = False
try:
    import matplotlib
    matplotlib.use("Agg")  # headless backend, must be set before pyplot
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgb

    _HAS_MPL = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _lab_to_rgb(L: float, a: float, b: float) -> Tuple[float, float, float]:
    """Convert CIE L*a*b* to sRGB (approximate, no colour-science dep).

    Uses a simple linear transform through XYZ with D65 illuminant.
    Returns (R, G, B) in [0, 1] range.
    """
    # Reference white D65
    Xn, Yn, Zn = 95.047, 100.0, 108.883

    # Lab → XYZ (reverse of CIE76)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    def _f_inv(t: float) -> float:
        delta = 6.0 / 29.0
        if t > delta:
            return t ** 3
        else:
            return 3.0 * delta ** 2 * (t - 4.0 / 29.0)

    X = Xn * _f_inv(fx)
    Y = Yn * _f_inv(fy)
    Z = Zn * _f_inv(fz)

    # XYZ → linear sRGB (D65)
    r_lin = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    # Gamma correction
    def _gamma(u: float) -> float:
        if u > 0.0031308:
            return 1.055 * (u ** (1.0 / 2.4)) - 0.055
        else:
            return 12.92 * u

    return (
        float(np.clip(_gamma(r_lin / 100.0), 0, 1)),
        float(np.clip(_gamma(g_lin / 100.0), 0, 1)),
        float(np.clip(_gamma(b_lin / 100.0), 0, 1)),
    )


def _delta_e76(L1: float, a1: float, b1: float,
               L2: float, a2: float, b2: float) -> float:
    """CIE76 ΔE colour difference."""
    return float(np.sqrt((L1 - L2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_target_lab(run_dir: Path) -> Optional[Tuple[float, float, float]]:
    """Load target Lab from best_genome.json component_scores."""
    best_path = run_dir / "best_genome.json"
    if not best_path.exists():
        return None
    try:
        data = json.loads(best_path.read_text(encoding="utf-8"))
        comps = data.get("component_scores", {})
        L = comps.get("L_target")
        a = comps.get("a_target")
        b = comps.get("b_target")
        if L is not None and a is not None and b is not None:
            return (float(L), float(a), float(b))
    except Exception:
        pass
    return None


def _load_history(run_dir: Path) -> Optional[pd.DataFrame]:
    """Load ga_history.csv."""
    path = run_dir / "ga_history.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "generation" not in df.columns or "best_fitness" not in df.columns:
            return None
        return df
    except Exception:
        return None


def _load_spectra(run_dir: Path) -> Optional[List[Dict[str, Any]]]:
    """Load reflectance spectra from spectra/ directory if present.

    Expects .npy or .json files in a 'spectra' subdirectory.
    Returns a list of {generation, wavelengths, reflectance} dicts.
    """
    spec_dir = run_dir / "spectra"
    if not spec_dir.is_dir():
        return None

    spectra: List[Dict[str, Any]] = []
    for fpath in sorted(spec_dir.iterdir()):
        if fpath.suffix == ".npy":
            try:
                arr = np.load(str(fpath))
                # Infer generation from filename stem
                stem = fpath.stem
                gen = None
                for part in stem.replace("-", "_").split("_"):
                    if part.isdigit():
                        gen = int(part)
                        break
                spectra.append({
                    "generation": gen or 0,
                    "source": str(fpath),
                    "wavelengths": list(range(380, 781, 5)),
                    "reflectance": arr.tolist() if isinstance(arr, np.ndarray) else arr,
                })
            except Exception:
                pass
        elif fpath.suffix == ".json":
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                wl = data.get("wavelengths", list(range(380, 781, 5)))
                ref = data.get("reflectance", data.get("spectrum", []))
                gen = data.get("generation", 0)
                if ref:
                    spectra.append({
                        "generation": int(gen),
                        "source": str(fpath),
                        "wavelengths": wl,
                        "reflectance": ref,
                    })
            except Exception:
                pass

    return spectra if spectra else None


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------


def _draw_swatch(ax, rgb: Tuple[float, float, float],
                 label: str, edgecolor: str = "black") -> None:
    """Draw a filled colour swatch rectangle with a label."""
    ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, color=rgb,
                                     ec=edgecolor, lw=2))
    ax.text(0.5, -0.1, label, ha="center", va="top",
            fontsize=9, transform=ax.transData)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor(edgecolor)
        spine.set_linewidth(2)


def render_frame(
    generation: int,
    num_generations: int,
    target_lab: Tuple[float, float, float],
    best_lab: Tuple[float, float, float],
    best_fitness: float,
    mean_fitness: float,
    hist_df: pd.DataFrame,
    spectra_data: Optional[List[Dict[str, Any]]],
    figsize: Tuple[float, float],
    output_path: Path,
) -> None:
    """Render a single frame and save to output_path.

    Parameters
    ----------
    generation : int
        Current generation index (0-based).
    num_generations : int
        Total number of generations (for title).
    target_lab : (float, float, float)
        Target L*a*b*.
    best_lab : (float, float, float)
        Current best L*a*b* for this generation.
    best_fitness : float
        Best fitness for this generation.
    mean_fitness : float
        Mean fitness for this generation.
    hist_df : pd.DataFrame
        Full GA history with 'generation', 'best_fitness', 'mean_fitness' columns.
    spectra_data : list of dict or None
        Optional spectra data.
    figsize : (float, float)
        Figure size in inches.
    output_path : Path
        Where to save the PNG.
    """
    target_rgb = _lab_to_rgb(*target_lab)
    best_rgb = _lab_to_rgb(*best_lab)

    delta_e = _delta_e76(*target_lab, *best_lab)

    # Determine grid layout
    has_spectra = spectra_data is not None and len(spectra_data) > 0
    if has_spectra:
        fig, axes = plt.subplots(2, 3, figsize=figsize,
                                 gridspec_kw={"height_ratios": [1, 1]})
        ax_swatch_tgt = axes[0, 0]
        ax_swatch_best = axes[0, 1]
        ax_de = axes[0, 2]
        ax_conv = axes[1, 0]
        ax_conv2 = axes[1, 1]
        ax_spec = axes[1, 2]
    else:
        fig, axes = plt.subplots(2, 3, figsize=figsize,
                                 gridspec_kw={"height_ratios": [1, 1]})
        ax_swatch_tgt = axes[0, 0]
        ax_swatch_best = axes[0, 1]
        ax_de = axes[0, 2]
        ax_conv = axes[1, 0]
        ax_conv2 = axes[1, 1]
        ax_spec = axes[1, 2]

    # ---- Title ----
    fig.suptitle(
        f"Generation {generation + 1} / {num_generations}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ---- Target swatch ----
    _draw_swatch(ax_swatch_tgt, target_rgb,
                 f"Target\nL*={target_lab[0]:.1f} "
                 f"a*={target_lab[1]:.1f} "
                 f"b*={target_lab[2]:.1f}",
                 edgecolor="black")

    # ---- Current best swatch ----
    _draw_swatch(ax_swatch_best, best_rgb,
                 f"Best\nL*={best_lab[0]:.1f} "
                 f"a*={best_lab[1]:.1f} "
                 f"b*={best_lab[2]:.1f}",
                 edgecolor="blue")

    # ---- ΔE panel ----
    ax_de.text(0.5, 0.6, f"ΔE = {delta_e:.2f}", ha="center", va="center",
              fontsize=28, fontweight="bold")
    ax_de.text(0.5, 0.3, f"Fitness = {best_fitness:.4f}", ha="center",
              va="center", fontsize=14)
    ax_de.set_xlim(0, 1)
    ax_de.set_ylim(0, 1)
    ax_de.set_xticks([])
    ax_de.set_yticks([])
    ax_de.set_title("Colour Difference", fontsize=10)

    # ---- Convergence curve (best_fitness) ----
    gens = hist_df["generation"].values
    bests = hist_df["best_fitness"].values
    means = hist_df["mean_fitness"].values

    ax_conv.plot(gens, bests, "b-o", markersize=4, label="Best fitness")
    ax_conv.plot(gens, means, "r--s", markersize=3, label="Mean fitness")
    ax_conv.axvline(x=generation, color="gray", linestyle=":",
                    alpha=0.7, label="Current")
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Fitness")
    ax_conv.set_title("Fitness Convergence", fontsize=10)
    ax_conv.legend(fontsize=7, loc="lower right")
    ax_conv.grid(True, alpha=0.3)

    # ---- second convergence panel (Lab traces) ----
    if "best_L" in hist_df.columns:
        ax_conv2.plot(gens, hist_df["best_L"].values, "g-o",
                      markersize=4, label="L*")
    if "best_a" in hist_df.columns:
        ax_conv2.plot(gens, hist_df["best_a"].values, "m--s",
                      markersize=3, label="a*")
    if "best_b" in hist_df.columns:
        ax_conv2.plot(gens, hist_df["best_b"].values, "c--^",
                      markersize=3, label="b*")
    ax_conv2.axvline(x=generation, color="gray", linestyle=":",
                     alpha=0.7, label="Current")
    ax_conv2.set_xlabel("Generation")
    ax_conv2.set_ylabel("Lab value")
    ax_conv2.set_title("Best Lab Traces", fontsize=10)
    ax_conv2.legend(fontsize=7, loc="best")
    ax_conv2.grid(True, alpha=0.3)

    # ---- Optional spectra panel ----
    if has_spectra:
        for spec in spectra_data:
            wl = np.array(spec.get("wavelengths", []), dtype=float)
            ref = np.array(spec.get("reflectance", []), dtype=float)
            if len(wl) > 0 and len(ref) > 0:
                gen_s = spec.get("generation", 0)
                label_s = f"Gen {gen_s}"
                ax_spec.plot(wl, ref, label=label_s, alpha=0.8)
        ax_spec.set_xlabel("Wavelength (nm)")
        ax_spec.set_ylabel("Reflectance")
        ax_spec.set_title("Reflectance Spectra", fontsize=10)
        ax_spec.legend(fontsize=7, loc="best")
        ax_spec.grid(True, alpha=0.3)
        ax_spec.set_xlim(380, 780)
    else:
        ax_spec.text(0.5, 0.5, "No spectra data", ha="center", va="center",
                     fontsize=12, color="gray")
        ax_spec.set_xticks([])
        ax_spec.set_yticks([])
        ax_spec.set_title("Reflectance Spectra", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Video encoding
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is on PATH."""
    return shutil.which("ffmpeg") is not None


def encode_mp4(frame_dir: Path, output_path: Path, fps: int,
               frame_size: Optional[Tuple[int, int]] = None) -> bool:
    """Encode frames into an MP4 video using ffmpeg.

    Parameters
    ----------
    frame_dir : Path
        Directory containing frame_0000.png, frame_0001.png, ...
    output_path : Path
        Destination for the MP4 file.
    fps : int
        Frames per second.
    frame_size : tuple of (width, height) or None
        Optional explicit frame size.  If None, ffmpeg auto-detects.

    Returns
    -------
    success : bool
    """
    if not _ffmpeg_available():
        print("INFO: ffmpeg not found on PATH — cannot encode MP4.",
              file=sys.stderr)
        return False

    pattern = str(frame_dir / "frame_%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
    ]
    if frame_size is not None:
        cmd.extend(["-s", f"{frame_size[0]}x{frame_size[1]}"])

    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: ffmpeg failed: {exc.stderr}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("ERROR: ffmpeg binary not found.", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main rendering logic
# ---------------------------------------------------------------------------


def render_best_run_video(
    run_dir: Path,
    output_dir: Path,
    fps: int = 5,
    frame_size: Optional[Tuple[int, int]] = None,
    overwrite: bool = False,
    keep_frames: bool = False,
) -> int:
    """Render video for a single GA run directory.

    Returns 0 on success, 1 on failure.
    """
    # ---- Validate run directory ----
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    # ---- Load data ----
    history = _load_history(run_dir)
    if history is None:
        print(f"ERROR: no valid ga_history.csv in {run_dir}", file=sys.stderr)
        return 1

    target_lab = _load_target_lab(run_dir)
    if target_lab is None:
        print("WARNING: target Lab not found in best_genome.json. "
              "Using last generation's best Lab as target.", file=sys.stderr)
        last_best_l = history["best_L"].iloc[-1]
        last_best_a = history["best_a"].iloc[-1]
        last_best_b = history["best_b"].iloc[-1]
        target_lab = (float(last_best_l), float(last_best_a), float(last_best_b))

    spectra_data = _load_spectra(run_dir)

    num_generations = len(history)
    gens = history["generation"].values

    # ---- Prepare output ----
    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    if frame_dir.exists():
        if not overwrite:
            print(f"ERROR: frame directory exists: {frame_dir} "
                  f"(use --overwrite to replace)", file=sys.stderr)
            return 1
        else:
            shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure size ----
    if frame_size is not None:
        figsize = (frame_size[0] / 100.0, frame_size[1] / 100.0)
    else:
        figsize = (16.0, 9.0)

    # ---- Render frames ----
    print(f"Rendering {num_generations} frames to {frame_dir}/ ...")
    for idx in tqdm(range(num_generations), desc="Frames", unit="frame"):
        gen = int(gens[idx])
        best_l = float(history["best_L"].iloc[idx])
        best_a = float(history["best_a"].iloc[idx])
        best_b = float(history["best_b"].iloc[idx])
        best_fit = float(history["best_fitness"].iloc[idx])
        mean_fit = float(history["mean_fitness"].iloc[idx])

        frame_path = frame_dir / f"frame_{idx:04d}.png"
        render_frame(
            generation=gen,
            num_generations=num_generations,
            target_lab=target_lab,
            best_lab=(best_l, best_a, best_b),
            best_fitness=best_fit,
            mean_fitness=mean_fit,
            hist_df=history,
            spectra_data=spectra_data,
            figsize=figsize,
            output_path=frame_path,
        )

    # ---- Encode MP4 ----
    mp4_path = output_dir / "best_run_video.mp4"
    if _ffmpeg_available():
        print("Encoding MP4 ...")
        success = False
        with tqdm(total=1, desc="Encoding") as pbar:
            success = encode_mp4(frame_dir, mp4_path, fps, frame_size)
            pbar.update(1)

        if success:
            print(f"Video saved: {mp4_path}")
        else:
            print("WARNING: MP4 encoding failed.  Frames preserved.",
                  file=sys.stderr)
    else:
        print("INFO: ffmpeg not found — frames preserved, no MP4 generated.",
              file=sys.stderr)

    # ---- Cleanup frames ----
    if not keep_frames:
        print("Cleaning up frames ...")
        shutil.rmtree(frame_dir)
        print(f"Removed: {frame_dir}/")
    else:
        print(f"Frames kept: {frame_dir}/")

    print(f"Done.  Output directory: {output_dir}/")
    return 0


# ---------------------------------------------------------------------------
# Synthetic data generator (for smoke testing)
# ---------------------------------------------------------------------------


def _generate_synthetic_run(output_dir: Path, num_generations: int = 10) -> Path:
    """Create a synthetic GA run directory for smoke testing.

    Returns the path to the run directory.
    """
    run_dir = output_dir / "synthetic_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    target_lab = (65.0, 12.0, 18.0)

    # Create history CSV with realistic-looking convergence
    rng = np.random.default_rng(42)
    n = num_generations
    generations = np.arange(n)
    best_fitness = 0.5 + 2.5 * (1 - np.exp(-generations / 3.0)) + 0.05 * rng.normal(size=n)
    best_fitness = np.clip(best_fitness, 0.1, 5.0)
    mean_fitness = best_fitness - 0.3 * (1 + 0.5 * rng.random(n))
    worst_fitness = best_fitness - 0.5 * (1 + rng.random(n))
    median_fitness = best_fitness - 0.15 * (1 + 0.3 * rng.random(n))
    std_fitness = 0.1 + 0.3 * rng.random(n)
    diversity = 1.5 * np.exp(-generations / 5.0) + 0.1 * rng.random(n)
    elapsed_s = 0.05 + 0.02 * rng.random(n)

    # Lab values converging toward target
    best_L = target_lab[0] + (80 - target_lab[0]) * np.exp(-generations / 4.0) \
        + 2 * rng.normal(size=n)
    best_a = target_lab[1] + (5 - target_lab[1]) * np.exp(-generations / 4.0) \
        + 1.5 * rng.normal(size=n)
    best_b = target_lab[2] + (25 - target_lab[2]) * np.exp(-generations / 4.0) \
        + 2 * rng.normal(size=n)

    # Genome parameter names (core 19)
    pnames = [
        "melanin", "blood_layer1", "blood_layer2", "spo2",
        "g_layer0", "g_layer1", "g_layer2",
        "d_layer0", "d_layer1",
        "amp_layer0", "amp_layer1", "amp_layer2",
        "water_layer0", "water_layer1", "water_layer2",
        "fat_layer2",
        "n_mult_layer0", "n_mult_layer1", "n_mult_layer2",
    ]

    fieldnames = [
        "generation",
        "best_fitness", "mean_fitness", "worst_fitness",
        "median_fitness", "std_fitness", "diversity", "elapsed_s",
        "best_L", "best_a", "best_b",
    ] + [f"best_{n}" for n in pnames]

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        row = {
            "generation": int(generations[i]),
            "best_fitness": f"{best_fitness[i]:.6f}",
            "mean_fitness": f"{mean_fitness[i]:.6f}",
            "worst_fitness": f"{worst_fitness[i]:.6f}",
            "median_fitness": f"{median_fitness[i]:.6f}",
            "std_fitness": f"{std_fitness[i]:.6f}",
            "diversity": f"{diversity[i]:.6f}",
            "elapsed_s": f"{elapsed_s[i]:.4f}",
            "best_L": f"{best_L[i]:.4f}",
            "best_a": f"{best_a[i]:.4f}",
            "best_b": f"{best_b[i]:.4f}",
        }
        for pn in pnames:
            val = 0.3 + 0.6 * rng.random()
            row[f"best_{pn}"] = f"{val:.6e}"
        rows.append(row)

    with open(run_dir / "ga_history.csv", "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Create best_genome.json
    last_best_f = best_fitness[-1]
    last_L = float(best_L[-1])
    last_a = float(best_a[-1])
    last_b = float(best_b[-1])

    best_genome = {
        "genome": {pn: 0.5 for pn in pnames},
        "fitness": float(last_best_f),
        "component_scores": {
            "err_L": (last_L - target_lab[0]) / 60.0,
            "err_a": (last_a - target_lab[1]) / 20.0,
            "err_b": (last_b - target_lab[2]) / 25.0,
            "total_err": float(np.sqrt(
                ((last_L - target_lab[0]) / 60.0) ** 2 +
                ((last_a - target_lab[1]) / 20.0) ** 2 +
                ((last_b - target_lab[2]) / 25.0) ** 2
            )),
            "L_sim": last_L,
            "a_sim": last_a,
            "b_sim": last_b,
            "L_target": target_lab[0],
            "a_target": target_lab[1],
            "b_target": target_lab[2],
        },
        "lab_sim": [last_L, last_a, last_b],
        "generation": n - 1,
        "run_seed": 42,
        "population_size": 20,
        "mutation_rate": 0.2,
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
    }
    (run_dir / "best_genome.json").write_text(
        json.dumps(best_genome, indent=2), encoding="utf-8"
    )

    # Create population_final.json (truncated)
    pop = {
        "seed": 42,
        "population_size": 20,
        "generations_completed": n,
        "individuals": [
            {
                "genome": {pn: 0.5 for pn in pnames},
                "fitness": float(best_fitness[-1]),
                "component_scores": {},
                "lab_sim": [float(best_L[-1]), float(best_a[-1]), float(best_b[-1])],
                "generation": n - 1,
            }
        ],
        "timestamp_utc": "2026-01-01T00:00:00+00:00",
    }
    (run_dir / "population_final.json").write_text(
        json.dumps(pop, indent=2), encoding="utf-8"
    )

    # Create optional synthetic spectra
    spectra_dir = run_dir / "spectra"
    spectra_dir.mkdir(exist_ok=True)
    for i in range(0, n, max(1, n // 3)):
        wl = np.arange(380, 781, 5, dtype=float)
        # Synthetic reflectance spectrum that converges toward target
        progress = i / max(1, n - 1)
        ref = 0.3 + 0.5 * (1 - progress) * np.exp(-((wl - 550) / 80) ** 2) \
            + 0.1 * progress * np.sin(wl / 100.0)
        ref = np.clip(ref, 0.01, 0.99)
        spec_path = spectra_dir / f"spectrum_gen_{i:04d}.npy"
        np.save(str(spec_path), ref)

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Render best-run video from optical GA results.\n\n"
            "Reads a run folder (MCX or PyXOpto branch) containing "
            "ga_history.csv and optional spectra, generates per-frame "
            "PNGs showing convergence, swatches, and ΔE, then encodes "
            "an MP4 if ffmpeg is available."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io_group = ap.add_argument_group("Input / Output")
    io_group.add_argument(
        "--run-dir", type=str, default=None,
        help="Path to the GA run folder containing ga_history.csv",
    )
    io_group.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for frames and video (default: --run-dir/video)",
    )

    # Video settings
    vid_group = ap.add_argument_group("Video settings")
    vid_group.add_argument(
        "--fps", type=int, default=5,
        help="Frames per second in output MP4 (default: 5)",
    )
    vid_group.add_argument(
        "--frame-width", type=int, default=1600,
        help="Frame width in pixels (default: 1600)",
    )
    vid_group.add_argument(
        "--frame-height", type=int, default=900,
        help="Frame height in pixels (default: 900)",
    )

    # Behaviour flags
    flags_group = ap.add_argument_group("Behaviour")
    flags_group.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing frames directory",
    )
    flags_group.add_argument(
        "--keep-frames", action="store_true",
        help="Keep per-frame PNGs after encoding MP4",
    )

    # Synthetic test
    test_group = ap.add_argument_group("Synthetic test")
    test_group.add_argument(
        "--smoke-test", action="store_true",
        help="Generate a synthetic GA run and render video from it, "
        "then exit",
    )
    test_group.add_argument(
        "--smoke-generations", type=int, default=10,
        help="Number of synthetic generations (default: 10)",
    )
    test_group.add_argument(
        "--smoke-output", type=str, default="/tmp/ga_video_smoke",
        help="Output directory for smoke test (default: /tmp/ga_video_smoke)",
    )

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    if not _HAS_MPL:
        print("ERROR: matplotlib is required for rendering.  Install with: "
              "pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    # ---- Smoke test mode ----
    if args.smoke_test:
        smoke_out = Path(args.smoke_output)
        smoke_out.mkdir(parents=True, exist_ok=True)
        run_dir = _generate_synthetic_run(smoke_out, args.smoke_generations)
        print(f"Generated synthetic run: {run_dir}")
        out_dir = smoke_out / "video"
        ret = render_best_run_video(
            run_dir=run_dir,
            output_dir=out_dir,
            fps=args.fps,
            frame_size=(args.frame_width, args.frame_height),
            overwrite=True,
            keep_frames=args.keep_frames,
        )
        sys.exit(ret)

    # ---- Real mode ----
    if args.run_dir is None:
        print("ERROR: --run-dir is required (use --smoke-test for a "
              "synthetic demo)", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "video"

    ret = render_best_run_video(
        run_dir=run_dir,
        output_dir=output_dir,
        fps=args.fps,
        frame_size=(args.frame_width, args.frame_height),
        overwrite=args.overwrite,
        keep_frames=args.keep_frames,
    )
    sys.exit(ret)


if __name__ == "__main__":
    main()
