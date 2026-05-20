"""
MC backend wrapper for the optical-parameter GA.

Provides a unified interface to Monte Carlo simulation backends
(xopto MCML, MCX) for realistic forward modelling.

Xopto auto-discovery
--------------------
When ``xopto`` is not installed via pip, this module automatically
appends the friend-bundle path::

    external_refs/friend_bundle/extracted/Pyxopto - Murilo Version/pyxopto

and retries the import.  Logging shows which path was used.
"""

from __future__ import annotations

import contextlib
import io
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Xopto auto-discovery
# ---------------------------------------------------------------------------

_FRIEND_BUNDLE_XOPTO_PATH = (
    Path(__file__).resolve().parents[2]
    / "external_refs"
    / "friend_bundle"
    / "extracted"
    / "Pyxopto - Murilo Version"
    / "pyxopto"
)

_HAS_XOPTO = False
_XOPTO_SOURCE: str = "unavailable"


def _ensure_xopto_import() -> bool:
    """Try to import xopto; auto-append friend-bundle path if needed.

    Returns True if xopto is importable after the attempt.
    """
    global _HAS_XOPTO, _XOPTO_SOURCE

    # First try normal import
    try:
        from xopto.mcml import mc as _xopto_mc  # noqa: F401
        from xopto.materials import skin as _xopto_skin  # noqa: F401
        from xopto.util import color as _xopto_color  # noqa: F401

        _HAS_XOPTO = True
        _XOPTO_SOURCE = "pip-installed or already on sys.path"
        return True
    except ImportError:
        pass

    # Try friend-bundle path (only if it exists)
    bundle_root = _FRIEND_BUNDLE_XOPTO_PATH
    if bundle_root.exists() and bundle_root.is_dir():
        bundle_path = str(bundle_root.resolve())
        if bundle_path not in _sys_path_present():
            import sys

            sys.path.insert(0, bundle_path)
    else:
        bundle_path = str(bundle_root)

    try:
        from xopto.mcml import mc as _xopto_mc  # noqa: F401
        from xopto.materials import skin as _xopto_skin  # noqa: F401
        from xopto.util import color as _xopto_color  # noqa: F401

        _HAS_XOPTO = True
        _XOPTO_SOURCE = f"friend-bundle: {bundle_path}"
        logger.info("xopto loaded from friend bundle: %s", bundle_path)
        return True
    except ImportError as exc:
        _XOPTO_SOURCE = f"unavailable ({exc})"
        return False


def _sys_path_present() -> set:
    import sys

    return set(sys.path)


# Run auto-discovery at module level.
# Suppress stdout during import to silence third-party noise
# (e.g. xopto.materials.absorption.bilirubin has a debug print(data)).
with contextlib.redirect_stdout(io.StringIO()):
    _ensure_xopto_import()


def check_xopto() -> bool:
    """Return True if the xopto library is available.

    On first call (or module import), auto-discovery appends the
    friend-bundle path and retries.  Logs the source to stderr.
    """
    if not _HAS_XOPTO:
        logger.warning("xopto not available. Source: %s", _XOPTO_SOURCE)
    return _HAS_XOPTO


def xopto_source() -> str:
    """Return a human-readable string describing how xopto was loaded."""
    return _XOPTO_SOURCE


def run_xopto_simulation(
    genome: Dict[str, float],
    wavelengths_nm: Optional[List[int]] = None,
    num_photons: int = 1_000_000,
    with_specular: bool = True,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Run a multi-wavelength MCML simulation via xopto.

    .. note::
        Requires ``xopto`` to be importable (auto-discovery attempted).

    Parameters
    ----------
    genome : dict
        Biophysical parameter dict (see genome_encoding_optical).
    wavelengths_nm : list of int, optional
        Wavelengths in nm. Defaults to 380–780, 5 nm step.
    num_photons : int
        Photons per wavelength.
    with_specular : bool
        Include specular reflectance.

    Returns
    -------
    reflectance : np.ndarray
        Reflectance spectrum, shape (n_wavelengths,).
    bulk_props : dict
        Per-layer mu_a and mu_s for all wavelengths.

    Raises
    ------
    RuntimeError
        If xopto is not importable.
    """
    if not _HAS_XOPTO:
        raise RuntimeError(
            "xopto is required for realistic forward modelling.\n"
            "Checked pip-installed location and friend-bundle path:\n"
            f"  {_FRIEND_BUNDLE_XOPTO_PATH}\n"
            "Install with: pip install xopto\n"
            "Or use surrogate mode: --forward-mode surrogate"
        )

    # Defer to the implementation in forward_model
    from scripts.optical_ga.forward_model import run_realistic

    lab, reflectance, rgb, bulk_props = run_realistic(
        genome, num_photons=num_photons, with_specular=with_specular, backend="xopto"
    )
    return reflectance, bulk_props


# ---------------------------------------------------------------------------
# MCX wrapper
# ---------------------------------------------------------------------------


def check_mcx(binary: str = "mcx") -> bool:
    """Return True if the MCX binary is available on PATH."""
    return shutil.which(binary) is not None


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from MCX terminal output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _compute_specular_fresnel(n_tissue: float) -> float:
    """Fresnel power reflection at normal incidence: air (n=1) → tissue."""
    return ((n_tissue - 1.0) / (n_tissue + 1.0)) ** 2


def run_mcx_simulation(
    genome: Dict[str, float],
    wavelengths_nm: Optional[List[int]] = None,
    num_photons: int = 100_000,
    with_specular: bool = True,
    mcx_binary: str = "mcx",
    work_dir: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, List[float]]]:
    """Run a multi-wavelength MCX simulation from a genome.

    For each wavelength, builds a small 3-layer voxel volume (40×40×60),
    runs MCX with Fresnel boundary conditions, and extracts the
    reflectance via energy balance from MCX stdout.

    This helper is intentionally scoped to GA parity runs and does not accept
    arbitrary label volumes from disk. For full-geometry MCX workflows, use
    `scripts/simulation/mcx_build_volume.py` + `scripts/simulation/mcx_batch_runner.py`.

    Parameters
    ----------
    genome : dict
        Biophysical parameter dict.
    wavelengths_nm : list of int, optional
        Wavelengths in nm. Defaults to 380–780 nm, 5 nm step.
    num_photons : int
        Photons per wavelength (default 100K; use 10K-50K for GA sweeps).
    with_specular : bool
        Add Fresnel specular component to reflectance.
    mcx_binary : str
        Path or name of MCX binary.
    work_dir : str, optional
        Scratch directory for MCX temp files (created if needed).

    Returns
    -------
    reflectance : np.ndarray
        Reflectance spectrum, shape (n_wavelengths,).
    bulk_props : dict
        Per-layer mu_a and mu_s for all wavelengths.

    Raises
    ------
    RuntimeError
        If MCX binary not found or simulation fails.
    """
    if not check_mcx(mcx_binary):
        raise RuntimeError(
            f"MCX binary {mcx_binary!r} not found on PATH. "
            "See http://mcx.space/ for installation."
        )

    from scripts.optical_ga.genome_encoding_optical import WAVELENGTHS
    from scripts.optical_ga.forward_model import (
        _epidermis_scattering_m_inv,
        _dermis_scattering_m_inv,
        _subcutis_scattering_m_inv,
        _estimate_mua_melanin,
        _estimate_mua_blood,
        _estimate_mua_water,
        _estimate_mua_fat,
        _DEFAULT_N,
    )

    if wavelengths_nm is None:
        wavelengths_nm = WAVELENGTHS

    wavelengths_m = [wl * 1e-9 for wl in wavelengths_nm]
    num_wl = len(wavelengths_nm)

    # Unpack genome
    melanin = genome.get("melanin", 0.1)
    blood1 = genome.get("blood_layer1", 0.02)
    blood2 = genome.get("blood_layer2", 0.02)
    spo2 = genome.get("spo2", 0.75)
    water0 = genome.get("water_layer0", 0.65)
    water1 = genome.get("water_layer1", 0.7)
    water2 = genome.get("water_layer2", 0.7)
    fat = genome.get("fat_layer2", 0.2)
    amp0 = genome.get("amp_layer0", 1.0)
    amp1 = genome.get("amp_layer1", 1.0)
    amp2 = genome.get("amp_layer2", 1.0)
    n0 = _DEFAULT_N[0] * genome.get("n_mult_layer0", 1.0)
    n1 = _DEFAULT_N[1] * genome.get("n_mult_layer1", 1.0)
    n2 = _DEFAULT_N[2] * genome.get("n_mult_layer2", 1.0)
    g0 = genome.get("g_layer0", 0.9)
    g1 = genome.get("g_layer1", 0.9)
    g2 = genome.get("g_layer2", 0.9)

    # Compute specular Fresnel reflection at air–epidermis interface
    R_spec = _compute_specular_fresnel(n0)

    # Pre-compute per-layer properties for all wavelengths
    bulk_props: Dict[str, List[float]] = {
        "layer0_mua": [], "layer0_mus": [],
        "layer1_mua": [], "layer1_mus": [],
        "layer2_mua": [], "layer2_mus": [],
        "bulk_mu_a": [], "bulk_mu_s": [],
    }

    for i in range(num_wl):
        wl_nm = wavelengths_nm[i]
        wl_m = wavelengths_m[i]

        # Absorption
        mua0 = _estimate_mua_melanin(wl_nm, melanin) + _estimate_mua_water(wl_nm, water0)
        mua1 = _estimate_mua_blood(wl_nm, blood1, spo2) + _estimate_mua_water(wl_nm, water1)
        mua2 = (
            _estimate_mua_blood(wl_nm, blood2, spo2)
            + _estimate_mua_water(wl_nm, water2)
            + _estimate_mua_fat(wl_nm, fat)
        )

        # Scattering
        mus0 = amp0 * _epidermis_scattering_m_inv(wl_m)
        mus1 = amp1 * _dermis_scattering_m_inv(wl_m)
        mus2 = amp2 * _subcutis_scattering_m_inv(wl_m)

        bulk_props["layer0_mua"].append(mua0)
        bulk_props["layer0_mus"].append(mus0)
        bulk_props["layer1_mua"].append(mua1)
        bulk_props["layer1_mus"].append(mus1)
        bulk_props["layer2_mua"].append(mua2)
        bulk_props["layer2_mus"].append(mus2)

        # Thickness-weighted bulk (for reference)
        d0 = genome.get("d_layer0", 8e-5)
        d1 = genome.get("d_layer1", 3e-4)
        d2_eff = 0.03
        total_d = d0 + d1 + d2_eff
        bulk_props["bulk_mu_a"].append(
            (mua0 * d0 + mua1 * d1 + mua2 * d2_eff) / total_d
        )
        bulk_props["bulk_mu_s"].append(
            (mus0 * d0 + mus1 * d1 + mus2 * d2_eff) / total_d
        )

    # ------------------------------------------------------------------
    # MCX volume building (reused across wavelengths)
    # ------------------------------------------------------------------
    # The label geometry stays the same; only media properties change.
    Z, Y, X = 60, 40, 40  # depth, height, width
    vol = np.zeros((Z, Y, X), dtype=np.uint8)
    # Epidermis starts at z=10 (air above)
    epid_depth_vox = max(2, int(genome.get("d_layer0", 8e-5) / 10e-6))
    derm_depth_vox = max(4, int(genome.get("d_layer1", 3e-4) / 10e-6))
    epi_end = min(Z, 10 + epid_depth_vox)
    derm_end = min(Z, epi_end + derm_depth_vox)
    vol[10:epi_end, :, :] = 1  # epidermis
    vol[epi_end:derm_end, :, :] = 2  # dermis
    vol[derm_end:, :, :] = 3  # subcutis

    # Resolve work directory
    if work_dir is None:
        work_dir_obj = tempfile.mkdtemp(prefix="mcx_ga_")
    else:
        work_dir_obj = Path(work_dir)
        work_dir_obj.mkdir(parents=True, exist_ok=True)
    work_dir_obj = Path(work_dir_obj)

    vol_path = work_dir_obj / "mcx_volume.raw"
    vol.tofile(str(vol_path))

    # Compute dims in MCX order [X, Y, Z]
    dim_mcx = [X, Y, Z]

    # Source position (centre XY, top Z)
    src_cx, src_cy = X // 2, Y // 2

    # ------------------------------------------------------------------
    # Wavelength loop
    # ------------------------------------------------------------------
    reflectance_spectrum = np.empty(num_wl, dtype=float)

    for idx in range(num_wl):
        wl_nm = wavelengths_nm[idx]
        mua_epi = bulk_props["layer0_mua"][idx] / 1000.0  # convert 1/m → 1/mm
        mus_epi = bulk_props["layer0_mus"][idx] / 1000.0
        mua_der = bulk_props["layer1_mua"][idx] / 1000.0
        mus_der = bulk_props["layer1_mus"][idx] / 1000.0
        mua_sub = bulk_props["layer2_mua"][idx] / 1000.0
        mus_sub = bulk_props["layer2_mus"][idx] / 1000.0

        media = [
            {"mua": 0.0001, "mus": 0.0001, "g": 1.0, "n": 1.0},  # air
            {"mua": max(1e-6, mua_epi), "mus": max(1e-6, mus_epi), "g": float(g0), "n": float(n0)},  # epi
            {"mua": max(1e-6, mua_der), "mus": max(1e-6, mus_der), "g": float(g1), "n": float(n1)},  # dermis
            {"mua": max(1e-6, mua_sub), "mus": max(1e-6, mus_sub), "g": float(g2), "n": float(n2)},  # subcutis
        ]

        config = {
            "Session": {
                "ID": f"ga_mcx_wl{idx:03d}_{wl_nm}nm",
                "Photons": int(num_photons),
                "RNGSeed": int(42 + idx),
                "DoNormalize": True,
                "DoAutoThread": True,
                "DoSaveVolume": False,
                "DoPartialPath": False,
                "OutputFormat": "mc2",
                "OutputType": "F",
            },
            "Forward": {"T0": 0.0, "T1": 1e-9, "Dt": 1e-9},
            "Domain": {
                "MediaFormat": "byte",
                "LengthUnit": 1.0,
                "Media": media,
                "Dim": dim_mcx,
                "VolumeFile": str(vol_path.resolve()),
                "OriginType": 0,
            },
            "Optode": {
                "Source": {
                    "Type": "pencil",
                    "Pos": [float(src_cx), float(src_cy), 0.0],
                    "Dir": [0.0, 0.0, 1.0, 0.0],
                    "Param1": [0.0, 0.0, 0.0, 0.0],
                    "Param2": [0.0, 0.0, 0.0, 0.0],
                },
                "Detector": [
                    {
                        "Pos": [float(src_cx), float(src_cy), 0.0],
                        "R": 1.0,
                    },
                ],
            },
        }

        config_path = work_dir_obj / f"config_wl{idx:03d}.json"
        config_path.write_text(
            __import__("json").dumps(config, indent=2), encoding="utf-8"
        )

        try:
            proc = subprocess.run(
                [mcx_binary, "-f", str(config_path.resolve()), "-b", "r"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(work_dir_obj),
            )
        except FileNotFoundError:
            raise RuntimeError(
                f"MCX binary {mcx_binary!r} not found. "
                f"Install from http://mcx.space/"
            )
        except subprocess.TimeoutExpired:
            logger.warning("MCX timed out at wavelength %d nm", wl_nm)
            reflectance_spectrum[idx] = 0.5  # fallback
            continue

        if proc.returncode != 0:
            raise RuntimeError(
                f"MCX failed (rc={proc.returncode}) at wavelength {wl_nm}nm. "
                f"Stderr: {proc.stderr[-300:]}"
            )

        # Parse absorbed fraction from stdout
        clean_out = _strip_ansi(proc.stdout)
        match = re.search(r"absorbed:\s*([\d.]+)%", clean_out)
        if not match:
            logger.warning(
                "Could not parse MCX absorption at %d nm. "
                "Stdout: %s", wl_nm, clean_out[-150:]
            )
            reflectance_spectrum[idx] = 0.5
            continue

        absorbed_pct = float(match.group(1)) / 100.0

        # Energy balance:
        #   R_spec = Fresnel reflection (excluded from MCX's absorbed fraction)
        #   Of photons entering tissue: absorbed_pct absorbed, rest diffusely reflected
        #   Total reflectance = R_spec + (1 - R_spec) * (1 - absorbed_pct)
        if with_specular:
            R_val = R_spec + (1.0 - R_spec) * (1.0 - absorbed_pct)
        else:
            R_val = (1.0 - R_spec) * (1.0 - absorbed_pct)

        reflectance_spectrum[idx] = float(np.clip(R_val, 0.0, 1.0))

    return reflectance_spectrum, bulk_props


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------

BACKENDS: Dict[str, str] = {
    "xopto": "xopto MCML (via run_xopto_simulation)",
    "mcx": "MCX binary (via run_mcx_simulation)",
}


def get_available_backends() -> List[str]:
    """Return list of available MC backends."""
    available = []
    if _HAS_XOPTO:
        available.append("xopto")
    if check_mcx():
        available.append("mcx")
    return available
