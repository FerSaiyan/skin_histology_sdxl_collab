#!/usr/bin/env python
"""
Colorimetry utilities for the optical-parameter GA.

Provides L\\*a\\*b\\* computation from reflectance spectra, ITA (Individual
Typology Angle), and related colour-space helpers.

Wraps the ``colour-science`` library when available; provides fallback
approximations for CI / smoke tests.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional colour-science import
# ---------------------------------------------------------------------------

_HAS_COLOUR = False
try:
    import colour  # noqa: F401
    from colour import SpectralDistribution, sd_to_XYZ, XYZ_to_Lab, XYZ_to_sRGB

    _HAS_COLOUR = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Wavelengths (must match genome_encoding_optical.WAVELENGTHS)
# ---------------------------------------------------------------------------

_WAVELENGTHS = list(range(380, 781, 5))


# ---------------------------------------------------------------------------
# ITA (Individual Typology Angle)
# ---------------------------------------------------------------------------

def calculate_ita(L: float, b: float, eps: float = 1e-8) -> float:
    """Compute the Individual Typology Angle from CIE L\\* and b\\*.

    Parameters
    ----------
    L : float
        CIE L\\* (lightness).
    b : float
        CIE b\\* (yellow-blue axis).
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    ita : float
        ITA angle in degrees.
    """
    return float(np.degrees(np.arctan((L - 50.0) / (b + eps))))


def classify_ita(ita: float) -> str:
    """Classify ITA angle into a skin type category."""
    if ita > 55:
        return "Very Light"
    elif ita > 41:
        return "Light"
    elif ita > 28:
        return "Intermediate"
    elif ita > 10:
        return "Tan"
    elif ita > -30:
        return "Brown"
    else:
        return "Dark"


# ---------------------------------------------------------------------------
# CIELAB from reflectance spectrum
# ---------------------------------------------------------------------------

def _compute_white_xyz(
    cmfs_name: str = "CIE 1964 10 Degree Standard Observer",
) -> np.ndarray:
    """Compute the XYZ tristimulus values for a perfect diffuser (100% R)."""
    if not _HAS_COLOUR:
        raise RuntimeError("colour-science required")
    from colour.colorimetry import MSDS_CMFS

    illuminant = colour.SDS_ILLUMINANTS["D65"]
    white_sd = colour.SpectralDistribution(
        dict(zip(_WAVELENGTHS, np.ones(len(_WAVELENGTHS))))
    )
    cmfs = MSDS_CMFS[cmfs_name]
    return sd_to_XYZ(white_sd, illuminant=illuminant, cmfs=cmfs)


def _xyz_to_lab_manual(XYZ: np.ndarray, XYZ_n: np.ndarray) -> np.ndarray:
    """Manual CIE L\\*a\\*b\\* computation (bypasses colour-science API quirks).

    Parameters
    ----------
    XYZ : (3,) array
        CIE XYZ tristimulus values of the stimulus.
    XYZ_n : (3,) array
        CIE XYZ tristimulus values of the reference white point.

    Returns
    -------
    lab : (3,) array
        CIE L\\*, a\\*, b\\*.
    """
    X, Y, Z = XYZ
    Xn, Yn, Zn = XYZ_n

    def _f(t):
        """CIE non-linear function."""
        delta = 6 / 29
        if t > delta ** 3:
            return np.cbrt(t)
        else:
            return t / (3 * delta ** 2) + 4 / 29

    fx = _f(X / Xn)
    fy = _f(Y / Yn)
    fz = _f(Z / Zn)

    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    return np.array([L, a, b])


def reflectance_to_lab(
    reflectance: np.ndarray,
    wavelengths: Optional[List[int]] = None,
    cmfs_name: str = "CIE 1964 10 Degree Standard Observer",
) -> Optional[Tuple[float, float, float]]:
    """Compute CIE L\\*a\\*b\\* from a reflectance spectrum.

    Uses ``colour-science`` for XYZ integration; computes Lab manually
    to avoid version-specific whitepoint API issues.

    Parameters
    ----------
    reflectance : np.ndarray
        Reflectance values at each wavelength (same length as wavelengths).
    wavelengths : list of int, optional
        Wavelengths in nm. Defaults to 380–780 nm, 5 nm step.
    cmfs_name : str
        Name of the standard observer colour matching functions.

    Returns
    -------
    lab : (float, float, float) or None
        CIE L\\*, a\\*, b\\* values, or None if colour-science is unavailable.
    """
    if not _HAS_COLOUR:
        return None

    if wavelengths is None:
        wavelengths = _WAVELENGTHS

    from colour.colorimetry import MSDS_CMFS

    illuminant = colour.SDS_ILLUMINANTS["D65"]
    cmfs = MSDS_CMFS[cmfs_name]

    sd_data = dict(zip(wavelengths, reflectance))
    sd = SpectralDistribution(sd_data)
    xyz = sd_to_XYZ(sd, illuminant=illuminant, cmfs=cmfs)

    # Whitepoint is XYZ of perfect reflecting diffuser under same illumination
    whitepoint = _compute_white_xyz(cmfs_name)
    lab = _xyz_to_lab_manual(xyz, whitepoint)
    return (float(lab[0]), float(lab[1]), float(lab[2]))


def reflectance_to_rgb(
    reflectance: np.ndarray,
    wavelengths: Optional[List[int]] = None,
    cmfs_name: str = "CIE 1964 10 Degree Standard Observer",
) -> Optional[Tuple[float, float, float]]:
    """Compute sRGB from a reflectance spectrum.

    Returns None if colour-science is unavailable.
    """
    if not _HAS_COLOUR:
        return None

    if wavelengths is None:
        wavelengths = _WAVELENGTHS

    from colour.colorimetry import MSDS_CMFS

    illuminant = colour.SDS_ILLUMINANTS["D65"]
    cmfs = MSDS_CMFS[cmfs_name]

    sd_data = dict(zip(wavelengths, reflectance))
    sd = SpectralDistribution(sd_data)
    xyz = sd_to_XYZ(sd, illuminant=illuminant, cmfs=cmfs)
    rgb = colour.XYZ_to_sRGB(xyz)
    rgb = np.clip(rgb, 0, 1)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


# ---------------------------------------------------------------------------
# Synthetic Lab → approximate reflectance (for surrogate mode)
# ---------------------------------------------------------------------------

def lab_to_approx_reflectance(
    L: float,
    a: float,
    b: float,
    wavelengths: Optional[List[int]] = None,
) -> np.ndarray:
    """Generate a synthetic reflectance spectrum from L\\*a\\*b\\* values.

    This is a rough analytical reverse mapping intended for surrogate mode.
    It should NOT be used for scientific computations.

    Uses a logistic-shaped reflectance curve modulated by L (lightness),
    a (red-green), and b (yellow-blue).
    """
    if wavelengths is None:
        wavelengths = _WAVELENGTHS

    wl = np.array(wavelengths, dtype=np.float64)
    # Base reflectance shape (tissue-like spectrum)
    base = 0.3 + 0.5 / (1.0 + np.exp(-(wl - 550.0) / 60.0))

    # Modulate by L* (lightness scales overall reflectance)
    L_norm = max(0.0, min(100.0, L)) / 100.0
    base *= 0.3 + 0.7 * L_norm

    # Modulate by a* (red-green: enhance red @ ~580nm, green ~520nm)
    a_norm = max(-60.0, min(60.0, a)) / 60.0
    red_peak = np.exp(-((wl - 580.0) ** 2) / (2 * 50.0 ** 2))
    green_peak = np.exp(-((wl - 520.0) ** 2) / (2 * 50.0 ** 2))
    base += 0.05 * a_norm * (red_peak - green_peak)

    # Modulate by b* (yellow-blue: enhance yellow ~570nm)
    b_norm = max(-60.0, min(60.0, b)) / 60.0
    yellow_peak = np.exp(-((wl - 570.0) ** 2) / (2 * 50.0 ** 2))
    base += 0.05 * b_norm * yellow_peak

    return np.clip(base, 0.01, 0.99)
