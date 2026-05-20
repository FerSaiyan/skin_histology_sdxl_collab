#!/usr/bin/env python
"""
Forward model for the optical-parameter GA.

Simulates skin reflectance from a set of biophysical parameters.

Two modes are supported:

1. **Surrogate mode** (default, lightweight)
   - Uses an analytical approximation to generate reflectance spectra and
     L\\*a\\*b\\* from the genome parameters.
   - No external dependencies beyond numpy.
   - Suitable for CI, smoke tests, and development.

2. **Realistic mode** (requires external MC backend, strict by default)
   - Expects either the ``xopto`` library (for MCML-based simulation) or
     ``mcx`` (Monte Carlo eXtreme) to be installed.
   - When neither backend is available a ``RuntimeError`` is raised with
     an actionable installation hint **— there is no implicit fallback**
     to the surrogate.  Callers that wish to fall back must catch the
     exception or pass ``allow_surrogate_fallback`` to the caller
     (e.g. ``run_optical_ga_batch_compare.py``).

Wavelength handling: 380–780 nm, 5 nm step (81 wavelengths) as configured
in ``genome_encoding_optical``.
"""

from __future__ import annotations

import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from scripts.optical_ga.genome_encoding_optical import (
    WAVELENGTHS,
    get_param_defs,
    num_params,
    param_names,
)

from scripts.optical_ga.colorimetry import (
    reflectance_to_lab,
    reflectance_to_rgb,
)

# ---------------------------------------------------------------------------
# Constants (from Salomatina et al.)
# ---------------------------------------------------------------------------

_A_PRIME_EPI = 66.7
_FRAY_EPI = 0.29
_BMIE_EPI = 0.689

_A_PRIME_DERM = 43.6
_FRAY_DERM = 0.41
_BMIE_DERM = 0.562

# Subcutis power-law parameters
_SUBCUTIS_MU_S_500 = 2.84  # mm^-1 at 500 nm
_SUBCUTIS_POWER = -1.03

# Default refractive indices
_DEFAULT_N = [1.6, 1.4, 1.44]  # epidermis, dermis, subcutis

# ---------------------------------------------------------------------------
# Scattering functions (from reference)
# ---------------------------------------------------------------------------


def _mu_s_prime_ray_mie_cm(
    lambda_nm: float, a_prime_cm: float, fRay: float, bMie: float
) -> float:
    """Reduced scattering coefficient by Rayleigh+Mie formula (cm^-1)."""
    wl_ratio = lambda_nm / 500.0
    return a_prime_cm * (fRay * (wl_ratio ** -4) + (1.0 - fRay) * (wl_ratio ** -bMie))


def _epidermis_scattering_m_inv(wavelength_m: float) -> float:
    """Epidermis scattering, returns 1/m."""
    return _mu_s_prime_ray_mie_cm(
        wavelength_m * 1e9, _A_PRIME_EPI, _FRAY_EPI, _BMIE_EPI
    ) * 100.0


def _dermis_scattering_m_inv(wavelength_m: float) -> float:
    """Dermis scattering, returns 1/m."""
    return _mu_s_prime_ray_mie_cm(
        wavelength_m * 1e9, _A_PRIME_DERM, _FRAY_DERM, _BMIE_DERM
    ) * 100.0


def _subcutis_scattering_m_inv(wavelength_m: float) -> float:
    """Subcutis scattering via single power law, returns 1/m."""
    wl_nm = wavelength_m * 1e9
    mu_s_mm = _SUBCUTIS_MU_S_500 * ((wl_nm / 500.0) ** _SUBCUTIS_POWER)
    return mu_s_mm * 1000.0


# ---------------------------------------------------------------------------
# Absorption component estimates (for surrogate mode)
# ---------------------------------------------------------------------------
# In the surrogate, we estimate effective mu_a and mu_s per layer per
# wavelength using simple chromophore approximations.  Full MC simulation
# (xopto/mcx) computes these properly from the SkinLayer model.

# Rough extinction coefficients (arbitrary units, for surrogate signa only)
# These are NOT accurate — they just give the GA a plausible optimisation
# landscape.


def _estimate_mua_melanin(wl_nm: float, melanin_conc: float) -> float:
    """Rough melanin absorption estimate (1/m)."""
    # Melanin absorption decreases roughly as λ^-3
    wl_ratio = max(0.1, wl_nm / 500.0)
    return melanin_conc * 1000.0 * (wl_ratio ** -3.0)


def _estimate_mua_blood(
    wl_nm: float, blood_conc: float, spo2: float
) -> float:
    """Rough blood absorption estimate (1/m)."""
    # Haemoglobin absorption peaks at ~420, 540, 575 nm
    peak_420 = np.exp(-((wl_nm - 420.0) ** 2) / (2 * 20.0 ** 2))
    peak_540 = np.exp(-((wl_nm - 540.0) ** 2) / (2 * 15.0 ** 2))
    peak_575 = np.exp(-((wl_nm - 575.0) ** 2) / (2 * 15.0 ** 2))
    # Oxygenated has stronger 540/575, deoxy has stronger 555
    oxy_factor = 0.5 + 0.5 * spo2
    return blood_conc * 200.0 * (peak_420 + oxy_factor * (peak_540 + peak_575))


def _estimate_mua_water(wl_nm: float, water_conc: float) -> float:
    """Rough water absorption estimate (1/m)."""
    # Water absorption rises significantly beyond ~900 nm; minimal in visible
    wl_ratio = wl_nm / 500.0
    return water_conc * 0.1 * (wl_ratio ** 2.0)


def _estimate_mua_fat(wl_nm: float, fat_conc: float) -> float:
    """Rough fat absorption estimate (1/m)."""
    wl_ratio = wl_nm / 500.0
    return fat_conc * 0.05 * (wl_ratio ** 1.5)


# ---------------------------------------------------------------------------
# Surrogate forward model
# ---------------------------------------------------------------------------


def run_surrogate(
    genome: Dict[str, float],
    num_photons: int = 1000000,  # kept for API compat, unused in surrogate
    with_specular: bool = True,
) -> Tuple[
    Tuple[float, float, float],
    np.ndarray,
    Tuple[float, float, float],
    Dict[str, List[float]],
]:
    """Run the surrogate forward model.

    This is a lightweight analytical approximation suitable for smoke tests
    and CI.  It computes effective mu_a and mu_s per layer, combines them,
    and produces a reflectance spectrum using a simple diffusion-style
    approximation.

    Parameters
    ----------
    genome : dict
        Biophysical parameter dict (from genome_encoding_optical).
    num_photons : int
        Ignored in surrogate mode (kept for API compat with real mode).
    with_specular : bool
        If True, adds a small specular component to reflectance.

    Returns
    -------
    lab : (float, float, float)
        CIE L\\*, a\\*, b\\*.
    reflectance : np.ndarray
        Reflectance spectrum, shape (81,) for 380–780 nm.
    rgb : (float, float, float)
        Approximate sRGB values.
    bulk_props : dict
        Per-layer mu_a and mu_s (keys: layer0_mua, layer0_mus, ...).
    """
    # Unpack genome
    melanin = genome.get("melanin", 0.1)
    blood1 = genome.get("blood_layer1", 0.02)
    blood2 = genome.get("blood_layer2", 0.02)
    spo2 = genome.get("spo2", 0.75)
    d0 = genome.get("d_layer0", 8e-5)
    d1 = genome.get("d_layer1", 3e-4)
    d2 = 0.03  # effective subcutis thickness (m)
    amp0 = genome.get("amp_layer0", 1.0)
    amp1 = genome.get("amp_layer1", 1.0)
    amp2 = genome.get("amp_layer2", 1.0)
    water0 = genome.get("water_layer0", 0.65)
    water1 = genome.get("water_layer1", 0.7)
    water2 = genome.get("water_layer2", 0.7)
    fat = genome.get("fat_layer2", 0.2)

    wavelengths_m = np.array([wl * 1e-9 for wl in WAVELENGTHS])
    wavelengths_nm = np.array(WAVELENGTHS)
    num_wl = len(WAVELENGTHS)

    bulk_props: Dict[str, List[float]] = {
        "layer0_mua": [], "layer0_mus": [],
        "layer1_mua": [], "layer1_mus": [],
        "layer2_mua": [], "layer2_mus": [],
        "bulk_mu_a": [], "bulk_mu_s": [],
    }

    for i in range(num_wl):
        wl_nm = wavelengths_nm[i]
        wl_m = wavelengths_m[i]

        # Layer 0 (epidermis)
        mua0 = _estimate_mua_melanin(wl_nm, melanin) + _estimate_mua_water(wl_nm, water0)
        mus0 = amp0 * _epidermis_scattering_m_inv(wl_m)

        # Layer 1 (dermis)
        mua1 = (
            _estimate_mua_blood(wl_nm, blood1, spo2)
            + _estimate_mua_water(wl_nm, water1)
        )
        mus1 = amp1 * _dermis_scattering_m_inv(wl_m)

        # Layer 2 (subcutis)
        mua2 = (
            _estimate_mua_blood(wl_nm, blood2, spo2)
            + _estimate_mua_water(wl_nm, water2)
            + _estimate_mua_fat(wl_nm, fat)
        )
        mus2 = amp2 * _subcutis_scattering_m_inv(wl_m)

        bulk_props["layer0_mua"].append(mua0)
        bulk_props["layer0_mus"].append(mus0)
        bulk_props["layer1_mua"].append(mua1)
        bulk_props["layer1_mus"].append(mus1)
        bulk_props["layer2_mua"].append(mua2)
        bulk_props["layer2_mus"].append(mus2)

        # Effective bulk properties (thickness-weighted)
        total_d = d0 + d1 + d2
        mu_a_eff = (mua0 * d0 + mua1 * d1 + mua2 * d2) / total_d
        mu_s_eff = (mus0 * d0 + mus1 * d1 + mus2 * d2) / total_d
        bulk_props["bulk_mu_a"].append(mu_a_eff)
        bulk_props["bulk_mu_s"].append(mu_s_eff)

    # --- Approximate reflectance from bulk properties ---
    # Use a simple diffusion-inspired approximation.
    # The effective path length is calibrated so that default parameters
    # (light skin) produce reflectance ≈ 0.3-0.6 (L* ~ 60-80).
    bulk_mu_a = np.array(bulk_props["bulk_mu_a"])
    bulk_mu_s = np.array(bulk_props["bulk_mu_s"])

    # Effective attenuation coefficient (diffusion approximation)
    # μ_eff = sqrt(3 * μa * (μa + μs'))
    mu_eff = np.sqrt(3.0 * bulk_mu_a * (bulk_mu_a + bulk_mu_s) + 1e-12)

    # Calibrated effective path length (m).
    # This is NOT the physical thickness — it's tuned to produce realistic
    # reflectance values for the surrogate's approximate mu_a/mu_s estimates.
    # Value 0.005 gives reflectance ≈ 0.4-0.6 for light skin defaults.
    PATH_LENGTH_CALIBRATION = 0.005
    reflectance = np.exp(-mu_eff * PATH_LENGTH_CALIBRATION)

    # Add specular component (~4% Fresnel reflection)
    if with_specular:
        specular = 0.04 * np.ones_like(reflectance)
        reflectance = np.clip(reflectance + specular, 0.0, 1.0)

    # Compute Lab and RGB (if colour-science available)
    lab = reflectance_to_lab(reflectance)
    rgb = reflectance_to_rgb(reflectance)

    # Fallback if colour-science not available
    if lab is None:
        # Rough Lab approximation from reflectance
        mean_r = float(np.mean(reflectance))
        lab = (mean_r * 100.0, 0.0, 0.0)
    if rgb is None:
        mean_r = float(np.mean(reflectance))
        rgb = (mean_r, mean_r, mean_r)

    return lab, reflectance, rgb, bulk_props


# ---------------------------------------------------------------------------
# Realistic mode placeholder
# ---------------------------------------------------------------------------


def run_realistic(
    genome: Dict[str, float],
    num_photons: int = 1_000_000,
    with_specular: bool = True,
    backend: str = "auto",
) -> Tuple[
    Tuple[float, float, float],
    np.ndarray,
    Tuple[float, float, float],
    Dict[str, List[float]],
]:
    """Run the realistic forward model using an external MC backend.

    .. note::

        This mode requires the ``xopto`` library (for MCML-based simulation)
        or ``mcx`` (Monte Carlo eXtreme).  If neither is available, a clear
        error is raised.

    Parameters
    ----------
    genome : dict
        Biophysical parameter dict.
    num_photons : int
        Number of photons for MC simulation (default 1M).
    with_specular : bool
        Include specular reflectance.
    backend : str
        Backend selection: ``"auto"``, ``"xopto"``, or ``"mcx"``.

    Returns
    -------
    Same signature as ``run_surrogate``.
    """
    # Check xopto availability via mc_wrapper (handles auto-discovery)
    from scripts.optical_ga.mc_wrapper import check_xopto, check_mcx

    _xopto_available = check_xopto()
    _mcx_available = check_mcx()

    if backend == "auto":
        if _xopto_available:
            backend = "xopto"
        elif _mcx_available:
            backend = "mcx"
        else:
            raise RuntimeError(
                "Realistic forward model requires xopto (pip install xopto) or "
                "the MCX binary (http://mcx.space/).\n"
                "Neither was found.  Use surrogate mode (--mode surrogate) for "
                "lightweight smoke tests without external dependencies."
            )

    if backend == "xopto":
        if not _xopto_available:
            raise RuntimeError(
                "xopto backend selected but xopto is not available. "
                "Run with backend='mcx' or backend='auto'."
            )
        return _run_xopto(genome, num_photons, with_specular)

    elif backend == "mcx":
        if not _mcx_available:
            raise RuntimeError(
                "MCX backend selected but MCX binary is not available. "
                "Run with backend='xopto' or backend='auto'."
            )
        return _run_mcx(genome, num_photons, with_specular)

    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Use 'auto', 'xopto', or 'mcx'."
        )


def _run_xopto(
    genome: Dict[str, float],
    num_photons: int = 1_000_000,
    with_specular: bool = True,
) -> Tuple[
    Tuple[float, float, float],
    np.ndarray,
    Tuple[float, float, float],
    Dict[str, List[float]],
]:
    """Run forward model using xopto MCML backend.

    Wraps the reference script logic from ``find_ita_graph_all.py``.
    Uses mc_wrapper's auto-discovery to ensure xopto is importable.
    """
    # Ensure xopto is importable via mc_wrapper's auto-discovery
    from scripts.optical_ga.mc_wrapper import _ensure_xopto_import

    _ensure_xopto_import()

    # Deferred imports (now xopto should be on sys.path)
    from xopto.mcml import mc
    from xopto.materials import skin
    import colour
    from colour import SpectralDistribution, sd_to_XYZ

    wavelengths_nm = np.arange(380, 781, 5)
    wavelengths_m = wavelengths_nm * 1e-9
    num_wl = len(wavelengths_nm)

    model = skin.Skin3()

    # Scattering functions with amplitude multipliers
    def _epidermis_scattering(wl):
        return genome.get("amp_layer0", 1.0) * _epidermis_scattering_m_inv(wl)

    def _dermis_scattering(wl):
        return genome.get("amp_layer1", 1.0) * _dermis_scattering_m_inv(wl)

    def _subcutis_scattering(wl):
        return genome.get("amp_layer2", 1.0) * _subcutis_scattering_m_inv(wl)

    model[0].musr = _epidermis_scattering
    model[1].musr = _dermis_scattering
    model[2].musr = _subcutis_scattering

    # Assign genome parameters
    model[0].melanin = genome.get("melanin", 0.1)
    model[0].water = genome.get("water_layer0", 0.65)
    model[0].d = genome.get("d_layer0", 8e-5)
    model[0].g = genome.get("g_layer0", 0.9)
    model[0].baseline_absorption = 0.0

    model[1].blood = genome.get("blood_layer1", 0.02)
    model[1].water = genome.get("water_layer1", 0.7)
    model[1].spo2 = genome.get("spo2", 0.75)
    model[1].d = genome.get("d_layer1", 3e-4)
    model[1].g = genome.get("g_layer1", 0.9)
    model[1].baseline_absorption = 0.0

    # Optional dermal chromophores
    model[1].bilirubin = genome.get("bilirubin_layer1", 0.0)
    model[1].biliverdin = genome.get("biliverdin_layer1", 0.0)
    model[1].cohb = genome.get("cohb_layer1", 0.0)
    model[1].methb = genome.get("methb_layer1", 0.0)

    model[2].blood = genome.get("blood_layer2", 0.02)
    model[2].water = genome.get("water_layer2", 0.7)
    model[2].spo2 = genome.get("spo2", 0.75)
    model[2].g = genome.get("g_layer2", 0.9)
    model[2].fat = genome.get("fat_layer2", 0.2)
    model[2].baseline_absorption = 0.0

    # Refractive indices
    model[0].n = _DEFAULT_N[0] * genome.get("n_mult_layer0", 1.0)
    model[1].n = _DEFAULT_N[1] * genome.get("n_mult_layer1", 1.0)
    model[2].n = _DEFAULT_N[2] * genome.get("n_mult_layer2", 1.0)

    # Compute per-layer bulk properties
    bulk_props: Dict[str, List[float]] = {
        "layer0_mua": [], "layer0_mus": [],
        "layer1_mua": [], "layer1_mus": [],
        "layer2_mua": [], "layer2_mus": [],
    }
    for layer in range(3):
        for wl in wavelengths_nm:
            wl_m = wl * 1e-9
            bulk_props[f"layer{layer}_mua"].append(float(model[layer].mua(wl_m)))
            bulk_props[f"layer{layer}_mus"].append(float(model[layer].musr(wl_m)))

    # Effective thickness-weighted bulk properties
    d0 = genome.get("d_layer0", 8e-5)
    d1 = genome.get("d_layer1", 3e-4)
    d2_eff = 0.03
    total_d = d0 + d1 + d2_eff

    bulk_props["bulk_mu_a"] = []
    bulk_props["bulk_mu_s"] = []
    for i in range(num_wl):
        mu_a_eff = (
            bulk_props["layer0_mua"][i] * d0
            + bulk_props["layer1_mua"][i] * d1
            + bulk_props["layer2_mua"][i] * d2_eff
        ) / total_d
        mu_s_eff = (
            bulk_props["layer0_mus"][i] * d0
            + bulk_props["layer1_mus"][i] * d1
            + bulk_props["layer2_mus"][i] * d2_eff
        ) / total_d
        bulk_props["bulk_mu_a"].append(mu_a_eff)
        bulk_props["bulk_mu_s"].append(mu_s_eff)

    # MC simulation at each wavelength
    nominal_wl = 550e-9
    layers_mc = model.create_mc_layers(nominal_wl)

    incidence_angle = np.deg2rad(45.0)
    source = mc.mcsource.Line(
        direction=[np.sin(incidence_angle), 0.0, np.cos(incidence_angle)]
    )
    # Use a wide acceptance angle (cosmin=0.5 ≈ 60° half-angle) to capture
    # most of the diffuse reflectance.  The original code used
    # CIE1964.fov (10°) which was too narrow.
    observer_cosmin = 0.5
    detectors = mc.mcdetector.Detectors(
        top=mc.mcdetector.Total(cosmin=observer_cosmin),
        specular=mc.mcdetector.Total(cosmin=observer_cosmin),
    )

    # Try GPU, fall back to CPU
    try:
        gpu = mc.clinfo.gpu()
        mc_obj = mc.Mc(layers_mc, source, detectors, cl_devices=gpu)
    except Exception:
        mc_obj = mc.Mc(layers_mc, source, detectors)
    mc_obj.rmax = 50.0e-3

    reflectance_spectrum = np.empty(num_wl, dtype=float)
    for idx, wl in enumerate(wavelengths_m):
        model.update_mc_layers(mc_obj.layers, wl)
        sim_result = mc_obj.run(num_photons)

        if isinstance(sim_result, (list, tuple)):
            if len(sim_result) == 3:
                _, _, det_res = sim_result
            elif len(sim_result) == 4:
                _, _, _, det_res = sim_result
            else:
                raise ValueError(f"Unexpected MC output length: {len(sim_result)}")
        else:
            raise ValueError("MC run did not return a tuple/list")

        r_val = det_res.top.reflectance
        if with_specular:
            r_val += det_res.specular.reflectance
        reflectance_spectrum[idx] = r_val

    # Compute Lab and RGB
    from scripts.optical_ga.colorimetry import (
        _compute_white_xyz,
        _xyz_to_lab_manual,
    )

    sd_data = dict(zip(wavelengths_nm, reflectance_spectrum))
    sd = SpectralDistribution(sd_data)
    xyz = sd_to_XYZ(sd, illuminant=colour.SDS_ILLUMINANTS["D65"],
                     cmfs=colour.colorimetry.MSDS_CMFS[
                         "CIE 1964 10 Degree Standard Observer"
                     ])
    whitepoint = _compute_white_xyz("CIE 1964 10 Degree Standard Observer")
    lab_vals = _xyz_to_lab_manual(xyz, whitepoint)
    lab = (float(lab_vals[0]), float(lab_vals[1]), float(lab_vals[2]))

    rgb_vals = colour.XYZ_to_sRGB(xyz)
    rgb_vals = np.clip(rgb_vals, 0, 1)
    rgb = (float(rgb_vals[0]), float(rgb_vals[1]), float(rgb_vals[2]))

    return lab, reflectance_spectrum, rgb, bulk_props


# ---------------------------------------------------------------------------
# MCX-backed forward model
# ---------------------------------------------------------------------------


def _run_mcx(
    genome: Dict[str, float],
    num_photons: int = 1_000_000,
    with_specular: bool = True,
) -> Tuple[
    Tuple[float, float, float],
    np.ndarray,
    Tuple[float, float, float],
    Dict[str, List[float]],
]:
    """Run forward model using the MCX binary backend.

    Builds a 3-layer voxel volume, runs MCX per wavelength, extracts
    reflectance via energy balance from MCX stdout, and converts to
    L\\*a\\*b\\* via the existing colorimetry path.

    The MCX branch does NOT route through xopto.

    Parameters
    ----------
    genome : dict
        Biophysical parameter dict.
    num_photons : int
        Photons per wavelength.  Use 50K–100K for GA sweeps,
        500K+ for production fidelity.
    with_specular : bool
        Include Fresnel specular component.

    Returns
    -------
    lab : (float, float, float)
    reflectance : np.ndarray  (81 wavelengths)
    rgb : (float, float, float)
    bulk_props : dict
    """
    from scripts.optical_ga.mc_wrapper import run_mcx_simulation

    wavelengths_nm = list(np.arange(380, 781, 5))  # 81 wavelengths

    # Delegate to the MCX runner
    reflectance, bulk_props = run_mcx_simulation(
        genome,
        wavelengths_nm=wavelengths_nm,
        num_photons=num_photons,
        with_specular=with_specular,
    )

    # Convert to Lab and RGB via existing colorimetry
    lab = reflectance_to_lab(reflectance)
    rgb = reflectance_to_rgb(reflectance)

    # Fallback if colour-science not available
    if lab is None:
        mean_r = float(np.mean(reflectance))
        lab = (mean_r * 100.0, 0.0, 0.0)
    if rgb is None:
        mean_r = float(np.mean(reflectance))
        rgb = (mean_r, mean_r, mean_r)

    return lab, reflectance, rgb, bulk_props


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FORWARD_MODES = {
    "surrogate": run_surrogate,
    "realistic": run_realistic,
}


def run_forward(
    genome: Dict[str, float],
    mode: str = "surrogate",
    num_photons: int = 1_000_000,
    with_specular: bool = True,
) -> Tuple[
    Tuple[float, float, float],
    np.ndarray,
    Tuple[float, float, float],
    Dict[str, List[float]],
]:
    """Run the forward model in the specified mode.

    Parameters
    ----------
    genome : dict
        Biophysical parameter dict.
    mode : str
        ``"surrogate"`` (default, lightweight) or ``"realistic"`` (MC backend).
    num_photons : int
        Number of photons (only used in realistic mode).
    with_specular : bool
        Include specular reflectance.

    Returns
    -------
    lab : (float, float, float)
    reflectance : np.ndarray
    rgb : (float, float, float)
    bulk_props : dict
    """
    runner = FORWARD_MODES.get(mode)
    if runner is None:
        raise ValueError(
            f"Unknown forward mode: {mode!r}. "
            f"Available: {list(FORWARD_MODES.keys())}"
        )

    if mode == "realistic":
        # Determine which backend will be used
        from scripts.optical_ga.mc_wrapper import check_xopto, check_mcx

        if check_xopto():
            backend_hint = "xopto (MCML)"
        elif check_mcx():
            backend_hint = "MCX binary"
        else:
            backend_hint = "none (will raise)"
        print(
            f"INFO: Running realistic forward model (backend: {backend_hint}).\n"
            "  If not available, use --mode surrogate for lightweight mode.",
            file=sys.stderr,
        )

    return runner(genome, num_photons=num_photons, with_specular=with_specular)
