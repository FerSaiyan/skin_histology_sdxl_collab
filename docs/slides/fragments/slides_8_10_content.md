# Slides 8–10: Multi-Physics Pipeline

---

## optical-optimization

### Optical Parameter Optimization (Genetic Algorithm)

**Goal** — fit a 19-parameter skin model (melanin concentration, blood volume,
oxygen saturation, scattering amplitude, layer thicknesses) so that the
predicted reflectance matches a **target skin colour** (L\*a\*b\* / ITA°).

**Genome overview:**

| Category | Parameters | Range |
|---|---|---|
| Epidermal melanin | melanin_concentration | 0–40 % |
| Dermal blood | blood_volume_fraction, O2_saturation | 0.5–10 %, 50–100 % |
| Scattering | reduced_scattering_coeff (500 nm), scattering_power | 10–50 cm⁻¹, 0.5–2.0 |
| Geometry | epidermal_thickness, dermal_thickness | 0.04–0.15 mm, 0.5–3.0 mm |
| Auxiliary | refractive_indices, chromophore baselines | fixed / narrow bounds |

**Fitness:** colour-difference error in L\*a\*b\* space with normalised
denominators per channel. Tournament selection, uniform crossover, Gaussian
mutation with adaptive std-dev.

**Forward model modes:**

| Mode | Engine | When to use |
|---|---|---|
| Surrogate | Analytical reflectance (Kubelka–Munk–style) | Quick scans, no external deps |
| Realistic | MCML via PyXOpto (or MCX) | High-fidelity fits, production |

**Air–epidermis interface:** PCA normal estimation from the 3D surface mesh
(`select_orient_tile_for_incidence.py`) ensures tiles are oriented consistently
for incidence-angle-aware simulation.

**Pipeline bridge:** `build_histoseg_label_volume.py` → `label_to_optical_priors.py`
→ `run_optical_ga_from_labels.py` — all outputs are deterministic `.npy` /
`.json` / `.npz`.

> **Murilo port context:** The GA loop and forward-modelling infrastructure
> are ported and adapted from Murilo's original skin-optics optimisation
> framework. Key differences in our port: (1) integration with 3D label
> volumes instead of 2D patches, (2) dual forward-mode architecture
> (surrogate + MCML) for faster iteration, (3) direct colour target
> specification in L\*a\*b\* with ITA° conversion for Fitzpatrick-style
> classification.

**Speaker hints:**
- Emphasise the *two forward modes* — surrogate for rapid prototyping,
  MCML-based for publication-grade fits.
- The GA is not a black box: each genome can be visualised as reflectance
  curve + skin colour swatch.
- Mention that the Murilo port saves ~80 % re-implementation time vs
  starting from scratch.

---

## mcx-vs-pyxopto

### MCX vs PyXOpto — Light Transport Engine Comparison

| Property | MCX (Monte Carlo eXtreme) | PyXOpto (xopto) |
|---|---|---|
| **Engine** | GPU-accelerated (CUDA C++) | CPU (Python + C extensions) |
| **Input format** | Binary voxel volume + JSON config | Python API, layered geometry |
| **Speed** | ~10³–10⁴× faster at scale | Adequate for single-1D fits |
| **Output** | Fluence / absorption volume (.nii/.npy) | Reflectance / transmittance curves |
| **Pipeline role** | `mcx_build_volume.py` → `mcx_batch_runner.py` | `mc_wrapper.py` in GA forward model |
| **Status** | Scaffolded, batch compare ready | Integrated, production use |
| **Strengths** | High-res 3D volumes, batch studies, GPU scaling | Zero GPU dep, easy debugging, pip-installable |
| **Limitations** | Binary volume conversion step, CUDA-only | Slower for voxelised 3D, single-threaded |

### Why MCX Is Strategic for Future 3D

- **Scalable batch simulation** — the same `mcx_batch_runner.py` infrastructure
  can sweep over hundreds of optical-property realisations in minutes on a single
  GPU, enabling uncertainty quantification and sensitivity analysis.

- **Native 3D voxel support** — MCX operates directly on voxelised label volumes
  (the same `.npy` arrays we already produce), so there is no geometry
  abstraction gap between segmentation and simulation.

- **Future multi-GPU & volumetric heating** — MCX can export per-voxel absorbed
  energy density (W/m³), which is the direct source term for thermal finite-difference
  solvers. This closes the *optical → thermal* coupling loop without interpolation
  or resampling.

- **Head-to-head validation** — `batch_compare` runs both MCX and PyXOpto with
  shared parameter bounds and identical tissue models, giving confidence in
  the simpler CPU model during development while MCX is the production target.

**Speaker hints:**
- MCX is *not* a drop-in replacement for PyXOpto — it requires a volume build step.
- The batch-compare script is the key tool for trust calibration between the two engines.
- Frame MCX as an *investment*: once the volume pipeline is solid, every future
  study gets GPU-speed light transport for free.

---

## thermal-coupling

### Thermal Coupling — Optical Absorption → Bioheat

The optical simulation (MCX or PyXOpto) produces a **volumetric absorption
density** Q(r) [W/m³]. This becomes the heat source in the Pennes bioheat
equation:

```
ρ c ∂T/∂t = ∇·(k ∇T) + w_b c_b (T_a − T) + Q_met + Q(r)
```

| Symbol | Quantity | Units | Source |
|---|---|---|---|
| ρ | Density | kg/m³ | Label-dependent |
| c | Specific heat | J/(kg·K) | Label-dependent |
| k | Thermal conductivity | W/(m·K) | Label-dependent |
| w_b | Blood perfusion rate | s⁻¹ | Label-dependent |
| c_b | Blood specific heat | J/(kg·K) | Constant |
| T_a | Arterial temperature | °C | 37 °C |
| Q_met | Metabolic heat generation | W/m³ | Label-dependent |
| **Q(r)** | **Optical absorption (MCX output)** | **W/m³** | **From light transport** |

**Pipeline:** NIfTI volume → `thermal_build_model.py` (property arrays) →
`thermal_solve.py` (explicit finite-difference Pennes solver) →
`thermal_visualise.py` (temperature PNG + summary JSON).

**Configuration:**
- Initial condition: uniform 37 °C.
- Boundary conditions: fixed temperature or convective (Robin-type).
- Source modes: `none` (metabolic-only baseline), `spherical` (localised
  heating), `optical` (MCX-derived Q(r) — the production mode).
- All scripts run without GPU; smoke tests verify end-to-end numerical stability.

**Key coupling insight:** The optical absorption field Q(r) is the *only*
channel through which tissue microstructure (melanin, blood, collagen)
enters the thermal model. This means:
1. Inpainting quality directly affects the thermal source term.
2. The same label-to-property mapping used for optical priors is reused for
   thermal properties — no separate segmentation needed.
3. Sensitivity of the thermal field to inpainting artefacts can be quantified
   via perturbed-Q runs.

**Speaker hints:**
- The equation uses plain ASCII-friendly notation for slides — adapt to
  LaTeX for proceedings.
- Emphasise that Q(r) from MCX is a 3D array with the same voxel grid as the
  NIfTI volume — no resampling.
- Thermal validation target: compare against known experimental data (e.g.,
  laser-skin heating studies) once the optical GA is calibrated.
- Future: couple thermal back to optical via temperature-dependent chromophore
  spectra (thermochromism) for fully coupled multi-physics.
