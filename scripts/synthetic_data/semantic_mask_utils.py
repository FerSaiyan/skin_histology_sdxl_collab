#!/usr/bin/env python3
"""Geometry-aware semantic mask helpers for class-conditioned histology inpainting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy import ndimage as ndi


CLASS_NAME_BY_ID: Dict[int, str] = {
    0: "background",
    1: "epidermis",
    2: "reticular_dermis",
    3: "papillary_dermis",
    4: "dermis",
    5: "keratin",
    6: "inflammation",
    7: "hair_follicles",
    8: "glands",
    9: "basal_cell_carcinoma",
    10: "squamous_cell_carcinoma",
    11: "intraepidermal_carcinoma",
}

PROMPT_NAME_BY_ID: Dict[int, str] = {
    1: "epidermis",
    2: "reticular dermis",
    3: "papillary dermis",
    4: "dermis",
    5: "keratin",
    6: "inflammation",
    7: "hair follicle",
    8: "skin gland",
    9: "basal cell carcinoma",
    10: "squamous cell carcinoma",
    11: "intraepidermal carcinoma",
}

TARGET_CLASS_IDS = tuple(sorted(PROMPT_NAME_BY_ID))
CANCER_CLASS_IDS = (9, 10, 11)


@dataclass(frozen=True)
class SemanticMaskConfig:
    min_component_pixels: int = 256
    min_mask_pixels: int = 256
    max_mask_area_frac: float = 0.16
    interior_min_purity: float = 0.90
    boundary_min_purity: float = 0.60
    boundary_max_purity: float = 0.92
    min_minor_radius: float = 3.0
    max_major_radius: float = 72.0
    max_attempts: int = 80
    ring_px: int = 10


def class_prompt(class_id: int) -> str:
    """Return a stable natural-language caption for one semantic target class."""
    if class_id not in PROMPT_NAME_BY_ID:
        raise ValueError(f"Unsupported target class id: {class_id}")
    return f"H&E stained skin histology, target tissue: {PROMPT_NAME_BY_ID[class_id]}, realistic microscopy"


def _ellipse_mask(
    shape: Tuple[int, int],
    center_yx: Tuple[float, float],
    major_radius: float,
    minor_radius: float,
    angle_rad: float,
) -> np.ndarray:
    """Rasterize a rotated ellipse as a boolean array."""
    h, w = shape
    cy, cx = center_yx
    y, x = np.ogrid[:h, :w]
    dy = y - float(cy)
    dx = x - float(cx)
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    # Local coordinates with u along the major axis.
    u = dx * c + dy * s
    v = -dx * s + dy * c
    major = max(float(major_radius), 1.0)
    minor = max(float(minor_radius), 1.0)
    return (u / major) ** 2 + (v / minor) ** 2 <= 1.0


def _component_geometry(component: np.ndarray) -> Tuple[float, float, float]:
    """Return (orientation_rad, major_sigma, minor_sigma) from component PCA."""
    yy, xx = np.where(component)
    if len(xx) < 3:
        return 0.0, 1.0, 1.0
    pts = np.stack([xx.astype(np.float64), yy.astype(np.float64)], axis=1)
    pts -= pts.mean(axis=0, keepdims=True)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order], 1e-6)
    vec = vecs[:, order[0]]
    angle = float(np.arctan2(vec[1], vec[0]))
    return angle, float(np.sqrt(vals[0])), float(np.sqrt(vals[1]))


def mask_metrics(label_map: np.ndarray, mask: np.ndarray, target_class_id: int) -> Dict[str, float | int]:
    """Compute target purity/coverage and basic mask geometry metrics."""
    m = np.asarray(mask, dtype=bool)
    target = np.asarray(label_map) == int(target_class_id)
    mask_pixels = int(m.sum())
    target_pixels = int(target.sum())
    target_inside = int(np.logical_and(m, target).sum())
    purity = float(target_inside / max(mask_pixels, 1))
    coverage = float(target_inside / max(target_pixels, 1))
    return {
        "mask_pixels": mask_pixels,
        "mask_area_fraction": float(mask_pixels / max(m.size, 1)),
        "target_pixels_in_tile": target_pixels,
        "target_pixels_in_mask": target_inside,
        "target_purity": purity,
        "target_class_coverage": coverage,
    }


def neighbor_class_histogram(
    label_map: np.ndarray,
    mask: np.ndarray,
    ring_px: int = 10,
) -> Dict[int, int]:
    """Count semantic classes in a ring immediately outside a mask."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return {}
    dilated = ndi.binary_dilation(m, iterations=max(1, int(ring_px)))
    ring = np.logical_and(dilated, ~m)
    vals, counts = np.unique(np.asarray(label_map)[ring], return_counts=True)
    return {int(k): int(v) for k, v in zip(vals.tolist(), counts.tolist())}


def adjacency_counts(label_map: np.ndarray, num_classes: int = 12) -> np.ndarray:
    """Return symmetric boundary-contact counts between semantic classes."""
    labels = np.asarray(label_map, dtype=np.int32)
    out = np.zeros((num_classes, num_classes), dtype=np.int64)
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        diff = a != b
        if not np.any(diff):
            continue
        aa = a[diff].ravel()
        bb = b[diff].ravel()
        valid = (aa >= 0) & (aa < num_classes) & (bb >= 0) & (bb < num_classes)
        aa = aa[valid]
        bb = bb[valid]
        np.add.at(out, (aa, bb), 1)
        np.add.at(out, (bb, aa), 1)
    return out


def _component_candidates(target: np.ndarray, min_component_pixels: int):
    labeled, n = ndi.label(target)
    if n <= 0:
        return []
    counts = np.bincount(labeled.ravel())
    comps = []
    for idx in range(1, len(counts)):
        if int(counts[idx]) >= int(min_component_pixels):
            comps.append((idx, int(counts[idx]), labeled == idx))
    comps.sort(key=lambda x: x[1], reverse=True)
    return comps


def _pick_weighted_index(rng: np.random.Generator, weights: np.ndarray) -> int:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return int(rng.integers(0, len(weights)))
    probs = weights / total
    return int(rng.choice(len(weights), p=probs))


def sample_semantic_mask(
    label_map: np.ndarray,
    target_class_id: int,
    rng: np.random.Generator,
    mode: str,
    config: SemanticMaskConfig,
) -> Dict[str, object] | None:
    """Sample a round/elliptical mask aligned to one semantic class.

    ``mode='interior'`` seeks high target purity. ``mode='boundary'`` deliberately
    crosses a real semantic boundary while keeping the requested class dominant.
    Thin classes are handled with elongated ellipses aligned to component PCA.
    """
    if mode not in {"interior", "boundary"}:
        raise ValueError("mode must be 'interior' or 'boundary'")
    labels = np.asarray(label_map, dtype=np.int32)
    target = labels == int(target_class_id)
    components = _component_candidates(target, int(config.min_component_pixels))
    if not components:
        return None

    h, w = labels.shape
    max_pixels = int(round(float(config.max_mask_area_frac) * labels.size))
    comp_weights = np.array([np.sqrt(float(size)) for _, size, _ in components], dtype=np.float64)

    for _ in range(max(1, int(config.max_attempts))):
        comp_idx = _pick_weighted_index(rng, comp_weights)
        component_id, component_pixels, component = components[comp_idx]
        dist_inside = ndi.distance_transform_edt(component)
        angle, major_sigma, minor_sigma = _component_geometry(component)

        if mode == "interior":
            yy, xx = np.where(component)
            if len(xx) == 0:
                continue
            distances = dist_inside[yy, xx]
            # Bias centers toward the interior, but retain some variation.
            center_idx = _pick_weighted_index(rng, np.maximum(distances, 0.5) ** 2)
            cy, cx = float(yy[center_idx]), float(xx[center_idx])
            local_clearance = float(max(distances[center_idx], 1.0))

            # For thick structures this is nearly circular; for thin elongated
            # structures the PCA-aligned major axis grows while the minor axis
            # respects local boundary clearance.
            minor = min(
                max(float(config.min_minor_radius), local_clearance * float(rng.uniform(0.55, 0.90))),
                float(config.max_major_radius),
            )
            elongation = float(np.clip(major_sigma / max(minor_sigma, 1.0), 1.0, 5.0))
            major = min(
                float(config.max_major_radius),
                max(minor, minor * float(rng.uniform(1.0, min(2.8, elongation + 0.5)))),
            )
            use_angle = angle + float(rng.normal(0.0, 0.10))
        else:
            # Choose a target pixel near the target/non-target boundary.
            eroded = ndi.binary_erosion(component, iterations=1)
            boundary = np.logical_and(component, ~eroded)
            yy, xx = np.where(boundary)
            if len(xx) == 0:
                continue
            center_idx = int(rng.integers(0, len(xx)))
            cy, cx = float(yy[center_idx]), float(xx[center_idx])
            # Boundary masks are moderately round but may follow elongated layers.
            base = float(rng.uniform(8.0, min(48.0, float(config.max_major_radius))))
            elongation = float(np.clip(major_sigma / max(minor_sigma, 1.0), 1.0, 3.5))
            minor = max(float(config.min_minor_radius), base * float(rng.uniform(0.45, 0.85)))
            major = min(float(config.max_major_radius), base * float(rng.uniform(0.9, min(1.8, elongation))))
            use_angle = angle + float(rng.normal(0.0, 0.18))

        mask = _ellipse_mask(
            (h, w),
            center_yx=(cy, cx),
            major_radius=major,
            minor_radius=minor,
            angle_rad=use_angle,
        )
        metrics = mask_metrics(labels, mask, target_class_id)
        if int(metrics["mask_pixels"]) < int(config.min_mask_pixels):
            continue
        if int(metrics["mask_pixels"]) > max_pixels:
            continue
        purity = float(metrics["target_purity"])
        if mode == "interior":
            if purity < float(config.interior_min_purity):
                continue
        else:
            if purity < float(config.boundary_min_purity):
                continue
            if purity > float(config.boundary_max_purity):
                continue

        return {
            "mask": mask,
            "mode": mode,
            "component_id": int(component_id),
            "component_pixels": int(component_pixels),
            "center_x": float(cx),
            "center_y": float(cy),
            "major_radius": float(major),
            "minor_radius": float(minor),
            "angle_deg": float(np.degrees(use_angle)),
            **metrics,
            "neighbor_histogram": neighbor_class_histogram(labels, mask, ring_px=int(config.ring_px)),
        }
    return None


def valid_target_classes(
    label_map: np.ndarray,
    config: SemanticMaskConfig,
    class_ids: Iterable[int] = TARGET_CLASS_IDS,
) -> Tuple[int, ...]:
    """Return target IDs with at least one component large enough to attempt a mask."""
    labels = np.asarray(label_map)
    valid = []
    for cid in class_ids:
        target = labels == int(cid)
        comps = _component_candidates(target, int(config.min_component_pixels))
        if comps:
            valid.append(int(cid))
    return tuple(valid)
