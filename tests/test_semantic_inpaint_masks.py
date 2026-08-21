from __future__ import annotations

import numpy as np

from scripts.synthetic_data.semantic_mask_utils import (
    SemanticMaskConfig,
    adjacency_counts,
    class_prompt,
    sample_semantic_mask,
    valid_target_classes,
)


def _cfg(**overrides):
    base = dict(
        min_component_pixels=64,
        min_mask_pixels=64,
        max_mask_area_frac=0.25,
        interior_min_purity=0.90,
        boundary_min_purity=0.45,
        boundary_max_purity=0.90,
        min_minor_radius=2.0,
        max_major_radius=32.0,
        max_attempts=300,
        ring_px=4,
    )
    base.update(overrides)
    return SemanticMaskConfig(**base)


def test_interior_mask_stays_on_requested_class():
    labels = np.zeros((128, 128), dtype=np.uint8)
    labels[20:110, 10:118] = 4
    result = sample_semantic_mask(labels, 4, np.random.default_rng(11), "interior", _cfg())
    assert result is not None
    assert float(result["target_purity"]) >= 0.90
    assert int(result["mask_pixels"]) >= 64


def test_thin_layer_can_use_elongated_ellipse():
    labels = np.zeros((128, 128), dtype=np.uint8)
    labels[48:56, 10:118] = 1
    cfg = _cfg(min_component_pixels=128, min_mask_pixels=40, interior_min_purity=0.85)
    result = sample_semantic_mask(labels, 1, np.random.default_rng(5), "interior", cfg)
    assert result is not None
    assert float(result["target_purity"]) >= 0.85
    assert float(result["major_radius"]) >= float(result["minor_radius"])


def test_boundary_mask_contains_target_and_neighbor_context():
    labels = np.zeros((128, 128), dtype=np.uint8)
    labels[:, :64] = 3
    labels[:, 64:] = 4
    cfg = _cfg(boundary_min_purity=0.35, boundary_max_purity=0.85)
    result = sample_semantic_mask(labels, 3, np.random.default_rng(17), "boundary", cfg)
    assert result is not None
    purity = float(result["target_purity"])
    assert 0.35 <= purity <= 0.85
    neighbors = result["neighbor_histogram"]
    assert 4 in neighbors or 3 in neighbors


def test_valid_target_classes_and_adjacency():
    labels = np.zeros((64, 64), dtype=np.uint8)
    labels[:, :20] = 1
    labels[:, 20:45] = 3
    labels[:, 45:] = 4
    valid = valid_target_classes(labels, _cfg(min_component_pixels=32, min_mask_pixels=16))
    assert 1 in valid and 3 in valid and 4 in valid
    adj = adjacency_counts(labels)
    assert adj[1, 3] > 0
    assert adj[3, 4] > 0
    assert adj[1, 4] == 0


def test_prompt_uses_explicit_histology_class():
    text = class_prompt(10)
    assert "squamous cell carcinoma" in text
    assert "target tissue" in text
