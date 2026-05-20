from __future__ import annotations

import numpy as np

from scripts.optical_ga.build_tiles_for_simulation import (
    _air_epidermis_interface_px,
    _iter_grid_positions,
)


def test_iter_grid_positions_covers_image_edges():
    coords = list(_iter_grid_positions(h=1000, w=1000, tile_size=512, stride=256))
    assert coords[0] == (0, 0)
    # Ensure last coordinate reaches the exact boundary-aligned window.
    assert coords[-1] == (488, 488)


def test_air_epidermis_interface_count_nonzero_for_simple_split():
    mask = np.zeros((16, 16), dtype=np.int32)
    mask[:, 8:] = 1
    px = _air_epidermis_interface_px(mask)
    assert px > 0
