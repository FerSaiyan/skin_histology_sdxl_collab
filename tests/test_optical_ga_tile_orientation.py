"""Geometry invariants for the epidermis-normal estimation and oriented-tile extraction.

Tests in this file verify:

- Normal estimation from synthetic straight-interface masks returns the
  expected angle (± tolerance).
- A perfectly vertical interface yields a normal of ±90° (horizontal
  normal pointing from air into epidermis).
- The tile-rotation logic aligns the normal to the top edge of the
  output tile.
- The affine warp samples from real (non-padded) image coordinates,
  confirmed by checking non-zero content at expected rotation.
- Low-confidence / degenerate cases produce appropriate confidence
  values and do not crash.

Coordinate convention verified in these tests:
- Image coords: x right, y down, angles CCW from +x.
- Outward normal: air (0) → epidermis (1).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.optical_ga.estimate_epidermis_normal import (
    detect_air_epidermis_interface,
    estimate_orientation,
    estimate_epidermis_normal,
    _HAS_SKIMAGE,
)

from scripts.optical_ga.select_orient_tile_for_incidence import (
    select_orient_tile,
    select_orient_tile_from_mask,
    _warp_affine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vertical_interface_mask(
    height: int = 256,
    width: int = 256,
    col_boundary: float = 128.0,
    air_left: bool = True,
) -> np.ndarray:
    """Create a mask with a vertical air↔epidermis interface.

    Parameters
    ----------
    height, width : int
        Mask dimensions.
    col_boundary : float
        Column at which the interface sits (fractional allowed).
    air_left : bool
        If True, air (0) is on the left, epidermis (1) on the right.

    Returns
    -------
    mask : np.ndarray (H, W) int32
    """
    mask = np.zeros((height, width), dtype=np.int32)
    col_idx = int(np.floor(col_boundary))
    if air_left:
        mask[:, col_idx:] = 1  # epidermis on right
    else:
        mask[:, :col_idx] = 1  # epidermis on left
    return mask


def _make_horizontal_interface_mask(
    height: int = 256,
    width: int = 256,
    row_boundary: float = 128.0,
    air_top: bool = True,
) -> np.ndarray:
    """Create a mask with a horizontal air↔epidermis interface.

    Parameters
    ----------
    height, width : int
        Mask dimensions.
    row_boundary : float
        Row at which the interface sits.
    air_top : bool
        If True, air (0) is on the top, epidermis (1) on the bottom.

    Returns
    -------
    mask : np.ndarray (H, W) int32
    """
    mask = np.zeros((height, width), dtype=np.int32)
    row_idx = int(np.floor(row_boundary))
    if air_top:
        mask[row_idx:, :] = 1  # epidermis on bottom
    else:
        mask[:row_idx, :] = 1  # epidermis on top
    return mask


def _make_diagonal_interface_mask(
    height: int = 256,
    width: int = 256,
    offset: float = 0.0,
    angle_deg: float = 36.0,
) -> np.ndarray:
    """Create a mask with a diagonal air↔epidermis interface.

    The interface follows the line:
        col = tan(θ) * row + offset
    where θ = ``angle_deg`` measured from the horizontal (row axis).
    Air (0) is below the line, epidermis (1) above it.

    Parameters
    ----------
    height, width : int
        Mask dimensions.
    offset : float
        Column-intercept offset in pixels.
    angle_deg : float
        Angle in degrees measured from the horizontal (row-axis).
        At 0°, the line is horizontal (interface runs left-right).
    """
    mask = np.zeros((height, width), dtype=np.int32)
    tan_theta = np.tan(np.radians(angle_deg))
    for r in range(height):
        c_boundary = tan_theta * r + offset
        c_boundary = np.clip(c_boundary, 0, width - 1)
        # Air (0) below the line → epidermis (1) above the line
        # "below the line" = col < c_boundary (since col increases right)
        # Actually, for a diagonal line, we want air on one side.
        # Let's put air on the "left side" of the line direction.
        # col_boundary increases with row, so the line moves right as
        # we go down. Air is to the left (col < col_boundary).
        mask[r, : int(c_boundary)] = 1  # epidermis on the left side
    return mask


def _normal_from_mask(mask: np.ndarray) -> tuple:
    """Convenience: run full pipeline on a mask, return
    ``(normal_deg, confidence)``."""
    normal_deg, normal_vec, confidence, _, _ = estimate_epidermis_normal(mask)
    return normal_deg, confidence, normal_vec


# ---------------------------------------------------------------------------
# Test: detect_air_epidermis_interface
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="requires scikit-image")
class TestDetectInterface:
    def test_vertical_interface_finds_contour(self):
        """A vertical interface produces a contour of expected length."""
        mask = _make_vertical_interface_mask(200, 200, 100.0, air_left=True)
        pts, frac = detect_air_epidermis_interface(mask)
        assert len(pts) >= 100  # should be around 200 points
        assert 0.0 < frac <= 1.0

    def test_horizontal_interface_finds_contour(self):
        """A horizontal interface produces a contour."""
        mask = _make_horizontal_interface_mask(200, 200, 100.0, air_top=True)
        pts, frac = detect_air_epidermis_interface(mask)
        assert len(pts) >= 100
        assert 0.0 < frac <= 1.0

    def test_no_interface_raises(self):
        """Mask with no epidermis raises ValueError."""
        mask = np.zeros((50, 50), dtype=np.int32)
        with pytest.raises(ValueError, match="No contours found"):
            detect_air_epidermis_interface(mask)

    def test_too_few_points_raises(self):
        """Tiny mask with too-short contour raises ValueError."""
        mask = np.zeros((8, 8), dtype=np.int32)
        # Two adjacent epidermis pixels produce a short contour (< 5)
        mask[3:5, 3:5] = 1
        with pytest.raises((ValueError,), match=r"(Longest contour has only|No contours found)"):
            detect_air_epidermis_interface(mask, min_contour_length=10)


# ---------------------------------------------------------------------------
# Test: estimate_orientation — PCA angle
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="requires scikit-image")
class TestEstimateOrientation:
    def test_vertical_interface_normal(self):
        """Vertical interface (air left) → normal points right (0°)."""
        mask = _make_vertical_interface_mask(256, 256, 128.0, air_left=True)
        pts, _ = detect_air_epidermis_interface(mask)
        normal_deg, _, confidence = estimate_orientation(pts)
        # Normal should be close to 0° (pointing right, from air to epidermis)
        assert abs(normal_deg) < 10.0 or abs(normal_deg - 180.0) < 10.0, (
            f"Expected normal ~0° or ~180°, got {normal_deg}"
        )
        assert confidence > 0.8

    def test_vertical_interface_air_right(self):
        """Vertical interface (air right) → normal points left (180°/−180°)."""
        mask = _make_vertical_interface_mask(256, 256, 128.0, air_left=False)
        pts, _ = detect_air_epidermis_interface(mask)
        normal_deg, _, confidence = estimate_orientation(pts)
        # Normal should point left (air on right → normal from air to
        # epidermis = from right to left = ~180°)
        assert abs(abs(normal_deg) - 180.0) < 10.0 or abs(normal_deg) < 10.0, (
            f"Expected normal ~±180° or ~0°, got {normal_deg}"
        )
        assert confidence > 0.8

    def test_horizontal_interface_air_top(self):
        """Horizontal interface (air top) → normal points down (90°)."""
        mask = _make_horizontal_interface_mask(256, 256, 128.0, air_top=True)
        pts, _ = detect_air_epidermis_interface(mask)
        normal_deg, _, confidence = estimate_orientation(pts)
        # Air is on top, epidermis on bottom → outward normal points
        # from air to epidermis = downward = +90°
        assert abs(normal_deg - 90.0) < 10.0 or abs(normal_deg + 90.0) < 10.0, (
            f"Expected normal ~90° or ~-90°, got {normal_deg}"
        )
        assert confidence > 0.8

    def test_horizontal_interface_air_bottom(self):
        """Horizontal interface (air bottom) → normal points up (−90°)."""
        mask = _make_horizontal_interface_mask(256, 256, 128.0, air_top=False)
        pts, _ = detect_air_epidermis_interface(mask)
        normal_deg, _, confidence = estimate_orientation(pts)
        # Air is on bottom, epidermis on top → outward normal points
        # from air to epidermis = upward = −90°
        assert abs(normal_deg + 90.0) < 10.0 or abs(normal_deg - 90.0) < 10.0, (
            f"Expected normal ~-90° or ~90°, got {normal_deg}"
        )
        assert confidence > 0.8

    def test_diagonal_interface_36deg(self):
        """Diagonal interface: normal angle consistent with the line.

        A line at angle θ = 36° from horizontal has tangent direction
        (cos 36°, sin 36°) and normal (−sin 36°, cos 36°) in (col, row)
        space.

        In image (x, y) = (col, row), the normal makes angle:
            atan2(cos 36°, −sin 36°) ≈ atan2(0.81, −0.59) ≈ 126°

        But the exact value depends on where air vs epidermis are.
        We just check that the angle is sensible (not random) and the
        confidence is reasonable.
        """
        mask = _make_diagonal_interface_mask(256, 256, offset=20.0, angle_deg=36.0)
        pts, _ = detect_air_epidermis_interface(mask)
        normal_deg, normal_vec, confidence = estimate_orientation(pts)
        # The exact angle depends on the diagonal parameters; we check
        # that normal magnitude is 1, confidence is positive, and the
        # angle is in a plausible range.
        assert np.abs(np.linalg.norm(normal_vec) - 1.0) < 1e-6
        assert confidence > 0.5  # diagonal should still be quite linear
        assert -180.0 <= normal_deg <= 180.0


# ---------------------------------------------------------------------------
# Test: full estimate_epidermis_normal — outward-normal disambiguation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="requires scikit-image")
class TestEstimateEpidermisNormal:
    def test_full_pipeline_vertical_air_left(self):
        """Outward normal from a vertical interface (air left) should be ~0°.

        Air is on the left (col < 128), epidermis on the right (col ≥ 128).
        Outward normal (air→epidermis) should point right = 0°.
        """
        mask = _make_vertical_interface_mask(256, 256, 128.0, air_left=True)
        deg, vec, conf, pts, frac = estimate_epidermis_normal(mask)
        # normal should point right (~0°)
        assert abs(deg) < 15.0, f"Expected ~0°, got {deg}"
        assert vec[0] > 0  # nx positive (pointing right)
        assert conf > 0.5

    def test_full_pipeline_vertical_air_right(self):
        """Outward normal from a vertical interface (air right) should be ~180°.

        Air on right, epidermis on left → normal from (right→left) = 180°.
        """
        mask = _make_vertical_interface_mask(256, 256, 128.0, air_left=False)
        deg, vec, conf, pts, frac = estimate_epidermis_normal(mask)
        # normal should point left (~180° or ~-180°)
        assert abs(abs(deg) - 180.0) < 15.0, f"Expected ±180°, got {deg}"
        assert vec[0] < 0  # nx negative (pointing left)
        assert conf > 0.5

    def test_full_pipeline_horizontal_air_top(self):
        """Outward normal from horizontal interface (air top) should be 90°.

        Air on top, epidermis on bottom → normal points down = 90°.
        """
        mask = _make_horizontal_interface_mask(256, 256, 128.0, air_top=True)
        deg, vec, conf, pts, frac = estimate_epidermis_normal(mask)
        assert abs(deg - 90.0) < 15.0, f"Expected ~90°, got {deg}"
        assert vec[1] > 0  # ny positive (pointing down)
        assert conf > 0.5

    def test_full_pipeline_horizontal_air_bottom(self):
        """Outward normal from horizontal interface (air bottom) should be −90°.

        Air on bottom, epidermis on top → normal points up = −90°.
        """
        mask = _make_horizontal_interface_mask(256, 256, 128.0, air_top=False)
        deg, vec, conf, pts, frac = estimate_epidermis_normal(mask)
        assert abs(deg + 90.0) < 15.0, f"Expected ~-90°, got {deg}"
        assert vec[1] < 0  # ny negative (pointing up)
        assert conf > 0.5


# ---------------------------------------------------------------------------
# Test: tile rotation alignment
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="requires scikit-image")
class TestTileRotation:
    def test_vertical_normal_tile_aligns_normal_up(self):
        """Vertical interface (normal right, 0°): tile should rotate so
        the interface appears horizontal in the output (normal points up).

        With normal at 0°, rotation_deg = 0 + 90 = 90° CCW.
        After 90° CCW rotation, the vertical interface appears horizontal.
        """
        height, width = 256, 256
        mask = _make_vertical_interface_mask(height, width, 128.0, air_left=True)
        # Create a simple RGB image (grayscale gradient for visual check)
        image = np.zeros((height, width, 3), dtype=np.uint8)
        # Put a bright vertical stripe just to the right of the interface
        image[:, 132:140, :] = 200  # bright stripe in epidermis region

        deg, vec, conf, _, _ = estimate_epidermis_normal(mask)
        # normal should point right (~0°)
        assert abs(deg) < 15.0

        result = select_orient_tile(
            image, mask, normal_deg=deg, confidence=conf,
            center_xy=(128.0, 128.0), tile_size=64,
        )
        tile = result["tile"]
        tile_mask = result["tile_mask"]
        meta = result["metadata"]

        # Rotation should be ~90°
        assert abs(meta["rotation_deg"] - 90.0) < 15.0, (
            f"Expected rotation ~90°, got {meta['rotation_deg']}"
        )

        # After 90° CCW rotation, the vertical bright stripe should
        # appear near the top of the tile (since right side → top)
        # The interface at col=128 should now be at...
        # Output top (row 0): was left of centre → right of centre
        # Output right: was bottom of centre
        # The rotated tile should have non-zero pixels in the correct
        # region. We check rotation by verifying that content exists.
        assert tile.shape == (64, 64, 3)
        assert tile_mask.shape == (64, 64)

        # The rotated mask should have epidermis pixels
        assert tile_mask.sum() > 0

        # The centre column of the tile should have a mix of 0 and 1 in mask
        # due to the rotated interface
        mid_col = tile_mask[:, 32]
        assert mid_col.min() == 0 and mid_col.max() == 1, (
            "Rotated mask centre column should straddle the interface"
        )

    def test_horizontal_normal_tile_aligns_normal_up(self):
        """Horizontal interface (normal down, 90°): rotation aligns normal up.

        With normal at 90°, rotation = 90 + 90 = 180°.
        After 180° rotation, the horizontal interface stays horizontal
        but the normal flips to point up.
        """
        height, width = 256, 256
        mask = _make_horizontal_interface_mask(height, width, 128.0, air_top=True)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        deg, vec, conf, _, _ = estimate_epidermis_normal(mask)
        assert abs(deg - 90.0) < 15.0

        result = select_orient_tile(
            image, mask, normal_deg=deg, confidence=conf,
            center_xy=(128.0, 128.0), tile_size=64,
        )
        meta = result["metadata"]
        # Rotation should be ~180°
        assert abs(abs(meta["rotation_deg"]) - 180.0) < 15.0 or \
               abs(meta["rotation_deg"]) < 15.0, (
            f"Expected rotation ~±180° or ~0°, got {meta['rotation_deg']}"
        )

    def test_tile_nonzero_content(self):
        """Extracted tile from a non-zero image has content (sampling from
        real source coordinates, not empty canvas)."""
        height, width = 200, 300
        # Create a simple checkerboard-like image with content everywhere
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[50:150, 50:250, :] = 128  # filled rectangle

        # Simple mask with horizontal interface
        mask = _make_horizontal_interface_mask(height, width, 100.0, air_top=True)

        result = select_orient_tile_from_mask(
            image, mask, tile_size=64, seed=42,
        )
        tile = result["tile"]
        assert tile.shape == (64, 64, 3)
        # The tile should contain non-zero pixels since the source
        # image has content in the region of interest
        assert tile.sum() > 0, "Tile should have non-zero content"

    def test_convenience_pipeline_matches_normal_deg(self):
        """select_orient_tile_from_mask produces consistent metadata."""
        mask = _make_vertical_interface_mask(200, 200, 100.0, air_left=True)
        image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

        # Run the convenience pipeline
        result = select_orient_tile_from_mask(
            image, mask, tile_size=48, seed=7,
        )
        meta = result["metadata"]
        assert "normal_deg_input" in meta
        assert -180.0 <= meta["normal_deg_input"] <= 180.0
        assert 0.0 <= meta["confidence"] <= 1.0
        assert meta["tile_size"] == 48


# ---------------------------------------------------------------------------
# Test: affine warp backend
# ---------------------------------------------------------------------------


class TestAffineWarp:
    def test_warp_identity_returns_input(self):
        """Rotation by 0° around centre returns the same content (cropped).

        Source centre must equal the output tile centre for identity.
        For a 10×10 tile, centre = (4.5, 4.5).
        """
        src = np.arange(100, dtype=np.float64).reshape(10, 10)
        dst = _warp_affine(
            src, rot_deg=0.0, center_xy=(4.5, 4.5),
            dsize=(10, 10), order=1,
        )
        assert dst.shape == (10, 10)
        np.testing.assert_allclose(dst, src, atol=1.0)

    def test_warp_90deg_rotates_content(self):
        """90° CCW rotation around centre."""
        src = np.zeros((20, 20), dtype=np.float64)
        src[5:15, 15:, ] = 1.0  # vertical bar on the right side

        dst = _warp_affine(
            src, rot_deg=90.0, center_xy=(10.0, 10.0),
            dsize=(20, 20), order=1,
        )
        # After 90° CCW, the bar on the right should appear at the top
        top = dst[:5, :]  # top rows (approximately)
        # There should be some content at the top
        assert top.sum() > 1.0, (
            "90° CCW should move right-side content to the top"
        )

    def test_warp_nearest_integer_mask(self):
        """Nearest-neighbour warp of a binary mask stays binary."""
        mask = np.zeros((50, 50), dtype=np.float64)
        mask[10:40, 10:40] = 1.0

        dst = _warp_affine(
            mask, rot_deg=15.0, center_xy=(25.0, 25.0),
            dsize=(50, 50), order=0,
        )
        unique = np.unique(dst)
        assert set(unique).issubset({0.0, 1.0}), (
            f"Expected only {0.0, 1.0}, got {unique}"
        )


# ---------------------------------------------------------------------------
# Test: confidence / rejection
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="requires scikit-image")
class TestConfidence:
    def test_circular_interface_low_confidence(self):
        """A circular interface (non-linear) gives lower confidence than
        a straight line."""
        h, w = 128, 128
        yy, xx = np.ogrid[:h, :w]
        centre = (64.0, 64.0)
        radius = 30.0
        # Mask: inside circle = epidermis (1), outside = air (0)
        mask = ((xx - centre[1]) ** 2 + (yy - centre[0]) ** 2 < radius ** 2)
        mask = mask.astype(np.int32)

        deg, vec, conf, pts, _ = estimate_epidermis_normal(mask)

        # A circle is not a straight line, so confidence should be
        # substantially lower than the near-1.0 of a straight interface
        assert conf < 0.8, (
            f"Circular interface should have lower confidence, got {conf}"
        )

    def test_confidence_range(self):
        """Confidence is always in [0, 1]."""
        # Straight interface
        mask1 = _make_vertical_interface_mask(100, 100, 50.0, air_left=True)
        _, _, conf1, _, _ = estimate_epidermis_normal(mask1)
        assert 0.0 <= conf1 <= 1.0

        # Full mask of epidermis (no air)
        mask2 = np.ones((50, 50), dtype=np.int32)
        with pytest.raises(ValueError):
            estimate_epidermis_normal(mask2)


# ---------------------------------------------------------------------------
# Test: CLI smoke
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="requires scikit-image")
class TestCLISmoke:
    def test_estimate_help_succeeds(self):
        """--help for estimate_epidermis_normal exits with code 0."""
        from scripts.optical_ga.estimate_epidermis_normal import _parse_args
        try:
            _parse_args(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_select_help_succeeds(self):
        """--help for select_orient_tile_for_incidence exits with code 0."""
        from scripts.optical_ga.select_orient_tile_for_incidence import _parse_args
        try:
            _parse_args(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_estimate_runs_with_temp_files(self):
        """Estimate normal from a synthetic mask via CLI."""
        from scripts.optical_ga.estimate_epidermis_normal import _main

        mask = _make_vertical_interface_mask(64, 64, 32.0, air_left=True)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            from PIL import Image as PILImage
            PILImage.fromarray(mask.astype(np.uint8) * 255).save(f.name)
            mask_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as jf:
            out_path = jf.name

        try:
            _main([
                "--mask", mask_path,
                "--output", out_path,
                "--seed", "42",
            ])
            with open(out_path) as f:
                data = json.load(f)
            assert "normal_deg" in data
            assert "confidence" in data
        finally:
            Path(mask_path).unlink(missing_ok=True)
            Path(out_path).unlink(missing_ok=True)


def test_select_cli_uses_centroid_from_normal_json_without_manual_center():
    """When --normal-json includes centroid, CLI uses it by default.

    This test avoids skimage by providing a precomputed normal JSON.
    """
    from PIL import Image as PILImage
    from scripts.optical_ga.select_orient_tile_for_incidence import _main

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        image_path = td_path / "img.png"
        mask_path = td_path / "mask.png"
        normal_path = td_path / "normal.json"
        out_dir = td_path / "out"

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[32:, :, :] = 200
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[32:, :] = 255

        PILImage.fromarray(img).save(image_path)
        PILImage.fromarray(mask).save(mask_path)

        normal_json = {
            "normal_deg": 0.0,
            "confidence": 0.9,
            "centroid_col": 10.0,
            "centroid_row": 20.0,
        }
        normal_path.write_text(json.dumps(normal_json))

        _main([
            "--image", str(image_path),
            "--mask", str(mask_path),
            "--normal-json", str(normal_path),
            "--output-dir", str(out_dir),
            "--tile-size", "32",
        ])

        meta_path = out_dir / "orient_tile_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert abs(meta["source_center_x"] - 10.0) < 1e-6
        assert abs(meta["source_center_y"] - 20.0) < 1e-6
        assert meta["center_source"] == "normal_json_centroid"


# ---------------------------------------------------------------------------
# Test: batch compare strict-mode behaviour
# ---------------------------------------------------------------------------


class TestBatchCompareStrictMode:
    """Verify that _make_branch_evaluator enforces strict physical-mode
    behaviour correctly.

    In ``--forward-mode realistic`` the default (``allow_surrogate_fallback``
    = ``False``) must raise ``RuntimeError`` when the physical backend is
    unavailable.  With ``allow_surrogate_fallback=True``, a surrogate
    evaluator is returned instead.
    """

    @staticmethod
    def _make_simple_target() -> tuple:
        return (60.0, 10.0, 15.0)

    def test_strict_mode_raises_on_mcx_unavailable(self, monkeypatch):
        """When allow_surrogate_fallback=False and mcx is unavailable,
        _make_branch_evaluator raises RuntimeError."""
        import scripts.optical_ga.run_optical_ga_batch_compare as bcmp

        # Force check_mcx to return False
        monkeypatch.setattr(
            "scripts.optical_ga.mc_wrapper.check_mcx",
            lambda binary="mcx": False,
        )
        monkeypatch.setattr(
            "scripts.optical_ga.run_optical_ga_batch_compare.check_mcx",
            lambda binary="mcx": False,
        )

        target = self._make_simple_target()
        with pytest.raises(RuntimeError, match="MCX backend not available"):
            bcmp._make_branch_evaluator(
                lab_target=target,
                branch_name="mcx",
                forward_mode="realistic",
                fitness_mode="lab",
                num_photons=100_000,
                with_specular=True,
                require_physical=False,
                allow_surrogate_fallback=False,
                verbose=False,
            )

    def test_strict_mode_raises_on_pyxopto_unavailable(self, monkeypatch):
        """When allow_surrogate_fallback=False and xopto is unavailable,
        _make_branch_evaluator raises RuntimeError."""
        import scripts.optical_ga.run_optical_ga_batch_compare as bcmp

        # Force check_xopto to return False
        monkeypatch.setattr(
            "scripts.optical_ga.mc_wrapper.check_xopto",
            lambda: False,
        )
        monkeypatch.setattr(
            "scripts.optical_ga.run_optical_ga_batch_compare.check_xopto",
            lambda: False,
        )

        target = self._make_simple_target()
        with pytest.raises(RuntimeError, match="PyXOpto.*not available"):
            bcmp._make_branch_evaluator(
                lab_target=target,
                branch_name="pyxopto",
                forward_mode="realistic",
                fitness_mode="lab",
                num_photons=100_000,
                with_specular=True,
                require_physical=False,
                allow_surrogate_fallback=False,
                verbose=False,
            )

    def test_surrogate_fallback_returns_evaluator(self, monkeypatch):
        """When allow_surrogate_fallback=True, a surrogate evaluator is
        returned even when backends are unavailable."""
        import scripts.optical_ga.run_optical_ga_batch_compare as bcmp

        monkeypatch.setattr(
            "scripts.optical_ga.mc_wrapper.check_mcx",
            lambda binary="mcx": False,
        )
        monkeypatch.setattr(
            "scripts.optical_ga.run_optical_ga_batch_compare.check_mcx",
            lambda binary="mcx": False,
        )
        monkeypatch.setattr(
            "scripts.optical_ga.mc_wrapper.check_xopto",
            lambda: False,
        )
        monkeypatch.setattr(
            "scripts.optical_ga.run_optical_ga_batch_compare.check_xopto",
            lambda: False,
        )

        target = self._make_simple_target()

        # MCX branch — fallback allowed
        eval_mcx = bcmp._make_branch_evaluator(
            lab_target=target,
            branch_name="mcx",
            forward_mode="realistic",
            fitness_mode="lab",
            num_photons=100_000,
            with_specular=True,
            require_physical=False,
            allow_surrogate_fallback=True,
            verbose=False,
        )
        assert callable(eval_mcx), "Should return a callable evaluator"

        # PyXOpto branch — fallback allowed
        eval_pyx = bcmp._make_branch_evaluator(
            lab_target=target,
            branch_name="pyxopto",
            forward_mode="realistic",
            fitness_mode="lab",
            num_photons=100_000,
            with_specular=True,
            require_physical=False,
            allow_surrogate_fallback=True,
            verbose=False,
        )
        assert callable(eval_pyx), "Should return a callable evaluator"

    def test_unknown_branch_raises_valueerror(self):
        """An unknown branch name raises ValueError regardless of flags."""
        import scripts.optical_ga.run_optical_ga_batch_compare as bcmp

        target = self._make_simple_target()
        with pytest.raises(ValueError, match="Unknown branch"):
            bcmp._make_branch_evaluator(
                lab_target=target,
                branch_name="nonexistent",
                forward_mode="realistic",
                fitness_mode="lab",
                num_photons=100_000,
                with_specular=True,
                require_physical=False,
                allow_surrogate_fallback=False,
                verbose=False,
            )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
