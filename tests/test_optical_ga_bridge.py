"""Tests for the Histo-Seg → optical-GA bridge scripts."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.optical_ga.build_histoseg_label_volume import (
    _rgb_to_class_map,
    _extrude_to_3d,
    _compute_histogram,
    HISTOSEG_COLOR_TO_CLASS_ID,
    CLASS_NAME_BY_ID,
)
from scripts.optical_ga.label_to_optical_priors import (
    _load_priors_config,
    _load_label_volume,
    _compute_label_counts,
    _build_prior_index_map,
    _validate_priors_table,
    DEFAULT_PRIORS_TABLE,
    _GENERIC_FALLBACK_PRIORS,
    _EXPECTED_PRIOR_KEYS,
)


# ---------------------------------------------------------------------------
# Tests for build_histoseg_label_volume.py
# ---------------------------------------------------------------------------


class TestBuildHistosegLabelVolume:
    def test_rgb_to_class_map_all_known(self):
        """All 12 colors from the canonical palette are correctly mapped."""
        h, w = 12, 1
        rgb_list = list(HISTOSEG_COLOR_TO_CLASS_ID.keys())
        expected_ids = [HISTOSEG_COLOR_TO_CLASS_ID[c] for c in rgb_list]
        mask = np.array(rgb_list, dtype=np.uint8).reshape(h, w, 3)

        class_map, unknowns = _rgb_to_class_map(mask)
        assert len(unknowns) == 0
        for i, cid in enumerate(expected_ids):
            assert class_map[i, 0] == cid, (
                f"Color {rgb_list[i]} → expected class {cid}, got {class_map[i, 0]}"
            )

    def test_rgb_to_class_map_unknown_color(self):
        """Unknown RGB values are mapped to class 0 (background) and reported."""
        mask = np.zeros((5, 5, 3), dtype=np.uint8)
        mask[2, 2, :] = [42, 42, 42]  # not in palette
        class_map, unknowns = _rgb_to_class_map(mask)
        # The unknown pixel should be mapped to 0
        assert class_map[2, 2] == 0
        assert len(unknowns) >= 1
        assert (42, 42, 42) in unknowns

    def test_rgb_to_class_map_background_black(self):
        """Background black (0,0,0) maps to class 0."""
        mask = np.zeros((4, 4, 3), dtype=np.uint8)
        class_map, unknowns = _rgb_to_class_map(mask)
        assert np.all(class_map == 0)
        assert len(unknowns) == 0

    def test_rgb_to_class_map_epidermis(self):
        """Epidermis gray (224,224,224) maps to class 1."""
        mask = np.full((2, 2, 3), [224, 224, 224], dtype=np.uint8)
        class_map, unknowns = _rgb_to_class_map(mask)
        assert np.all(class_map == 1)
        assert len(unknowns) == 0

    def test_extrude_to_3d_depth1(self):
        """depth=1 adds a leading singleton axis."""
        arr = np.zeros((10, 10), dtype=np.uint8)
        vol = _extrude_to_3d(arr, 1)
        assert vol.shape == (1, 10, 10)

    def test_extrude_to_3d_depth3(self):
        """depth=3 repeats the 2D map 3 times along axis 0."""
        arr = np.zeros((10, 10), dtype=np.uint8)
        vol = _extrude_to_3d(arr, 3)
        assert vol.shape == (3, 10, 10)
        assert np.all(vol[0] == arr)
        assert np.all(vol[1] == arr)
        assert np.all(vol[2] == arr)

    def test_extrude_to_3d_depth_invalid(self):
        """depth < 1 raises ValueError."""
        arr = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="Depth must be >= 1"):
            _extrude_to_3d(arr, 0)

    def test_compute_histogram(self):
        """Histogram counts expected pixel fractions."""
        arr = np.array([
            [0, 1],
            [2, 1],
        ], dtype=np.uint8)
        hist = _compute_histogram(arr)
        assert hist.get("background") == 1
        assert hist.get("epidermis") == 2
        assert hist.get("reticular_dermis") == 1

    def test_class_names_match_ids(self):
        """class_name lookup has all 12 entries."""
        for cid in range(12):
            assert cid in CLASS_NAME_BY_ID
        assert CLASS_NAME_BY_ID[0] == "background"
        assert CLASS_NAME_BY_ID[1] == "epidermis"
        assert CLASS_NAME_BY_ID[9] == "basal_cell_carcinoma"

    def test_cli_help(self):
        """The CLI --help flag prints usage and exits 0."""
        # Test via subprocess to avoid sys.exit side effects
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.optical_ga.build_histoseg_label_volume",
             "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Build a class-ID label volume" in result.stdout
        assert "--mask" in result.stdout


# ---------------------------------------------------------------------------
# Tests for label_to_optical_priors.py
# ---------------------------------------------------------------------------


class TestLabelToOpticalPriors:
    def test_load_priors_config_default(self):
        """None input returns built-in defaults."""
        table = _load_priors_config(None)
        assert len(table) == 12
        assert "0" in table
        assert table["0"]["class_name"] == "background"

    def test_load_priors_config_custom(self):
        """Custom JSON path loads and returns content."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "metadata": {"test": True},
                "5": {"class_name": "custom", "priors": {"melanin_min": 0.0}},
            }, f)
            fname = f.name
        try:
            table = _load_priors_config(fname)
            assert "5" in table
            assert table["5"]["class_name"] == "custom"
            assert "metadata" not in table  # stripped
        finally:
            Path(fname).unlink()

    def test_load_priors_config_missing_file(self):
        """Missing file falls back to defaults with a warning."""
        table = _load_priors_config("/tmp/nonexistent_priors.json")
        assert len(table) == 12

    def test_load_label_volume_2d(self):
        """Load a 2D label .npy array."""
        arr = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, arr)
            fname = f.name
        try:
            loaded = _load_label_volume(fname)
            assert np.array_equal(loaded, arr)
        finally:
            Path(fname).unlink()

    def test_load_label_volume_3d(self):
        """Load a 3D label .npy array."""
        arr = np.zeros((3, 4, 5), dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, arr)
            fname = f.name
        try:
            loaded = _load_label_volume(fname)
            assert loaded.shape == (3, 4, 5)
        finally:
            Path(fname).unlink()

    def test_load_label_volume_float_conversion(self):
        """Float arrays are converted to int32."""
        arr = np.array([[0.0, 1.0], [2.0, 3.0]])
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, arr)
            fname = f.name
        try:
            loaded = _load_label_volume(fname)
            assert loaded.dtype.kind == "i"
            assert np.array_equal(loaded, arr.astype(np.int32))
        finally:
            Path(fname).unlink()

    def test_label_counts(self):
        """_compute_label_counts tallies correctly."""
        arr = np.array([[0, 1], [2, 1]], dtype=np.int32)
        counts = _compute_label_counts(arr)
        assert counts == {"0": 1, "1": 2, "2": 1}

    def test_validate_priors_table_ok(self):
        """Default table has no validation issues."""
        issues = _validate_priors_table(DEFAULT_PRIORS_TABLE)
        assert issues == []

    def test_validate_priors_table_missing_priors(self):
        """Missing 'priors' key is flagged."""
        bad = {"0": {"class_name": "test"}}
        issues = _validate_priors_table(bad)
        assert len(issues) >= 1
        assert any("missing 'priors'" in i for i in issues)

    def test_validate_priors_table_missing_keys(self):
        """Missing individual prior keys are flagged."""
        bad = {"0": {"class_name": "test", "priors": {}}}
        issues = _validate_priors_table(bad)
        assert len(issues) > 10  # many missing
        assert any("melanin_min" in i for i in issues)

    def test_build_prior_index_map_known(self):
        """Known class IDs get their index."""
        vol = np.array([[0, 1], [4, 9]], dtype=np.uint8)
        index_map = _build_prior_index_map(vol, DEFAULT_PRIORS_TABLE)
        # class 0 → index 0, class 1 → index 1, class 4 → index 4, class 9 → index 9
        assert index_map[0, 0] == 0
        assert index_map[0, 1] == 1
        assert index_map[1, 0] == 4
        assert index_map[1, 1] == 9

    def test_build_prior_index_map_unknown_fallback(self):
        """Unknown class IDs fall back to index 0 when class '0' is present."""
        vol = np.array([[99]], dtype=np.uint8)
        index_map = _build_prior_index_map(vol, DEFAULT_PRIORS_TABLE)
        assert index_map[0, 0] == 0

    def test_build_prior_index_map_fallback_without_class0(self):
        """Custom config missing class '0': unknown IDs map to synthetic fallback slot, NOT index 0."""
        config_no_zero = {
            "1": {"class_name": "epidermis", "priors": {"melanin_min": 0.0}},
            "2": {"class_name": "dermis", "priors": {"melanin_min": 0.0}},
        }
        vol = np.array([[1, 2], [99, 1]], dtype=np.uint8)
        index_map = _build_prior_index_map(vol, config_no_zero)
        # class 1 → 0, class 2 → 1, unknown 99 → 2 (synthetic slot)
        assert index_map[0, 0] == 0, "class 1 should map to index 0"
        assert index_map[0, 1] == 1, "class 2 should map to index 1"
        assert index_map[1, 0] == 2, "unknown class 99 should map to synthetic fallback index 2, NOT index 0"
        assert index_map[1, 1] == 0, "class 1 should still map to index 0"

    def test_build_prior_index_map_fallback_with_class0_present(self):
        """Config WITH class '0': unknown IDs map to class-0's index, even if class 0 is not first sorted key."""
        config_with_zero_not_first = {
            "10": {"class_name": "ten", "priors": {"melanin_min": 0.0}},
            "0": {"class_name": "background", "priors": {"melanin_min": 0.0}},
        }
        vol = np.array([[10, 99]], dtype=np.uint8)
        index_map = _build_prior_index_map(vol, config_with_zero_not_first)
        # sorted keys: ["0", "10"] → 0→0, 10→1
        assert index_map[0, 0] == 1, "class 10 should map to index 1"
        assert index_map[0, 1] == 0, (
            "unknown class 99 should fall back to class-0 index (0), "
            "not the first-sorted key's index"
        )

    def test_build_prior_index_map_too_many_classes_fallback_overflow(self):
        """Config with 128+ classes and no class '0' makes fallback index exceed int8 → ValueError."""
        too_many = {}
        for i in range(129):  # 0..128 → 129 keys; then pop "0" → 128 keys
            too_many[str(i)] = {
                "class_name": f"class_{i}",
                "priors": {"melanin_min": 0.0},
            }
        # Remove "0" so fallback_index = len(128) = 128 > 127
        too_many.pop("0", None)
        vol = np.array([[0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds int8 capacity"):
            _build_prior_index_map(vol, too_many)

    def test_build_prior_index_map_too_many_classes_overflow_with_class0(self):
        """Config with 128 classes including '0' → fallback_index = 0, no overflow."""
        many_with_zero = {}
        for i in range(128):
            many_with_zero[str(i)] = {
                "class_name": f"class_{i}",
                "priors": {"melanin_min": 0.0},
            }
        # "0" is present → fallback_index = lookup[0] = 0
        vol = np.array([[200]], dtype=np.uint8)
        index_map = _build_prior_index_map(vol, many_with_zero)
        assert index_map[0, 0] == 0, "fallback with class-0 present should map to index 0"

    def test_generic_fallback_structure(self):
        """Generic fallback has the expected structure."""
        assert "class_name" in _GENERIC_FALLBACK_PRIORS
        assert "priors" in _GENERIC_FALLBACK_PRIORS
        for key in _EXPECTED_PRIOR_KEYS:
            assert key in _GENERIC_FALLBACK_PRIORS["priors"]

    def test_cli_help(self):
        """The CLI --help flag prints usage and exits 0."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.optical_ga.label_to_optical_priors",
             "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Convert a class-ID label map" in result.stdout
        assert "--label-npy" in result.stdout
