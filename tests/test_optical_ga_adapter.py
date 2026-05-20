"""Tests for the optical GA adapter (run_optical_ga_from_labels.py)."""

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

from scripts.optical_ga.run_optical_ga_from_labels import (
    _load_label_volume,
    _load_priors_json,
    _exclude_background,
    _derive_bounds_from_priors,
    _generate_seed_genomes,
    _identify_present_classes,
    _build_manifest,
    _PARAM_STEMS,
)


# ---------------------------------------------------------------------------
# Sample minimal priors data (2 classes + background)
# ---------------------------------------------------------------------------

_SAMPLE_PRIORS = {
    "0": {
        "class_name": "background",
        "class_id": 0,
        "count_voxels": 10,
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.001,
            "blood_layer1_min": 0.0, "blood_layer1_max": 0.001,
            "blood_layer2_min": 0.0, "blood_layer2_max": 0.001,
            "spo2_min": 0.5, "spo2_max": 0.5,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.1e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.5, "amp_layer0_max": 1.5,
            "amp_layer1_min": 0.5, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.5, "amp_layer2_max": 1.5,
            "water_layer0_min": 0.01, "water_layer0_max": 0.8,
            "water_layer1_min": 0.01, "water_layer1_max": 0.8,
            "water_layer2_min": 0.01, "water_layer2_max": 0.8,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.8,
            "n_mult_layer0_min": 0.8, "n_mult_layer0_max": 1.2,
            "n_mult_layer1_min": 0.8, "n_mult_layer1_max": 1.2,
            "n_mult_layer2_min": 0.8, "n_mult_layer2_max": 1.2,
        },
    },
    "1": {
        "class_name": "epidermis",
        "class_id": 1,
        "priors": {
            "melanin_min": 0.05, "melanin_max": 0.5,
            "blood_layer1_min": 0.0, "blood_layer1_max": 0.02,
            "blood_layer2_min": 0.0, "blood_layer2_max": 0.02,
            "spo2_min": 0.6, "spo2_max": 0.95,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.2e-4,
            "d_layer1_min": 0.1e-3, "d_layer1_max": 0.4e-3,
            "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.6, "amp_layer1_max": 1.4,
            "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.1, "water_layer0_max": 0.7,
            "water_layer1_min": 0.2, "water_layer1_max": 0.7,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
            "n_mult_layer0_min": 0.9, "n_mult_layer0_max": 1.15,
            "n_mult_layer1_min": 0.9, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.9, "n_mult_layer2_max": 1.15,
        },
    },
    "2": {
        "class_name": "reticular_dermis",
        "class_id": 2,
        "priors": {
            "melanin_min": 0.0, "melanin_max": 0.05,
            "blood_layer1_min": 0.005, "blood_layer1_max": 0.08,
            "blood_layer2_min": 0.005, "blood_layer2_max": 0.08,
            "spo2_min": 0.6, "spo2_max": 0.95,
            "g_layer0_min": 0.85, "g_layer0_max": 0.95,
            "g_layer1_min": 0.85, "g_layer1_max": 0.95,
            "g_layer2_min": 0.85, "g_layer2_max": 0.95,
            "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
            "d_layer1_min": 0.2e-3, "d_layer1_max": 0.5e-3,
            "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
            "amp_layer1_min": 0.7, "amp_layer1_max": 1.5,
            "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
            "water_layer0_min": 0.2, "water_layer0_max": 0.7,
            "water_layer1_min": 0.3, "water_layer1_max": 0.8,
            "water_layer2_min": 0.2, "water_layer2_max": 0.7,
            "fat_layer2_min": 0.05, "fat_layer2_max": 0.5,
            "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
            "n_mult_layer1_min": 0.9, "n_mult_layer1_max": 1.15,
            "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
        },
    },
}


class TestBoundsDerivation:
    """Tests for the bounds derivation logic."""

    def test_exclude_background_removes_class0(self):
        """Background (class 0) is excluded from non-bg set."""
        bg = _exclude_background(_SAMPLE_PRIORS)
        assert "0" not in bg
        assert "1" in bg
        assert "2" in bg

    def test_derive_bounds_two_classes(self):
        """Derived bounds from epidermis + dermis cover union of ranges."""
        d_min, d_max, d_log = _derive_bounds_from_priors(_SAMPLE_PRIORS)
        # melanin: epidermis [0.05, 0.5], dermis [0.0, 0.05] → [0.0, 0.5]
        assert d_min["melanin"] == 0.0
        assert d_max["melanin"] == 0.5
        # blood_layer1: epidermis [0.0, 0.02], dermis [0.005, 0.08] → [0.0, 0.08]
        assert d_min["blood_layer1"] == 0.0
        assert d_max["blood_layer1"] == 0.08
        # blood_layer2: both [0.0, 0.02] and [0.005, 0.08] → [0.0, 0.08]
        assert d_min["blood_layer2"] == 0.0
        assert d_max["blood_layer2"] == 0.08
        # spo2: both [0.6, 0.95] → [0.6, 0.95]
        assert d_min["spo2"] == 0.6
        assert d_max["spo2"] == 0.95

    def test_derive_bounds_water_content(self):
        """Water bounds are correctly intersected."""
        d_min, d_max, _ = _derive_bounds_from_priors(_SAMPLE_PRIORS)
        # water_layer0: epidermis [0.1, 0.7], dermis [0.2, 0.7] → [0.1, 0.7]
        assert d_min["water_layer0"] == 0.1
        assert d_max["water_layer0"] == 0.7
        # water_layer1: epidermis [0.2, 0.7], dermis [0.3, 0.8] → [0.2, 0.8]
        assert d_min["water_layer1"] == 0.2
        assert d_max["water_layer1"] == 0.8

    def test_derive_bounds_log_structure(self):
        """Derivation log contains expected metadata."""
        _, _, d_log = _derive_bounds_from_priors(_SAMPLE_PRIORS)
        assert d_log["num_non_background_classes"] == 2
        assert d_log["non_background_class_ids"] == [1, 2]
        assert d_log["excluded_background"] is True
        assert "per_parameter" in d_log
        assert len(d_log["per_parameter"]) == len(_PARAM_STEMS)
        # Check a few per-param entries
        melanin_entry = d_log["per_parameter"]["melanin"]
        assert "derived_min" in melanin_entry
        assert "derived_max" in melanin_entry
        assert "num_class_contributors" in melanin_entry

    def test_derive_bounds_all_params_present(self):
        """All 19 core parameters get derived bounds."""
        d_min, d_max, _ = _derive_bounds_from_priors(_SAMPLE_PRIORS)
        assert len(d_min) == 19
        assert len(d_max) == 19
        for p in _PARAM_STEMS:
            assert p in d_min
            assert p in d_max
            assert d_min[p] <= d_max[p]

    def test_derive_bounds_single_class(self):
        """With only one non-bg class, bounds equal that class's priors."""
        single = {"5": _SAMPLE_PRIORS["1"]}  # only epidermis
        d_min, d_max, d_log = _derive_bounds_from_priors(single)
        assert d_log["num_non_background_classes"] == 1
        # melanin: epidermis [0.05, 0.5]
        assert d_min["melanin"] == 0.05
        assert d_max["melanin"] == 0.5

    def test_derive_bounds_no_non_bg_fallback(self):
        """No non-background classes → fallback to global bounds."""
        only_bg = {"0": _SAMPLE_PRIORS["0"]}
        d_min, d_max, d_log = _derive_bounds_from_priors(only_bg)
        assert d_log["num_non_background_classes"] == 0
        assert "fallback_triggered" in d_log
        # Use global bounds: melanin [0.0, 0.5]
        assert d_min["melanin"] == 0.0
        assert d_max["melanin"] == 0.5

    def test_derive_bounds_empty(self):
        """Empty present classes → fallback to global bounds."""
        d_min, d_max, d_log = _derive_bounds_from_priors({})
        assert d_log["num_non_background_classes"] == 0
        assert "fallback_triggered" in d_log
        assert d_min["melanin"] == 0.0
        assert d_max["melanin"] == 0.5


class TestLoadFunctions:
    """Tests for loading helpers."""

    def test_load_label_volume(self):
        """Load a .npy label volume."""
        arr = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, arr)
            fname = f.name
        try:
            loaded = _load_label_volume(fname)
            assert np.array_equal(loaded, arr)
        finally:
            Path(fname).unlink()

    def test_load_label_volume_missing(self):
        """Missing file exits with SystemExit."""
        with pytest.raises(SystemExit):
            _load_label_volume("/tmp/nonexistent_file.npy")

    def test_load_priors_json(self):
        """Load a priors JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(_SAMPLE_PRIORS, f)
            fname = f.name
        try:
            loaded = _load_priors_json(fname)
            assert "0" in loaded
            assert "1" in loaded
        finally:
            Path(fname).unlink()

    def test_load_priors_json_with_per_class_wrapper(self):
        """Load a priors JSON that uses the per_class_priors wrapper key."""
        wrapped = {"per_class_priors": _SAMPLE_PRIORS, "metadata": {"test": True}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(wrapped, f)
            fname = f.name
        try:
            loaded = _load_priors_json(fname)
            assert "0" in loaded
            assert "metadata" not in loaded  # unwrapped
        finally:
            Path(fname).unlink()

    def test_load_priors_json_missing(self):
        """Missing file exits with SystemExit."""
        with pytest.raises(SystemExit):
            _load_priors_json("/tmp/nonexistent_priors.json")


class TestIdentifyPresentClasses:
    """Tests for identifying which classes are present in a volume."""

    def test_single_present(self):
        """Only class 1 present."""
        vol = np.ones((4, 4), dtype=np.uint8)
        present = _identify_present_classes(vol, _SAMPLE_PRIORS)
        assert "0" not in present  # not in vol
        assert "1" in present
        assert "2" not in present  # not in vol
        assert present["1"]["count_voxels"] == 16

    def test_multiple_present(self):
        """Classes 0, 1, 2 present."""
        vol = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        present = _identify_present_classes(vol, _SAMPLE_PRIORS)
        assert "0" in present
        assert "1" in present
        assert "2" in present
        assert present["0"]["count_voxels"] == 1
        assert present["1"]["count_voxels"] == 2
        assert present["2"]["count_voxels"] == 1

    def test_unknown_class_in_volume(self):
        """Class ID in volume but not in priors data gets a warning entry."""
        vol = np.array([[99]], dtype=np.uint8)
        present = _identify_present_classes(vol, {"0": _SAMPLE_PRIORS["0"]})
        assert "99" in present
        assert "warning" in present["99"]

    def test_metadata_keys_are_ignored(self):
        """Non-numeric keys in priors data are ignored safely."""
        vol = np.array([[1]], dtype=np.uint8)
        priors_with_metadata = {
            "metadata": {"schema": "v1"},
            "1": _SAMPLE_PRIORS["1"],
        }
        present = _identify_present_classes(vol, priors_with_metadata)
        assert "1" in present
        assert "metadata" not in present

    def test_counts_match(self):
        """Voxel counts are accurate."""
        vol = np.array([[0, 1, 1], [2, 2, 2]], dtype=np.uint8)
        present = _identify_present_classes(vol, _SAMPLE_PRIORS)
        assert present["0"]["count_voxels"] == 1
        assert present["1"]["count_voxels"] == 2
        assert present["2"]["count_voxels"] == 3


class TestSeedGeneration:
    """Tests for seed genome generation within derived bounds."""

    def test_seeds_within_derived_bounds(self):
        """Seed genomes are inside derived bounds."""
        import random
        d_min = {"melanin": 0.05, "blood_layer1": 0.0}
        d_max = {"melanin": 0.5, "blood_layer1": 0.08}
        rng = random.Random(42)
        seeds = _generate_seed_genomes(d_min, d_max, num_seeds=10, rng=rng)
        assert len(seeds) == 10
        for s in seeds:
            assert 0.05 <= s["melanin"] <= 0.5
            assert 0.0 <= s["blood_layer1"] <= 0.08

    def test_seeds_have_all_core_params(self):
        """All 19 core parameters are present in each seed."""
        import random
        from scripts.optical_ga.genome_encoding_optical import param_names
        d_min = {"melanin": 0.0, "blood_layer1": 0.0}
        d_max = {"melanin": 0.5, "blood_layer1": 0.1}
        rng = random.Random(42)
        seeds = _generate_seed_genomes(d_min, d_max, num_seeds=3, rng=rng)
        names = param_names(use_dermal_chromophores=False)
        for s in seeds:
            for n in names:
                assert n in s

    def test_seeds_zero_seeds(self):
        """num_seeds=0 returns empty list."""
        import random
        rng = random.Random(42)
        seeds = _generate_seed_genomes({}, {}, num_seeds=0, rng=rng)
        assert seeds == []


class TestManifestBuilder:
    """Tests for manifest construction."""

    def test_manifest_contains_expected_keys(self):
        """Manifest dict has all required top-level keys."""
        manifest = _build_manifest(
            label_npy="/tmp/test.npy",
            priors_json="/tmp/test.json",
            output_dir="/tmp/out",
            present_classes={"1": {"class_id": 1, "class_name": "epidermis", "count_voxels": 10}},
            derived_min={"melanin": 0.0},
            derived_max={"melanin": 0.5},
            derivation_log={"strategy": "test", "num_non_background_classes": 1},
            ga_args={"generations": 3, "population_size": 8},
            ga_outputs={"best_genome": "/tmp/out/best_genome.json"},
            elapsed_s=0.5,
        )
        assert manifest["manifest_type"] == "optical_ga_adapter_manifest"
        assert "inputs" in manifest
        assert "present_classes" in manifest
        assert "bounds_derivation" in manifest
        assert "ga_configuration" in manifest
        assert "ga_outputs" in manifest
        assert "elapsed_seconds" in manifest

    def test_manifest_present_classes(self):
        """Present classes include id, name, and count."""
        manifest = _build_manifest(
            label_npy="/tmp/test.npy",
            priors_json="/tmp/test.json",
            output_dir="/tmp/out",
            present_classes={
                "1": {"class_id": 1, "class_name": "epidermis", "count_voxels": 42},
                "2": {"class_id": 2, "class_name": "dermis", "count_voxels": 99},
            },
            derived_min={},
            derived_max={},
            derivation_log={},
            ga_args={},
            ga_outputs={},
            elapsed_s=0.0,
        )
        pc = manifest["present_classes"]
        assert pc["1"]["class_name"] == "epidermis"
        assert pc["1"]["count_voxels"] == 42
        assert pc["2"]["class_name"] == "dermis"
        assert pc["2"]["count_voxels"] == 99


class TestCliIntegration:
    """Smoke-test the CLI entrypoint (dry-run mode)."""

    def test_cli_help(self):
        """The CLI --help flag prints usage and exits 0."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.optical_ga.run_optical_ga_from_labels",
             "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Run optical GA from a label volume" in result.stdout
        assert "--label-npy" in result.stdout
        assert "--priors-json" in result.stdout
        assert "--output-dir" in result.stdout

    def test_cli_dry_run_with_synthetic_data(self):
        """Dry-run with synthetic label volume and priors creates expected output."""
        import subprocess
        import tempfile
        import os

        # Create synthetic label volume
        vol = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, vol)
            label_path = f.name

        # Create synthetic priors
        simple_priors = {
            "0": _SAMPLE_PRIORS["0"],
            "1": _SAMPLE_PRIORS["1"],
            "2": _SAMPLE_PRIORS["2"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(simple_priors, f)
            priors_path = f.name

        out_dir = tempfile.mkdtemp()

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "scripts.optical_ga.run_optical_ga_from_labels",
                    "--label-npy", label_path,
                    "--priors-json", priors_path,
                    "--output-dir", out_dir,
                    "--generations", "2",
                    "--population-size", "4",
                    "--seed", "42",
                    "--dry-run",
                ],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert "DRY RUN" in result.stdout
            assert "Present in vol" in result.stdout
            assert "Derivation log" in result.stdout
            assert "per_parameter" in result.stdout
        finally:
            os.unlink(label_path)
            os.unlink(priors_path)
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_cli_full_run_with_synthetic_data(self):
        """Full GA run with tiny synthetic data produces all artifacts."""
        import subprocess
        import tempfile
        import os

        # Create synthetic label volume (tiny)
        vol = np.array([[0, 1], [2, 1]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            np.save(f, vol)
            label_path = f.name

        simple_priors = {
            "0": _SAMPLE_PRIORS["0"],
            "1": _SAMPLE_PRIORS["1"],
            "2": _SAMPLE_PRIORS["2"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(simple_priors, f)
            priors_path = f.name

        out_dir = tempfile.mkdtemp()

        try:
            result = subprocess.run(
                [
                    sys.executable, "-m",
                    "scripts.optical_ga.run_optical_ga_from_labels",
                    "--label-npy", label_path,
                    "--priors-json", priors_path,
                    "--output-dir", out_dir,
                    "--generations", "2",
                    "--population-size", "4",
                    "--seed", "42",
                ],
                capture_output=True, text=True,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"

            # Check artifacts exist
            assert os.path.exists(os.path.join(out_dir, "adapter_manifest.json"))
            assert os.path.exists(os.path.join(out_dir, "best_genome.json"))
            assert os.path.exists(os.path.join(out_dir, "ga_history.csv"))
            assert os.path.exists(os.path.join(out_dir, "population_final.json"))

            # Validate manifest content
            with open(os.path.join(out_dir, "adapter_manifest.json")) as f:
                manifest = json.load(f)
            assert manifest["manifest_type"] == "optical_ga_adapter_manifest"
            assert "present_classes" in manifest
            assert "bounds_derivation" in manifest
            assert "ga_configuration" in manifest
            assert "ga_outputs" in manifest
            # Check present classes include classes in volume
            pc_keys = set(manifest["present_classes"].keys())
            assert "1" in pc_keys
            assert "2" in pc_keys

            # Check bounds derivation
            bd = manifest["bounds_derivation"]
            assert bd["num_non_background_classes"] == 2
            assert "per_parameter" in bd

            # Check GA output
            with open(os.path.join(out_dir, "best_genome.json")) as f:
                best = json.load(f)
            assert "genome" in best
            assert "fitness" in best
            assert len(best["genome"]) == 19

            # Check history
            import csv
            with open(os.path.join(out_dir, "ga_history.csv")) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2  # 2 generations

        finally:
            os.unlink(label_path)
            os.unlink(priors_path)
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)


class TestEdgeCases:
    """Edge cases for bounds derivation."""

    def test_derive_bounds_prior_with_narrow_range(self):
        """A class with a very narrow range still produces valid bounds."""
        single = {
            "7": {
                "class_name": "hair_follicles",
                "class_id": 7,
                "priors": {
                    "melanin_min": 0.2, "melanin_max": 0.5,
                    "blood_layer1_min": 0.005, "blood_layer1_max": 0.05,
                    "blood_layer2_min": 0.005, "blood_layer2_max": 0.05,
                    "spo2_min": 0.6, "spo2_max": 0.95,
                    "g_layer0_min": 0.85, "g_layer0_max": 0.95,
                    "g_layer1_min": 0.85, "g_layer1_max": 0.95,
                    "g_layer2_min": 0.85, "g_layer2_max": 0.95,
                    "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
                    "d_layer1_min": 0.15e-3, "d_layer1_max": 0.5e-3,
                    "amp_layer0_min": 0.6, "amp_layer0_max": 1.4,
                    "amp_layer1_min": 0.6, "amp_layer1_max": 1.4,
                    "amp_layer2_min": 0.6, "amp_layer2_max": 1.4,
                    "water_layer0_min": 0.1, "water_layer0_max": 0.6,
                    "water_layer1_min": 0.1, "water_layer1_max": 0.6,
                    "water_layer2_min": 0.1, "water_layer2_max": 0.6,
                    "fat_layer2_min": 0.05, "fat_layer2_max": 0.4,
                    "n_mult_layer0_min": 0.85, "n_mult_layer0_max": 1.1,
                    "n_mult_layer1_min": 0.85, "n_mult_layer1_max": 1.1,
                    "n_mult_layer2_min": 0.85, "n_mult_layer2_max": 1.1,
                },
            }
        }
        d_min, d_max, _ = _derive_bounds_from_priors(single)
        # Non-background only - single class
        assert d_min["melanin"] == 0.2
        assert d_max["melanin"] == 0.5
        assert d_min["fat_layer2"] == 0.05
        assert d_max["fat_layer2"] == 0.4

    def test_bounds_never_exceed_global(self):
        """Derived bounds are always clamped to global genome bounds."""
        # Create unrealistic priors that exceed global bounds
        extreme = {
            "9": {
                "class_name": "extreme",
                "class_id": 9,
                "priors": {
                    "melanin_min": -1.0, "melanin_max": 5.0,  # outside [0, 0.5]
                    "blood_layer1_min": -0.5, "blood_layer1_max": 2.0,
                    "blood_layer2_min": 0.0, "blood_layer2_max": 0.02,
                    "spo2_min": 0.6, "spo2_max": 0.95,
                    "g_layer0_min": 0.85, "g_layer0_max": 0.95,
                    "g_layer1_min": 0.85, "g_layer1_max": 0.95,
                    "g_layer2_min": 0.85, "g_layer2_max": 0.95,
                    "d_layer0_min": 3e-5, "d_layer0_max": 1.5e-4,
                    "d_layer1_min": 0.1e-3, "d_layer1_max": 0.5e-3,
                    "amp_layer0_min": 0.5, "amp_layer0_max": 1.5,
                    "amp_layer1_min": 0.5, "amp_layer1_max": 1.5,
                    "amp_layer2_min": 0.5, "amp_layer2_max": 1.5,
                    "water_layer0_min": 0.01, "water_layer0_max": 0.8,
                    "water_layer1_min": 0.01, "water_layer1_max": 0.8,
                    "water_layer2_min": 0.01, "water_layer2_max": 0.8,
                    "fat_layer2_min": 0.05, "fat_layer2_max": 0.8,
                    "n_mult_layer0_min": 0.8, "n_mult_layer0_max": 1.2,
                    "n_mult_layer1_min": 0.8, "n_mult_layer1_max": 1.2,
                    "n_mult_layer2_min": 0.8, "n_mult_layer2_max": 1.2,
                },
            }
        }
        d_min, d_max, _ = _derive_bounds_from_priors(extreme)
        # Melanin global bounds are [0.0, 0.5]
        assert d_min["melanin"] >= 0.0
        assert d_max["melanin"] <= 0.5
        # blood_layer1 global bounds are [0.0, 0.1]
        assert d_min["blood_layer1"] >= 0.0
        assert d_max["blood_layer1"] <= 0.1
