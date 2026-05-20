"""Tests for the multi-physics pipeline manifest schema.

Ensures that ``build_manifest()`` produces the correct top-level keys,
per-step status/command fields, and summary counts.

The pipeline module uses only stdlib + numpy, so it can be imported safely
without external dependencies.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.simulation.run_mvp_multiphysics_pipeline import (
    build_manifest,
    _STEP_ORDER,
    _step_name,
    _skipped_result,
    _artifacts_exist,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_minimal_args() -> Namespace:
    """Create a Namespace with all attributes the manifest builder expects."""
    return Namespace(
        optical_ga_forward_mode="surrogate",
        optical_ga_fitness_mode="lab",
        optical_ga_target_L=60.0,
        optical_ga_target_a=10.0,
        optical_ga_target_b=15.0,
        optical_ga_generations=3,
        optical_ga_population_size=8,
        optical_ga_mutation_rate=0.2,
        optical_ga_mutation_strength=0.1,
        optical_ga_elite_fraction=0.1,
        optical_ga_tournament_size=3,
        optical_ga_seed=42,
        optical_ga_use_dermal_chromophores=False,
        skip_optical_ga=False,
        skip_mcx=False,
        skip_thermal=False,
        mcx_run=False,
        mcx_binary="mcx",
        mcx_photons=None,
        thermal_dt=0.01,
        thermal_num_steps=50,
        thermal_source_mode="spherical",
        thermal_source_radius_vox=2.0,
        thermal_source_power=5e5,
        thermal_source_center=None,
        fail_fast=False,
    )


def _completed_result() -> Dict[str, Any]:
    return {
        "status": "completed",
        "returncode": 0,
        "command": ["python", "some_script.py", "--flag"],
        "error": None,
        "stdout_snippet": ["ok", "done"],
        "stderr_snippet": [],
    }


def _failed_result(msg: str = "Something went wrong") -> Dict[str, Any]:
    return {
        "status": "failed",
        "returncode": 1,
        "command": ["python", "bad_script.py"],
        "error": msg,
        "stdout_snippet": [],
        "stderr_snippet": ["ERROR: something went wrong"],
    }


# ---------------------------------------------------------------------------
# Manifest top-level structure
# ---------------------------------------------------------------------------

class TestManifestTopLevel:
    def test_has_required_keys(self):
        args = _make_minimal_args()
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/out")

        assert "pipeline" in manifest
        assert manifest["pipeline"] == "mvp_multiphysics"
        assert "timestamp_utc" in manifest
        assert "input_args" in manifest
        assert "steps" in manifest
        assert "summary" in manifest

    def test_pipeline_constant(self):
        """The pipeline field must always be 'mvp_multiphysics'."""
        args = _make_minimal_args()
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/out")
        assert manifest["pipeline"] == "mvp_multiphysics"

    def test_timestamp_is_iso_format(self):
        args = _make_minimal_args()
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/out")
        ts = manifest["timestamp_utc"]
        # Basic ISO-8601 check: contains 'T' and ends with timezone
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z") or "+" in ts[19:]


# ---------------------------------------------------------------------------
# Manifest input_args
# ---------------------------------------------------------------------------

class TestInputArgs:
    def test_contains_label_volume(self):
        args = _make_minimal_args()
        manifest = build_manifest(args, {}, "/tmp/my_volume.npy", "/tmp/out")
        assert manifest["input_args"]["label_volume"] == "/tmp/my_volume.npy"

    def test_contains_output_dir(self):
        args = _make_minimal_args()
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/my_output")
        assert manifest["input_args"]["output_dir"] == "/tmp/my_output"

    def test_contains_optical_ga_flags(self):
        args = _make_minimal_args()
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/out")
        ia = manifest["input_args"]
        assert "optical_ga_forward_mode" in ia
        assert "optical_ga_fitness_mode" in ia
        assert "optical_ga_generations" in ia
        assert "skip_optical_ga" in ia
        assert "skip_mcx" in ia
        assert "skip_thermal" in ia
        assert "mcx_run" in ia
        assert "mcx_binary" in ia
        assert "thermal_dt" in ia
        assert "fail_fast" in ia

    def test_mcx_photons_none_preserved(self):
        args = _make_minimal_args()  # mcx_photons=None
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/out")
        assert manifest["input_args"]["mcx_photons"] is None

    def test_mcx_photons_valued(self):
        args = _make_minimal_args()
        args.mcx_photons = 1000000
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/out")
        assert manifest["input_args"]["mcx_photons"] == 1000000


# ---------------------------------------------------------------------------
# Per-step entries
# ---------------------------------------------------------------------------

class TestPerStep:
    def test_step_has_required_fields(self):
        args = _make_minimal_args()
        steps = {
            "optical_ga": _completed_result(),
        }
        manifest = build_manifest(args, steps, "/tmp/vol.npy", "/tmp/out")
        entry = manifest["steps"]["optical_ga"]

        assert "step_name" in entry
        assert entry["step_name"] == "Optical GA"
        assert "status" in entry
        assert entry["status"] == "completed"
        assert "returncode" in entry
        assert entry["returncode"] == 0
        assert "command" in entry
        assert entry["command"] == ["python", "some_script.py", "--flag"]
        assert "error" in entry

    def test_step_missing_results_not_in_manifest(self):
        """Steps not present in the steps dict should not appear."""
        args = _make_minimal_args()
        manifest = build_manifest(
            args,
            {"optical_ga": _completed_result()},
            "/tmp/vol.npy",
            "/tmp/out",
        )
        assert "optical_ga" in manifest["steps"]
        assert "mcx_build" not in manifest["steps"]

    def test_failed_step_has_error_field(self):
        args = _make_minimal_args()
        steps = {"mcx_build": _failed_result("MCX binary not found")}
        manifest = build_manifest(args, steps, "/tmp/vol.npy", "/tmp/out")
        entry = manifest["steps"]["mcx_build"]
        assert entry["status"] == "failed"
        assert entry["error"] == "MCX binary not found"

    def test_stdout_snippet_stored_as_summary(self):
        args = _make_minimal_args()
        steps = {"optical_ga": _completed_result()}
        manifest = build_manifest(args, steps, "/tmp/vol.npy", "/tmp/out")
        assert "stdout_summary" in manifest["steps"]["optical_ga"]

    def test_stdout_snippet_trimmed_to_5(self):
        """stdout_summary should be at most 5 lines."""
        args = _make_minimal_args()
        long_result = _completed_result()
        long_result["stdout_snippet"] = [f"line{i}" for i in range(20)]
        steps = {"optical_ga": long_result}
        manifest = build_manifest(args, steps, "/tmp/vol.npy", "/tmp/out")
        assert len(manifest["steps"]["optical_ga"]["stdout_summary"]) <= 5

    def test_step_name_human_readable(self):
        """step_name should be human-readable, not the internal ID."""
        assert _step_name("optical_ga") == "Optical GA"
        assert _step_name("mcx_build") == "MCX Build Volume"
        assert _step_name("mcx_batch") == "MCX Batch Runner"
        assert _step_name("thermal_solve") == "Thermal Solve (Pennes)"
        assert _step_name("unknown_id") == "unknown_id"  # fallback


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummary:
    def test_all_completed(self):
        args = _make_minimal_args()
        steps: Dict[str, Dict[str, Any]] = {}
        for sid in _STEP_ORDER:
            steps[sid] = _completed_result()
        manifest = build_manifest(args, steps, "/tmp/vol.npy", "/tmp/out")
        s = manifest["summary"]
        assert s["total_steps"] == len(_STEP_ORDER)
        assert s["completed"] == len(_STEP_ORDER)
        assert s["failed"] == 0
        assert s["skipped"] == 0
        assert s["errors"] == 0
        assert s["has_errors"] is False

    def test_mixed_statuses(self):
        args = _make_minimal_args()
        steps: Dict[str, Dict[str, Any]] = {
            "optical_ga": _completed_result(),
            "mcx_build": _failed_result(),
            "mcx_batch": _skipped_result("Skipped as mcx is disabled"),
        }
        manifest = build_manifest(args, steps, "/tmp/vol.npy", "/tmp/out")
        s = manifest["summary"]
        assert s["total_steps"] == 3
        assert s["completed"] == 1
        assert s["failed"] == 1
        assert s["skipped"] == 1
        assert s["has_errors"] is True

    def test_empty_steps(self):
        args = _make_minimal_args()
        manifest = build_manifest(args, {}, "/tmp/vol.npy", "/tmp/out")
        s = manifest["summary"]
        assert s["total_steps"] == 0
        assert s["completed"] == 0
        assert s["has_errors"] is False

    def test_timeout_counts_as_error(self):
        args = _make_minimal_args()
        steps = {
            "thermal_solve": {
                "status": "timeout",
                "returncode": None,
                "command": ["pennes_solver"],
                "error": "Timed out",
            }
        }
        manifest = build_manifest(args, steps, "/tmp/vol.npy", "/tmp/out")
        assert manifest["summary"]["has_errors"] is True
        assert manifest["summary"]["errors"] == 0
        assert manifest["summary"]["failed"] == 0


# ---------------------------------------------------------------------------
# _skipped_result helper
# ---------------------------------------------------------------------------

class TestSkippedResult:
    def test_structure(self):
        result = _skipped_result("MCX binary not found")
        assert result["status"] == "skipped"
        assert result["returncode"] is None
        assert result["command"] == []
        assert result["error"] == "MCX binary not found"
        assert result["stdout_snippet"] is None
        assert result["stderr_snippet"] is None


# ---------------------------------------------------------------------------
# _artifacts_exist
# ---------------------------------------------------------------------------

class TestArtifactsExist:
    def test_optical_ga_artifacts(self, tmp_path: Path):
        ga_dir = tmp_path / "optical_ga"
        ga_dir.mkdir(parents=True)
        for fname in ("best_genome.json", "ga_history.csv", "population_final.json"):
            (ga_dir / fname).write_text("{}")

        artifacts = _artifacts_exist(tmp_path, "optical_ga")
        assert len(artifacts) == 3

    def test_optical_ga_artifacts_partial(self, tmp_path: Path):
        ga_dir = tmp_path / "optical_ga"
        ga_dir.mkdir(parents=True)
        (ga_dir / "best_genome.json").write_text("{}")

        artifacts = _artifacts_exist(tmp_path, "optical_ga")
        assert len(artifacts) == 1
        assert "best_genome.json" in artifacts[0]

    def test_optical_ga_artifacts_none(self, tmp_path: Path):
        artifacts = _artifacts_exist(tmp_path, "optical_ga")
        assert artifacts == []

    def test_mcx_artifacts(self, tmp_path: Path):
        mcx_dir = tmp_path / "mcx"
        mcx_dir.mkdir(parents=True)
        for fname in ("mcx_volume.npy", "mcx_media_table.json", "mcx_config.json", "mcx_build_manifest.json"):
            (mcx_dir / fname).write_text("{}")

        artifacts = _artifacts_exist(tmp_path, "mcx_build")
        assert len(artifacts) == 4


# ---------------------------------------------------------------------------
# _STEP_ORDER consistency
# ---------------------------------------------------------------------------

class TestStepOrder:
    def test_all_steps_have_names(self):
        """Every step in _STEP_ORDER should have a human-readable name."""
        for sid in _STEP_ORDER:
            name = _step_name(sid)
            assert name != sid, f"Step '{sid}' has no human-readable name"
            assert len(name) > 2

    def test_step_order_is_complete(self):
        """The step order should cover all current components."""
        expected = [
            "optical_ga",
            "mcx_build",
            "mcx_batch",
            "thermal_build",
            "thermal_solve",
            "thermal_visualise",
        ]
        assert _STEP_ORDER == expected
