"""Tests for config -> CLI mapping in ``run_mvp_multiphysics_from_config.py``.

Covers:
  - build_pipeline_command() mapping behavior
  - CLI override handling (label_volume, output_dir, fail_fast)
  - GA / MCX / Thermal option mapping
  - Error edge cases in config validation

These tests import the module directly; PyYAML must be installed.
"""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict

import pytest

yaml = pytest.importorskip("yaml")  # skip entire file if PyYAML not installed

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.simulation.run_mvp_multiphysics_from_config import (
    build_pipeline_command,
    _get_overrides,
    _resolve_path,
    load_config,
    _REQUIRED_TOP_LEVEL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_config() -> Dict[str, Any]:
    """Return a config with only the two required fields."""
    return {
        "label_volume": "/tmp/test_volume.npy",
        "output_dir": "/tmp/test_output",
    }


def _cmd_contains(cmd: list[str], needle: str) -> bool:
    """Check if the resolved command string includes *needle*."""
    return needle in " ".join(cmd)


# ---------------------------------------------------------------------------
# build_pipeline_command — required paths
# ---------------------------------------------------------------------------

class TestRequiredPaths:
    def test_label_volume_present(self):
        config = _minimal_config()
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert _cmd_contains(cmd, "--label-volume")
        assert _cmd_contains(cmd, "/tmp/test_volume.npy")

    def test_output_dir_present(self):
        config = _minimal_config()
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert _cmd_contains(cmd, "--output-dir")
        assert _cmd_contains(cmd, "/tmp/test_output")

    def test_pipeline_script_path(self):
        config = _minimal_config()
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        script_expected = str(
            _REPO_ROOT / "scripts" / "simulation" / "run_mvp_multiphysics_pipeline.py"
        )
        # cmd[0] = sys.executable, cmd[1] = pipeline script path
        assert cmd[1] == script_expected


# ---------------------------------------------------------------------------
# build_pipeline_command — GA options
# ---------------------------------------------------------------------------

class TestGAOptions:
    def test_disabled_adds_skip_ga(self):
        config = _minimal_config()
        config["ga"] = {"enabled": False}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--skip-ga" in cmd

    def test_mock_adds_ga_mock(self):
        config = _minimal_config()
        config["ga"] = {"enabled": True, "mock": True}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--ga-mock" in cmd

    def test_fitness_json_adds_flag(self):
        config = _minimal_config()
        config["ga"] = {
            "enabled": True, "mock": False,
            "fitness_json": "/abs/path/fitness.json",
        }
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        idx = cmd.index("--ga-fitness-json")
        assert cmd[idx + 1] == "/abs/path/fitness.json"

    def test_ga_loop_params_mapped(self):
        config = _minimal_config()
        config["ga"] = {
            "enabled": True, "mock": True,
            "generations": 10,
            "population_size": 20,
            "mutation_rate": 0.3,
            "mutation_strength": 0.05,
            "elite_fraction": 0.15,
            "tournament_size": 4,
            "seed": 123,
        }
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        cmd_str = " ".join(cmd)
        assert "--ga-generations 10" in cmd_str
        assert "--ga-population-size 20" in cmd_str
        assert "--ga-mutation-rate 0.3" in cmd_str
        assert "--ga-mutation-strength 0.05" in cmd_str
        assert "--ga-elite-fraction 0.15" in cmd_str
        assert "--ga-tournament-size 4" in cmd_str
        assert "--ga-seed 123" in cmd_str

    def test_ga_params_absent_by_default(self):
        """When no GA section is present, defaults to enabled+no extra flags."""
        config = _minimal_config()
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        # Should NOT have any --ga-* flags since default mock=True doesn't add flag
        # Actually, it adds --ga-mock. Let me check...
        # In build_pipeline_command: ga = config.get("ga", {})
        # if not ga.get("enabled", True): skip
        # else: ga_mock = ga.get("mock", True); if ga_mock: cmd.append("--ga-mock")
        # So default (no ga in config) means ga.get("mock", True) = True, adds --ga-mock
        # But no generation/population flags because those are absent from config
        assert "--ga-mock" in cmd
        assert not any(f in " ".join(cmd) for f in [
            "--ga-generations", "--ga-population-size", "--ga-mutation-rate",
            "--ga-mutation-strength", "--ga-elite-fraction", "--ga-tournament-size",
            "--ga-seed",
        ])


# ---------------------------------------------------------------------------
# build_pipeline_command — MCX options
# ---------------------------------------------------------------------------

class TestMCXOptions:
    def test_disabled_adds_skip_mcx(self):
        config = _minimal_config()
        config["mcx"] = {"enabled": False}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--skip-mcx" in cmd

    def test_dry_run_default_no_mcx_run(self):
        """Dry-run mode should not add --mcx-run."""
        config = _minimal_config()
        config["mcx"] = {"enabled": True, "mode": "dry-run"}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--mcx-run" not in cmd

    def test_run_mode_adds_mcx_run(self):
        config = _minimal_config()
        config["mcx"] = {"enabled": True, "mode": "run"}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--mcx-run" in cmd

    def test_mcx_binary_path(self):
        config = _minimal_config()
        config["mcx"] = {"enabled": True, "mcx_binary": "/opt/mcx/bin/mcx"}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        idx = cmd.index("--mcx-binary")
        assert cmd[idx + 1] == "/opt/mcx/bin/mcx"

    def test_photons(self):
        config = _minimal_config()
        config["mcx"] = {"enabled": True, "photons": 500000}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        idx = cmd.index("--mcx-photons")
        assert cmd[idx + 1] == "500000"


# ---------------------------------------------------------------------------
# build_pipeline_command — Thermal options
# ---------------------------------------------------------------------------

class TestThermalOptions:
    def test_disabled_adds_skip_thermal(self):
        config = _minimal_config()
        config["thermal"] = {"enabled": False}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--skip-thermal" in cmd

    def test_dt_and_num_steps(self):
        config = _minimal_config()
        config["thermal"] = {"enabled": True, "dt": 0.005, "num_steps": 100}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        cmd_str = " ".join(cmd)
        assert "--thermal-dt 0.005" in cmd_str
        assert "--thermal-num-steps 100" in cmd_str

    def test_source_mode_none(self):
        config = _minimal_config()
        config["thermal"] = {"enabled": True, "source": {"mode": "none"}}
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--thermal-source-mode none" in " ".join(cmd)

    def test_source_mode_spherical(self):
        config = _minimal_config()
        config["thermal"] = {
            "enabled": True,
            "source": {
                "mode": "spherical",
                "radius_vox": 3.0,
                "power": 1e6,
                "center": [1.0, 2.0, 3.0],
            },
        }
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        cmd_str = " ".join(cmd)
        assert "--thermal-source-mode spherical" in cmd_str
        assert "--thermal-source-radius-vox 3.0" in cmd_str
        assert "--thermal-source-power 1000000.0" in cmd_str
        assert "--thermal-source-center 1.0 2.0 3.0" in cmd_str

    def test_source_center_default_not_set(self):
        """When center is None, no --thermal-source-center flag."""
        config = _minimal_config()
        config["thermal"] = {
            "enabled": True,
            "source": {"mode": "spherical", "center": None},
        }
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--thermal-source-center" not in cmd


# ---------------------------------------------------------------------------
# build_pipeline_command — error handling & runtime
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_fail_fast_from_config(self):
        config = _minimal_config()
        config["fail_fast"] = True
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--fail-fast" in cmd

    def test_fail_fast_override_wins(self):
        config = _minimal_config()
        config["fail_fast"] = False
        cmd = build_pipeline_command(
            config, {"fail_fast": "true"}, _REPO_ROOT, Path("/tmp")
        )
        assert "--fail-fast" in cmd

    def test_timeout_mapped(self):
        config = _minimal_config()
        config["timeout"] = 120
        cmd = build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
        assert "--timeout 120" in " ".join(cmd)


# ---------------------------------------------------------------------------
# build_pipeline_command — CLI overrides
# ---------------------------------------------------------------------------

class TestCLIOverrides:
    def test_label_volume_override(self):
        config = _minimal_config()  # /tmp/test_volume.npy
        cmd = build_pipeline_command(
            config, {"label_volume": "/override/vol.npy"}, _REPO_ROOT, Path("/tmp")
        )
        idx = cmd.index("--label-volume")
        assert cmd[idx + 1] == "/override/vol.npy"

    def test_output_dir_override(self):
        config = _minimal_config()  # /tmp/test_output
        cmd = build_pipeline_command(
            config, {"output_dir": "/override/out"}, _REPO_ROOT, Path("/tmp")
        )
        idx = cmd.index("--output-dir")
        assert cmd[idx + 1] == "/override/out"


# ---------------------------------------------------------------------------
# _get_overrides
# ---------------------------------------------------------------------------

class TestGetOverrides:
    def test_populated(self):
        ns = Namespace(label_volume="/custom/vol.npy", output_dir="/custom/out", fail_fast=True)
        overrides = _get_overrides(ns)
        assert overrides["label_volume"] == "/custom/vol.npy"
        assert overrides["output_dir"] == "/custom/out"
        assert overrides["fail_fast"] == "true"

    def test_empty(self):
        ns = Namespace(label_volume=None, output_dir=None, fail_fast=False)
        overrides = _get_overrides(ns)
        assert overrides == {}

    def test_fail_fast_false(self):
        ns = Namespace(label_volume=None, output_dir=None, fail_fast=False)
        overrides = _get_overrides(ns)
        assert "fail_fast" not in overrides


# ---------------------------------------------------------------------------
# _resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath:
    def test_absolute_stays_absolute(self):
        result = _resolve_path("/absolute/path.npy", Path("/config/dir"))
        assert result == "/absolute/path.npy"

    def test_relative_resolved_to_config_dir(self):
        result = _resolve_path("relative/path.npy", Path("/config/dir"))
        expected = str((Path("/config/dir") / "relative/path.npy").resolve())
        assert result == expected


# ---------------------------------------------------------------------------
# load_config — validation
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_missing_file_exits(self, tmp_path: Path):
        missing = str(tmp_path / "nonexistent.yaml")
        with pytest.raises(SystemExit):
            load_config(missing)

    def test_missing_required_keys_exits(self, tmp_path: Path):
        cfg = tmp_path / "bad.yaml"
        cfg.write_text("ga:\n  enabled: true\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            load_config(str(cfg))

    def test_valid_config_passes(self, tmp_path: Path):
        cfg = tmp_path / "good.yaml"
        cfg.write_text(
            "label_volume: /tmp/vol.npy\noutput_dir: /tmp/out\n",
            encoding="utf-8",
        )
        result = load_config(str(cfg))
        assert result["label_volume"] == "/tmp/vol.npy"
        assert result["output_dir"] == "/tmp/out"

    def test_required_top_level_keys(self):
        assert "label_volume" in _REQUIRED_TOP_LEVEL
        assert "output_dir" in _REQUIRED_TOP_LEVEL
        assert len(_REQUIRED_TOP_LEVEL) == 2


# ---------------------------------------------------------------------------
# Edge cases / error handling in config mapping
# ---------------------------------------------------------------------------

class TestConfigErrorHandling:
    def test_invalid_mcx_mode_exits(self):
        config = _minimal_config()
        config["mcx"] = {"enabled": True, "mode": "invalid_mode"}
        with pytest.raises(SystemExit):
            build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))

    def test_invalid_thermal_source_mode_exits(self):
        config = _minimal_config()
        config["thermal"] = {"enabled": True, "source": {"mode": "invalid"}}
        with pytest.raises(SystemExit):
            build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))

    def test_invalid_thermal_source_center_exits(self):
        config = _minimal_config()
        config["thermal"] = {
            "enabled": True,
            "source": {"center": [1.0, 2.0]},  # only 2 elements
        }
        with pytest.raises(SystemExit):
            build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))

    def test_ga_mock_false_without_fitness_json_exits(self):
        config = _minimal_config()
        config["ga"] = {"enabled": True, "mock": False}  # no fitness_json
        with pytest.raises(SystemExit):
            build_pipeline_command(config, {}, _REPO_ROOT, Path("/tmp"))
