from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image
import pytest

from src.sequence_utils import SequenceValidationError, validate_contiguous_selection


REPO_ROOT = Path(__file__).resolve().parents[1]


def _touch_png(path: Path, size: tuple[int, int] = (8, 6)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (255, 255, 255)).save(path)


def test_zenodo_filtered_a_files_are_not_contiguous(tmp_path: Path) -> None:
    names = ["2_01_a.png", "2_01_b.png", "2_01_c.png", "2_02_a.png", "2_02_b.png"]
    for name in names:
        _touch_png(tmp_path / name)

    with pytest.raises(SequenceValidationError):
        validate_contiguous_selection([tmp_path / "2_01_a.png", tmp_path / "2_02_a.png"])


def test_zenodo_full_source_block_is_contiguous(tmp_path: Path) -> None:
    paths = []
    for name in ["2_01_a.png", "2_01_b.png", "2_01_c.png", "2_02_a.png"]:
        path = tmp_path / name
        _touch_png(path)
        paths.append(path)


    validate_contiguous_selection(paths)


def test_numeric_slice_gap_is_rejected(tmp_path: Path) -> None:
    paths = []
    for name in ["slice_0000.png", "slice_0001.png", "slice_0003.png"]:
        path = tmp_path / name
        _touch_png(path)
        paths.append(path)

    with pytest.raises(SequenceValidationError):
        validate_contiguous_selection(paths)


def test_build_sequential_stack_rejects_filtered_zenodo_glob(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    for name in ["2_01_a.png", "2_01_b.png", "2_01_c.png", "2_02_a.png"]:
        _touch_png(source_dir / name)

    output_dir = tmp_path / "stack"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/3d/build_sequential_slice_stack.py"),
        "--source-glob",
        str(source_dir / "*_a.png"),
        "--output-dir",
        str(output_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "not contiguous" in (result.stderr + result.stdout)


def test_build_sequential_stack_writes_contiguous_manifest(tmp_path: Path) -> None:
    source_dir = tmp_path / "raw"
    for index, name in enumerate(["2_01_a.png", "2_01_b.png", "2_01_c.png", "2_02_a.png"]):
        _touch_png(source_dir / name, size=(8 + index, 6 + index))

    output_dir = tmp_path / "stack"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts/3d/build_sequential_slice_stack.py"),
        "--source-glob",
        str(source_dir / "*.png"),
        "--output-dir",
        str(output_dir),
        "--num-slices",
        "3",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=True)

    assert (output_dir / "slice_0000.png").exists()
    assert (output_dir / "slice_0001.png").exists()
    assert (output_dir / "slice_0002.png").exists()
    assert (output_dir / "sequence_manifest.json").exists()
