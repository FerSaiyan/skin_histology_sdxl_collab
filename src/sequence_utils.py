"""Utilities for ordering and validating sequential histology slice paths."""

from __future__ import annotations

import glob
import re
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class SequenceValidationError(ValueError):
    """Raised when selected slice paths are not a contiguous sequence."""


def natural_sort_key(value: str | Path) -> list[object]:
    """Human-friendly path sort key, so slice_2 sorts before slice_10."""
    text = str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def natural_sorted_paths(paths: Iterable[str | Path]) -> list[Path]:
    return sorted((Path(p) for p in paths), key=lambda p: natural_sort_key(str(p)))


def collect_ordered_paths(pattern: str) -> list[Path]:
    paths = natural_sorted_paths(glob.glob(pattern))
    if not paths:
        raise SequenceValidationError(f"No files match glob pattern: {pattern}")
    return paths


def parse_slice_index(path: str | Path) -> int | None:
    """Parse common slice_0001-style numeric indices."""
    stem = Path(path).stem
    match = re.search(r"(?:^|[_-])slice[_-]?(\d+)$", stem, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|[_-])(\d+)$", stem)
    if match:
        return int(match.group(1))
    return None


def parse_zenodo_cropped_name(path: str | Path) -> tuple[int, int, str] | None:
    """Parse Zenodo melanoma cropped_slices names like 2_01_a.png."""
    match = re.fullmatch(r"(\d+)_(\d+)_([A-Za-z]+)", Path(path).stem)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3).lower()


def default_reference_paths(paths: list[str | Path]) -> list[Path] | None:
    """Return full sibling reference order for Zenodo cropped_slices selections."""
    path_objs = [Path(p) for p in paths]
    if not path_objs:
        return None
    parents = {p.parent.resolve() for p in path_objs}
    if len(parents) != 1:
        return None
    parsed = [parse_zenodo_cropped_name(p) for p in path_objs]
    if not all(item is not None for item in parsed):
        return None

    parent = path_objs[0].parent
    case_ids = {item[0] for item in parsed if item is not None}
    siblings = [p for p in parent.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
    reference = []
    for sibling in siblings:
        parsed_sibling = parse_zenodo_cropped_name(sibling)
        if parsed_sibling and parsed_sibling[0] in case_ids:
            reference.append(sibling)
    return natural_sorted_paths(reference) if reference else None


def validate_contiguous_selection(
    paths: list[str | Path],
    *,
    reference_paths: list[str | Path] | None = None,
    context: str = "slice selection",
) -> None:
    """Validate that paths form a contiguous sequence.

    For Zenodo melanoma cropped_slices names, validation is against all sibling
    files for the same case prefix. This rejects filtered subsets like *_a.png
    because they skip the intervening *_b/*_c files in the source order.
    """
    path_objs = [Path(p) for p in paths]
    if len(path_objs) != len({p.resolve() for p in path_objs}):
        raise SequenceValidationError(f"{context}: duplicate slice paths detected")

    if not path_objs:
        raise SequenceValidationError(f"{context}: no slice paths provided")

    if reference_paths is None:
        reference_paths = default_reference_paths(path_objs)

    if reference_paths is not None:
        reference = natural_sorted_paths(reference_paths)
        by_resolved = {p.resolve(): i for i, p in enumerate(reference)}
        missing = [p for p in path_objs if p.resolve() not in by_resolved]
        if missing:
            missing_list = ", ".join(p.name for p in missing[:5])
            raise SequenceValidationError(
                f"{context}: selected file(s) are absent from reference order: {missing_list}"
            )
        positions = [by_resolved[p.resolve()] for p in path_objs]
        expected_positions = list(range(min(positions), max(positions) + 1))
        if positions != expected_positions:
            expected_names = [reference[i].name for i in expected_positions]
            selected_names = [p.name for p in path_objs]
            raise SequenceValidationError(
                f"{context}: selected files are not contiguous in source order. "
                f"Selected first/last: {selected_names[0]} -> {selected_names[-1]}; "
                f"expected contiguous block: {expected_names[0]} -> {expected_names[-1]} "
                f"({len(expected_names)} files), got {len(selected_names)} files."
            )
        return

    indices = [parse_slice_index(p) for p in path_objs]
    if all(index is not None for index in indices):
        numeric = [int(index) for index in indices if index is not None]
        expected = list(range(numeric[0], numeric[-1] + 1))
        if numeric != expected:
            raise SequenceValidationError(
                f"{context}: numeric slice indices are not contiguous: "
                f"{numeric[0]} -> {numeric[-1]} with {len(numeric)} files"
            )


def sequence_summary(paths: list[str | Path]) -> dict[str, object]:
    path_objs = natural_sorted_paths(paths)
    summary: dict[str, object] = {
        "num_slices": len(path_objs),
        "first": path_objs[0].name if path_objs else None,
        "last": path_objs[-1].name if path_objs else None,
        "contiguous": False,
    }
    try:
        validate_contiguous_selection(path_objs)
        summary["contiguous"] = True
    except SequenceValidationError as exc:
        summary["contiguous_error"] = str(exc)
    return summary
