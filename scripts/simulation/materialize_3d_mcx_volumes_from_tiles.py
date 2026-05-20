#!/usr/bin/env python
"""
Materialize small 3D label volumes from tiles_for_simulation label tiles.

This script bridges the 2D oriented tile dataset
(`data/artifacts/tiles_for_simulation/label_id/*.npy`) to 3D label volumes
that can be consumed by MCX tooling.

For each selected tile, it builds a `(D,H,W)` volume by repeating the 2D class
map along Z, then writes:
  - `volumes/<tile_id>__D{depth}.npy`
  - `meta/<tile_id>__D{depth}.json`
  - `manifest.json`
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _load_manifest_ids(manifest_csv: Path) -> List[str]:
    if not manifest_csv.exists():
        return []
    out: List[str] = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = str(row.get("tile_id", "")).strip()
            if tid:
                out.append(tid)
    return out


def _resolve_tile_paths(label_dir: Path, tile_ids: Optional[List[str]]) -> List[Path]:
    if tile_ids:
        paths: List[Path] = []
        for tid in tile_ids:
            p = label_dir / f"{tid}.npy"
            if p.exists():
                paths.append(p)
        return paths
    return sorted(label_dir.glob("*.npy"))


def _sample_paths(paths: List[Path], max_volumes: int, seed: int) -> List[Path]:
    if max_volumes <= 0 or len(paths) <= max_volumes:
        return paths
    rng = np.random.default_rng(seed)
    idx = np.arange(len(paths))
    rng.shuffle(idx)
    keep = np.sort(idx[:max_volumes])
    return [paths[int(i)] for i in keep]


def _volume_meta(tile_id: str, arr2d: np.ndarray, depth: int, out_name: str) -> Dict[str, Any]:
    uniq, counts = np.unique(arr2d, return_counts=True)
    hist = {str(int(k)): int(v) for k, v in zip(uniq.tolist(), counts.tolist())}
    return {
        "tile_id": tile_id,
        "source_shape_2d": [int(arr2d.shape[0]), int(arr2d.shape[1])],
        "depth": int(depth),
        "output_volume": out_name,
        "present_class_ids": [int(x) for x in uniq.tolist()],
        "class_histogram_2d": hist,
    }


def _build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Build 3D extruded class-ID volumes from tiles_for_simulation label tiles."
    )
    ap.add_argument(
        "--tiles-root",
        default="data/artifacts/tiles_for_simulation",
        help="tiles_for_simulation root directory.",
    )
    ap.add_argument(
        "--label-dir",
        default="",
        help="Override label tile directory (default: <tiles-root>/label_id).",
    )
    ap.add_argument(
        "--tiles-manifest-csv",
        default="",
        help="Optional tile manifest CSV to preserve tile ordering.",
    )
    ap.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for generated volumes and metadata.",
    )
    ap.add_argument(
        "--depth",
        type=int,
        default=16,
        help="Extrusion depth D (default: 16).",
    )
    ap.add_argument(
        "--max-volumes",
        type=int,
        default=0,
        help="Max number of volumes to generate (0 = all).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed when --max-volumes > 0.",
    )
    ap.add_argument(
        "--tile-ids",
        nargs="*",
        default=None,
        help="Explicit tile IDs to process (without .npy extension).",
    )
    return ap


def main() -> int:
    args = _build_cli().parse_args()

    if args.depth < 1:
        raise SystemExit(f"--depth must be >= 1, got {args.depth}")
    if args.max_volumes < 0:
        raise SystemExit(f"--max-volumes must be >= 0, got {args.max_volumes}")

    tiles_root = Path(args.tiles_root).resolve()
    label_dir = Path(args.label_dir).resolve() if args.label_dir else (tiles_root / "label_id")
    if not label_dir.exists():
        raise SystemExit(f"Label directory not found: {label_dir}")

    manifest_csv = (
        Path(args.tiles_manifest_csv).resolve()
        if args.tiles_manifest_csv
        else (tiles_root / "tiles_manifest.csv")
    )

    tile_ids = args.tile_ids
    if tile_ids is None:
        ordered_ids = _load_manifest_ids(manifest_csv)
        if ordered_ids:
            paths = _resolve_tile_paths(label_dir, ordered_ids)
        else:
            paths = _resolve_tile_paths(label_dir, None)
    else:
        paths = _resolve_tile_paths(label_dir, tile_ids)

    if not paths:
        raise SystemExit("No label tiles found to materialize.")

    paths = _sample_paths(paths, int(args.max_volumes), int(args.seed))

    out_root = Path(args.output_dir).resolve()
    out_vol = out_root / "volumes"
    out_meta = out_root / "meta"
    out_root.mkdir(parents=True, exist_ok=True)
    out_vol.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for p in paths:
        tile_id = p.stem
        arr = np.load(str(p))
        if arr.ndim != 2:
            raise SystemExit(f"Expected 2D class map at {p}, got shape {arr.shape}")

        arr_i = arr.astype(np.int32)
        vol = np.repeat(arr_i[np.newaxis, ...], int(args.depth), axis=0)

        vol_name = f"{tile_id}__D{int(args.depth)}.npy"
        meta_name = f"{tile_id}__D{int(args.depth)}.json"
        vol_path = out_vol / vol_name
        meta_path = out_meta / meta_name
        np.save(str(vol_path), vol)

        meta = _volume_meta(tile_id=tile_id, arr2d=arr_i, depth=int(args.depth), out_name=vol_name)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        records.append(
            {
                "tile_id": tile_id,
                "source_label_tile": str(p.resolve()),
                "volume_npy": str(vol_path.resolve()),
                "meta_json": str(meta_path.resolve()),
                "present_class_ids": meta["present_class_ids"],
            }
        )

    manifest = {
        "tiles_root": str(tiles_root),
        "label_dir": str(label_dir),
        "depth": int(args.depth),
        "num_volumes": len(records),
        "records": records,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Generated {len(records)} volume(s)")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
