from __future__ import annotations

import csv

import numpy as np
from PIL import Image

from scripts.optical_ga.build_tiles_for_simulation import (
    BuildOptions,
    MANIFEST_COLUMNS,
    TileFilters,
    _air_epidermis_interface_px,
    _block_integral,
    _iter_grid_positions,
    _process_pair,
    _rgb_mask_to_class_map,
    _window_count,
)
from scripts.optical_ga.build_histoseg_label_volume import HISTOSEG_COLOR_TO_CLASS_ID
from scripts.segmentation.histoseg_tile_dataset import HistosegSimulationTileDataset


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


def test_interface_count_excludes_epidermis_dermis_boundary():
    labels = np.full((16, 16), 4, dtype=np.uint8)
    labels[:, :8] = 1
    assert _air_epidermis_interface_px(labels) == 0


def test_rgb_mask_lookup_preserves_known_classes():
    colors = list(HISTOSEG_COLOR_TO_CLASS_ID)
    rgb = np.array(colors, dtype=np.uint8).reshape(1, len(colors), 3)
    expected = np.array(
        [HISTOSEG_COLOR_TO_CLASS_ID[color] for color in colors], dtype=np.uint8
    ).reshape(1, -1)
    np.testing.assert_array_equal(_rgb_mask_to_class_map(rgb), expected)


def test_block_integral_matches_direct_window_counts():
    binary = np.zeros((24, 28), dtype=bool)
    binary[3:19, 7:24] = True
    integral, scale = _block_integral(binary, 4)
    for y0, x0 in [(0, 0), (4, 8), (8, 12), (5, 7)]:
        actual = _window_count(binary, integral, scale, y0, x0, 12)
        expected = int(binary[y0:y0 + 12, x0:x0 + 12].sum())
        assert actual == expected


def test_segmentation_mode_writes_only_required_outputs(tmp_path):
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    labels = np.full((64, 64), 4, dtype=np.uint8)
    labels[:8] = 0
    labels[8:16] = 1
    class_to_color = {class_id: color for color, class_id in HISTOSEG_COLOR_TO_CLASS_ID.items()}
    mask_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    for class_id, color in class_to_color.items():
        mask_rgb[labels == class_id] = color

    image_path = tmp_path / "slide.jpg"
    mask_path = tmp_path / "slide.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask_rgb).save(mask_path)
    output_dir = tmp_path / "tiles"
    (output_dir / "rgb").mkdir(parents=True)
    (output_dir / "label_id").mkdir()
    options = BuildOptions(
        mode="segmentation",
        tile_size=32,
        stride=32,
        screen_downsample=4,
        filters=TileFilters(0.25, 1.0, 0.02, 8, 0.55),
        max_tiles_per_image=0,
        strict_air_epidermis=False,
        write_label_vis=False,
        write_epidermis_mask=False,
        write_tile_meta=False,
        png_compress_level=1,
        rgb_format="png",
        jpeg_quality=95,
        storage="files",
        staging_dir="",
    )
    rows, stats, _ = _process_pair(
        {"image_path": str(image_path), "mask_path": str(mask_path), "slice_id": "slide"},
        tmp_path,
        output_dir,
        options,
    )

    assert stats["num_candidates"] == 4
    assert stats["num_keep"] == 4
    assert len(rows) == 4
    assert all(row["label_vis_path"] == "" for row in rows)
    assert all(row["epidermis_mask_path"] == "" for row in rows)
    assert all(row["tile_meta_path"] == "" for row in rows)
    assert len(list((output_dir / "rgb").glob("*.png"))) == 4
    assert len(list((output_dir / "label_id").glob("*.npy"))) == 4
    assert not (output_dir / "label_vis").exists()
    assert not (output_dir / "epidermis_mask").exists()
    assert not (output_dir / "meta").exists()


def test_hdf5_segmentation_shard_loads_through_training_dataset(tmp_path):
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    labels = np.full((64, 64), 4, dtype=np.uint8)
    labels[:8] = 0
    labels[8:16] = 1
    class_to_color = {class_id: color for color, class_id in HISTOSEG_COLOR_TO_CLASS_ID.items()}
    mask_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    for class_id, color in class_to_color.items():
        mask_rgb[labels == class_id] = color
    image_path = tmp_path / "shard_slide.jpg"
    mask_path = tmp_path / "shard_slide.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask_rgb).save(mask_path)
    output_dir = tmp_path / "sharded_tiles"
    (output_dir / "shards").mkdir(parents=True)
    options = BuildOptions(
        mode="segmentation",
        tile_size=32,
        stride=32,
        screen_downsample=4,
        filters=TileFilters(0.25, 1.0, 0.02, 8, 0.55),
        max_tiles_per_image=0,
        strict_air_epidermis=False,
        write_label_vis=False,
        write_epidermis_mask=False,
        write_tile_meta=False,
        png_compress_level=1,
        rgb_format="jpg",
        jpeg_quality=95,
        storage="hdf5",
        staging_dir=str(tmp_path / "staging"),
    )
    (tmp_path / "staging").mkdir()
    rows, stats, _ = _process_pair(
        {"image_path": str(image_path), "mask_path": str(mask_path), "slice_id": "shard_slide"},
        tmp_path,
        output_dir,
        options,
    )
    assert stats["num_keep"] == 4
    assert len(list((output_dir / "shards").glob("*.h5"))) == 1
    split_path = output_dir / "splits.csv"
    with split_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*MANIFEST_COLUMNS, "split"])
        writer.writeheader()
        writer.writerows([{**row, "split": "train"} for row in rows])
    dataset = HistosegSimulationTileDataset(
        tiles_root=output_dir,
        splits_csv=split_path,
        split="train",
        encoder_size=32,
    )
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 32, 32)
    assert tuple(sample["label"].shape) == (32, 32)
    assert set(sample["label"].unique().tolist()).issubset({0, 1, 4})
