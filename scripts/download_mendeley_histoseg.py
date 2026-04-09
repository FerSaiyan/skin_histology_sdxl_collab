#!/usr/bin/env python
"""
Download Histo-Seg (Mendeley Data) files through the public API.

Examples:
  python scripts/download_mendeley_histoseg.py \
      --dataset-id vccj8mp2cg \
      --expected-version 2 \
      --manifest-out data/metadata/histo_seg_manifest.json

  python scripts/download_mendeley_histoseg.py \
      --dataset-id vccj8mp2cg \
      --expected-version 2 \
      --manifest-out data/metadata/histo_seg_manifest.json \
      --download \
      --download-dir data/raw/histo_seg_v2 \
      --skip-existing
"""

from __future__ import annotations

import argparse
import json
import ssl
from pathlib import Path
from typing import Dict, List
from urllib.request import Request, urlopen


def _fetch_json(url: str, timeout_sec: int = 60) -> Dict:
    req = Request(url, headers={"User-Agent": "skin-histology-sdxl-collab/1.0"})
    context = ssl.create_default_context()
    with urlopen(req, timeout=timeout_sec, context=context) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _download_file(url: str, dst: Path, timeout_sec: int = 120) -> None:
    req = Request(url, headers={"User-Agent": "skin-histology-sdxl-collab/1.0"})
    context = ssl.create_default_context()
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urlopen(req, timeout=timeout_sec, context=context) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)


def _extract_files(dataset_json: Dict, include_ext: List[str]) -> List[Dict]:
    include_set = {x.strip().lower().lstrip(".") for x in include_ext if x.strip()}
    selected: List[Dict] = []

    for file_obj in dataset_json.get("files", []):
        name = str(file_obj.get("filename", ""))
        if "." not in name:
            continue
        ext = name.rsplit(".", 1)[-1].lower()
        if include_set and ext not in include_set:
            continue

        content = file_obj.get("content_details", {}) or {}
        selected.append(
            {
                "filename": name,
                "mendeley_file_id": file_obj.get("id"),
                "download_url": content.get("download_url"),
                "view_url": content.get("view_url"),
                "sha256": content.get("sha256_hash"),
                "content_type": content.get("content_type"),
                "size_bytes": content.get("size", 0),
            }
        )

    selected.sort(key=lambda x: x["filename"])
    return selected


def _write_manifest(manifest_out: Path, payload: Dict) -> None:
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_manifest(path: Path) -> Dict:
    if not path.is_file():
        raise SystemExit(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch/download Histo-Seg dataset from Mendeley public API.")
    ap.add_argument("--dataset-id", default="", help="Mendeley dataset id (e.g. vccj8mp2cg).")
    ap.add_argument("--expected-version", type=int, default=None, help="Optional expected dataset version.")
    ap.add_argument("--manifest-out", default="", help="Where to write the dataset manifest JSON.")
    ap.add_argument("--manifest-in", default="", help="Use an existing manifest JSON instead of calling the API.")
    ap.add_argument("--download", action="store_true", help="If set, download files to --download-dir.")
    ap.add_argument("--download-dir", default=None, help="Output directory for files when --download is used.")
    ap.add_argument("--include-ext", default="jpg,png", help="Comma-separated extensions to keep (default: jpg,png).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip files that already exist with matching size.")
    ap.add_argument("--max-files", type=int, default=0, help="Optional limit for debug runs (0 = all).")
    ap.add_argument("--timeout-sec", type=int, default=120, help="HTTP timeout in seconds.")
    args = ap.parse_args()

    manifest_in = str(args.manifest_in or "").strip()
    manifest_out = str(args.manifest_out or "").strip()
    include_ext = [x.strip() for x in args.include_ext.split(",") if x.strip()]

    if manifest_in:
        manifest = _load_manifest(Path(manifest_in))
        files = manifest.get("files", []) or []
        if args.max_files and args.max_files > 0:
            files = files[: args.max_files]
        total_size = int(sum(int(f.get("size_bytes", 0) or 0) for f in files))
        print(f"[Mendeley] Using manifest: {manifest_in} ({len(files)} files)")
        print(f"[Mendeley] Total selected size: {total_size / (1024 ** 3):.3f} GiB")
    else:
        dataset_id = args.dataset_id.strip()
        if not dataset_id:
            raise SystemExit("--dataset-id is required unless --manifest-in is provided")

        api_url = f"https://data.mendeley.com/public-api/datasets/{dataset_id}"
        print(f"[Mendeley] Fetching metadata: {api_url}")
        ds = _fetch_json(api_url, timeout_sec=max(30, int(args.timeout_sec)))

        resolved_version = ds.get("version")
        if args.expected_version is not None and int(resolved_version) != int(args.expected_version):
            raise SystemExit(
                f"Version mismatch: expected {args.expected_version}, got {resolved_version}. "
                "Update --expected-version if this is intentional."
            )

        files = _extract_files(ds, include_ext=include_ext)
        if args.max_files and args.max_files > 0:
            files = files[: args.max_files]

        total_size = int(sum(int(f.get("size_bytes", 0) or 0) for f in files))
        manifest = {
            "dataset_id": dataset_id,
            "dataset_name": ds.get("name"),
            "description": ds.get("description"),
            "requested_version": args.expected_version,
            "resolved_version": resolved_version,
            "api_url": api_url,
            "include_ext": include_ext,
            "file_count": len(files),
            "total_size_bytes": total_size,
            "files": files,
        }

        if manifest_out:
            _write_manifest(Path(manifest_out), manifest)
            print(f"[Mendeley] Manifest saved: {manifest_out} ({len(files)} files)")
        print(f"[Mendeley] Total selected size: {total_size / (1024 ** 3):.3f} GiB")

    if not args.download:
        return

    if not args.download_dir:
        raise SystemExit("--download requires --download-dir")

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Mendeley] Download target: {download_dir}")

    n_done = 0
    n_skip = 0
    for i, file_meta in enumerate(files, start=1):
        filename = str(file_meta["filename"])
        size_bytes = int(file_meta.get("size_bytes") or 0)
        url = str(file_meta.get("download_url") or "")
        if not url:
            print(f"[WARN] Missing download URL for {filename}, skipping.")
            continue

        dst = download_dir / filename
        if args.skip_existing and dst.exists() and dst.is_file() and (size_bytes <= 0 or dst.stat().st_size == size_bytes):
            n_skip += 1
            print(f"[{i}/{len(files)}] SKIP {filename}")
            continue

        print(f"[{i}/{len(files)}] GET  {filename}")
        _download_file(url, dst, timeout_sec=max(60, int(args.timeout_sec)))
        n_done += 1

    print(f"[Mendeley] Download complete. downloaded={n_done} skipped={n_skip}")


if __name__ == "__main__":
    main()
