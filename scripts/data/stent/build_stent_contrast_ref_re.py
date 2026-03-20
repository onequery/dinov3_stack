#!/usr/bin/env python3
"""
Rebuild Stent-Contrast / Stent-Ref style PNG datasets from DICOMs.

Input structure (example):
  <dcm_root>/<split>/<class>/<patient>/<study_date>/XA/<series>.dcm

Output structures:
  <ref_root>/<split>/<class>/<patient>/<study_date>/XA/<series>.png
  <contrast_root>/<split>/<class>/<patient>/<study_date>/XA/<series>.png

Ref image:
- frame index 0 from each DICOM.

Contrast image:
- frame index from --label-json (filename key: patient/study_date/XA/<series>.dcm).
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm


def normalize_to_uint8(img: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, low, high)
    img = (img - low) / (high - low)
    img = (img * 255.0).astype(np.uint8)
    return img


def get_bits_stored_range(ds) -> tuple[int, int]:
    bits_stored = int(ds.BitsStored)
    pixel_repr = int(ds.PixelRepresentation)  # 0=unsigned, 1=signed
    if pixel_repr == 0:
        return 0, (2**bits_stored) - 1
    return -(2 ** (bits_stored - 1)), (2 ** (bits_stored - 1)) - 1


def extract_frame_uint8(pixel: np.ndarray, ds, frame_index: int) -> np.ndarray:
    if "PixelData" not in ds:
        raise ValueError("No PixelData")

    if pixel.ndim == 3:
        if frame_index < 0 or frame_index >= pixel.shape[0]:
            raise IndexError(f"Frame index out of range: {frame_index}")
        frame = pixel[frame_index]
    else:
        if frame_index not in (0, -1):
            raise IndexError(f"Single-frame DICOM, index {frame_index} is invalid")
        frame = pixel

    try:
        pixel_voi = apply_voi_lut(frame, ds)
        if pixel_voi is not None:
            low, high = float(pixel_voi.min()), float(pixel_voi.max())
            return normalize_to_uint8(pixel_voi, low, high)
    except Exception:
        pass

    if "WindowCenter" in ds and "WindowWidth" in ds:
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        wc = float(wc[0] if isinstance(wc, (list, pydicom.multival.MultiValue)) else wc)
        ww = float(ww[0] if isinstance(ww, (list, pydicom.multival.MultiValue)) else ww)
        low = wc - ww / 2.0
        high = wc + ww / 2.0
        return normalize_to_uint8(frame, low, high)

    low, high = get_bits_stored_range(ds)
    return normalize_to_uint8(frame, low, high)


def iter_label_records(json_path: Path) -> Iterable[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        first = f.read(1)
        if not first:
            return
        f.seek(0)

        if first == "[":
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def build_label_index_map(label_json: Path, frame_index_base: int) -> dict[str, int]:
    label_map: dict[str, int] = {}
    for record in tqdm(iter_label_records(label_json), desc="Reading label json"):
        filename = record.get("filename")
        data = record.get("data")
        if not filename or not isinstance(data, list) or not data:
            continue
        item = data[0]
        if not isinstance(item, dict):
            continue
        raw_index = item.get("index")
        if raw_index is None:
            continue
        try:
            index = int(raw_index) - frame_index_base
        except (TypeError, ValueError):
            continue
        if index < 0:
            continue
        label_map[str(Path(filename).as_posix())] = index
    return label_map


def save_png(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


def write_summary(
    summary_path: Path,
    dcm_root: Path,
    ref_root: Path,
    contrast_root: Path,
    frame_index_base: int,
    stats: dict[str, int],
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("========== Stent Contrast/Ref rebuild summary ==========\n")
        f.write(f"dcm_root: {dcm_root.resolve()}\n")
        f.write(f"ref_root: {ref_root.resolve()}\n")
        f.write(f"contrast_root: {contrast_root.resolve()}\n")
        f.write(f"frame_index_base: {frame_index_base}\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n\n")
        for key in sorted(stats.keys()):
            f.write(f"{key}: {stats[key]}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcm-root", type=Path, default=Path("input/stent_split_dcm_re"))
    parser.add_argument("--label-json", type=Path, default=Path("input/frames_prediction.json"))
    parser.add_argument("--ref-root", type=Path, default=Path("input/Stent-Ref-re"))
    parser.add_argument("--contrast-root", type=Path, default=Path("input/Stent-Contrast-re"))
    parser.add_argument(
        "--reuse-ref-root",
        type=Path,
        default=Path("input/Stent-Ref"),
        help="Optional existing Stent-Ref root to reuse PNGs from",
    )
    parser.add_argument(
        "--reuse-contrast-root",
        type=Path,
        default=Path("input/Stent-Contrast"),
        help="Optional existing Stent-Contrast root to reuse PNGs from",
    )
    parser.add_argument("--frame-index-base", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip writing output file if it already exists",
    )
    return parser.parse_args()


def build_png_index(root: Path) -> dict[str, Path]:
    """
    Build index key:
      patient/study_date/XA/file.png
    from dataset layout:
      split/class/patient/study_date/XA/file.png
    """
    if not root.exists():
        return {}

    index: dict[str, Path] = {}
    for png_path in root.rglob("*.png"):
        rel = png_path.relative_to(root)
        if len(rel.parts) < 5:
            continue
        key = str(Path(*rel.parts[2:]).as_posix())
        if key not in index:
            index[key] = png_path
    return index


def main() -> None:
    args = parse_args()

    if not args.dcm_root.exists():
        raise FileNotFoundError(f"dcm root not found: {args.dcm_root}")
    if not args.label_json.exists():
        raise FileNotFoundError(f"label json not found: {args.label_json}")

    label_map = build_label_index_map(args.label_json, args.frame_index_base)
    reuse_ref_index = build_png_index(args.reuse_ref_root)
    reuse_contrast_index = build_png_index(args.reuse_contrast_root)
    dcm_files = sorted(args.dcm_root.rglob("*.dcm"))
    if not dcm_files:
        raise RuntimeError(f"No DCM files found under {args.dcm_root}")

    stats = {
        "dcm_total": len(dcm_files),
        "ref_saved": 0,
        "contrast_saved": 0,
        "ref_reused": 0,
        "contrast_reused": 0,
        "ref_skipped_existing": 0,
        "contrast_skipped_existing": 0,
        "contrast_missing_label": 0,
        "ref_failed": 0,
        "contrast_failed": 0,
    }

    missing_label_path = args.contrast_root / "_logs" / "missing_label_files.txt"
    failed_path = args.contrast_root / "_logs" / "failed_files.txt"
    missing_label_path.parent.mkdir(parents=True, exist_ok=True)

    missing_labels: list[str] = []
    failed_files: list[str] = []

    for dcm_path in tqdm(dcm_files, desc="Extracting ref/contrast"):
        rel = dcm_path.relative_to(args.dcm_root)
        if len(rel.parts) < 5:
            failed_files.append(f"[layout] {rel.as_posix()}")
            continue

        ref_out = (args.ref_root / rel).with_suffix(".png")
        contrast_out = (args.contrast_root / rel).with_suffix(".png")

        rel_label_key = str(Path(*rel.parts[2:]).as_posix())
        contrast_idx = label_map.get(rel_label_key)
        rel_png_key = str(Path(*rel.parts[2:]).with_suffix(".png").as_posix())
        if contrast_idx is None:
            stats["contrast_missing_label"] += 1
            missing_labels.append(rel_label_key)

        need_ref = not (args.skip_existing and ref_out.exists())
        need_contrast = contrast_idx is not None and not (
            args.skip_existing and contrast_out.exists()
        )

        if not need_ref:
            stats["ref_skipped_existing"] += 1
        if contrast_idx is not None and not need_contrast:
            stats["contrast_skipped_existing"] += 1
        if not need_ref and not need_contrast:
            continue

        ref_done = False
        contrast_done = False

        if need_ref:
            src_ref_png = reuse_ref_index.get(rel_png_key)
            if src_ref_png is not None and src_ref_png.exists():
                ref_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_ref_png, ref_out)
                stats["ref_reused"] += 1
                ref_done = True

        if need_contrast:
            src_contrast_png = reuse_contrast_index.get(rel_png_key)
            if src_contrast_png is not None and src_contrast_png.exists():
                contrast_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_contrast_png, contrast_out)
                stats["contrast_reused"] += 1
                contrast_done = True

        if ref_done and (contrast_done or not need_contrast):
            continue

        if contrast_done and not need_ref:
            continue

        try:
            ds = pydicom.dcmread(dcm_path, force=True)
            pixel = ds.pixel_array.astype(np.float32)
        except Exception as e:
            failed_files.append(f"[load] {rel.as_posix()} | {e}")
            if need_ref and not ref_done:
                stats["ref_failed"] += 1
            if need_contrast and not contrast_done:
                stats["contrast_failed"] += 1
            continue

        if need_ref and not ref_done:
            try:
                ref_img = extract_frame_uint8(pixel, ds, 0)
                save_png(ref_img, ref_out)
                stats["ref_saved"] += 1
            except Exception as e:
                failed_files.append(f"[ref] {rel.as_posix()} | {e}")
                stats["ref_failed"] += 1

        if need_contrast and not contrast_done:
            try:
                contrast_img = extract_frame_uint8(pixel, ds, int(contrast_idx))
                save_png(contrast_img, contrast_out)
                stats["contrast_saved"] += 1
            except Exception as e:
                failed_files.append(f"[contrast] {rel.as_posix()} | {e}")
                stats["contrast_failed"] += 1

    if missing_labels:
        with missing_label_path.open("w", encoding="utf-8") as f:
            for item in missing_labels:
                f.write(item + "\n")

    if failed_files:
        with failed_path.open("w", encoding="utf-8") as f:
            for item in failed_files:
                f.write(item + "\n")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_summary(
        summary_path=args.contrast_root / "_logs" / f"rebuild_summary_{ts}.txt",
        dcm_root=args.dcm_root,
        ref_root=args.ref_root,
        contrast_root=args.contrast_root,
        frame_index_base=args.frame_index_base,
        stats=stats,
    )

    print("Done.")
    for k in sorted(stats.keys()):
        print(f"{k}: {stats[k]}")


if __name__ == "__main__":
    main()
