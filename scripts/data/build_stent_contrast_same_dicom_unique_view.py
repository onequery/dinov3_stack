#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, TextIO

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm


DEFAULT_BASE_IMAGE_ROOT = Path("input/Stent-Contrast-unique-view")
DEFAULT_DCM_ROOT = Path("input/stent_split_dcm_unique_view")
DEFAULT_LABEL_JSON = Path("input/frames_prediction.json")
DEFAULT_OUTPUT_ROOT = Path("input/Stent-Contrast-same-dicom-unique-view")
DEFAULT_MANIFEST_NAME = "manifest_same_dicom_master.csv"
DEFAULT_SUMMARY_NAME = "summary_same_dicom_build.csv"
DEFAULT_FRAME_INDEX_BASE = 0
FRAME_OFFSETS = (-3, 3)
SPLITS = ("train", "valid", "test")


class TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass(frozen=True)
class BaseImageRow:
    split: str
    class_name: str
    patient_id: str
    study_id: str
    series_name: str
    base_png_path: Path
    dcm_path: Path
    dcm_rel_path: str
    dicom_id: str
    contrast_frame_index: int


@dataclass(frozen=True)
class ExtractedRow:
    split: str
    class_name: str
    patient_id: str
    study_id: str
    dicom_rel_path: str
    dicom_id: str
    contrast_frame_index: int
    frame_index: int
    frame_offset: int
    img_path: str
    source_contrast_png: str


class BuildError(RuntimeError):
    pass


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_console_and_file_logging(
    output_root: Path,
    log_file_arg: str | None,
    default_prefix: str,
) -> tuple[Path, TextIO, TextIO, TextIO]:
    if log_file_arg:
        log_path = Path(log_file_arg).expanduser()
        if not log_path.is_absolute():
            log_path = output_root / log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_root / f"{default_prefix}_{stamp}.log"
    ensure_dir(log_path.parent)
    file_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, file_handle)
    sys.stderr = TeeStream(original_stderr, file_handle)
    log(f"Console output is mirrored to log file: {log_path}")
    return log_path, file_handle, original_stdout, original_stderr


def restore_console_logging(
    file_handle: TextIO,
    original_stdout: TextIO,
    original_stderr: TextIO,
) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    file_handle.close()


def normalize_to_uint8(img: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        return np.zeros_like(img, dtype=np.uint8)
    img = np.clip(img, low, high)
    img = (img - low) / (high - low)
    img = (img * 255.0).astype(np.uint8)
    return img


def get_bits_stored_range(ds) -> tuple[int, int]:
    bits_stored = int(ds.BitsStored)
    pixel_repr = int(ds.PixelRepresentation)
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
    with json_path.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        if not first:
            return
        handle.seek(0)

        if first == "[":
            data = json.load(handle)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield item
            return

        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
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


def collect_base_pngs(base_image_root: Path) -> list[Path]:
    return sorted(base_image_root.rglob("*.png"))


def sample_paths_per_split(paths: Sequence[Path], root: Path, max_images_per_split: int | None, seed: int) -> list[Path]:
    if max_images_per_split is None or max_images_per_split <= 0:
        return list(paths)
    grouped: dict[str, list[Path]] = {split: [] for split in SPLITS}
    for path in paths:
        rel = path.relative_to(root)
        grouped.setdefault(rel.parts[0], []).append(path)
    rng = np.random.default_rng(int(seed))
    sampled: list[Path] = []
    for split in SPLITS:
        split_paths = sorted(grouped.get(split, []))
        if len(split_paths) <= max_images_per_split:
            sampled.extend(split_paths)
            continue
        selected = np.sort(rng.choice(len(split_paths), size=max_images_per_split, replace=False))
        sampled.extend([split_paths[idx] for idx in selected])
    return sorted(sampled)


def build_base_rows(
    base_image_root: Path,
    dcm_root: Path,
    label_map: dict[str, int],
    max_images_per_split: int | None,
    seed: int,
) -> tuple[list[BaseImageRow], pd.DataFrame]:
    png_paths = sample_paths_per_split(
        collect_base_pngs(base_image_root),
        root=base_image_root,
        max_images_per_split=max_images_per_split,
        seed=seed,
    )
    rows: list[BaseImageRow] = []
    summary_rows: list[dict[str, object]] = []
    for png_path in png_paths:
        rel = png_path.relative_to(base_image_root)
        if len(rel.parts) < 5:
            raise BuildError(f"Unexpected base image layout: {png_path}")
        split, class_name, patient_id, study_id = rel.parts[:4]
        series_name = png_path.stem
        dcm_rel = rel.with_suffix(".dcm")
        dcm_path = dcm_root / dcm_rel
        dcm_rel_path = dcm_rel.as_posix()
        dicom_id = Path(patient_id, study_id, "XA", series_name).as_posix()
        label_key = Path(patient_id, study_id, "XA", f"{series_name}.dcm").as_posix()
        contrast_frame_index = label_map.get(label_key)
        rows.append(
            BaseImageRow(
                split=split,
                class_name=class_name,
                patient_id=patient_id,
                study_id=study_id,
                series_name=series_name,
                base_png_path=png_path,
                dcm_path=dcm_path,
                dcm_rel_path=dcm_rel_path,
                dicom_id=dicom_id,
                contrast_frame_index=int(contrast_frame_index) if contrast_frame_index is not None else -1,
            )
        )
        summary_rows.append(
            {
                "split": split,
                "class_name": class_name,
                "patient_id": patient_id,
                "study_id": study_id,
                "series_name": series_name,
                "base_png_path": png_path.as_posix(),
                "dcm_path": dcm_path.as_posix(),
                "dcm_rel_path": dcm_rel_path,
                "dicom_id": dicom_id,
                "label_key": label_key,
                "has_dcm": int(dcm_path.exists()),
                "has_label": int(contrast_frame_index is not None),
                "contrast_frame_index": int(contrast_frame_index) if contrast_frame_index is not None else "",
            }
        )
    return rows, pd.DataFrame(summary_rows)


def save_png(arr: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, mode="L").save(out_path)


def build_frame_output_path(output_root: Path, row: BaseImageRow, frame_offset: int, frame_index: int) -> Path:
    rel = row.base_png_path.relative_to(DEFAULT_BASE_IMAGE_ROOT if False else row.base_png_path.parents[5])
    stem = row.series_name
    suffix = "relm3" if frame_offset < 0 else "relp3"
    return output_root / rel.parent / f"{stem}__{suffix}_idx{frame_index:04d}.png"


def extract_rows(
    base_rows: Sequence[BaseImageRow],
    base_image_root: Path,
    output_root: Path,
    skip_existing: bool,
) -> tuple[list[ExtractedRow], pd.DataFrame, dict[str, int]]:
    extracted_rows: list[ExtractedRow] = []
    split_class_stats: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {
        "base_count": 0,
        "usable_count": 0,
        "dropped_count": 0,
        "output_image_count": 0,
    })
    global_stats = {
        "base_count": 0,
        "usable_dicom_count": 0,
        "dropped_dicom_count": 0,
        "output_image_count": 0,
        "missing_dcm_count": 0,
        "missing_label_count": 0,
        "range_invalid_count": 0,
        "save_failed_count": 0,
        "skip_existing_count": 0,
    }

    base_root_resolved = base_image_root.resolve()
    for row in tqdm(base_rows, desc="Building same-dicom dataset"):
        key = (row.split, row.class_name)
        split_class_stats[key]["base_count"] += 1
        global_stats["base_count"] += 1

        if not row.dcm_path.exists():
            split_class_stats[key]["dropped_count"] += 1
            global_stats["dropped_dicom_count"] += 1
            global_stats["missing_dcm_count"] += 1
            continue
        if row.contrast_frame_index < 0:
            split_class_stats[key]["dropped_count"] += 1
            global_stats["dropped_dicom_count"] += 1
            global_stats["missing_label_count"] += 1
            continue

        ds_header = pydicom.dcmread(str(row.dcm_path), stop_before_pixels=True, force=True)
        n_frames = int(getattr(ds_header, "NumberOfFrames", 1))
        frame_indices = [row.contrast_frame_index + offset for offset in FRAME_OFFSETS]
        if not all(0 <= idx < n_frames for idx in frame_indices):
            split_class_stats[key]["dropped_count"] += 1
            global_stats["dropped_dicom_count"] += 1
            global_stats["range_invalid_count"] += 1
            continue

        rel = row.base_png_path.resolve().relative_to(base_root_resolved)
        out_paths = []
        all_outputs_exist = True
        for frame_offset, frame_index in zip(FRAME_OFFSETS, frame_indices):
            suffix = "relm3" if frame_offset < 0 else "relp3"
            out_path = output_root / rel.parent / f"{row.series_name}__{suffix}_idx{frame_index:04d}.png"
            out_paths.append((frame_offset, frame_index, out_path))
            all_outputs_exist = all_outputs_exist and out_path.exists()
        if skip_existing and all_outputs_exist:
            global_stats["skip_existing_count"] += len(out_paths)
        else:
            ds = pydicom.dcmread(str(row.dcm_path), force=True)
            pixel = ds.pixel_array.astype(np.float32)
            try:
                for frame_offset, frame_index, out_path in out_paths:
                    if skip_existing and out_path.exists():
                        global_stats["skip_existing_count"] += 1
                        continue
                    img = extract_frame_uint8(pixel, ds, frame_index)
                    save_png(img, out_path)
            except Exception as exc:
                log(f"[SKIP] failed to extract {row.dcm_rel_path} | {exc}")
                split_class_stats[key]["dropped_count"] += 1
                global_stats["dropped_dicom_count"] += 1
                global_stats["save_failed_count"] += 1
                continue

        split_class_stats[key]["usable_count"] += 1
        split_class_stats[key]["output_image_count"] += 2
        global_stats["usable_dicom_count"] += 1
        global_stats["output_image_count"] += 2
        for frame_offset, frame_index, out_path in out_paths:
            extracted_rows.append(
                ExtractedRow(
                    split=row.split,
                    class_name=row.class_name,
                    patient_id=row.patient_id,
                    study_id=row.study_id,
                    dicom_rel_path=row.dcm_rel_path,
                    dicom_id=row.dicom_id,
                    contrast_frame_index=row.contrast_frame_index,
                    frame_index=frame_index,
                    frame_offset=frame_offset,
                    img_path=out_path.as_posix(),
                    source_contrast_png=row.base_png_path.as_posix(),
                )
            )

    summary_rows: list[dict[str, object]] = []
    for split in SPLITS:
        for class_name in ("stent", "no_stent"):
            stats = split_class_stats[(split, class_name)]
            summary_rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "base_count": int(stats["base_count"]),
                    "usable_count": int(stats["usable_count"]),
                    "dropped_count": int(stats["dropped_count"]),
                    "output_image_count": int(stats["output_image_count"]),
                }
            )
    return extracted_rows, pd.DataFrame(summary_rows), global_stats


def validate_extracted_manifest(manifest: pd.DataFrame, output_root: Path) -> None:
    if manifest.empty:
        raise BuildError("Derived same-dicom manifest is empty.")
    counts = manifest["dicom_id"].value_counts().sort_index()
    if int(counts.min()) != 2 or int(counts.max()) != 2:
        bad = counts[counts != 2].head(10).to_dict()
        raise BuildError(f"Each dicom_id must map to exactly 2 images. Violations: {bad}")
    frame_sets = manifest.groupby("dicom_id")["frame_offset"].apply(lambda s: tuple(sorted(int(v) for v in s.tolist())))
    bad_offsets = frame_sets[frame_sets != FRAME_OFFSETS]
    if not bad_offsets.empty:
        raise BuildError(f"Unexpected frame offsets detected: {bad_offsets.head(10).to_dict()}")
    total_pngs = sum(1 for split in SPLITS for _ in (output_root / split).rglob("*.png"))
    if total_pngs != len(manifest):
        raise BuildError(f"PNG count mismatch: manifest_rows={len(manifest)} actual_pngs={total_pngs}")
    missing = [path for path in manifest["img_path"].tolist() if not Path(path).exists()]
    if missing:
        raise BuildError(f"Missing generated PNGs: {missing[:10]}")


def write_text_summary(
    output_path: Path,
    args: argparse.Namespace,
    split_summary_df: pd.DataFrame,
    global_stats: dict[str, int],
    manifest_rows: int,
    log_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("========== Stent Contrast Same-DICOM unique-view build summary ==========\n")
        handle.write(f"base_image_root: {Path(args.base_image_root).resolve()}\n")
        handle.write(f"dcm_root: {Path(args.dcm_root).resolve()}\n")
        handle.write(f"label_json: {Path(args.label_json).resolve()}\n")
        handle.write(f"output_root: {Path(args.output_root).resolve()}\n")
        handle.write(f"manifest_name: {args.manifest_name}\n")
        handle.write(f"summary_name: {args.summary_name}\n")
        handle.write(f"frame_index_base: {args.frame_index_base}\n")
        handle.write(f"max_images_per_split: {args.max_images_per_split}\n")
        handle.write(f"seed: {args.seed}\n")
        handle.write(f"timestamp: {datetime.now().isoformat()}\n")
        handle.write(f"log_path: {log_path.resolve()}\n")
        handle.write("\n")
        handle.write(f"manifest_rows: {manifest_rows}\n")
        for key in sorted(global_stats):
            handle.write(f"{key}: {global_stats[key]}\n")
        handle.write("\n")
        handle.write(split_summary_df.to_csv(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-image-root", type=Path, default=DEFAULT_BASE_IMAGE_ROOT)
    parser.add_argument("--dcm-root", type=Path, default=DEFAULT_DCM_ROOT)
    parser.add_argument("--label-json", type=Path, default=DEFAULT_LABEL_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--frame-index-base", type=int, default=DEFAULT_FRAME_INDEX_BASE, choices=[0, 1])
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_root)
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        args.output_root,
        args.log_file,
        default_prefix="build_stent_contrast_same_dicom_unique_view",
    )
    try:
        if not args.base_image_root.exists():
            raise FileNotFoundError(f"base image root not found: {args.base_image_root}")
        if not args.dcm_root.exists():
            raise FileNotFoundError(f"dcm root not found: {args.dcm_root}")
        if not args.label_json.exists():
            raise FileNotFoundError(f"label json not found: {args.label_json}")

        np.random.seed(int(args.seed))
        log(f"Arguments: {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2, sort_keys=True)}")
        label_map = build_label_index_map(args.label_json, args.frame_index_base)
        log(f"Loaded label map entries: {len(label_map)}")

        base_rows, base_summary_df = build_base_rows(
            base_image_root=args.base_image_root,
            dcm_root=args.dcm_root,
            label_map=label_map,
            max_images_per_split=args.max_images_per_split,
            seed=int(args.seed),
        )
        log(f"Base contrast PNG count considered: {len(base_rows)}")

        extracted_rows, split_summary_df, global_stats = extract_rows(
            base_rows=base_rows,
            base_image_root=args.base_image_root,
            output_root=args.output_root,
            skip_existing=bool(args.skip_existing),
        )

        manifest_df = pd.DataFrame([row.__dict__ for row in extracted_rows]).sort_values(
            ["split", "class_name", "dicom_id", "frame_offset"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        validate_extracted_manifest(manifest_df, args.output_root)

        manifest_path = args.output_root / args.manifest_name
        summary_csv_path = args.output_root / args.summary_name
        base_audit_path = args.output_root / "summary_same_dicom_source_audit.csv"
        summary_txt_path = args.output_root / "summary_same_dicom_build.txt"

        manifest_df.to_csv(manifest_path, index=False)
        split_summary_df.to_csv(summary_csv_path, index=False)
        base_summary_df.to_csv(base_audit_path, index=False)
        write_text_summary(
            output_path=summary_txt_path,
            args=args,
            split_summary_df=split_summary_df,
            global_stats=global_stats,
            manifest_rows=len(manifest_df),
            log_path=log_path,
        )

        log(f"Saved manifest: {manifest_path}")
        log(f"Saved split summary: {summary_csv_path}")
        log(f"Saved source audit: {base_audit_path}")
        log(f"Saved text summary: {summary_txt_path}")
        log("Same-DICOM dataset build completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
