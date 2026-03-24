#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIQUE_VIEW_ROOT = REPO_ROOT / "input/Stent-Contrast-unique-view"
DEFAULT_SAME_DICOM_ROOT = REPO_ROOT / "input/Stent-Contrast-same-dicom-unique-view"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/fm_improve_exp1-input_policy/input_v2_percentile_canonicalization/downstream_only/input_v2/cache"
)
PNG_EXT = ".png"
POLICY_NAME = "input_v2_percentile_canonicalization"


@dataclass(frozen=True)
class CacheOutputs:
    unique_view_root: Path
    same_dicom_root: Path
    same_dicom_manifest: Path
    metadata_json: Path
    summary_csv: Path
    integrity_csv: Path


def canonicalize_percentile_uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape={image.shape}")
    arr = image.astype(np.float32, copy=False)
    low = float(np.percentile(arr, 0.5))
    high = float(np.percentile(arr, 99.5))
    if not np.isfinite(low) or not np.isfinite(high) or (high - low) < 1.0:
        return image.copy()
    clipped = np.clip(arr, low, high)
    scaled = (clipped - low) / (high - low)
    out = np.clip(np.rint(scaled * 255.0), 0, 255).astype(np.uint8)
    return out


def hash_records(records: Iterable[dict[str, object]]) -> str:
    payload = json.dumps(list(records), sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def iter_png_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob(f"*{PNG_EXT}") if path.is_file())


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_source_path(dataset_root: Path, raw_value: object) -> Path:
    value = str(raw_value)
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (dataset_root / candidate).resolve()


def infer_relative_image_path(source_path: Path, dataset_root: Path) -> Path:
    resolved_root = dataset_root.resolve()
    try:
        return source_path.relative_to(resolved_root)
    except ValueError:
        parts = source_path.parts
        for split in ("train", "valid", "test"):
            if split in parts:
                idx = parts.index(split)
                return Path(*parts[idx:])
        raise ValueError(f"Unable to infer relative dataset path for: {source_path}")


def rewrite_manifest_paths(
    manifest_df: pd.DataFrame,
    source_root: Path,
    cached_root: Path,
) -> pd.DataFrame:
    if "img_path" not in manifest_df.columns:
        raise ValueError("same-DICOM manifest does not contain an 'img_path' column.")
    rewritten = manifest_df.copy()
    new_paths: list[str] = []
    for value in rewritten["img_path"].tolist():
        source_path = resolve_source_path(source_root, value)
        rel_path = infer_relative_image_path(source_path, source_root)
        new_paths.append(str((cached_root / rel_path).resolve()))
    rewritten["img_path"] = new_paths
    return rewritten


def validate_same_dicom_integrity(manifest_df: pd.DataFrame) -> pd.DataFrame:
    dicom_col = next((c for c in ("dicom_id", "dicom_uid", "dicom_key") if c in manifest_df.columns), None)
    offset_col = next((c for c in ("frame_offset", "relative_offset", "offset", "rel_offset") if c in manifest_df.columns), None)
    if "split" not in manifest_df.columns or dicom_col is None:
        summary = pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "required_columns_missing",
                    "columns": ",".join(manifest_df.columns.astype(str).tolist()),
                }
            ]
        )
        return summary

    records: list[dict[str, object]] = []
    for (split, dicom_id), group in manifest_df.groupby(["split", dicom_col], sort=True):
        row: dict[str, object] = {
            "split": split,
            "dicom_id": dicom_id,
            "num_rows": int(len(group)),
            "is_pair": int(len(group) == 2),
        }
        if offset_col is not None:
            offsets = sorted(group[offset_col].tolist())
            row["offsets"] = json.dumps(offsets)
            row["has_expected_offsets"] = int(offsets == [-3, 3])
        else:
            row["offsets"] = ""
            row["has_expected_offsets"] = -1
        records.append(row)
    return pd.DataFrame.from_records(records)


def build_cache_dataset(
    source_root: Path,
    output_root: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    for src_path in iter_png_paths(source_root):
        rel_path = src_path.relative_to(source_root)
        dst_path = output_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {src_path}")
        mapped = canonicalize_percentile_uint8(image)
        if not cv2.imwrite(str(dst_path), mapped):
            raise ValueError(f"Failed to write cached image: {dst_path}")
        split = rel_path.parts[0] if rel_path.parts else "unknown"
        split_counts[split] = split_counts.get(split, 0) + 1
        rows.append(
            {
                "split": split,
                "rel_path": str(rel_path),
                "source_path": str(src_path.resolve()),
                "cached_path": str(dst_path.resolve()),
            }
        )
    return pd.DataFrame.from_records(rows), split_counts


def build_input_v2_cache(
    unique_view_root: Path,
    same_dicom_root: Path,
    same_dicom_manifest: Path,
    output_root: Path,
    overwrite: bool,
) -> CacheOutputs:
    unique_view_root = unique_view_root.resolve()
    same_dicom_root = same_dicom_root.resolve()
    same_dicom_manifest = same_dicom_manifest.resolve()
    output_root = output_root.resolve()

    unique_cache_root = output_root / "unique_view"
    same_cache_root = output_root / "same_dicom"
    ensure_clean_dir(unique_cache_root, overwrite=overwrite)
    ensure_clean_dir(same_cache_root, overwrite=overwrite)

    unique_manifest_df, unique_counts = build_cache_dataset(unique_view_root, unique_cache_root)
    same_manifest_df, same_counts = build_cache_dataset(same_dicom_root, same_cache_root)

    unique_manifest_path = output_root / "cache_manifest_unique_view.csv"
    same_cache_manifest_path = output_root / "cache_manifest_same_dicom.csv"
    unique_manifest_df.to_csv(unique_manifest_path, index=False)
    same_manifest_df.to_csv(same_cache_manifest_path, index=False)

    original_same_dicom_manifest = pd.read_csv(same_dicom_manifest)
    rewritten_same_dicom_manifest = rewrite_manifest_paths(
        original_same_dicom_manifest,
        source_root=same_dicom_root,
        cached_root=same_cache_root,
    )
    rewritten_manifest_path = same_cache_root / "manifest_same_dicom_master.csv"
    rewritten_same_dicom_manifest.to_csv(rewritten_manifest_path, index=False)

    integrity_df = validate_same_dicom_integrity(rewritten_same_dicom_manifest)
    integrity_path = output_root / "summary_same_dicom_integrity.csv"
    integrity_df.to_csv(integrity_path, index=False)

    summary_rows = [
        {"dataset": "unique_view", "split": split, "num_images": count}
        for split, count in sorted(unique_counts.items())
    ]
    summary_rows.extend(
        {"dataset": "same_dicom", "split": split, "num_images": count}
        for split, count in sorted(same_counts.items())
    )
    summary_df = pd.DataFrame.from_records(summary_rows)
    summary_path = output_root / "summary_input_v2_cache.csv"
    summary_df.to_csv(summary_path, index=False)

    manifest_hash = hash_records(
        {
            "rel_path": row["rel_path"],
            "cached_path": row["cached_path"],
        }
        for row in unique_manifest_df.to_dict("records")
    )
    metadata = {
        "policy_name": POLICY_NAME,
        "unique_view_root": str(unique_view_root),
        "same_dicom_root": str(same_dicom_root),
        "same_dicom_manifest": str(same_dicom_manifest),
        "output_root": str(output_root),
        "unique_view_cache_root": str(unique_cache_root),
        "same_dicom_cache_root": str(same_cache_root),
        "rewritten_same_dicom_manifest": str(rewritten_manifest_path),
        "summary_csv": str(summary_path),
        "integrity_csv": str(integrity_path),
        "percentile_low": 0.5,
        "percentile_high": 99.5,
        "manifest_hash": manifest_hash,
        "unique_view_counts": unique_counts,
        "same_dicom_counts": same_counts,
    }
    metadata_path = output_root / "cache_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return CacheOutputs(
        unique_view_root=unique_cache_root,
        same_dicom_root=same_cache_root,
        same_dicom_manifest=rewritten_manifest_path,
        metadata_json=metadata_path,
        summary_csv=summary_path,
        integrity_csv=integrity_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique-view-root", default=str(DEFAULT_UNIQUE_VIEW_ROOT))
    parser.add_argument("--same-dicom-root", default=str(DEFAULT_SAME_DICOM_ROOT))
    parser.add_argument("--same-dicom-manifest", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    same_dicom_manifest = (
        Path(args.same_dicom_manifest)
        if args.same_dicom_manifest
        else Path(args.same_dicom_root) / "manifest_same_dicom_master.csv"
    )
    outputs = build_input_v2_cache(
        unique_view_root=Path(args.unique_view_root),
        same_dicom_root=Path(args.same_dicom_root),
        same_dicom_manifest=same_dicom_manifest,
        output_root=Path(args.output_root),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(
        {
            "unique_view_root": str(outputs.unique_view_root),
            "same_dicom_root": str(outputs.same_dicom_root),
            "same_dicom_manifest": str(outputs.same_dicom_manifest),
            "metadata_json": str(outputs.metadata_json),
            "summary_csv": str(outputs.summary_csv),
            "integrity_csv": str(outputs.integrity_csv),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
