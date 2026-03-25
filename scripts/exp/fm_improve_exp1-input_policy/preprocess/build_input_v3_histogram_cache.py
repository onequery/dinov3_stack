#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from input_standardization_common import (  # noqa: E402
    compute_reference_cdf,
    histogram_match_uint8,
    read_grayscale,
    write_grayscale,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIQUE_VIEW_ROOT = REPO_ROOT / "input/Stent-Contrast-unique-view"
DEFAULT_SAME_DICOM_ROOT = REPO_ROOT / "input/Stent-Contrast-same-dicom-unique-view"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/fm_improve_exp1-input_policy/input_v3_histogram_standardization/downstream_only/input_v3/cache"
)
PNG_EXT = ".png"
POLICY_NAME = "input_v3_histogram_standardization"


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [CACHE] {message}", flush=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass(frozen=True)
class CacheOutputs:
    unique_view_root: Path
    same_dicom_root: Path
    same_dicom_manifest: Path
    metadata_json: Path
    summary_csv: Path
    integrity_csv: Path


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

    repo_option = (REPO_ROOT / candidate).resolve()
    if repo_option.exists():
        return repo_option

    dataset_option = (dataset_root / candidate).resolve()
    if dataset_option.exists():
        return dataset_option

    parts = candidate.parts
    for split in ("train", "valid", "test"):
        if split in parts:
            split_relative = Path(*parts[parts.index(split):])
            split_option = (dataset_root / split_relative).resolve()
            if split_option.exists():
                return split_option

    return dataset_option


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
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "required_columns_missing",
                    "columns": ",".join(manifest_df.columns.astype(str).tolist()),
                }
            ]
        )

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


def build_reference_cdf_artifact(unique_view_root: Path, output_root: Path) -> tuple[Path, Path, int]:
    import numpy as np

    train_root = unique_view_root / "train"
    train_paths = iter_png_paths(train_root)
    if not train_paths:
        raise RuntimeError(f"No train PNGs discovered under {train_root}")

    started_at = time.time()
    log(f"START reference_cdf | source_root={train_root} | num_images={len(train_paths)}")
    ref_cdf = compute_reference_cdf(train_paths)
    cdf_path = output_root / "reference_cdf_unique_view_train.npy"
    meta_path = output_root / "reference_cdf_unique_view_train.json"
    cdf_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "policy_name": POLICY_NAME,
        "source_root": str(train_root.resolve()),
        "num_train_images": len(train_paths),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with cdf_path.open("wb") as f:
        np.save(f, ref_cdf)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(
        f"DONE reference_cdf | num_images={len(train_paths)} "
        f"| elapsed={format_duration(time.time() - started_at)}"
    )
    return cdf_path, meta_path, len(train_paths)


def load_reference_cdf(path: Path):
    import numpy as np

    return np.load(path)


def build_cache_dataset(
    source_root: Path,
    output_root: Path,
    *,
    dataset_name: str,
    ref_cdf,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    png_paths = iter_png_paths(source_root)
    total_images = len(png_paths)
    started_at = time.time()
    log(f"START dataset={dataset_name} | total_images={total_images} | source_root={source_root} | output_root={output_root}")
    for index, src_path in enumerate(png_paths, start=1):
        rel_path = src_path.relative_to(source_root)
        dst_path = output_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        image = read_grayscale(src_path)
        mapped = histogram_match_uint8(image, ref_cdf)
        write_grayscale(dst_path, mapped)
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
        if index == 1 or index == total_images or index % progress_every == 0:
            elapsed = max(1e-9, time.time() - started_at)
            rate = index / elapsed
            remaining = (total_images - index) / rate if rate > 0 else None
            log(
                f"PROGRESS dataset={dataset_name} | done={index}/{total_images} "
                f"| elapsed={format_duration(elapsed)} | remaining_est={format_duration(remaining)}"
            )
    log(
        f"DONE dataset={dataset_name} | total_images={total_images} "
        f"| elapsed={format_duration(time.time() - started_at)}"
    )
    return pd.DataFrame.from_records(rows), split_counts


def build_input_v3_cache(
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
    ensure_clean_dir(output_root, overwrite=overwrite)
    ensure_clean_dir(unique_cache_root, overwrite=False)
    ensure_clean_dir(same_cache_root, overwrite=False)

    ref_cdf_path, ref_meta_path, num_train_images = build_reference_cdf_artifact(
        unique_view_root=unique_view_root,
        output_root=output_root,
    )
    ref_cdf = load_reference_cdf(ref_cdf_path)

    unique_manifest_df, unique_counts = build_cache_dataset(
        unique_view_root,
        unique_cache_root,
        dataset_name="unique_view",
        ref_cdf=ref_cdf,
    )
    same_manifest_df, same_counts = build_cache_dataset(
        same_dicom_root,
        same_cache_root,
        dataset_name="same_dicom",
        ref_cdf=ref_cdf,
    )

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
    summary_path = output_root / "summary_input_v3_cache.csv"
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
        "reference_cdf_npy": str(ref_cdf_path),
        "reference_cdf_json": str(ref_meta_path),
        "reference_train_image_count": int(num_train_images),
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
    outputs = build_input_v3_cache(
        unique_view_root=Path(args.unique_view_root),
        same_dicom_root=Path(args.same_dicom_root),
        same_dicom_manifest=same_dicom_manifest,
        output_root=Path(args.output_root),
        overwrite=bool(args.overwrite),
    )
    print(
        json.dumps(
            {
                "unique_view_root": str(outputs.unique_view_root),
                "same_dicom_root": str(outputs.same_dicom_root),
                "same_dicom_manifest": str(outputs.same_dicom_manifest),
                "metadata_json": str(outputs.metadata_json),
                "summary_csv": str(outputs.summary_csv),
                "integrity_csv": str(outputs.integrity_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
