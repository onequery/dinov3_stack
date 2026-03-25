#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pydicom

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from border_suppression_common import (  # noqa: E402
    DEFAULT_BLUR_SIGMA,
    DEFAULT_CENTER_X,
    DEFAULT_CENTER_Y,
    DEFAULT_FEATHER_WIDTH,
    DEFAULT_RADIUS,
    TARGET_MODEL_NAME_NORMALIZED,
    apply_border_suppression,
    make_three_panel,
    normalize_manufacturer_model_name,
    overlay_mask_boundary,
    safe_name_from_rel_path,
)
from input_standardization_common import read_grayscale, write_grayscale  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIQUE_VIEW_ROOT = REPO_ROOT / "input/Stent-Contrast-unique-view"
DEFAULT_SAME_DICOM_ROOT = REPO_ROOT / "input/Stent-Contrast-same-dicom-unique-view"
DEFAULT_SAME_DICOM_MANIFEST = DEFAULT_SAME_DICOM_ROOT / "manifest_same_dicom_master.csv"
DEFAULT_DCM_ROOT = REPO_ROOT / "input/stent_split_dcm_unique_view"
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/fm_improve_exp1-input_policy/input_v4_border_suppression/downstream_only/input_v4/cache"
)
PNG_EXT = ".png"
POLICY_NAME = "input_v4_border_suppression"
PREVIEW_LIMIT_PER_GROUP = 3
TARGET_IMAGE_SHAPE = (512, 512)


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [CACHE] {message}", flush=True)


@dataclass(frozen=True)
class CacheOutputs:
    unique_view_root: Path
    same_dicom_root: Path
    same_dicom_manifest: Path
    metadata_json: Path
    summary_csv: Path
    integrity_csv: Path
    application_csv: Path


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


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
            split_relative = Path(*parts[parts.index(split) :])
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


def build_same_dicom_dicom_lookup(manifest_df: pd.DataFrame, source_root: Path, dcm_root: Path) -> dict[str, Path]:
    required = {"img_path", "dicom_rel_path"}
    missing = required.difference(manifest_df.columns)
    if missing:
        raise ValueError(f"same-DICOM manifest missing required columns: {sorted(missing)}")
    lookup: dict[str, Path] = {}
    for row in manifest_df.to_dict("records"):
        source_path = resolve_source_path(source_root, row["img_path"])
        rel_path = infer_relative_image_path(source_path, source_root)
        lookup[str(rel_path)] = (dcm_root / Path(str(row["dicom_rel_path"]))).resolve()
    return lookup


def read_model_name_normalized(dicom_path: Path, cache: dict[Path, str]) -> str | None:
    cached = cache.get(dicom_path)
    if cached is not None:
        return cached
    try:
        dataset = pydicom.dcmread(
            str(dicom_path),
            stop_before_pixels=True,
            specific_tags=["ManufacturerModelName"],
        )
    except Exception:
        return None
    normalized = normalize_manufacturer_model_name(getattr(dataset, "ManufacturerModelName", None))
    cache[dicom_path] = normalized
    return normalized


def resolve_unique_view_dicom_path(rel_path: Path, dcm_root: Path) -> Path:
    return (dcm_root / rel_path).with_suffix(".dcm").resolve()


def maybe_write_preview_panel(
    *,
    preview_root: Path,
    category: str,
    dataset_name: str,
    rel_path: Path,
    panel: object,
    records: list[dict[str, object]],
    counter: Counter,
) -> None:
    if counter[category] >= PREVIEW_LIMIT_PER_GROUP:
        return
    category_dir = preview_root / category
    category_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_name_from_rel_path(rel_path.parts)
    out_path = category_dir / f"{dataset_name}__{safe_name}__{category}.png"
    import cv2
    cv2.imwrite(str(out_path), panel)
    records.append(
        {
            "category": category,
            "dataset": dataset_name,
            "rel_path": str(rel_path),
            "panel_path": str(out_path.resolve()),
        }
    )
    counter[category] += 1


def build_cache_dataset(
    source_root: Path,
    output_root: Path,
    *,
    dataset_name: str,
    dcm_root: Path,
    same_dicom_dicom_lookup: dict[str, Path] | None,
    preview_root: Path,
    preview_records: list[dict[str, object]],
    preview_counter: Counter,
    center_x: int,
    center_y: int,
    radius: int,
    blur_sigma: float,
    feather_width: int,
    progress_every: int = 250,
) -> tuple[pd.DataFrame, dict[str, int], Counter, int]:
    rows: list[dict[str, object]] = []
    split_counts: dict[str, int] = {}
    status_counts: Counter = Counter()
    target_device_count = 0
    model_cache: dict[Path, str] = {}
    png_paths = iter_png_paths(source_root)
    total_images = len(png_paths)
    started_at = time.time()
    log(f"START dataset={dataset_name} | total_images={total_images} | source_root={source_root} | output_root={output_root}")
    for index, src_path in enumerate(png_paths, start=1):
        rel_path = src_path.relative_to(source_root)
        dst_path = output_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        split = rel_path.parts[0] if rel_path.parts else "unknown"
        split_counts[split] = split_counts.get(split, 0) + 1

        if same_dicom_dicom_lookup is None:
            dicom_path = resolve_unique_view_dicom_path(rel_path, dcm_root)
        else:
            dicom_path = same_dicom_dicom_lookup.get(str(rel_path), Path())

        model_name_normalized = ""
        status = "copied_model_mismatch"
        modified = False
        source_image = None

        if not dicom_path or not dicom_path.exists():
            shutil.copy2(src_path, dst_path)
            status = "copied_missing_dicom"
        else:
            model_name_normalized = read_model_name_normalized(dicom_path, model_cache) or ""
            if model_name_normalized == TARGET_MODEL_NAME_NORMALIZED:
                target_device_count += 1
                source_image = read_grayscale(src_path)
                if tuple(source_image.shape) != TARGET_IMAGE_SHAPE:
                    shutil.copy2(src_path, dst_path)
                    status = "copied_shape_mismatch"
                else:
                    suppressed, mask, _background, _alpha = apply_border_suppression(
                        source_image,
                        center_x=center_x,
                        center_y=center_y,
                        radius=radius,
                        blur_sigma=blur_sigma,
                        feather_width=feather_width,
                    )
                    write_grayscale(dst_path, suppressed)
                    status = "modified"
                    modified = True
                    overlay = overlay_mask_boundary(source_image, mask)
                    panel = make_three_panel(source_image, suppressed, overlay)
                    maybe_write_preview_panel(
                        preview_root=preview_root,
                        category="modified",
                        dataset_name=dataset_name,
                        rel_path=rel_path,
                        panel=panel,
                        records=preview_records,
                        counter=preview_counter,
                    )
            else:
                shutil.copy2(src_path, dst_path)
                status = "copied_model_mismatch"

        if status == "copied_model_mismatch":
            if source_image is None:
                source_image = read_grayscale(src_path)
            control_panel = make_three_panel(
                source_image,
                source_image,
                source_image,
                labels=("Original", "Untouched", f"Control: {model_name_normalized or 'UNKNOWN'}"),
            )
            maybe_write_preview_panel(
                preview_root=preview_root,
                category="control",
                dataset_name=dataset_name,
                rel_path=rel_path,
                panel=control_panel,
                records=preview_records,
                counter=preview_counter,
            )

        status_counts[status] += 1
        rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "rel_path": str(rel_path),
                "source_path": str(src_path.resolve()),
                "cached_path": str(dst_path.resolve()),
                "dicom_path": str(dicom_path) if dicom_path else "",
                "model_name_normalized": model_name_normalized,
                "status": status,
                "modified": int(modified),
            }
        )
        if index == 1 or index == total_images or index % progress_every == 0:
            elapsed = max(1e-9, time.time() - started_at)
            rate = index / elapsed
            remaining = (total_images - index) / rate if rate > 0 else None
            log(
                f"PROGRESS dataset={dataset_name} | done={index}/{total_images} | elapsed={format_duration(elapsed)} | remaining_est={format_duration(remaining)}"
            )
    log(
        f"DONE dataset={dataset_name} | total_images={total_images} | target_device_images={target_device_count} | modified={status_counts['modified']} | copied_model_mismatch={status_counts['copied_model_mismatch']} | copied_shape_mismatch={status_counts['copied_shape_mismatch']} | copied_missing_dicom={status_counts['copied_missing_dicom']} | elapsed={format_duration(time.time() - started_at)}"
    )
    return pd.DataFrame.from_records(rows), split_counts, status_counts, target_device_count


def build_input_v4_cache(
    unique_view_root: Path,
    same_dicom_root: Path,
    same_dicom_manifest: Path,
    dcm_root: Path,
    output_root: Path,
    overwrite: bool,
    *,
    center_x: int = DEFAULT_CENTER_X,
    center_y: int = DEFAULT_CENTER_Y,
    radius: int = DEFAULT_RADIUS,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    feather_width: int = DEFAULT_FEATHER_WIDTH,
) -> CacheOutputs:
    unique_view_root = unique_view_root.resolve()
    same_dicom_root = same_dicom_root.resolve()
    same_dicom_manifest = same_dicom_manifest.resolve()
    dcm_root = dcm_root.resolve()
    output_root = output_root.resolve()

    unique_cache_root = output_root / "unique_view"
    same_cache_root = output_root / "same_dicom"
    preview_root = output_root / "qa_previews"
    ensure_clean_dir(output_root, overwrite=overwrite)
    ensure_clean_dir(unique_cache_root, overwrite=False)
    ensure_clean_dir(same_cache_root, overwrite=False)
    ensure_clean_dir(preview_root, overwrite=False)

    original_same_dicom_manifest = pd.read_csv(same_dicom_manifest)
    same_dicom_lookup = build_same_dicom_dicom_lookup(original_same_dicom_manifest, same_dicom_root, dcm_root)

    preview_records: list[dict[str, object]] = []
    preview_counter: Counter = Counter()

    unique_manifest_df, unique_counts, unique_status_counts, unique_target_count = build_cache_dataset(
        unique_view_root,
        unique_cache_root,
        dataset_name="unique_view",
        dcm_root=dcm_root,
        same_dicom_dicom_lookup=None,
        preview_root=preview_root,
        preview_records=preview_records,
        preview_counter=preview_counter,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        blur_sigma=blur_sigma,
        feather_width=feather_width,
    )
    same_manifest_df, same_counts, same_status_counts, same_target_count = build_cache_dataset(
        same_dicom_root,
        same_cache_root,
        dataset_name="same_dicom",
        dcm_root=dcm_root,
        same_dicom_dicom_lookup=same_dicom_lookup,
        preview_root=preview_root,
        preview_records=preview_records,
        preview_counter=preview_counter,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        blur_sigma=blur_sigma,
        feather_width=feather_width,
    )

    unique_manifest_path = output_root / "cache_manifest_unique_view.csv"
    same_cache_manifest_path = output_root / "cache_manifest_same_dicom.csv"
    unique_manifest_df.to_csv(unique_manifest_path, index=False)
    same_manifest_df.to_csv(same_cache_manifest_path, index=False)

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
    summary_path = output_root / "summary_input_v4_cache.csv"
    summary_df.to_csv(summary_path, index=False)

    application_df = pd.concat([unique_manifest_df, same_manifest_df], ignore_index=True)
    application_summary = (
        application_df.groupby(["dataset", "split", "status"], sort=True)
        .size()
        .reset_index(name="count")
    )
    overall_application = (
        application_df.groupby(["dataset", "status"], sort=True)
        .size()
        .reset_index(name="count")
    )
    overall_application.insert(1, "split", "all")
    application_summary = pd.concat([application_summary, overall_application], ignore_index=True)
    application_path = output_root / "summary_input_v4_application.csv"
    application_summary.to_csv(application_path, index=False)

    preview_summary_path = preview_root / "summary_qa_previews.csv"
    pd.DataFrame.from_records(preview_records).to_csv(preview_summary_path, index=False)

    manifest_hash = hash_records(
        {
            "dataset": row["dataset"],
            "rel_path": row["rel_path"],
            "cached_path": row["cached_path"],
            "status": row["status"],
        }
        for row in application_df.to_dict("records")
    )
    metadata = {
        "policy_name": POLICY_NAME,
        "unique_view_root": str(unique_view_root),
        "same_dicom_root": str(same_dicom_root),
        "same_dicom_manifest": str(same_dicom_manifest),
        "dcm_root": str(dcm_root),
        "output_root": str(output_root),
        "unique_view_cache_root": str(unique_cache_root),
        "same_dicom_cache_root": str(same_cache_root),
        "rewritten_same_dicom_manifest": str(rewritten_manifest_path),
        "summary_csv": str(summary_path),
        "application_csv": str(application_path),
        "integrity_csv": str(integrity_path),
        "preview_summary_csv": str(preview_summary_path),
        "preview_root": str(preview_root),
        "manifest_hash": manifest_hash,
        "target_model_name_normalized": TARGET_MODEL_NAME_NORMALIZED,
        "target_image_shape": list(TARGET_IMAGE_SHAPE),
        "suppression_parameters": {
            "center_x": center_x,
            "center_y": center_y,
            "radius": radius,
            "blur_sigma": blur_sigma,
            "feather_width": feather_width,
        },
        "unique_view_counts": unique_counts,
        "same_dicom_counts": same_counts,
        "unique_view_status_counts": dict(unique_status_counts),
        "same_dicom_status_counts": dict(same_status_counts),
        "unique_view_target_device_images": int(unique_target_count),
        "same_dicom_target_device_images": int(same_target_count),
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
        application_csv=application_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unique-view-root", default=str(DEFAULT_UNIQUE_VIEW_ROOT))
    parser.add_argument("--same-dicom-root", default=str(DEFAULT_SAME_DICOM_ROOT))
    parser.add_argument("--same-dicom-manifest", default=str(DEFAULT_SAME_DICOM_MANIFEST))
    parser.add_argument("--dcm-root", default=str(DEFAULT_DCM_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_input_v4_cache(
        unique_view_root=Path(args.unique_view_root),
        same_dicom_root=Path(args.same_dicom_root),
        same_dicom_manifest=Path(args.same_dicom_manifest),
        dcm_root=Path(args.dcm_root),
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
                "application_csv": str(outputs.application_csv),
                "integrity_csv": str(outputs.integrity_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
