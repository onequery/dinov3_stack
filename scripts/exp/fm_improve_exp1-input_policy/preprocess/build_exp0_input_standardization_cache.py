#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, TextIO

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from input_standardization_common import (  # noqa: E402
    ALLOWED_SPLITS,
    ALLOWED_VARIANTS,
    canonicalize_norm_v1,
    canonicalize_norm_v2,
    compute_reference_cdf,
    ensure_dir,
    histogram_match_uint8,
    read_grayscale,
    sha256_records,
    write_grayscale,
)


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


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def setup_logging(output_root: Path, log_file_arg: str | None) -> tuple[TextIO, TextIO, TextIO]:
    ensure_dir(output_root)
    if log_file_arg:
        log_path = Path(log_file_arg)
        if not log_path.is_absolute():
            log_path = output_root / log_path
    else:
        log_path = output_root / f"build_exp0_input_standardization_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    ensure_dir(log_path.parent)
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, fh)
    sys.stderr = TeeStream(original_stderr, fh)
    log(f"Console output is mirrored to log file: {log_path}")
    return fh, original_stdout, original_stderr


def restore_logging(fh: TextIO, original_stdout: TextIO, original_stderr: TextIO) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    fh.close()


def sample_unique_manifest(df: pd.DataFrame, max_images_per_split: int | None, seed: int) -> pd.DataFrame:
    if not max_images_per_split or max_images_per_split <= 0:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    sampled = []
    for split in ALLOWED_SPLITS:
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue
        if len(split_df) <= max_images_per_split:
            sampled.append(split_df)
            continue
        idx = np.sort(rng.choice(split_df.index.to_numpy(), size=max_images_per_split, replace=False))
        sampled.append(df.loc[idx].copy())
    if not sampled:
        return df.iloc[0:0].copy()
    out = pd.concat(sampled, axis=0, ignore_index=True)
    return out.sort_values(["split", "class_name", "patient_id", "study_id", "source_rel_path"]).reset_index(drop=True)


def sample_same_dicom_manifest(df: pd.DataFrame, max_images_per_split: int | None, seed: int) -> pd.DataFrame:
    if not max_images_per_split or max_images_per_split <= 0:
        return df.copy().reset_index(drop=True)
    max_pairs = max(1, int(max_images_per_split) // 2)
    rng = np.random.default_rng(seed)
    sampled = []
    for split in ALLOWED_SPLITS:
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue
        pair_ids = split_df["dicom_id"].drop_duplicates().tolist()
        if len(pair_ids) > max_pairs:
            chosen = set(rng.choice(np.array(pair_ids, dtype=object), size=max_pairs, replace=False).tolist())
            split_df = split_df[split_df["dicom_id"].isin(chosen)].copy()
        sampled.append(split_df)
    if not sampled:
        return df.iloc[0:0].copy()
    out = pd.concat(sampled, axis=0, ignore_index=True)
    return out.sort_values(["split", "class_name", "patient_id", "study_id", "dicom_id", "frame_offset"]).reset_index(drop=True)


def build_unique_source_manifest(image_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(image_root.rglob("*.png")):
        rel = path.relative_to(image_root)
        if len(rel.parts) < 6 or rel.parts[0] not in ALLOWED_SPLITS:
            continue
        split, class_name, patient_id, study_id = rel.parts[:4]
        rows.append(
            {
                "split": split,
                "class_name": class_name,
                "patient_id": str(patient_id),
                "study_id": str(study_id),
                "source_img_path": str(path.resolve()),
                "source_rel_path": rel.as_posix(),
            }
        )
    if not rows:
        raise RuntimeError(f"No PNG files discovered under {image_root}")
    df = pd.DataFrame(rows)
    return df.sort_values(["split", "class_name", "patient_id", "study_id", "source_rel_path"]).reset_index(drop=True)


def load_same_dicom_source_manifest(image_root: Path) -> pd.DataFrame:
    manifest_path = image_root / "manifest_same_dicom_master.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing same-DICOM manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = [
        "split",
        "class_name",
        "patient_id",
        "study_id",
        "dicom_rel_path",
        "dicom_id",
        "contrast_frame_index",
        "frame_index",
        "frame_offset",
        "img_path",
        "source_contrast_png",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"same-DICOM manifest missing columns: {missing}")
    df = df.copy()
    df["source_img_path"] = df["img_path"].astype(str)
    return df.sort_values(["split", "class_name", "patient_id", "study_id", "dicom_id", "frame_offset"]).reset_index(drop=True)


def build_reference_artifact(norm2_unique_manifest: pd.DataFrame, ref_output_path: Path) -> np.ndarray:
    train_paths = [Path(p) for p in norm2_unique_manifest.loc[norm2_unique_manifest["split"] == "train", "cached_img_path"].tolist()]
    if not train_paths:
        raise RuntimeError("No train images available for norm_v3 reference histogram.")
    ref_cdf = compute_reference_cdf(train_paths)
    ensure_dir(ref_output_path.parent)
    np.save(ref_output_path, ref_cdf)
    meta = {
        "num_train_images": len(train_paths),
        "created_at": datetime.now().isoformat(),
        "ref_path": str(ref_output_path),
    }
    ref_output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return ref_cdf


def manifest_hash(df: pd.DataFrame, columns: Sequence[str]) -> str:
    return sha256_records(df.loc[:, list(columns)].itertuples(index=False, name=None))


def rewrite_cached_paths(df: pd.DataFrame, source_root: Path, cache_root: Path) -> pd.DataFrame:
    out = df.copy()
    cached_paths = []
    for src in out["source_img_path"].astype(str).tolist():
        src_path = Path(src)
        rel = src_path.resolve().relative_to(source_root.resolve())
        cached_paths.append(str((cache_root / rel).resolve()))
    out["cached_img_path"] = cached_paths
    return out


def apply_variant_to_image(img_path: Path, variant: str, ref_cdf: np.ndarray | None) -> tuple[np.ndarray, dict[str, float | int]]:
    img = read_grayscale(img_path)
    if variant == "norm_v1":
        return canonicalize_norm_v1(img)
    if variant == "norm_v2":
        return canonicalize_norm_v2(img)
    if variant == "norm_v3":
        if ref_cdf is None:
            raise ValueError("ref_cdf is required for norm_v3")
        out = histogram_match_uint8(img, ref_cdf)
        return out, {
            "norm_low": 0.0,
            "norm_high": 255.0,
            "border_mask_area": 0,
            "border_mask_fraction": 0.0,
            "fallback_used": 0,
        }
    raise ValueError(f"Unsupported variant: {variant}")


def clear_if_needed(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    ensure_dir(path)


def build_variant(
    variant: str,
    output_root: Path,
    unique_view_root: Path,
    same_dicom_root: Path,
    seed: int,
    max_images_per_split: int | None,
    overwrite: bool,
) -> Path:
    if variant not in ALLOWED_VARIANTS:
        raise ValueError(f"Unsupported variant: {variant}")

    variant_root = output_root / variant
    cache_root = variant_root / "cache"
    unique_cache_root = cache_root / "unique_view"
    same_cache_root = cache_root / "same_dicom"

    if variant in {"norm_v1", "norm_v2"}:
        clear_if_needed(variant_root, overwrite)
        unique_source = build_unique_source_manifest(unique_view_root)
        same_source = load_same_dicom_source_manifest(same_dicom_root)
        unique_source = sample_unique_manifest(unique_source, max_images_per_split=max_images_per_split, seed=seed)
        same_source = sample_same_dicom_manifest(same_source, max_images_per_split=max_images_per_split, seed=seed)
        ref_cdf = None
        source_unique_root = unique_view_root
        source_same_root = same_dicom_root
    else:
        dependency_root = output_root / "norm_v2"
        dep_unique_manifest = dependency_root / "cache" / "unique_view" / "manifest_unique_view_cache.csv"
        dep_same_manifest = dependency_root / "cache" / "same_dicom" / "manifest_same_dicom_master.csv"
        if not dep_unique_manifest.exists() or not dep_same_manifest.exists():
            log("norm_v3 requires norm_v2 cache; building dependency first.")
            build_variant(
                variant="norm_v2",
                output_root=output_root,
                unique_view_root=unique_view_root,
                same_dicom_root=same_dicom_root,
                seed=seed,
                max_images_per_split=max_images_per_split,
                overwrite=overwrite,
            )
        clear_if_needed(variant_root, overwrite)
        unique_source = pd.read_csv(dep_unique_manifest)
        same_source = pd.read_csv(dep_same_manifest)
        if "cached_img_path" not in unique_source.columns or "cached_img_path" not in same_source.columns:
            raise ValueError("norm_v3 dependency manifests must contain 'cached_img_path'.")
        unique_source["source_img_path"] = unique_source["cached_img_path"].astype(str)
        same_source["source_img_path"] = same_source["cached_img_path"].astype(str)
        source_unique_root = dependency_root / "cache" / "unique_view"
        source_same_root = dependency_root / "cache" / "same_dicom"
        ref_path = variant_root / "reference_cdf_norm_v2_train_unique_view.npy"
        ref_cdf = build_reference_artifact(unique_source, ref_path)

    unique_manifest = rewrite_cached_paths(unique_source, source_unique_root, unique_cache_root)
    same_manifest = rewrite_cached_paths(same_source, source_same_root, same_cache_root)

    unique_rows = []
    for row in unique_manifest.itertuples(index=False):
        src = Path(getattr(row, "source_img_path"))
        dst = Path(getattr(row, "cached_img_path"))
        out, stats = apply_variant_to_image(src, variant=variant, ref_cdf=ref_cdf)
        write_grayscale(dst, out)
        unique_rows.append(
            {
                **row._asdict(),
                **stats,
            }
        )
    unique_manifest_out = pd.DataFrame(unique_rows)
    unique_manifest_out.to_csv(unique_cache_root / "manifest_unique_view_cache.csv", index=False)

    same_rows = []
    for row in same_manifest.itertuples(index=False):
        src = Path(getattr(row, "source_img_path"))
        dst = Path(getattr(row, "cached_img_path"))
        out, stats = apply_variant_to_image(src, variant=variant, ref_cdf=ref_cdf)
        write_grayscale(dst, out)
        payload = {**row._asdict(), **stats}
        payload["img_path"] = str(dst.resolve())
        same_rows.append(payload)
    same_manifest_out = pd.DataFrame(same_rows)
    same_manifest_out.to_csv(same_cache_root / "manifest_same_dicom_master.csv", index=False)

    unique_hash = manifest_hash(unique_manifest_out, ["split", "class_name", "patient_id", "study_id", "source_rel_path", "cached_img_path"])
    same_hash = manifest_hash(same_manifest_out, ["split", "class_name", "patient_id", "study_id", "dicom_id", "frame_offset", "cached_img_path"])

    summary_rows = []
    for dataset_name, df in [("unique_view", unique_manifest_out), ("same_dicom", same_manifest_out)]:
        for split, split_df in df.groupby("split", sort=True):
            summary_rows.append(
                {
                    "variant": variant,
                    "dataset": dataset_name,
                    "split": split,
                    "num_images": int(len(split_df)),
                    "num_unique_source_paths": int(split_df["source_img_path"].nunique()),
                    "num_cached_paths": int(split_df["cached_img_path"].nunique()),
                }
            )
    summary_df = pd.DataFrame(summary_rows).sort_values(["dataset", "split"]).reset_index(drop=True)
    summary_df.to_csv(variant_root / "summary_cache_build.csv", index=False)

    run_meta = {
        "variant": variant,
        "seed": int(seed),
        "max_images_per_split": max_images_per_split,
        "overwrite": bool(overwrite),
        "created_at": datetime.now().isoformat(),
        "unique_manifest_hash": unique_hash,
        "same_dicom_manifest_hash": same_hash,
        "unique_view_root": str(unique_view_root.resolve()),
        "same_dicom_root": str(same_dicom_root.resolve()),
    }
    (variant_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True), encoding="utf-8")
    log(
        f"Built {variant} cache | unique_view_images={len(unique_manifest_out)} | "
        f"same_dicom_images={len(same_manifest_out)} | unique_hash={unique_hash[:8]} | same_hash={same_hash[:8]}"
    )
    return variant_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exp-0 input-standardization PNG caches.")
    parser.add_argument("--variant", choices=ALLOWED_VARIANTS, required=True)
    parser.add_argument("--unique-view-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--same-dicom-root", default="input/Stent-Contrast-same-dicom-unique-view")
    parser.add_argument("--output-root", default="outputs/exp1_fm_improve/ablation/exp0_input_standardization")
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    fh, original_stdout, original_stderr = setup_logging(out_root, args.log_file)
    try:
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        build_variant(
            variant=args.variant,
            output_root=out_root,
            unique_view_root=Path(args.unique_view_root).resolve(),
            same_dicom_root=Path(args.same_dicom_root).resolve(),
            seed=int(args.seed),
            max_images_per_split=args.max_images_per_split,
            overwrite=bool(args.overwrite),
        )
    finally:
        restore_logging(fh, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
