#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import pydicom


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

DEFAULT_IMAGE_ROOT = REPO_ROOT / "input/Stent-Contrast-unique-view"
DEFAULT_DCM_ROOT = REPO_ROOT / "input/stent_split_dcm_unique_view"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "input/contrast_benchmark/1_global/1_view_classification"
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
ALLOWED_SPLITS = ("train", "valid", "test")

VIEW_LABELS_9WAY = [
    "up_left",
    "up_center",
    "up_right",
    "center_left",
    "center_center",
    "center_right",
    "down_left",
    "down_center",
    "down_right",
]
VIEW_INDEX = {label: idx for idx, label in enumerate(VIEW_LABELS_9WAY)}
VIEW_DIR_MAP = {
    "up_left": "1_RAO_cranial",
    "center_left": "2_RAO_straight",
    "down_left": "3_RAO_caudal",
    "up_center": "4_AP_cranial",
    "center_center": "5_AP_straight",
    "down_center": "6_AP_caudal",
    "up_right": "7_LAO_cranial",
    "center_right": "8_LAO_straight",
    "down_right": "9_LAO_caudal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--dcm-root", type=Path, default=DEFAULT_DCM_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def clean_numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def quantize_horizontal(primary: float) -> str:
    if primary < -10.0:
        return "left"
    if primary > 10.0:
        return "right"
    return "center"


def quantize_vertical(secondary: float) -> str:
    if secondary < -10.0:
        return "down"
    if secondary > 10.0:
        return "up"
    return "center"


def build_view_label(primary: float | None, secondary: float | None) -> tuple[str | None, str | None, str | None]:
    if primary is None or secondary is None:
        return None, None, None
    horizontal = quantize_horizontal(primary)
    vertical = quantize_vertical(secondary)
    return horizontal, vertical, f"{vertical}_{horizontal}"


def iter_image_paths(image_root: Path) -> List[Path]:
    return sorted([p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def remove_contents(output_root: Path) -> None:
    if not output_root.exists():
        return
    for child in output_root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def relative_parts(image_root: Path, image_path: Path) -> tuple[str, str, str, str, str, Path]:
    rel = image_path.resolve().relative_to(image_root.resolve())
    if len(rel.parts) < 6:
        raise ValueError(f"Unexpected image path under dataset root: {image_path}")
    split, stent_label, patient_id, study_id, modality = rel.parts[:5]
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unexpected split in image path: {image_path}")
    if modality != "XA":
        raise ValueError(f"Unexpected modality in image path: {image_path}")
    return split, stent_label, patient_id, study_id, modality, rel


def read_angles(dcm_path: Path) -> tuple[float | None, float | None]:
    dcm = pydicom.dcmread(
        str(dcm_path),
        stop_before_pixels=True,
        specific_tags=["PositionerPrimaryAngle", "PositionerSecondaryAngle"],
        force=True,
    )
    primary = clean_numeric(getattr(dcm, "PositionerPrimaryAngle", None))
    secondary = clean_numeric(getattr(dcm, "PositionerSecondaryAngle", None))
    return primary, secondary


def build_dataset(
    image_root: Path,
    dcm_root: Path,
    output_root: Path,
    *,
    log_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    images = iter_image_paths(image_root)
    if not images:
        raise FileNotFoundError(f"No images found under {image_root}")

    rows: List[Dict[str, object]] = []
    drop_rows: List[Dict[str, object]] = []

    for idx, image_path in enumerate(images, start=1):
        split, stent_label, patient_id, study_id, modality, rel = relative_parts(image_root, image_path)
        dcm_rel = rel.with_suffix(".dcm")
        dcm_path = dcm_root / dcm_rel
        if not dcm_path.exists():
            raise FileNotFoundError(f"Missing DICOM for {image_path}: {dcm_path}")

        primary, secondary = read_angles(dcm_path)
        horizontal, vertical, view_label = build_view_label(primary, secondary)
        if view_label is None:
            drop_rows.append(
                {
                    "source_image_rel_path": str(rel),
                    "source_dicom_rel_path": str(dcm_rel),
                    "split": split,
                    "stent_label": stent_label,
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "drop_reason": "missing_view_angle",
                }
            )
            continue

        view_dir = VIEW_DIR_MAP[view_label]
        out_rel = Path(split) / view_dir / patient_id / study_id / modality / image_path.name
        out_path = output_root / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, out_path)

        rows.append(
            {
                "source_image_rel_path": str(rel),
                "source_dicom_rel_path": str(dcm_rel),
                "output_image_rel_path": str(out_rel),
                "split": split,
                "stent_label": stent_label,
                "patient_id": patient_id,
                "study_id": study_id,
                "modality": modality,
                "PositionerPrimaryAngle": primary,
                "PositionerSecondaryAngle": secondary,
                "view_horizontal_10deg": horizontal,
                "view_vertical_10deg": vertical,
                "view_label_9way": view_label,
                "view_label_index": int(VIEW_INDEX[view_label]),
                "view_dir_name": view_dir,
            }
        )

        if log_every > 0 and idx % log_every == 0:
            print(f"[BUILD] processed={idx}/{len(images)} copied={len(rows)} dropped={len(drop_rows)}")

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No samples copied into contrast_benchmark view classification dataset.")
    manifest = manifest.sort_values(
        ["split", "view_dir_name", "patient_id", "study_id", "output_image_rel_path"]
    ).reset_index(drop=True)
    manifest.insert(0, "image_id", np.arange(len(manifest), dtype=np.int64))
    drop_df = pd.DataFrame(drop_rows)
    return manifest, drop_df


def validate_output_structure(output_root: Path, manifest: pd.DataFrame) -> None:
    for split in ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split]
        if split_df.empty:
            raise ValueError(f"No rows found for split={split}")
        missing_views = [view for view in VIEW_DIR_MAP.values() if view not in set(split_df["view_dir_name"].tolist())]
        if missing_views:
            raise ValueError(f"Split {split} is missing view directories: {missing_views}")
        for rel_path in split_df["output_image_rel_path"].tolist():
            if not (output_root / rel_path).exists():
                raise FileNotFoundError(f"Missing copied output image: {output_root / rel_path}")


def write_reports(output_root: Path, manifest: pd.DataFrame, drop_df: pd.DataFrame, image_root: Path, dcm_root: Path) -> None:
    manifest.to_csv(output_root / "manifest_view_classification.csv", index=False)
    if drop_df.empty:
        pd.DataFrame(columns=["source_image_rel_path", "source_dicom_rel_path", "split", "stent_label", "patient_id", "study_id", "drop_reason"]).to_csv(
            output_root / "dropped_view_classification.csv", index=False
        )
    else:
        drop_df.to_csv(output_root / "dropped_view_classification.csv", index=False)

    summary_split = (
        manifest.groupby("split", sort=True)
        .agg(
            num_images=("image_id", "count"),
            num_patients=("patient_id", "nunique"),
            num_studies=("study_id", "nunique"),
        )
        .reset_index()
        .sort_values("split")
    )
    summary_split.to_csv(output_root / "summary_counts_by_split.csv", index=False)

    summary_split_view = (
        manifest.groupby(["split", "view_dir_name"], sort=True)
        .agg(
            num_images=("image_id", "count"),
            num_patients=("patient_id", "nunique"),
            num_studies=("study_id", "nunique"),
            mean_primary_angle=("PositionerPrimaryAngle", "mean"),
            mean_secondary_angle=("PositionerSecondaryAngle", "mean"),
        )
        .reset_index()
        .sort_values(["split", "view_dir_name"])
    )
    summary_split_view.to_csv(output_root / "summary_counts_by_split_view.csv", index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "image_root": str(image_root.resolve()),
        "dcm_root": str(dcm_root.resolve()),
        "output_root": str(output_root.resolve()),
        "num_images": int(len(manifest)),
        "num_dropped": int(len(drop_df)),
        "view_label_9way_to_dir": {k: VIEW_DIR_MAP[k] for k in VIEW_LABELS_9WAY},
        "horizontal_rule": {"primary_angle_lt_-10": "left/RAO", "primary_angle_gt_10": "right/LAO", "otherwise": "center/AP"},
        "vertical_rule": {"secondary_angle_lt_-10": "down/caudal", "secondary_angle_gt_10": "up/cranial", "otherwise": "center/straight"},
    }
    (output_root / "build_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    lines = [
        "# Contrast Benchmark — View Classification",
        "",
        f"- source image root: `{image_root}`",
        f"- source dicom root: `{dcm_root}`",
        f"- output root: `{output_root}`",
        "- task: 9-way view classification",
        "- output layout: `split/view_dir_name/patient_id/study_id/XA/*.png`",
        "- original `stent/no_stent` hierarchy is removed from the training dataset layout",
        "- angle bins:",
        "  - horizontal: RAO / AP / LAO from `PositionerPrimaryAngle` using thresholds `-10`, `+10`",
        "  - vertical: cranial / straight / caudal from `PositionerSecondaryAngle` using thresholds `-10`, `+10`",
        "",
        "## View Directory Mapping",
        "",
    ]
    for label in VIEW_LABELS_9WAY:
        lines.append(f"- `{label}` -> `{VIEW_DIR_MAP[label]}`")
    lines.extend(
        [
            "",
            "## Key Files",
            "",
            "- `manifest_view_classification.csv`",
            "- `dropped_view_classification.csv`",
            "- `summary_counts_by_split.csv`",
            "- `summary_counts_by_split_view.csv`",
            "- `build_metadata.json`",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    image_root = args.image_root.resolve()
    dcm_root = args.dcm_root.resolve()
    output_root = args.output_root.resolve()

    if not image_root.exists():
        raise FileNotFoundError(f"Missing image root: {image_root}")
    if not dcm_root.exists():
        raise FileNotFoundError(f"Missing dcm root: {dcm_root}")

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output root is not empty: {output_root}. Re-run with --overwrite.")

    output_root.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        remove_contents(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    print(f"[START] image_root={image_root}")
    print(f"[START] dcm_root={dcm_root}")
    print(f"[START] output_root={output_root}")
    manifest, drop_df = build_dataset(image_root, dcm_root, output_root, log_every=int(args.log_every))
    validate_output_structure(output_root, manifest)
    write_reports(output_root, manifest, drop_df, image_root, dcm_root)
    print(
        f"[DONE] copied={len(manifest)} dropped={len(drop_df)} output_root={output_root}"
    )


if __name__ == "__main__":
    main()
