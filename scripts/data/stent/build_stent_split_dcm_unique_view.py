#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pydicom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a copy of stent_split_dcm with at most one DICOM per study/view."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("input/stent_split_dcm"),
        help="Source dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("input/stent_split_dcm_unique_view"),
        help="Destination dataset root.",
    )
    parser.add_argument(
        "--angle-round-decimals",
        type=int,
        default=1,
        help="Round positioner angles to this many decimals before defining a view.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing output root before writing. Use with care.",
    )
    return parser.parse_args()


def read_view_key(path: Path, angle_round_decimals: int) -> Tuple[str, float | None, float | None]:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    primary = ds.get("PositionerPrimaryAngle", None)
    secondary = ds.get("PositionerSecondaryAngle", None)
    if primary is None or secondary is None:
        return (f"missing::{path.name}", None, None)
    primary_f = round(float(primary), int(angle_round_decimals))
    secondary_f = round(float(secondary), int(angle_round_decimals))
    return (f"{primary_f:.{angle_round_decimals}f}|{secondary_f:.{angle_round_decimals}f}", primary_f, secondary_f)


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Remove it or rerun with --overwrite."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    logs_root = output_root / "_logs"

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    ensure_output_root(output_root, overwrite=bool(args.overwrite))
    logs_root.mkdir(parents=True, exist_ok=True)

    dcm_paths = sorted(input_root.rglob("*.dcm"))
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files found under {input_root}")

    groups: Dict[Tuple[str, str, str, str, str, str], List[Dict[str, object]]] = defaultdict(list)

    for idx, dcm_path in enumerate(dcm_paths, start=1):
        rel = dcm_path.relative_to(input_root)
        rel_parts = rel.parts
        if len(rel_parts) < 6:
            continue
        split_name, class_name, patient_id, study_id, modality = rel_parts[:5]
        view_key, primary, secondary = read_view_key(dcm_path, int(args.angle_round_decimals))
        group_key = (split_name, class_name, patient_id, study_id, modality, view_key)
        groups[group_key].append(
            {
                "rel_path": str(rel.as_posix()),
                "path": dcm_path,
                "view_key": view_key,
                "primary_angle": primary,
                "secondary_angle": secondary,
            }
        )
        if idx % 500 == 0:
            print(f"Scanned {idx}/{len(dcm_paths)} DICOMs")

    kept_rows: List[Dict[str, object]] = []
    dropped_rows: List[Dict[str, object]] = []
    duplicate_hist = Counter()

    for group_key, recs in sorted(groups.items()):
        recs_sorted = sorted(recs, key=lambda item: item["rel_path"])
        kept = recs_sorted[0]
        duplicate_hist[len(recs_sorted)] += 1

        src_path = Path(kept["path"])
        dst_path = output_root / kept["rel_path"]
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

        split_name, class_name, patient_id, study_id, modality, view_key = group_key
        kept_rows.append(
            {
                "split": split_name,
                "class_name": class_name,
                "patient_id": patient_id,
                "study_id": study_id,
                "modality": modality,
                "view_key": view_key,
                "primary_angle": kept["primary_angle"],
                "secondary_angle": kept["secondary_angle"],
                "kept_rel_path": kept["rel_path"],
                "group_size": len(recs_sorted),
            }
        )

        for dropped in recs_sorted[1:]:
            dropped_rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "modality": modality,
                    "view_key": view_key,
                    "primary_angle": dropped["primary_angle"],
                    "secondary_angle": dropped["secondary_angle"],
                    "kept_rel_path": kept["rel_path"],
                    "dropped_rel_path": dropped["rel_path"],
                    "group_size": len(recs_sorted),
                }
            )

    kept_path = logs_root / "kept_view_groups.csv"
    dropped_path = logs_root / "dropped_duplicate_dicoms.csv"
    hist_path = logs_root / "duplicate_group_histogram.csv"
    summary_path = logs_root / "dedup_summary.txt"

    with kept_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(kept_rows[0].keys()) if kept_rows else [])
        if kept_rows:
            writer.writeheader()
            writer.writerows(kept_rows)

    with dropped_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dropped_rows[0].keys()) if dropped_rows else [])
        if dropped_rows:
            writer.writeheader()
            writer.writerows(dropped_rows)

    with hist_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group_size", "num_groups"])
        writer.writeheader()
        for group_size, count in sorted(duplicate_hist.items()):
            writer.writerow({"group_size": group_size, "num_groups": count})

    original_count = len(dcm_paths)
    kept_count = len(kept_rows)
    dropped_count = len(dropped_rows)
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("========== stent_split_dcm unique-view summary ==========\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"input_root: {input_root}\n")
        f.write(f"output_root: {output_root}\n")
        f.write(f"angle_round_decimals: {int(args.angle_round_decimals)}\n")
        f.write(f"original_dicom_count: {original_count}\n")
        f.write(f"kept_dicom_count: {kept_count}\n")
        f.write(f"dropped_dicom_count: {dropped_count}\n")
        f.write(f"drop_ratio: {dropped_count / max(1, original_count):.6f}\n")
        f.write(f"num_unique_study_view_groups: {kept_count}\n")

    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Original DICOMs: {original_count:,}")
    print(f"Kept DICOMs: {kept_count:,}")
    print(f"Dropped duplicate-view DICOMs: {dropped_count:,}")
    print(f"Kept groups CSV: {kept_path}")
    print(f"Dropped CSV: {dropped_path}")
    print(f"Histogram CSV: {hist_path}")
    print(f"Summary TXT: {summary_path}")


if __name__ == "__main__":
    main()
