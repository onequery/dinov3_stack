#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pydicom


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze duplicate angiography views within each study from DICOM metadata."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        default="input/stent_split_dcm",
        help="Root containing split/class/patient/study/modality/*.dcm structure.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/stent_split_dcm_view_duplicates",
        help="Directory to save CSV summaries and figures.",
    )
    parser.add_argument(
        "--angle-round-decimals",
        type=int,
        default=1,
        help="Round primary/secondary angles to this many decimals before defining a view.",
    )
    return parser.parse_args()


def read_view_key(path: Path, angle_round_decimals: int) -> Tuple[str, float | None, float | None]:
    ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    primary = ds.get("PositionerPrimaryAngle", None)
    secondary = ds.get("PositionerSecondaryAngle", None)
    if primary is None or secondary is None:
        return ("missing", None, None)
    primary_f = round(float(primary), int(angle_round_decimals))
    secondary_f = round(float(secondary), int(angle_round_decimals))
    return (f"{primary_f:.{angle_round_decimals}f}|{secondary_f:.{angle_round_decimals}f}", primary_f, secondary_f)


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dcm_paths = sorted(input_root.rglob("*.dcm"))
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files found under {input_root}")

    study_to_views: Dict[Tuple[str, str, str, str], List[Dict[str, object]]] = defaultdict(list)
    records: List[Dict[str, object]] = []

    for idx, dcm_path in enumerate(dcm_paths, start=1):
        rel_parts = dcm_path.relative_to(input_root).parts
        if len(rel_parts) < 6:
            continue
        split_name, class_name, patient_id, study_id, modality = rel_parts[:5]
        view_key, primary, secondary = read_view_key(dcm_path, args.angle_round_decimals)
        record = {
            "split": split_name,
            "class_name": class_name,
            "patient_id": patient_id,
            "study_id": study_id,
            "modality": modality,
            "dcm_path": str(dcm_path.relative_to(input_root)),
            "view_key": view_key,
            "primary_angle": primary,
            "secondary_angle": secondary,
        }
        study_key = (split_name, class_name, patient_id, study_id)
        study_to_views[study_key].append(record)
        records.append(record)
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(dcm_paths)} DICOMs")

    per_dcm_df = pd.DataFrame(records)
    per_dcm_path = output_root / "study_view_per_dcm.csv"
    per_dcm_df.to_csv(per_dcm_path, index=False)

    per_view_rows: List[Dict[str, object]] = []
    duplicate_counter = Counter()
    study_summary_rows: List[Dict[str, object]] = []

    for study_key, recs in sorted(study_to_views.items()):
        split_name, class_name, patient_id, study_id = study_key
        counts = Counter(rec["view_key"] for rec in recs)
        distinct_views = len(counts)
        total_dicoms = len(recs)
        max_duplicate = max(counts.values()) if counts else 0
        missing_count = int(counts.get("missing", 0))
        study_summary_rows.append(
            {
                "split": split_name,
                "class_name": class_name,
                "patient_id": patient_id,
                "study_id": study_id,
                "num_dicoms": total_dicoms,
                "num_distinct_views": distinct_views,
                "max_duplicate_view_count": max_duplicate,
                "missing_view_dicoms": missing_count,
            }
        )
        by_key = defaultdict(list)
        for rec in recs:
            by_key[rec["view_key"]].append(rec)
        for view_key, view_recs in sorted(by_key.items(), key=lambda item: (-len(item[1]), item[0])):
            count = len(view_recs)
            duplicate_counter[count] += 1
            per_view_rows.append(
                {
                    "split": split_name,
                    "class_name": class_name,
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "view_key": view_key,
                    "primary_angle": view_recs[0]["primary_angle"],
                    "secondary_angle": view_recs[0]["secondary_angle"],
                    "duplicate_count_within_study": count,
                    "dcm_paths": json.dumps([rec["dcm_path"] for rec in view_recs], ensure_ascii=True),
                }
            )

    per_view_df = pd.DataFrame(per_view_rows).sort_values(
        ["duplicate_count_within_study", "split", "class_name", "patient_id", "study_id"],
        ascending=[False, True, True, True, True],
    )
    per_view_path = output_root / "study_view_groups.csv"
    per_view_df.to_csv(per_view_path, index=False)

    study_summary_df = pd.DataFrame(study_summary_rows).sort_values(
        ["split", "class_name", "patient_id", "study_id"]
    )
    study_summary_path = output_root / "study_view_summary.csv"
    study_summary_df.to_csv(study_summary_path, index=False)

    hist_df = pd.DataFrame(
        [
            {"duplicate_view_count": count, "num_study_view_groups": freq}
            for count, freq in sorted(duplicate_counter.items())
        ]
    )
    hist_path = output_root / "view_duplicate_histogram.csv"
    hist_df.to_csv(hist_path, index=False)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(
        hist_df["duplicate_view_count"].astype(int),
        hist_df["num_study_view_groups"].astype(int),
        color="#4C72B0",
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xlabel("Duplicate View Count Within Study")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Duplicate DICOM Views Within Study")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    figure_path = output_root / "fig_study_view_duplicate_histogram.png"
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    missing_total = int((per_dcm_df["view_key"] == "missing").sum())
    num_studies = len(study_to_views)
    print(f"Input root: {input_root}")
    print(f"Total DICOMs: {len(dcm_paths):,}")
    print(f"Total studies: {num_studies:,}")
    print(f"Missing angle-tag DICOMs: {missing_total:,}")
    print(f"Saved per-DICOM table: {per_dcm_path}")
    print(f"Saved grouped-view table: {per_view_path}")
    print(f"Saved study summary: {study_summary_path}")
    print(f"Saved histogram CSV: {hist_path}")
    print(f"Saved histogram figure: {figure_path}")


if __name__ == "__main__":
    main()
