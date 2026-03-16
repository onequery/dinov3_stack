#!/usr/bin/env python3
"""
Build a patient-disjoint split from input/stent_split_dcm-style data.

Input structure:
  <input_root>/<split>/<class>/<patient>/<...>.dcm

Output structure:
  <output_root>/<split>/<class>/<patient>/<...>.dcm

Notes:
- Split is performed at patient level (one patient belongs to exactly one split).
- Files are copied (source is never modified).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class FileRecord:
    class_name: str
    patient_id: str
    remainder: Path
    src_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("input/stent_split_dcm"),
        help="Root directory with train/valid/test/class/patient layout",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("input/stent_split_dcm_re"),
        help="Root directory to write patient-disjoint split",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing output root (skip existing files)",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, valid_ratio: float, test_ratio: float) -> None:
    total = train_ratio + valid_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    if train_ratio < 0 or valid_ratio < 0 or test_ratio < 0:
        raise ValueError("Ratios must be non-negative")


def collect_records(input_root: Path) -> tuple[dict[str, list[FileRecord]], set[str]]:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    grouped: dict[str, list[FileRecord]] = defaultdict(list)
    class_names: set[str] = set()

    for src_path in input_root.rglob("*.dcm"):
        rel = src_path.relative_to(input_root)
        if len(rel.parts) < 5:
            # Unexpected layout; skip safely.
            continue
        # expected: split / class / patient / ...
        class_name = rel.parts[1]
        patient_id = rel.parts[2]
        remainder = Path(*rel.parts[3:])

        grouped[patient_id].append(
            FileRecord(
                class_name=class_name,
                patient_id=patient_id,
                remainder=remainder,
                src_path=src_path,
            )
        )
        class_names.add(class_name)

    if not grouped:
        raise RuntimeError(f"No DCM files found under {input_root}")

    return grouped, class_names


def assign_patients(
    patient_ids: list[str],
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    validate_ratios(train_ratio, valid_ratio, test_ratio)

    rng = random.Random(seed)
    ids = sorted(patient_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(math.floor(n * train_ratio))
    n_valid = int(math.floor(n * valid_ratio))
    n_test = n - n_train - n_valid

    split_map: dict[str, str] = {}
    for pid in ids[:n_train]:
        split_map[pid] = "train"
    for pid in ids[n_train : n_train + n_valid]:
        split_map[pid] = "valid"
    for pid in ids[n_train + n_valid :]:
        split_map[pid] = "test"

    if len(split_map) != n or n_test < 0:
        raise RuntimeError("Failed to assign patient splits")
    return split_map


def ensure_output_root(output_root: Path, allow_existing_output: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing_output:
        raise FileExistsError(
            f"Output root exists and is not empty: {output_root}\n"
            "Use --allow-existing-output to continue and skip existing files."
        )
    output_root.mkdir(parents=True, exist_ok=True)


def copy_records(
    grouped: dict[str, list[FileRecord]],
    split_map: dict[str, str],
    output_root: Path,
) -> tuple[int, int]:
    copied = 0
    skipped_exists = 0
    for patient_id, records in grouped.items():
        split_name = split_map[patient_id]
        for rec in records:
            dst = output_root / split_name / rec.class_name / rec.patient_id / rec.remainder
            if dst.exists():
                skipped_exists += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec.src_path, dst)
            copied += 1
    return copied, skipped_exists


def build_stats(
    grouped: dict[str, list[FileRecord]],
    split_map: dict[str, str],
    class_names: set[str],
) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        stats[split] = {
            "patients": 0,
            "files_total": 0,
        }
        for cls in sorted(class_names):
            stats[split][f"files_{cls}"] = 0

    split_patients: dict[str, set[str]] = {k: set() for k in ("train", "valid", "test")}

    for patient_id, records in grouped.items():
        split = split_map[patient_id]
        split_patients[split].add(patient_id)
        for rec in records:
            stats[split]["files_total"] += 1
            stats[split][f"files_{rec.class_name}"] += 1

    for split in ("train", "valid", "test"):
        stats[split]["patients"] = len(split_patients[split])

    return stats


def write_logs(
    output_root: Path,
    input_root: Path,
    seed: int,
    ratios: tuple[float, float, float],
    split_map: dict[str, str],
    stats: dict[str, dict[str, int]],
    copied: int,
    skipped_exists: int,
) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_path = log_dir / f"split_summary_{ts}.txt"
    assign_csv_path = log_dir / f"patient_split_map_{ts}.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("========== Patient-disjoint split summary ==========\n")
        f.write(f"input_root: {input_root.resolve()}\n")
        f.write(f"output_root: {output_root.resolve()}\n")
        f.write(f"seed: {seed}\n")
        f.write(
            f"ratios(train/valid/test): {ratios[0]:.4f}/{ratios[1]:.4f}/{ratios[2]:.4f}\n"
        )
        f.write(f"unique_patients: {len(split_map)}\n")
        f.write(f"copied_files: {copied}\n")
        f.write(f"skipped_existing_files: {skipped_exists}\n\n")

        for split in ("train", "valid", "test"):
            f.write(f"[{split}]\n")
            for key in sorted(stats[split].keys()):
                f.write(f"{key}: {stats[split][key]}\n")
            f.write("\n")

        train_pat = {pid for pid, sp in split_map.items() if sp == "train"}
        valid_pat = {pid for pid, sp in split_map.items() if sp == "valid"}
        test_pat = {pid for pid, sp in split_map.items() if sp == "test"}
        f.write("disjoint_check:\n")
        f.write(f"train&valid: {len(train_pat & valid_pat)}\n")
        f.write(f"train&test: {len(train_pat & test_pat)}\n")
        f.write(f"valid&test: {len(valid_pat & test_pat)}\n")

    with assign_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "split"])
        for pid in sorted(split_map.keys()):
            writer.writerow([pid, split_map[pid]])

    print(f"[Done] Summary: {summary_path}")
    print(f"[Done] Patient map: {assign_csv_path}")


def main() -> None:
    args = parse_args()
    ensure_output_root(args.output_root, args.allow_existing_output)

    grouped, class_names = collect_records(args.input_root)
    split_map = assign_patients(
        patient_ids=list(grouped.keys()),
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
    )

    copied, skipped_exists = copy_records(grouped, split_map, args.output_root)
    stats = build_stats(grouped, split_map, class_names)
    write_logs(
        output_root=args.output_root,
        input_root=args.input_root,
        seed=args.seed,
        ratios=(args.train_ratio, args.valid_ratio, args.test_ratio),
        split_map=split_map,
        stats=stats,
        copied=copied,
        skipped_exists=skipped_exists,
    )


if __name__ == "__main__":
    main()
