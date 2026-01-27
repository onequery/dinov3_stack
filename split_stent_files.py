#!/usr/bin/env python3
"""
Move DCM files into stent / no_stent folders while preserving relative path.

Input:
- input/stent_present_filenames.txt  (stent = 0)
- input/stent_absent_filenames.txt   (stent = -1)

Source root:
- input/stent/  (contains files in nested structure like 10003493/20060405/XA/001.dcm)

Target:
input/
  stent_split/
    stent/
      <same relative path>
    no_stent/
      <same relative path>

Features:
- move (not copy)
- skip missing files (and log them)
- preserve directory structure
- write summary + missing list
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def read_relpaths(txt_path: Path) -> list[str]:
    """Read relative paths from a txt file, strip spaces, ignore empty lines."""
    rels: list[str] = []
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().lstrip("/").replace("\\", "/")
            if p:
                rels.append(p)
    return rels


def safe_move(src: Path, dst: Path, overwrite: bool = False) -> str:
    """
    Move src -> dst.
    Returns status: moved / skipped_exists / missing_src / skipped_same
    """
    if not src.exists():
        return "missing_src"

    if dst.exists():
        if not overwrite:
            return "skipped_exists"
        # overwrite: remove existing then move
        dst.unlink()

    dst.parent.mkdir(parents=True, exist_ok=True)

    # If src and dst are same path (rare but possible), skip
    try:
        if src.resolve() == dst.resolve():
            return "skipped_same"
    except Exception:
        pass

    shutil.move(str(src), str(dst))
    return "moved"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", default="input/stent", help="source root directory")
    parser.add_argument("--present-list", default="stent_present_filenames.txt", help="txt file for stent-present relpaths")
    parser.add_argument("--absent-list", default="stent_absent_filenames.txt", help="txt file for stent-absent relpaths")
    parser.add_argument("--out-root", default="input/stent_split", help="output root directory")
    parser.add_argument("--overwrite", action="store_true", help="overwrite if destination file exists")
    parser.add_argument("--dry-run", action="store_true", help="print actions without moving files")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    out_root = Path(args.out_root).resolve()

    present_txt = Path(args.present_list).resolve()
    absent_txt = Path(args.absent_list).resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")
    if not present_txt.exists():
        raise FileNotFoundError(f"Present list not found: {present_txt}")
    if not absent_txt.exists():
        raise FileNotFoundError(f"Absent list not found: {absent_txt}")

    stent_dir = out_root / "stent"
    no_stent_dir = out_root / "no_stent"

    stent_dir.mkdir(parents=True, exist_ok=True)
    no_stent_dir.mkdir(parents=True, exist_ok=True)

    present_paths = read_relpaths(present_txt)
    absent_paths = read_relpaths(absent_txt)

    # Avoid double processing if same relpath appears in both lists (shouldn't, but just in case)
    present_set = set(present_paths)
    absent_set = set(absent_paths)
    overlap = present_set & absent_set

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = out_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    summary_path = log_dir / f"split_summary_{timestamp}.txt"
    missing_path = log_dir / f"missing_files_{timestamp}.txt"
    overlap_path = log_dir / f"overlap_relpaths_{timestamp}.txt"

    moved = 0
    skipped_exists = 0
    missing = 0
    skipped_same = 0

    missing_items: list[str] = []

    def process(relpaths: list[str], target_base: Path, label: str):
        nonlocal moved, skipped_exists, missing, skipped_same, missing_items

        # de-dup while preserving order
        seen = set()
        ordered_unique = []
        for rp in relpaths:
            if rp in seen:
                continue
            seen.add(rp)
            ordered_unique.append(rp)

        for rp in ordered_unique:
            if rp in overlap:
                # skip overlapped paths here; we will log them
                continue

            src = src_root / rp
            dst = target_base / rp

            if args.dry_run:
                # only report
                if not src.exists():
                    missing += 1
                    missing_items.append(f"[{label}] {rp}")
                elif dst.exists() and not args.overwrite:
                    skipped_exists += 1
                else:
                    moved += 1
                continue

            status = safe_move(src, dst, overwrite=args.overwrite)
            if status == "moved":
                moved += 1
            elif status == "skipped_exists":
                skipped_exists += 1
            elif status == "missing_src":
                missing += 1
                missing_items.append(f"[{label}] {rp}")
            elif status == "skipped_same":
                skipped_same += 1

    # Process stent-present (stent=0)
    process(present_paths, stent_dir, label="stent")
    # Process stent-absent (stent=-1)
    process(absent_paths, no_stent_dir, label="no_stent")

    # Write logs
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("========== Stent split summary ==========\n")
        f.write(f"src_root: {src_root}\n")
        f.write(f"out_root: {out_root}\n")
        f.write(f"present_list: {present_txt}\n")
        f.write(f"absent_list: {absent_txt}\n")
        f.write(f"overwrite: {args.overwrite}\n")
        f.write(f"dry_run: {args.dry_run}\n\n")
        f.write(f"moved: {moved}\n")
        f.write(f"skipped_exists: {skipped_exists}\n")
        f.write(f"skipped_same: {skipped_same}\n")
        f.write(f"missing_src: {missing}\n")
        f.write(f"overlap_relpaths: {len(overlap)}\n")

    if missing_items:
        with missing_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(missing_items) + "\n")

    if overlap:
        with overlap_path.open("w", encoding="utf-8") as f:
            for rp in sorted(overlap):
                f.write(rp + "\n")

    print("\n[Done]")
    print(f"- moved: {moved}")
    print(f"- skipped_exists: {skipped_exists}")
    print(f"- skipped_same: {skipped_same}")
    print(f"- missing_src: {missing}")
    print(f"- overlap_relpaths: {len(overlap)}")
    print(f"- summary: {summary_path}")
    if missing_items:
        print(f"- missing list: {missing_path}")
    if overlap:
        print(f"- overlap list: {overlap_path}")


if __name__ == "__main__":
    main()
