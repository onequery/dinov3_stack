#!/usr/bin/env python3

import argparse
import csv
import os
import shutil
from collections import defaultdict
from pathlib import Path


DEFAULT_SELECTED_FRAMES_DIRNAME = "20241213_mpxa_selected_frames"


def _copy_file(src: Path, dst: Path, *, overwrite: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return False
    shutil.copy2(src, dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy matched MPXA (uuid).dcm and (uuid)_*.json pairs while preserving folder structure."
        )
    )
    parser.add_argument("--src", type=Path, default=Path("input/mpxa"))
    parser.add_argument("--dst", type=Path, default=Path("input/MPXA-Seg"))
    parser.add_argument(
        "--selected-frames-dirname",
        type=str,
        default=DEFAULT_SELECTED_FRAMES_DIRNAME,
        help="Top-level directory to skip (contains json-only selected frames).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in destination.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/mpxa_seg_copy_report.csv"),
        help="CSV path for mismatch report (json without dcm, dcm without json).",
    )
    args = parser.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst

    if not src_root.exists():
        raise SystemExit(f"Source not found: {src_root}")
    if not src_root.is_dir():
        raise SystemExit(f"Source is not a directory: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    copied_dcm = 0
    copied_json = 0
    skipped_existing = 0
    scanned_dirs = 0
    scanned_dcm = 0
    scanned_json = 0
    skipped_selected_frames_files = 0

    mismatch_rows: list[dict[str, str]] = []

    for dirpath, dirnames, filenames in os.walk(src_root):
        scanned_dirs += 1

        rel_dir = os.path.relpath(dirpath, src_root)
        rel_parts = [] if rel_dir == "." else rel_dir.split(os.sep)
        if rel_parts and rel_parts[0] == args.selected_frames_dirname:
            # Do not descend further (even if it has subdirs in the future)
            dirnames[:] = []
            skipped_selected_frames_files += sum(
                1
                for fn in filenames
                if (fn.endswith(".json") or fn.endswith(".dcm")) and not fn.startswith("._")
            )
            continue

        dcm_keys: set[str] = set()
        json_by_key: dict[str, list[str]] = defaultdict(list)

        for fn in filenames:
            if fn.startswith("._"):
                continue
            if fn.endswith(".dcm"):
                scanned_dcm += 1
                dcm_keys.add(fn[:-4])
            elif fn.endswith(".json"):
                scanned_json += 1
                key = fn.split("_", 1)[0]
                json_by_key[key].append(fn)

        json_keys = set(json_by_key.keys())
        matched_keys = dcm_keys & json_keys

        # Copy matched pairs
        for key in sorted(matched_keys):
            src_dcm = Path(dirpath) / f"{key}.dcm"
            dst_dcm = dst_root / rel_dir / src_dcm.name
            if _copy_file(src_dcm, dst_dcm, overwrite=args.overwrite):
                copied_dcm += 1
            else:
                skipped_existing += 1

            for json_fn in sorted(json_by_key[key]):
                src_json = Path(dirpath) / json_fn
                dst_json = dst_root / rel_dir / json_fn
                if _copy_file(src_json, dst_json, overwrite=args.overwrite):
                    copied_json += 1
                else:
                    skipped_existing += 1

        # Record mismatches (within same directory)
        for key in sorted(json_keys - dcm_keys):
            for json_fn in sorted(json_by_key[key]):
                mismatch_rows.append(
                    {
                        "kind": "json_missing_dcm_in_same_dir",
                        "rel_dir": rel_dir,
                        "key": key,
                        "json_path": str(Path(rel_dir) / json_fn),
                        "expected_dcm_path": str(Path(rel_dir) / f"{key}.dcm"),
                    }
                )

        for key in sorted(dcm_keys - json_keys):
            mismatch_rows.append(
                {
                    "kind": "dcm_missing_json_in_same_dir",
                    "rel_dir": rel_dir,
                    "key": key,
                    "json_path": "",
                    "expected_dcm_path": str(Path(rel_dir) / f"{key}.dcm"),
                }
            )

    with args.report.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "rel_dir",
                "key",
                "json_path",
                "expected_dcm_path",
            ],
        )
        w.writeheader()
        w.writerows(mismatch_rows)

    print(f"src_root: {src_root}")
    print(f"dst_root: {dst_root}")
    print(f"scanned_dirs: {scanned_dirs}")
    print(f"scanned_dcm: {scanned_dcm}")
    print(f"scanned_json: {scanned_json}")
    print(f"skipped_selected_frames_files: {skipped_selected_frames_files}")
    print(f"copied_dcm: {copied_dcm}")
    print(f"copied_json: {copied_json}")
    print(f"skipped_existing: {skipped_existing}")
    print(f"mismatch_rows: {len(mismatch_rows)}")
    print(f"report: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

