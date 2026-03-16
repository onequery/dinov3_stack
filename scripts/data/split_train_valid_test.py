#!/usr/bin/env python3
"""
Split DCM files into train / valid / test with ratio 0.6 / 0.2 / 0.2

- Move files (NOT copy)
- Preserve original directory structure
- Handle large datasets safely
- Skip missing files
- Deterministic split with fixed random seed

Current structure:
input/stent_split/
  ├── stent/
  └── no_stent/

Target structure:
input/stent_split/
  ├── train/
  │     ├── stent/
  │     └── no_stent/
  ├── valid/
  │     ├── stent/
  │     └── no_stent/
  └── test/
        ├── stent/
        └── no_stent/
"""

from pathlib import Path
import random
import shutil
import argparse
from math import floor
from datetime import datetime


def collect_dcm_files(root: Path):
    """Collect all .dcm files under root, return list of Paths."""
    return [p for p in root.rglob("*.dcm") if p.is_file()]


def split_indices(n, ratios=(0.6, 0.2, 0.2)):
    """Return indices for train / valid / test."""
    n_train = floor(n * ratios[0])
    n_valid = floor(n * ratios[1])
    n_test = n - n_train - n_valid
    return n_train, n_valid, n_test


def move_file(src: Path, src_root: Path, dst_root: Path, dry_run=False):
    """Move src to dst_root preserving relative path."""
    rel_path = src.relative_to(src_root)
    dst = dst_root / rel_path

    if not src.exists():
        return "missing"

    if dst.exists():
        return "exists"

    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    return "moved"


def process_class(
    class_name: str,
    base_dir: Path,
    out_dir: Path,
    seed: int,
    dry_run: bool
):
    print(f"\n[Processing class: {class_name}]")

    src_root = base_dir / class_name
    files = collect_dcm_files(src_root)

    if not files:
        print("  No files found, skipping.")
        return

    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train, n_valid, n_test = split_indices(n)

    splits = {
        "train": files[:n_train],
        "valid": files[n_train:n_train + n_valid],
        "test": files[n_train + n_valid:]
    }

    stats = {"moved": 0, "exists": 0, "missing": 0}

    for split, split_files in splits.items():
        dst_root = out_dir / split / class_name

        for f in split_files:
            status = move_file(
                src=f,
                src_root=src_root,
                dst_root=dst_root,
                dry_run=dry_run
            )
            stats[status] += 1

    print(f"  Total files: {n}")
    print(f"  Train / Valid / Test: {n_train} / {n_valid} / {n_test}")
    print(f"  Moved: {stats['moved']}, Exists: {stats['exists']}, Missing: {stats['missing']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        default="input/stent_split",
        help="Directory containing stent/ and no_stent/"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without moving files"
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    out_dir = base_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n=== Train/Valid/Test split started ({timestamp}) ===")
    print(f"Base dir: {base_dir}")
    print(f"Dry run: {args.dry_run}")

    for cls in ["stent", "no_stent"]:
        process_class(
            class_name=cls,
            base_dir=base_dir,
            out_dir=out_dir,
            seed=args.seed,
            dry_run=args.dry_run
        )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
