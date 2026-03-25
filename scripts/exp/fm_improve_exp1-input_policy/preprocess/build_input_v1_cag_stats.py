#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent

import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.img_cls.input_policy import build_pre_normalize_eval_transform, eval_geometry_signature

IMAGE_EXTS = ("png", "jpg", "jpeg")


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_image_paths(split_root: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(split_root.rglob(f"*.{ext}"))
    return sorted(set(path.resolve() for path in paths if path.is_file()))


def compute_stats(image_paths: Iterable[Path], resize_size: int, center_crop_size: int) -> tuple[float, float, int, int]:
    transform = build_pre_normalize_eval_transform(resize_size, center_crop_size)
    total_sum = 0.0
    total_sum_sq = 0.0
    total_pixels = 0
    num_images = 0
    for img_path in image_paths:
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Failed to read grayscale image: {img_path}")
        tensor = transform(gray)
        arr = tensor.numpy().astype(np.float64, copy=False)
        total_sum += float(arr.sum())
        total_sum_sq += float(np.square(arr).sum())
        total_pixels += int(arr.size)
        num_images += 1
    if total_pixels <= 0:
        raise ValueError("No pixels found while computing input_v1 stats")
    mean = total_sum / float(total_pixels)
    variance = max(0.0, total_sum_sq / float(total_pixels) - mean * mean)
    std = float(np.sqrt(variance))
    if std <= 0.0:
        raise ValueError(f"Computed non-positive std for input_v1 stats: {std}")
    return float(mean), std, num_images, total_pixels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CAG normalization stats for input_v1_cag_stats_normalization.")
    parser.add_argument("--image-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--output-json",
        default="outputs/fm_improve_exp1-input_policy/input_v1_cag_stats_normalization/stats/cag_stats_unique_view_train.json",
    )
    parser.add_argument("--resize-size", type=int, default=480)
    parser.add_argument("--center-crop-size", type=int, default=448)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    split_root = image_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")
    output_json = Path(args.output_json).resolve()
    ensure_dir(output_json.parent)

    image_paths = collect_image_paths(split_root)
    if not image_paths:
        raise ValueError(f"No images found under {split_root}")

    log(f"Computing input_v1 stats from {len(image_paths)} images under {split_root}")
    mean, std, num_images, total_pixels = compute_stats(
        image_paths=image_paths,
        resize_size=args.resize_size,
        center_crop_size=args.center_crop_size,
    )
    payload = {
        "policy_name": "input_v1_cag_stats_normalization",
        "dataset_root": str(image_root),
        "split": str(args.split),
        "mean_scalar": mean,
        "std_scalar": std,
        "mean_rgb": [mean, mean, mean],
        "std_rgb": [std, std, std],
        "transform_signature": eval_geometry_signature(args.resize_size, args.center_crop_size),
        "num_images": int(num_images),
        "total_pixels": int(total_pixels),
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Wrote input_v1 stats to {output_json}")
    log(f"mean={mean:.8f} std={std:.8f}")


if __name__ == "__main__":
    main()
