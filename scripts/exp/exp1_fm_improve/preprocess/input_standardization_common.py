from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

ALLOWED_SPLITS = ("train", "valid", "test")
ALLOWED_VARIANTS = ("norm_v1", "norm_v2", "norm_v3")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_grayscale(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise ValueError(f"Failed to read grayscale PNG: {path}")
    return arr


def write_grayscale(path: Path, img: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok = cv2.imwrite(str(path), img)
    if not ok:
        raise ValueError(f"Failed to write PNG: {path}")


def robust_percentiles(values: np.ndarray, low_q: float = 0.5, high_q: float = 99.5) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 255.0
    low = float(np.percentile(values, low_q))
    high = float(np.percentile(values, high_q))
    return low, high


def rescale_to_uint8(img: np.ndarray, low: float, high: float) -> np.ndarray:
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1.0:
        return img.astype(np.uint8, copy=True)
    clipped = np.clip(img.astype(np.float32), low, high)
    scaled = (clipped - low) / (high - low)
    scaled = np.clip(np.rint(scaled * 255.0), 0.0, 255.0)
    return scaled.astype(np.uint8)


def canonicalize_norm_v1(img: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    low, high = robust_percentiles(img, 0.5, 99.5)
    out = rescale_to_uint8(img, low, high)
    return out, {
        "norm_low": low,
        "norm_high": high,
        "border_mask_area": 0,
        "border_mask_fraction": 0.0,
        "fallback_used": 0,
    }


def _border_zone_mask(shape: tuple[int, int], frac: float = 0.15) -> np.ndarray:
    h, w = shape
    dh = max(1, int(math.ceil(h * frac)))
    dw = max(1, int(math.ceil(w * frac)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:dh, :] = True
    mask[-dh:, :] = True
    mask[:, :dw] = True
    mask[:, -dw:] = True
    return mask


def detect_edge_connected_dark_border(img: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    h, w = img.shape
    p1 = float(np.percentile(img, 1.0))
    threshold = max(8.0, p1 + 4.0)
    border_zone = _border_zone_mask(img.shape, frac=0.15)
    candidate = (img.astype(np.float32) <= threshold) & border_zone
    if not np.any(candidate):
        return np.zeros_like(candidate, dtype=bool), {
            "border_threshold": threshold,
            "border_seed_count": 0,
            "border_mask_area": 0,
            "border_mask_fraction": 0.0,
        }

    num_labels, labels = cv2.connectedComponents(candidate.astype(np.uint8), connectivity=4)
    border_labels: set[int] = set()
    if num_labels > 1:
        edges = [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        for edge in edges:
            for value in np.unique(edge):
                if int(value) != 0:
                    border_labels.add(int(value))
    if not border_labels:
        return np.zeros_like(candidate, dtype=bool), {
            "border_threshold": threshold,
            "border_seed_count": 0,
            "border_mask_area": 0,
            "border_mask_fraction": 0.0,
        }

    border_mask = np.isin(labels, list(border_labels)) & candidate
    area = int(border_mask.sum())
    frac = float(area) / float(border_mask.size)
    if frac > 0.35:
        return np.zeros_like(candidate, dtype=bool), {
            "border_threshold": threshold,
            "border_seed_count": len(border_labels),
            "border_mask_area": area,
            "border_mask_fraction": frac,
        }
    return border_mask, {
        "border_threshold": threshold,
        "border_seed_count": len(border_labels),
        "border_mask_area": area,
        "border_mask_fraction": frac,
    }


def canonicalize_norm_v2(img: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    border_mask, border_stats = detect_edge_connected_dark_border(img)
    if not np.any(border_mask):
        out, stats = canonicalize_norm_v1(img)
        stats.update(border_stats)
        return out, stats

    valid_mask = ~border_mask
    valid = img[valid_mask]
    if valid.size < 32 or (float(valid.size) / float(img.size)) < 0.10:
        out, stats = canonicalize_norm_v1(img)
        stats.update(border_stats)
        stats["fallback_used"] = 1
        return out, stats

    low, high = robust_percentiles(valid, 0.5, 99.5)
    out = rescale_to_uint8(img, low, high)
    valid_norm = out[valid_mask]
    if valid_norm.size == 0:
        out, stats = canonicalize_norm_v1(img)
        stats.update(border_stats)
        stats["fallback_used"] = 1
        return out, stats

    fill_value = int(np.median(valid_norm))
    out = out.copy()
    out[border_mask] = fill_value
    stats = {
        "norm_low": low,
        "norm_high": high,
        "fill_value": fill_value,
        "fallback_used": 0,
    }
    stats.update(border_stats)
    return out, stats


def compute_reference_cdf(image_paths: Sequence[Path]) -> np.ndarray:
    hist = np.zeros(256, dtype=np.float64)
    for path in image_paths:
        img = read_grayscale(path)
        hist += np.bincount(img.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0.0:
        raise ValueError("Reference histogram is empty.")
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]
    return cdf.astype(np.float64)


def histogram_match_uint8(img: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    src_hist = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    if src_hist.sum() <= 0.0:
        return img.astype(np.uint8, copy=True)
    src_cdf = np.cumsum(src_hist)
    src_cdf /= src_cdf[-1]
    lut = np.interp(src_cdf, ref_cdf, np.arange(256, dtype=np.float64))
    mapped = np.take(lut, img.astype(np.int16), mode="clip")
    return np.clip(np.rint(mapped), 0.0, 255.0).astype(np.uint8)


def sha256_records(records: Iterable[Sequence[object]]) -> str:
    hasher = hashlib.sha256()
    for row in records:
        line = "\t".join(str(v) for v in row)
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()
