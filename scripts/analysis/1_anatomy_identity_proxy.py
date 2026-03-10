#!/usr/bin/env python3
"""
Analysis 1 — Anatomy Identity Proxy Analysis

Compare frozen backbone representations between:
1) ImageNet-1K pretrained backbone
2) CAG contrast pretrained backbone

Proxy definition:
- Positive: same patient + same study (excluding self-pairs)
- Negative: different patient + different study
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, TextIO, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.img_cls.model import Dinov3Backbone  # noqa: E402


DEFAULT_IMAGENET_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
DEFAULT_CAG_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)

PATTERN_8DIGIT = re.compile(r"^\d{8}$")


@dataclass
class BackboneStats:
    backbone_name: str
    n_positive_pairs: int
    n_negative_pairs: int
    positive_mean: float
    positive_std: float
    negative_mean: float
    negative_std: float
    gap: float
    auc_optional: float

    def to_row(self) -> Dict[str, float | int | str]:
        return {
            "backbone_name": self.backbone_name,
            "n_positive_pairs": self.n_positive_pairs,
            "n_negative_pairs": self.n_negative_pairs,
            "positive_mean": self.positive_mean,
            "positive_std": self.positive_std,
            "negative_mean": self.negative_mean,
            "negative_std": self.negative_std,
            "gap": self.gap,
            "auc_optional": self.auc_optional,
        }


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], transform: transforms.Compose):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb)


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

    def isatty(self) -> bool:
        for stream in self.streams:
            if hasattr(stream, "isatty") and stream.isatty():
                return True
        return False


class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total_steps = max(1, int(total_steps))
        self.completed_steps = 0
        self.start_time = time.time()

    def start_step(self, name: str) -> None:
        current = min(self.total_steps, self.completed_steps + 1)
        log(f"[PROGRESS] Step {current}/{self.total_steps} START | {name}")

    def finish_step(self, name: str) -> None:
        self.completed_steps = min(self.total_steps, self.completed_steps + 1)
        elapsed = max(0.0, time.time() - self.start_time)
        progress = 100.0 * self.completed_steps / self.total_steps
        avg_step = elapsed / max(1, self.completed_steps)
        remaining = max(0.0, avg_step * (self.total_steps - self.completed_steps))
        eta_dt = datetime.now() + timedelta(seconds=remaining)
        log(
            f"[PROGRESS] Step {self.completed_steps}/{self.total_steps} DONE | {name} | "
            f"progress={progress:.1f}% | elapsed={format_duration(elapsed)} | "
            f"remaining~{format_duration(remaining)} | eta={eta_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def setup_console_and_file_logging(
    output_root: Path,
    log_file_arg: str | None,
) -> Tuple[Path, TextIO, TextIO, TextIO]:
    if log_file_arg:
        log_path = Path(log_file_arg).expanduser()
        if not log_path.is_absolute():
            log_path = output_root / log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_root / f"1_anatomy_identity_proxy_{stamp}.log"

    ensure_dir(log_path.parent)
    file_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, file_handle)
    sys.stderr = TeeStream(original_stderr, file_handle)
    log(f"Console output is mirrored to log file: {log_path}")
    return log_path, file_handle, original_stdout, original_stderr


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_image_paths(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    all_paths: List[str] = []
    for ext in exts:
        all_paths.extend(glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True))
    return sorted(set(all_paths))


def hash_records(records: Sequence[Sequence[object]]) -> str:
    hasher = hashlib.sha256()
    for row in records:
        line = "\t".join(str(v) for v in row)
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> str:
    records = df.loc[:, list(columns)].itertuples(index=False, name=None)
    return hash_records(records)


def parse_patient_study_from_path(path: str) -> Tuple[str, str]:
    parts = Path(path).parts
    for i in range(len(parts) - 1):
        if PATTERN_8DIGIT.match(parts[i]) and PATTERN_8DIGIT.match(parts[i + 1]):
            return parts[i], parts[i + 1]
    raise ValueError(f"Could not parse patient_id/study_id from path: {path}")


def parse_split_and_class(path: str, root_dir: str) -> Tuple[str, str]:
    try:
        rel = Path(path).resolve().relative_to(Path(root_dir).resolve())
    except Exception:
        rel = Path(path)
    parts = rel.parts
    split = parts[0] if len(parts) >= 1 else "unknown_split"
    class_name = parts[1] if len(parts) >= 2 else "unknown_class"
    return split, class_name


def build_manifest(
    image_root: str,
    exts: Tuple[str, ...],
    max_images: int | None,
    seed: int,
) -> pd.DataFrame:
    paths = collect_image_paths(image_root, exts=exts)
    if not paths:
        raise ValueError(f"No images found under: {image_root}")

    rows = []
    for p in paths:
        patient_id, study_id = parse_patient_study_from_path(p)
        split, class_name = parse_split_and_class(p, image_root)
        rows.append(
            {
                "img_path": p,
                "patient_id": patient_id,
                "study_id": study_id,
                "split": split,
                "class_name": class_name,
            }
        )

    manifest = pd.DataFrame(rows).sort_values("img_path").reset_index(drop=True)

    if max_images is not None and max_images > 0 and max_images < len(manifest):
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(len(manifest), size=max_images, replace=False))
        manifest = manifest.iloc[selected].reset_index(drop=True)

    manifest.insert(0, "image_id", np.arange(len(manifest), dtype=np.int64))
    return manifest


def comb2(n: int) -> int:
    return (n * (n - 1)) // 2 if n >= 2 else 0


def compute_max_negative_pairs(manifest: pd.DataFrame) -> int:
    n = len(manifest)
    total_pairs = comb2(n)

    patient_counts = manifest.groupby("patient_id").size().to_numpy(dtype=np.int64)
    study_counts = manifest.groupby("study_id").size().to_numpy(dtype=np.int64)
    patient_study_counts = (
        manifest.groupby(["patient_id", "study_id"]).size().to_numpy(dtype=np.int64)
    )

    same_patient_pairs = int(np.sum([comb2(int(c)) for c in patient_counts]))
    same_study_pairs = int(np.sum([comb2(int(c)) for c in study_counts]))
    same_patient_study_pairs = int(np.sum([comb2(int(c)) for c in patient_study_counts]))

    max_negative = (
        total_pairs - same_patient_pairs - same_study_pairs + same_patient_study_pairs
    )
    return int(max(0, max_negative))


def build_positive_pairs(manifest: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    pos_i: List[int] = []
    pos_j: List[int] = []

    grouped = manifest.groupby(["patient_id", "study_id"])["image_id"]
    for _, image_ids_series in grouped:
        image_ids = image_ids_series.to_numpy(dtype=np.int64)
        m = len(image_ids)
        if m < 2:
            continue
        for a in range(m - 1):
            i = int(image_ids[a])
            for b in range(a + 1, m):
                j = int(image_ids[b])
                pos_i.append(i)
                pos_j.append(j)

    return np.asarray(pos_i, dtype=np.int64), np.asarray(pos_j, dtype=np.int64)


def sample_negative_pairs(
    manifest: pd.DataFrame,
    target_count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_count <= 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    patient = manifest["patient_id"].to_numpy()
    study = manifest["study_id"].to_numpy()
    n = len(manifest)

    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    neg_i: List[int] = []
    neg_j: List[int] = []

    attempts = 0
    max_attempts = max(500_000, target_count * 300)
    batch_size = max(10_000, min(200_000, target_count * 8))

    with tqdm(total=target_count, desc="Sampling negative pairs") as pbar:
        while len(neg_i) < target_count and attempts < max_attempts:
            i = rng.integers(0, n, size=batch_size, dtype=np.int64)
            j = rng.integers(0, n, size=batch_size, dtype=np.int64)
            attempts += batch_size

            mask_non_self = i != j
            i = i[mask_non_self]
            j = j[mask_non_self]

            lo = np.minimum(i, j)
            hi = np.maximum(i, j)

            mask_valid = (patient[lo] != patient[hi]) & (study[lo] != study[hi])
            lo = lo[mask_valid]
            hi = hi[mask_valid]

            for a, b in zip(lo.tolist(), hi.tolist()):
                key = int(a) * n + int(b)
                if key in selected:
                    continue
                selected.add(key)
                neg_i.append(int(a))
                neg_j.append(int(b))
                pbar.update(1)
                if len(neg_i) >= target_count:
                    break

    if len(neg_i) < target_count:
        raise RuntimeError(
            f"Failed to sample enough negative pairs. target={target_count}, got={len(neg_i)}"
        )

    return np.asarray(neg_i, dtype=np.int64), np.asarray(neg_j, dtype=np.int64)


def build_or_load_pair_index(
    manifest: pd.DataFrame,
    out_root: Path,
    negative_ratio: float,
    seed: int,
) -> pd.DataFrame:
    pair_index_path = out_root / "pair_index.csv"
    pair_meta_path = out_root / "pair_index_meta.json"

    manifest_hash = hash_dataframe(
        manifest, columns=["image_id", "img_path", "patient_id", "study_id"]
    )
    desired_meta = {
        "manifest_hash": manifest_hash,
        "negative_ratio": float(negative_ratio),
        "seed": int(seed),
        "n_images": int(len(manifest)),
    }

    if pair_index_path.exists() and pair_meta_path.exists():
        with open(pair_meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        if saved_meta == desired_meta:
            log(f"Reusing cached pair index: {pair_index_path}")
            pair_df = pd.read_csv(pair_index_path)
            required_cols = {"pair_id", "img_id_1", "img_id_2", "pair_type"}
            if not required_cols.issubset(set(pair_df.columns)):
                raise ValueError(
                    f"Cached pair index missing required columns: {required_cols - set(pair_df.columns)}"
                )
            return pair_df

    log("Building pair index...")
    pos_i, pos_j = build_positive_pairs(manifest)
    n_pos = int(len(pos_i))
    if n_pos == 0:
        raise ValueError("No positive pairs found. Cannot run anatomy identity proxy analysis.")

    max_negative = compute_max_negative_pairs(manifest)
    requested_negative = int(round(n_pos * float(negative_ratio)))
    n_neg = min(requested_negative, max_negative)
    if n_neg <= 0:
        raise ValueError("No valid negative pairs available for current dataset constraints.")

    log(
        f"Positive pairs: {n_pos:,} | Requested negatives: {requested_negative:,} | "
        f"Using negatives: {n_neg:,} (max_valid={max_negative:,})"
    )
    neg_i, neg_j = sample_negative_pairs(manifest, target_count=n_neg, seed=seed)

    pos_type = np.full(n_pos, "positive", dtype=object)
    neg_type = np.full(n_neg, "negative", dtype=object)

    all_i = np.concatenate([pos_i, neg_i], axis=0)
    all_j = np.concatenate([pos_j, neg_j], axis=0)
    all_type = np.concatenate([pos_type, neg_type], axis=0)

    pair_df = pd.DataFrame(
        {
            "pair_id": np.arange(len(all_i), dtype=np.int64),
            "img_id_1": all_i,
            "img_id_2": all_j,
            "pair_type": all_type,
        }
    )
    pair_df.to_csv(pair_index_path, index=False)
    with open(pair_meta_path, "w", encoding="utf-8") as f:
        json.dump(desired_meta, f, indent=2)

    return pair_df


def build_transform(resize_size: int, center_crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop((center_crop_size, center_crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def extract_global_representation(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, dict):
        if "x_norm_clstoken" in output and torch.is_tensor(output["x_norm_clstoken"]):
            tensor = output["x_norm_clstoken"]
        else:
            tensor_candidates = [v for v in output.values() if torch.is_tensor(v)]
            if not tensor_candidates:
                raise ValueError("Backbone output dictionary has no tensor values.")
            tensor = tensor_candidates[0]
    elif isinstance(output, (list, tuple)):
        if not output:
            raise ValueError("Backbone output list/tuple is empty.")
        first = output[0]
        if not torch.is_tensor(first):
            raise ValueError("Backbone output list/tuple first element is not a tensor.")
        tensor = first
    else:
        raise ValueError(f"Unsupported backbone output type: {type(output)}")

    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor[:, 0, :]
    raise ValueError(f"Unsupported output tensor shape: {tuple(tensor.shape)}")


def extract_or_load_features(
    manifest: pd.DataFrame,
    backbone_name: str,
    ckpt_path: str,
    args: argparse.Namespace,
    out_root: Path,
) -> torch.Tensor:
    features_path = out_root / f"features_{backbone_name}.pt"
    meta_path = out_root / f"features_{backbone_name}.meta.json"
    feature_index_path = out_root / "feature_index.csv"

    manifest_hash = hash_dataframe(manifest, columns=["image_id", "img_path"])
    expected_meta = {
        "manifest_hash": manifest_hash,
        "backbone_name": backbone_name,
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "model_name": args.model_name,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "resize_size": int(args.resize_size),
        "center_crop_size": int(args.center_crop_size),
    }

    if args.cache_features and features_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_meta = json.load(f)
        if cached_meta == expected_meta:
            log(f"Reusing feature cache for {backbone_name}: {features_path}")
            features = torch.load(features_path, map_location="cpu")
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"Cached features are not a tensor: {features_path}")
            if features.ndim != 2 or features.shape[0] != len(manifest):
                raise ValueError(
                    f"Cached features shape mismatch: {tuple(features.shape)} vs n_images={len(manifest)}"
                )
            return features

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found for {backbone_name}: {ckpt_path}")

    requested_device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        log(f"CUDA requested but unavailable in this environment. Falling back to CPU for {backbone_name}.")
        requested_device = "cpu"
    device = torch.device(requested_device)
    transform = build_transform(args.resize_size, args.center_crop_size)
    model = Dinov3Backbone(
        weights=ckpt_path,
        model_name=args.model_name,
        repo_dir=args.repo_dir,
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    image_paths = manifest["img_path"].tolist()
    dataset = ImagePathDataset(image_paths=image_paths, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    feats: List[torch.Tensor] = []

    log(f"Extracting features ({backbone_name}) on device={device} | n_images={len(image_paths):,}")
    with torch.no_grad():
        for x in tqdm(loader, desc=f"Extract-{backbone_name}"):
            x = x.to(device, non_blocking=True)
            out = model(x)
            feat = extract_global_representation(out)
            feat = F.normalize(feat, dim=1)
            feats.append(feat.cpu())

    features = torch.cat(feats, dim=0).contiguous().float()
    if features.shape[0] != len(manifest):
        raise ValueError(
            f"Feature length mismatch for {backbone_name}: {features.shape[0]} vs {len(manifest)}"
        )

    if args.cache_features:
        torch.save(features, features_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(expected_meta, f, indent=2)
        manifest.loc[:, ["image_id", "img_path"]].to_csv(feature_index_path, index=False)

    return features


def compute_pair_similarity(
    pair_df: pd.DataFrame,
    features: torch.Tensor,
) -> np.ndarray:
    idx1 = pair_df["img_id_1"].to_numpy(dtype=np.int64)
    idx2 = pair_df["img_id_2"].to_numpy(dtype=np.int64)
    feat_np = features.numpy()
    sims = np.sum(feat_np[idx1] * feat_np[idx2], axis=1)
    sims = np.clip(sims, -1.0, 1.0)
    return sims.astype(np.float32)


def build_pair_level_dataframe(
    pair_df: pd.DataFrame,
    manifest: pd.DataFrame,
    similarities: np.ndarray,
) -> pd.DataFrame:
    paths = manifest["img_path"].to_numpy()
    patient = manifest["patient_id"].to_numpy()
    study = manifest["study_id"].to_numpy()

    idx1 = pair_df["img_id_1"].to_numpy(dtype=np.int64)
    idx2 = pair_df["img_id_2"].to_numpy(dtype=np.int64)

    out = pd.DataFrame(
        {
            "img_path_1": paths[idx1],
            "img_path_2": paths[idx2],
            "patient_id_1": patient[idx1],
            "patient_id_2": patient[idx2],
            "study_id_1": study[idx1],
            "study_id_2": study[idx2],
            "pair_type": pair_df["pair_type"].to_numpy(),
            "cosine_similarity": similarities,
        }
    )
    return out


def summarize_similarity(
    backbone_name: str,
    pair_df: pd.DataFrame,
    similarities: np.ndarray,
) -> BackboneStats:
    pair_type = pair_df["pair_type"].to_numpy()
    pos = similarities[pair_type == "positive"]
    neg = similarities[pair_type == "negative"]

    labels = (pair_type == "positive").astype(np.int32)
    if np.unique(labels).shape[0] >= 2:
        auc = float(roc_auc_score(labels, similarities))
    else:
        auc = float("nan")

    return BackboneStats(
        backbone_name=backbone_name,
        n_positive_pairs=int(len(pos)),
        n_negative_pairs=int(len(neg)),
        positive_mean=float(np.mean(pos)) if len(pos) else float("nan"),
        positive_std=float(np.std(pos)) if len(pos) else float("nan"),
        negative_mean=float(np.mean(neg)) if len(neg) else float("nan"),
        negative_std=float(np.std(neg)) if len(neg) else float("nan"),
        gap=float(np.mean(pos) - np.mean(neg)) if len(pos) and len(neg) else float("nan"),
        auc_optional=auc,
    )


def save_histogram(
    similarities_df: pd.DataFrame,
    backbone_label: str,
    output_path: Path,
    xlim: Tuple[float, float] | None = None,
) -> None:
    pos = similarities_df.loc[similarities_df["pair_type"] == "positive", "cosine_similarity"].to_numpy()
    neg = similarities_df.loc[similarities_df["pair_type"] == "negative", "cosine_similarity"].to_numpy()

    plt.figure(figsize=(8, 6))
    plt.hist(pos, bins=80, density=True, alpha=0.60, label="Positive (same patient, same study)")
    plt.hist(neg, bins=80, density=True, alpha=0.60, label="Negative (diff patient, diff study)")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title(f"Similarity Distribution ({backbone_label})")
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_shared_hist_xlim(
    imagenet_df: pd.DataFrame,
    cag_df: pd.DataFrame,
) -> Tuple[float, float]:
    all_vals = np.concatenate(
        [
            imagenet_df["cosine_similarity"].to_numpy(dtype=np.float32),
            cag_df["cosine_similarity"].to_numpy(dtype=np.float32),
        ]
    )
    min_v = float(np.min(all_vals))
    max_v = float(np.max(all_vals))
    if max_v <= min_v:
        center = min_v
        min_v = center - 0.05
        max_v = center + 0.05
    else:
        margin = max(0.01, 0.03 * (max_v - min_v))
        min_v -= margin
        max_v += margin

    min_v = max(-1.0, min_v)
    max_v = min(1.0, max_v)
    if max_v <= min_v:
        min_v, max_v = -1.0, 1.0
    return (min_v, max_v)


def save_boxplot_compare(
    imagenet_df: pd.DataFrame,
    cag_df: pd.DataFrame,
    output_path: Path,
) -> None:
    data = [
        imagenet_df.loc[imagenet_df["pair_type"] == "positive", "cosine_similarity"].to_numpy(),
        imagenet_df.loc[imagenet_df["pair_type"] == "negative", "cosine_similarity"].to_numpy(),
        cag_df.loc[cag_df["pair_type"] == "positive", "cosine_similarity"].to_numpy(),
        cag_df.loc[cag_df["pair_type"] == "negative", "cosine_similarity"].to_numpy(),
    ]
    labels = [
        "ImageNet\nPositive",
        "ImageNet\nNegative",
        "CAG\nPositive",
        "CAG\nNegative",
    ]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("Cosine similarity")
    plt.title("Positive/Negative Similarity Comparison Across Backbones")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_gap_barplot(summary_df: pd.DataFrame, output_path: Path) -> None:
    names = summary_df["backbone_name"].tolist()
    gaps = summary_df["gap"].to_numpy(dtype=np.float64)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(names, gaps, color=["#4C72B0", "#DD8452"])
    for bar, gap in zip(bars, gaps.tolist()):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{gap:.4f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel("Gap = mean(pos) - mean(neg)")
    plt.title("Anatomy Identity Proxy Gap Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def format_float(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "nan"
    return f"{x:.6f}"


def write_markdown_summary(
    output_path: Path,
    summary_df: pd.DataFrame,
    n_images: int,
    n_patients: int,
    n_patient_study_groups: int,
    n_positive_pairs: int,
    n_negative_pairs: int,
) -> None:
    if set(summary_df["backbone_name"]) != {"imagenet", "cag"}:
        raise ValueError("Expected summary rows for both `imagenet` and `cag`.")

    row_img = summary_df[summary_df["backbone_name"] == "imagenet"].iloc[0]
    row_cag = summary_df[summary_df["backbone_name"] == "cag"].iloc[0]

    gap_img = float(row_img["gap"])
    gap_cag = float(row_cag["gap"])
    delta = gap_cag - gap_img

    if delta > 1e-6:
        interpretation = (
            "CAG-pretrained backbone shows a larger gap, suggesting stronger preservation "
            "of same-patient same-study coherence under the current anatomy identity proxy definition."
        )
    elif delta < -1e-6:
        interpretation = (
            "ImageNet-pretrained backbone shows a larger gap under this proxy. "
            "Current result does not indicate additional gain from CAG pretraining on this axis."
        )
    else:
        interpretation = (
            "Both backbones show similar gap under this proxy. "
            "Additional axes are needed to reveal potential CAG-specific gains."
        )

    lines = [
        "# Analysis 1 — Anatomy Identity Proxy",
        "",
        "## Scope",
        "- This analysis is limited to the proxy definition: `same patient same study` vs `diff patient diff study`.",
        "- It is **not** a direct vessel-aware measurement because vessel/view labels are unavailable.",
        "",
        "## Pair/Data Summary",
        f"- Number of images: {n_images:,}",
        f"- Unique patients: {n_patients:,}",
        f"- Unique patient-study groups: {n_patient_study_groups:,}",
        f"- Positive pairs: {n_positive_pairs:,}",
        f"- Negative pairs: {n_negative_pairs:,}",
        "",
        "## Backbone Statistics",
        "",
        "| Backbone | Positive mean | Positive std | Negative mean | Negative std | Gap | AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| ImageNet | {format_float(float(row_img['positive_mean']))} | "
            f"{format_float(float(row_img['positive_std']))} | "
            f"{format_float(float(row_img['negative_mean']))} | "
            f"{format_float(float(row_img['negative_std']))} | "
            f"{format_float(gap_img)} | "
            f"{format_float(float(row_img['auc_optional']))} |"
        ),
        (
            f"| CAG | {format_float(float(row_cag['positive_mean']))} | "
            f"{format_float(float(row_cag['positive_std']))} | "
            f"{format_float(float(row_cag['negative_mean']))} | "
            f"{format_float(float(row_cag['negative_std']))} | "
            f"{format_float(gap_cag)} | "
            f"{format_float(float(row_cag['auc_optional']))} |"
        ),
        "",
        "## Interpretation",
        f"- Gap difference (CAG - ImageNet): {delta:.6f}",
        f"- {interpretation}",
        "- Even if gaps are similar, this does not mean CAG pretraining is globally ineffective.",
        "- Follow-up analyses should include local-detail and vessel-aware axes when annotations become available.",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anatomy Identity Proxy analysis")
    parser.add_argument("--image-root", default="input/Stent-Contrast")
    parser.add_argument("--imagenet-ckpt", default=DEFAULT_IMAGENET_CKPT)
    parser.add_argument("--cag-ckpt", default=DEFAULT_CAG_CKPT)
    parser.add_argument("--output-root", default="outputs/6_anatomy_identity_proxy")

    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--cache-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable feature cache reuse/save.",
    )
    parser.add_argument("--resize-size", type=int, default=480)
    parser.add_argument("--center-crop-size", type=int, default=448)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--log-file",
        default=None,
        help=(
            "Optional log file path. If relative, it is resolved under --output-root. "
            "If omitted, a timestamped log file is created in --output-root."
        ),
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=["png", "jpg", "jpeg"],
        help="Image extensions to scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.output_root).resolve()
    ensure_dir(out_root)
    log_path: Path | None = None
    log_handle: TextIO | None = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        log_path, log_handle, original_stdout, original_stderr = setup_console_and_file_logging(
            output_root=out_root,
            log_file_arg=args.log_file,
        )
        log(f"Run args: {vars(args)}")
        tracker = ProgressTracker(total_steps=8)

        tracker.start_step("Build image manifest")
        manifest = build_manifest(
            image_root=args.image_root,
            exts=tuple(args.image_exts),
            max_images=args.max_images,
            seed=args.seed,
        )
        manifest_path = out_root / "image_manifest.csv"
        manifest.to_csv(manifest_path, index=False)
        log(f"Saved image manifest: {manifest_path}")

        n_images = len(manifest)
        n_patients = manifest["patient_id"].nunique()
        n_patient_study_groups = manifest.groupby(["patient_id", "study_id"]).ngroups
        log(
            f"Manifest summary | images={n_images:,} patients={n_patients:,} "
            f"patient_study_groups={n_patient_study_groups:,}"
        )
        tracker.finish_step("Build image manifest")

        tracker.start_step("Build/load pair index")
        pair_df = build_or_load_pair_index(
            manifest=manifest,
            out_root=out_root,
            negative_ratio=args.negative_ratio,
            seed=args.seed,
        )
        n_positive_pairs = int((pair_df["pair_type"] == "positive").sum())
        n_negative_pairs = int((pair_df["pair_type"] == "negative").sum())
        log(
            f"Pair index ready | positives={n_positive_pairs:,} negatives={n_negative_pairs:,}"
        )
        tracker.finish_step("Build/load pair index")

        ckpt_imagenet = str(Path(args.imagenet_ckpt).expanduser().resolve())
        ckpt_cag = str(Path(args.cag_ckpt).expanduser().resolve())

        tracker.start_step("Extract/load ImageNet features")
        features_imagenet = extract_or_load_features(
            manifest=manifest,
            backbone_name="imagenet",
            ckpt_path=ckpt_imagenet,
            args=args,
            out_root=out_root,
        )
        tracker.finish_step("Extract/load ImageNet features")

        tracker.start_step("Extract/load CAG features")
        features_cag = extract_or_load_features(
            manifest=manifest,
            backbone_name="cag",
            ckpt_path=ckpt_cag,
            args=args,
            out_root=out_root,
        )
        tracker.finish_step("Extract/load CAG features")

        tracker.start_step("Compute similarities and save pair CSVs")
        log("Computing pair-level cosine similarities...")
        sims_imagenet = compute_pair_similarity(pair_df=pair_df, features=features_imagenet)
        sims_cag = compute_pair_similarity(pair_df=pair_df, features=features_cag)

        pairs_imagenet_df = build_pair_level_dataframe(
            pair_df=pair_df, manifest=manifest, similarities=sims_imagenet
        )
        pairs_cag_df = build_pair_level_dataframe(
            pair_df=pair_df, manifest=manifest, similarities=sims_cag
        )

        pairs_imagenet_path = out_root / "pairs_similarity_imagenet.csv"
        pairs_cag_path = out_root / "pairs_similarity_cag.csv"
        pairs_imagenet_df.to_csv(pairs_imagenet_path, index=False)
        pairs_cag_df.to_csv(pairs_cag_path, index=False)
        log(f"Saved pair similarities: {pairs_imagenet_path}")
        log(f"Saved pair similarities: {pairs_cag_path}")
        tracker.finish_step("Compute similarities and save pair CSVs")

        tracker.start_step("Aggregate summary CSV")
        stats_imagenet = summarize_similarity(
            backbone_name="imagenet", pair_df=pair_df, similarities=sims_imagenet
        )
        stats_cag = summarize_similarity(
            backbone_name="cag", pair_df=pair_df, similarities=sims_cag
        )

        summary_df = pd.DataFrame([stats_imagenet.to_row(), stats_cag.to_row()])
        summary_csv = out_root / "summary_anatomy_identity_proxy.csv"
        summary_df.to_csv(summary_csv, index=False)
        log(f"Saved summary: {summary_csv}")
        tracker.finish_step("Aggregate summary CSV")

        tracker.start_step("Render figures")
        fig_hist_imagenet = out_root / "fig_similarity_hist_imagenet.png"
        fig_hist_cag = out_root / "fig_similarity_hist_cag.png"
        fig_box_compare = out_root / "fig_similarity_boxplot_compare.png"
        fig_gap_compare = out_root / "fig_gap_compare.png"
        hist_xlim = compute_shared_hist_xlim(
            imagenet_df=pairs_imagenet_df,
            cag_df=pairs_cag_df,
        )
        log(f"Using shared histogram x-axis range: [{hist_xlim[0]:.4f}, {hist_xlim[1]:.4f}]")

        save_histogram(
            similarities_df=pairs_imagenet_df,
            backbone_label="ImageNet",
            output_path=fig_hist_imagenet,
            xlim=hist_xlim,
        )
        save_histogram(
            similarities_df=pairs_cag_df,
            backbone_label="CAG",
            output_path=fig_hist_cag,
            xlim=hist_xlim,
        )
        save_boxplot_compare(
            imagenet_df=pairs_imagenet_df,
            cag_df=pairs_cag_df,
            output_path=fig_box_compare,
        )
        save_gap_barplot(summary_df=summary_df, output_path=fig_gap_compare)
        log(f"Saved figure: {fig_hist_imagenet}")
        log(f"Saved figure: {fig_hist_cag}")
        log(f"Saved figure: {fig_box_compare}")
        log(f"Saved figure: {fig_gap_compare}")
        tracker.finish_step("Render figures")

        tracker.start_step("Write markdown summary")
        md_path = out_root / "analysis_anatomy_identity_proxy.md"
        write_markdown_summary(
            output_path=md_path,
            summary_df=summary_df,
            n_images=n_images,
            n_patients=n_patients,
            n_patient_study_groups=n_patient_study_groups,
            n_positive_pairs=n_positive_pairs,
            n_negative_pairs=n_negative_pairs,
        )
        log(f"Saved markdown summary: {md_path}")
        tracker.finish_step("Write markdown summary")

        log(f"Done. Log file: {log_path}")
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
