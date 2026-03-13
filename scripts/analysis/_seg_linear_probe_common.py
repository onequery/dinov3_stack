#!/usr/bin/env python3
"""
Shared helpers for Local Analysis 2 / 2-1 segmentation linear probes.

Strict linear probe definition:
- frozen normalized patch tokens
- 1x1 conv classifier
- bilinear upsampling to image resolution
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, TextIO, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.img_cls.model import Dinov3Backbone  # noqa: E402
from src.img_seg.datasets import SegmentationDataset, collate_fn, valid_transforms  # noqa: E402
from src.img_seg.utils import get_label_mask, set_class_values  # noqa: E402


DEFAULT_IMAGENET_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
DEFAULT_CAG_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}


@dataclass
class SplitMetrics:
    backbone_name: str
    split: str
    best_lr: float
    epochs_trained: int
    num_images: int
    pixel_acc: float
    miou: float
    dice: float
    per_class_iou: List[float]
    per_class_dice: List[float]
    probe_params: int

    def to_row(self) -> Dict[str, object]:
        return {
            "backbone_name": self.backbone_name,
            "split": self.split,
            "best_lr": self.best_lr,
            "epochs_trained": self.epochs_trained,
            "num_images": self.num_images,
            "pixel_acc": self.pixel_acc,
            "miou": self.miou,
            "dice": self.dice,
            "per_class_iou": json.dumps(self.per_class_iou),
            "per_class_dice": json.dumps(self.per_class_dice),
            "probe_params": self.probe_params,
        }


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
        return any(hasattr(stream, "isatty") and stream.isatty() for stream in self.streams)


class ProgressTracker:
    def __init__(self, total_steps: int):
        self.total_steps = max(1, int(total_steps))
        self.completed_steps = 0
        self.start_time = time.time()
        self.current_step_name = ""
        self.current_step_started_at = self.start_time
        self.completed_step_history: List[Dict[str, float | str]] = []

    def start_step(self, name: str) -> None:
        self.current_step_name = str(name)
        self.current_step_started_at = time.time()
        current = min(self.total_steps, self.completed_steps + 1)
        log(f"[ETA][FULL-RUN] Step {current}/{self.total_steps} START | {name}")

    def finish_step(self, name: str) -> None:
        step_duration = max(0.0, time.time() - self.current_step_started_at)
        self.completed_step_history.append({"name": str(name), "seconds": float(step_duration)})
        self.completed_steps = min(self.total_steps, self.completed_steps + 1)
        elapsed = max(0.0, time.time() - self.start_time)
        progress = 100.0 * self.completed_steps / self.total_steps
        avg_step = elapsed / max(1, self.completed_steps)
        remaining = max(0.0, avg_step * (self.total_steps - self.completed_steps))
        eta_dt = datetime.now() + timedelta(seconds=remaining)
        log(
            f"[ETA][FULL-RUN] Step {self.completed_steps}/{self.total_steps} DONE | {name} | "
            f"progress={progress:.1f}% | elapsed={format_duration(elapsed)} | "
            f"remaining~{format_duration(remaining)} | eta={eta_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.current_step_name = ""

    def current_step_elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.current_step_started_at)

    def average_completed_short_step_seconds(self, default_seconds: float = 60.0) -> float:
        short_durations = [
            float(item["seconds"])
            for item in self.completed_step_history
            if not str(item["name"]).startswith("Run ")
        ]
        if short_durations:
            return sum(short_durations) / len(short_durations)
        return float(default_seconds)

    def live_eta_from_remaining(self, remaining_seconds: float, note: str = "") -> None:
        elapsed = max(0.0, time.time() - self.start_time)
        remaining = max(0.0, float(remaining_seconds))
        total_runtime = elapsed + remaining
        overall_progress = (elapsed / total_runtime) if total_runtime > 0.0 else 0.0
        eta_dt = datetime.now() + timedelta(seconds=remaining)
        current_step = self.current_step_name or "unknown"
        suffix = f" | {note}" if note else ""
        log(
            f"[ETA][FULL-RUN][LIVE] step={self.completed_steps + 1}/{self.total_steps} | "
            f"current_step={current_step} | overall_progress={overall_progress * 100.0:.1f}% | "
            f"elapsed={format_duration(elapsed)} | "
            f"remaining~{format_duration(remaining)} | eta={eta_dt.strftime('%Y-%m-%d %H:%M:%S')}{suffix}"
        )


class EpochETATracker:
    def __init__(self, backbone_name: str, lr_values: Sequence[float], max_epoch: int):
        self.backbone_name = backbone_name
        self.lr_values = [float(lr) for lr in lr_values]
        self.max_epoch = max(1, int(max_epoch))
        self.total_epoch_budget = max(1, len(self.lr_values) * self.max_epoch)
        self.completed_epochs = 0
        self.global_start = time.time()
        self.current_lr = None
        self.current_lr_index = 0
        self.current_lr_start = 0.0
        self.current_lr_epoch_times: List[float] = []

    def start_lr(self, lr: float, candidate_index: int) -> None:
        self.current_lr = float(lr)
        self.current_lr_index = int(candidate_index)
        self.current_lr_start = time.time()
        self.current_lr_epoch_times = []
        log(
            f"[ETA][TRAIN-SUBSTEP][{self.backbone_name}] LR candidate "
            f"{self.current_lr_index}/{len(self.lr_values)} START | "
            f"lr={self.current_lr}"
        )

    def finish_epoch(self, epoch: int, epoch_seconds: float) -> None:
        self.completed_epochs += 1
        self.current_lr_epoch_times.append(float(epoch_seconds))
        avg_epoch_this_lr = sum(self.current_lr_epoch_times) / max(1, len(self.current_lr_epoch_times))
        remaining_this_lr = max(0.0, (self.max_epoch - int(epoch)) * avg_epoch_this_lr)

        total_elapsed = max(0.0, time.time() - self.global_start)
        avg_epoch_total = total_elapsed / max(1, self.completed_epochs)
        remaining_total = max(0.0, (self.total_epoch_budget - self.completed_epochs) * avg_epoch_total)
        eta_dt = datetime.now() + timedelta(seconds=remaining_total)
        log(
            f"[ETA][TRAIN-SUBSTEP][{self.backbone_name}] lr={self.current_lr} "
            f"epoch={epoch}/{self.max_epoch} | "
            f"epoch_time={format_duration(epoch_seconds)} | avg_epoch={format_duration(avg_epoch_this_lr)} | "
            f"remaining_this_lr~{format_duration(remaining_this_lr)} | "
            f"remaining_substep_total~{format_duration(remaining_total)} | "
            f"eta={eta_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def skip_remaining_epochs(self, skipped_epochs: int) -> None:
        skipped = max(0, int(skipped_epochs))
        if skipped <= 0:
            return
        self.total_epoch_budget = max(self.completed_epochs, self.total_epoch_budget - skipped)

    def total_progress(self) -> float:
        return min(1.0, max(0.0, self.completed_epochs / max(1, self.total_epoch_budget)))

    def remaining_total_seconds(self) -> float:
        total_elapsed = max(0.0, time.time() - self.global_start)
        avg_epoch_total = total_elapsed / max(1, self.completed_epochs)
        return max(0.0, (self.total_epoch_budget - self.completed_epochs) * avg_epoch_total)


class MaskOnlyDataset(Dataset):
    def __init__(
        self,
        mask_paths: Sequence[str],
        img_size: Tuple[int, int],
        label_colors_list: List[List[int]],
        all_classes: List[str],
    ):
        self.mask_paths = list(mask_paths)
        self.width = int(img_size[0])
        self.height = int(img_size[1])
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.class_values = set_class_values(self.all_classes, self.all_classes)

    def __len__(self) -> int:
        return len(self.mask_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype(np.float32)
        if len(self.all_classes) == 2:
            fg = mask > 0
            mask[fg] = 255
            mask[np.logical_not(fg)] = 0
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        label_mask = get_label_mask(
            mask.astype(np.uint8),
            self.class_values,
            self.label_colors_list,
        ).astype(np.uint8)
        return torch.from_numpy(label_mask)


class CachedSegmentationDataset(Dataset):
    def __init__(self, features: torch.Tensor, targets: torch.Tensor):
        if len(features) != len(targets):
            raise ValueError(
                f"features/targets length mismatch: {len(features)} vs {len(targets)}"
            )
        self.features = features
        self.targets = targets

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


class StrictLinearSegProbe(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, output_size: Tuple[int, int]):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=True)
        self.output_size = (int(output_size[1]), int(output_size[0]))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(features)
        return F.interpolate(
            logits,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )


class SegmentationMetricsAccumulator:
    def __init__(self, num_classes: int):
        self.num_classes = int(num_classes)
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def add(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_np = pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
        target_np = target.detach().cpu().numpy().astype(np.int64).reshape(-1)
        valid = (target_np >= 0) & (target_np < self.num_classes)
        indices = self.num_classes * target_np[valid] + pred_np[valid]
        cm = np.bincount(indices, minlength=self.num_classes * self.num_classes).reshape(
            self.num_classes, self.num_classes
        )
        self.confusion += cm

    def compute(self) -> Dict[str, object]:
        return metrics_from_confusion(self.confusion)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_console_and_file_logging(
    output_root: Path,
    log_file_arg: str | None,
    default_prefix: str = "local_2_segmentation_linear_probe",
) -> Tuple[Path, TextIO, TextIO, TextIO]:
    if log_file_arg:
        log_path = Path(log_file_arg).expanduser()
        if not log_path.is_absolute():
            log_path = output_root / log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_root / f"{default_prefix}_{stamp}.log"

    ensure_dir(log_path.parent)
    file_handle = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, file_handle)
    sys.stderr = TeeStream(original_stderr, file_handle)
    log(f"Console output is mirrored to log file: {log_path}")
    return log_path, file_handle, original_stdout, original_stderr


def hash_records(records: Iterable[Sequence[object]]) -> str:
    hasher = hashlib.sha256()
    for row in records:
        hasher.update("\t".join(str(v) for v in row).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> str:
    return hash_records(df.loc[:, list(columns)].itertuples(index=False, name=None))


def collect_paths(root_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
    return sorted(
        [
            path
            for path in paths
            if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS
        ]
    )


def pair_image_mask_paths(
    image_paths: List[str],
    mask_paths: List[str],
    image_root: str,
    mask_root: str,
) -> Tuple[List[str], List[str]]:
    image_map = {os.path.relpath(path, image_root): path for path in image_paths}
    mask_map = {os.path.relpath(path, mask_root): path for path in mask_paths}
    shared = sorted(set(image_map.keys()) & set(mask_map.keys()))
    if not shared:
        raise ValueError(
            "No paired image/mask files found. "
            f"images_root={image_root}, masks_root={mask_root}"
        )
    return [image_map[key] for key in shared], [mask_map[key] for key in shared]


def maybe_subsample_pairs(
    image_paths: List[str],
    mask_paths: List[str],
    max_images: int | None,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if max_images is None or max_images <= 0 or max_images >= len(image_paths):
        return image_paths, mask_paths
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(image_paths), size=max_images, replace=False))
    return [image_paths[i] for i in chosen], [mask_paths[i] for i in chosen]


def load_seg_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Segmentation config must be a mapping.")
    return config


def build_or_load_manifest(
    split_name: str,
    image_paths: List[str],
    mask_paths: List[str],
    img_size: Tuple[int, int],
    patch_size: int,
    out_root: Path,
) -> pd.DataFrame:
    manifest_path = out_root / f"image_manifest_{split_name}.csv"
    meta_path = out_root / f"image_manifest_{split_name}.meta.json"
    pair_signature = hash_records(zip(image_paths, mask_paths))
    desired_meta = {
        "split": split_name,
        "pair_signature": pair_signature,
        "img_size": [int(img_size[0]), int(img_size[1])],
        "patch_size": int(patch_size),
        "n_images": int(len(image_paths)),
    }
    if manifest_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        if saved_meta == desired_meta:
            log(f"Reusing manifest: {manifest_path}")
            return pd.read_csv(manifest_path)

    df = pd.DataFrame(
        {
            "image_id": np.arange(len(image_paths), dtype=np.int64),
            "split": split_name,
            "image_path": image_paths,
            "mask_path": mask_paths,
        }
    )
    df.to_csv(manifest_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(desired_meta, f, indent=2)
    return df


def build_seg_dataset(
    manifest: pd.DataFrame,
    img_size: Tuple[int, int],
    all_classes: List[str],
    label_colors_list: List[List[int]],
) -> SegmentationDataset:
    return SegmentationDataset(
        image_paths=manifest["image_path"].tolist(),
        mask_paths=manifest["mask_path"].tolist(),
        tfms=valid_transforms(list(img_size)),
        label_colors_list=label_colors_list,
        classes_to_train=all_classes,
        all_classes=all_classes,
    )


def build_feature_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )


def build_mask_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=False,
    )


def extract_or_load_targets(
    split_name: str,
    manifest: pd.DataFrame,
    img_size: Tuple[int, int],
    label_colors_list: List[List[int]],
    all_classes: List[str],
    batch_size: int,
    num_workers: int,
    out_root: Path,
) -> torch.Tensor:
    targets_path = out_root / f"targets_{split_name}.pt"
    meta_path = out_root / f"targets_{split_name}.meta.json"
    manifest_hash = hash_dataframe(manifest, ["image_id", "image_path", "mask_path"])
    desired_meta = {
        "manifest_hash": manifest_hash,
        "img_size": [int(img_size[0]), int(img_size[1])],
        "n_images": int(len(manifest)),
    }
    if targets_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        if saved_meta == desired_meta:
            log(f"Reusing target cache: {targets_path}")
            cached = torch.load(targets_path, map_location="cpu")
            if isinstance(cached, torch.Tensor):
                return cached
            raise ValueError(f"Invalid target cache format: {targets_path}")

    dataset = MaskOnlyDataset(
        mask_paths=manifest["mask_path"].tolist(),
        img_size=img_size,
        label_colors_list=label_colors_list,
        all_classes=all_classes,
    )
    loader = build_mask_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    targets = torch.empty(
        (len(dataset), int(img_size[1]), int(img_size[0])),
        dtype=torch.uint8,
    )
    offset = 0
    for batch in tqdm(loader, desc=f"Targets-{split_name}"):
        batch = batch.to(dtype=torch.uint8)
        batch_size_actual = int(batch.shape[0])
        targets[offset : offset + batch_size_actual] = batch
        offset += batch_size_actual
    torch.save(targets, targets_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(desired_meta, f, indent=2)
    return targets


def resolve_device(device_arg: str | None) -> torch.device:
    requested = device_arg or ("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA requested but unavailable. Falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def extract_or_load_features(
    split_name: str,
    backbone_name: str,
    ckpt_path: str,
    manifest: pd.DataFrame,
    all_classes: List[str],
    label_colors_list: List[List[int]],
    args: argparse.Namespace,
    out_root: Path,
) -> torch.Tensor:
    features_path = out_root / f"features_{backbone_name}_{split_name}.pt"
    meta_path = out_root / f"features_{backbone_name}_{split_name}.meta.json"
    manifest_hash = hash_dataframe(manifest, ["image_id", "image_path", "mask_path"])
    grid_w = int(args.img_size[0]) // int(args.patch_size)
    grid_h = int(args.img_size[1]) // int(args.patch_size)
    desired_meta = {
        "manifest_hash": manifest_hash,
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "model_name": args.model_name,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "img_size": [int(args.img_size[0]), int(args.img_size[1])],
        "patch_size": int(args.patch_size),
        "grid_size": [grid_h, grid_w],
        "n_images": int(len(manifest)),
    }
    if args.cache_features and features_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        if saved_meta == desired_meta:
            log(f"Reusing feature cache: {features_path}")
            cached = torch.load(features_path, map_location="cpu")
            if isinstance(cached, torch.Tensor):
                return cached
            raise ValueError(f"Invalid feature cache format: {features_path}")

    device = resolve_device(args.device)
    wrapper = Dinov3Backbone(
        weights=ckpt_path,
        model_name=args.model_name,
        repo_dir=args.repo_dir,
    )
    backbone = wrapper.backbone_model.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    dataset = build_seg_dataset(
        manifest=manifest,
        img_size=(int(args.img_size[0]), int(args.img_size[1])),
        all_classes=all_classes,
        label_colors_list=label_colors_list,
    )
    loader = build_feature_loader(
        dataset=dataset,
        batch_size=int(args.feature_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    embed_dim = backbone.norm.normalized_shape[0]
    features = torch.empty((len(dataset), embed_dim, grid_h, grid_w), dtype=torch.float16)

    log(
        f"Extracting features ({backbone_name}, {split_name}) | "
        f"n_images={len(dataset):,} | grid={grid_h}x{grid_w} | device={device}"
    )
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract-{backbone_name}-{split_name}"):
            images = batch[0].to(device, non_blocking=(device.type == "cuda"))
            outputs = backbone.forward_features(images)
            patch_tokens = outputs["x_norm_patchtokens"]
            if patch_tokens.ndim != 3:
                raise ValueError(
                    f"Expected patch tokens [B, N, C], got {tuple(patch_tokens.shape)}"
                )
            batch_size_actual, token_count, channels = patch_tokens.shape
            if token_count != grid_h * grid_w:
                raise ValueError(
                    f"Patch token count mismatch: tokens={token_count}, grid={grid_h}x{grid_w}"
                )
            patch_tokens = (
                patch_tokens.reshape(batch_size_actual, grid_h, grid_w, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
                .cpu()
                .to(torch.float16)
            )
            features[offset : offset + batch_size_actual] = patch_tokens
            offset += batch_size_actual

    if args.cache_features:
        torch.save(features, features_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(desired_meta, f, indent=2)

    del backbone
    del wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features


def metrics_from_confusion(confusion: np.ndarray) -> Dict[str, object]:
    total = confusion.sum()
    diag = np.diag(confusion).astype(np.float64)
    gt_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = gt_count + pred_count - diag
    dice_denom = gt_count + pred_count
    per_class_iou = np.divide(
        diag,
        union,
        out=np.full_like(diag, np.nan, dtype=np.float64),
        where=union > 0,
    )
    per_class_dice = np.divide(
        2.0 * diag,
        dice_denom,
        out=np.full_like(diag, np.nan, dtype=np.float64),
        where=dice_denom > 0,
    )
    pixel_acc = float(diag.sum() / total) if total > 0 else float("nan")
    miou = float(np.nanmean(per_class_iou))
    dice = float(np.nanmean(per_class_dice))
    return {
        "pixel_acc": pixel_acc,
        "miou": miou,
        "dice": dice,
        "per_class_iou": per_class_iou.tolist(),
        "per_class_dice": per_class_dice.tolist(),
    }


def build_probe_loader(
    features: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    pin_memory: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = CachedSegmentationDataset(features=features, targets=targets)
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
        generator=generator,
    )


def set_random_seed(seed: int, strict_deterministic: bool = False) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if strict_deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def run_probe_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    is_train: bool,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
    desc: str,
) -> Dict[str, object]:
    if is_train:
        model.train()
    else:
        model.eval()
    meter = SegmentationMetricsAccumulator(num_classes=num_classes)
    running_loss = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for features, targets in tqdm(loader, desc=desc):
            features = features.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
            targets = targets.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda"))
            if is_train:
                assert optimizer is not None
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()
            running_loss += float(loss.item()) * int(features.shape[0])
            preds = logits.argmax(dim=1)
            meter.add(preds, targets)

    metrics = meter.compute()
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    return metrics


def clone_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def checkpoint_path_for_backbone(out_root: Path, backbone_name: str) -> Path:
    return out_root / f"probe_checkpoint_{backbone_name}.pt"


def resume_path_for_backbone(out_root: Path, backbone_name: str) -> Path:
    return out_root / f"probe_resume_{backbone_name}.pt"


def safe_torch_save(payload: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_torch_checkpoint_if_valid(path: Path, expected_signature: str) -> Dict[str, object] | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        return None
    if payload.get("signature") != expected_signature:
        return None
    return payload


def remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def build_probe_training_signature(
    backbone_name: str,
    backbone_ckpt_path: str,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_features: torch.Tensor,
    valid_targets: torch.Tensor,
    args: argparse.Namespace,
    deterministic_seed: int | None = None,
    strict_deterministic: bool = False,
) -> str:
    payload = {
        "backbone_name": backbone_name,
        "backbone_ckpt_path": str(Path(backbone_ckpt_path).expanduser().resolve()),
        "model_name": str(args.model_name),
        "repo_dir": str(Path(args.repo_dir).expanduser().resolve()),
        "img_size": [int(args.img_size[0]), int(args.img_size[1])],
        "patch_size": int(args.patch_size),
        "lr_grid": [float(x) for x in args.lr_grid],
        "max_epoch": int(args.max_epoch),
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "train_features_shape": list(train_features.shape),
        "train_features_dtype": str(train_features.dtype),
        "train_targets_shape": list(train_targets.shape),
        "train_targets_dtype": str(train_targets.dtype),
        "valid_features_shape": list(valid_features.shape),
        "valid_features_dtype": str(valid_features.dtype),
        "valid_targets_shape": list(valid_targets.shape),
        "valid_targets_dtype": str(valid_targets.dtype),
        "deterministic_seed": None if deterministic_seed is None else int(deterministic_seed),
        "strict_deterministic": bool(strict_deterministic),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def instantiate_probe_model(
    in_channels: int,
    output_size: Tuple[int, int],
    device: torch.device,
    num_classes: int = 2,
) -> StrictLinearSegProbe:
    return StrictLinearSegProbe(
        in_channels=in_channels,
        num_classes=num_classes,
        output_size=output_size,
    ).to(device)


def select_best_candidate(
    current_score: float,
    current_lr: float,
    best_score: float,
    best_lr: float | None,
    atol: float = 1e-12,
) -> bool:
    if current_score > best_score + atol:
        return True
    if abs(current_score - best_score) <= atol and (best_lr is None or current_lr < best_lr):
        return True
    return False


def train_probe_with_lr_search(
    backbone_name: str,
    backbone_ckpt_path: str,
    train_features: torch.Tensor,
    train_targets: torch.Tensor,
    valid_features: torch.Tensor,
    valid_targets: torch.Tensor,
    args: argparse.Namespace,
    out_root: Path,
    full_run_tracker: ProgressTracker | None = None,
    future_training_estimator: Callable[[float], float | Tuple[float, str]] | None = None,
    deterministic_seed: int | None = None,
    strict_deterministic: bool = False,
) -> Tuple[nn.Module, float, int, List[Dict[str, object]], Dict[str, object]]:
    device = resolve_device(args.device)
    num_classes = 2
    in_channels = int(train_features.shape[1])
    output_size = (int(args.img_size[0]), int(args.img_size[1]))
    probe_params = in_channels * num_classes + num_classes
    pin_memory = device.type == "cuda"
    criterion = nn.CrossEntropyLoss()

    valid_loader = build_probe_loader(
        features=valid_features,
        targets=valid_targets,
        batch_size=int(args.probe_batch_size),
        shuffle=False,
        pin_memory=pin_memory,
    )

    signature = build_probe_training_signature(
        backbone_name=backbone_name,
        backbone_ckpt_path=backbone_ckpt_path,
        train_features=train_features,
        train_targets=train_targets,
        valid_features=valid_features,
        valid_targets=valid_targets,
        args=args,
        deterministic_seed=deterministic_seed,
        strict_deterministic=strict_deterministic,
    )
    final_ckpt_path = checkpoint_path_for_backbone(out_root, backbone_name)
    resume_ckpt_path = resume_path_for_backbone(out_root, backbone_name)
    history_path = out_root / f"history_{backbone_name}.json"

    final_payload = load_torch_checkpoint_if_valid(final_ckpt_path, signature)
    if final_payload is not None:
        log(f"[{backbone_name}] Reusing final trained probe checkpoint: {final_ckpt_path}")
        best_model = instantiate_probe_model(
            in_channels=in_channels,
            output_size=output_size,
            device=device,
            num_classes=num_classes,
        )
        best_model.load_state_dict(final_payload["best_state"], strict=True)
        return (
            best_model,
            float(final_payload["best_lr"]),
            int(final_payload["best_epochs_trained"]),
            list(final_payload["best_history"]),
            dict(final_payload["lr_search_results"]),
        )

    resume_payload = load_torch_checkpoint_if_valid(resume_ckpt_path, signature)
    best_lr: float | None = None
    best_epochs_trained = 0
    best_history: List[Dict[str, object]] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_valid_score = -float("inf")
    lr_search_results: Dict[str, object] = {"probe_params": probe_params, "candidates": {}}
    eta_tracker = EpochETATracker(
        backbone_name=backbone_name,
        lr_values=[float(x) for x in args.lr_grid],
        max_epoch=int(args.max_epoch),
    )
    if resume_payload is not None:
        lr_search_results = dict(resume_payload.get("lr_search_results", lr_search_results))
        best_lr_value = resume_payload.get("best_lr")
        best_lr = float(best_lr_value) if best_lr_value is not None else None
        best_epochs_trained = int(resume_payload.get("best_epochs_trained", 0))
        best_history = list(resume_payload.get("best_history", []))
        best_state_value = resume_payload.get("best_state")
        best_state = best_state_value if isinstance(best_state_value, dict) else None
        best_valid_score = float(resume_payload.get("best_valid_score", best_valid_score))
        eta_state = resume_payload.get("eta_state", {})
        if isinstance(eta_state, dict):
            eta_tracker.completed_epochs = int(eta_state.get("completed_epochs", eta_tracker.completed_epochs))
            eta_tracker.total_epoch_budget = int(
                eta_state.get("total_epoch_budget", eta_tracker.total_epoch_budget)
            )
            elapsed_seconds = float(eta_state.get("elapsed_seconds", 0.0))
            if elapsed_seconds > 0.0:
                eta_tracker.global_start = time.time() - elapsed_seconds

    lr_values = [float(x) for x in args.lr_grid]
    start_lr_index = 1
    active_resume_state: Dict[str, object] | None = None
    if resume_payload is not None:
        stage = str(resume_payload.get("stage", ""))
        if stage == "between_candidates":
            start_lr_index = int(resume_payload.get("next_lr_index", 1))
            log(
                f"[{backbone_name}] Resuming after completed candidate(s) | "
                f"next_lr_index={start_lr_index}"
            )
        elif stage == "epoch":
            start_lr_index = int(resume_payload.get("current_lr_index", 1))
            active_resume_state = resume_payload
            log(
                f"[{backbone_name}] Resuming mid-candidate | "
                f"lr_index={start_lr_index} epoch={int(resume_payload.get('current_epoch', 0)) + 1}"
            )

    for lr_index, lr in enumerate(lr_values, start=1):
        if lr_index < start_lr_index:
            continue
        log(f"[{backbone_name}] LR search start | lr={lr}")
        eta_tracker.start_lr(lr=lr, candidate_index=lr_index)
        candidate_seed = None if deterministic_seed is None else int(deterministic_seed) + int(lr_index) * 1000
        if candidate_seed is not None:
            set_random_seed(candidate_seed, strict_deterministic=strict_deterministic)
        model = instantiate_probe_model(
            in_channels=in_channels,
            output_size=output_size,
            device=device,
            num_classes=num_classes,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        best_state_for_lr = clone_state_dict(model)
        best_metric_for_lr = -float("inf")
        best_epoch_for_lr = 0
        epochs_without_improve = 0
        history_for_lr: List[Dict[str, object]] = []
        start_epoch = 1

        if active_resume_state is not None and lr_index == start_lr_index:
            resume_lr = float(active_resume_state.get("current_lr", lr))
            if abs(resume_lr - float(lr)) > 1e-12:
                raise ValueError(
                    f"Resume lr mismatch for {backbone_name}: expected {lr}, got {resume_lr}"
                )
            model.load_state_dict(active_resume_state["model_state"], strict=True)
            optimizer.load_state_dict(active_resume_state["optimizer_state"])
            move_optimizer_state_to_device(optimizer, device)
            best_state_for_lr = active_resume_state["best_state_for_lr"]
            best_metric_for_lr = float(active_resume_state.get("best_metric_for_lr", best_metric_for_lr))
            best_epoch_for_lr = int(active_resume_state.get("best_epoch_for_lr", 0))
            epochs_without_improve = int(active_resume_state.get("epochs_without_improve", 0))
            history_for_lr = list(active_resume_state.get("history_for_lr", []))
            start_epoch = int(active_resume_state.get("current_epoch", 0)) + 1
            eta_lr_state = active_resume_state.get("eta_lr_state", {})
            if isinstance(eta_lr_state, dict):
                eta_tracker.current_lr_epoch_times = list(
                    map(float, eta_lr_state.get("current_lr_epoch_times", []))
                )
            log(
                f"[{backbone_name}] Loaded resume checkpoint | lr={lr} | "
                f"start_epoch={start_epoch}"
            )

        for epoch in range(start_epoch, int(args.max_epoch) + 1):
            epoch_start = time.time()
            train_generator = None
            if candidate_seed is not None:
                train_generator = torch.Generator()
                train_generator.manual_seed(candidate_seed + int(epoch))
            train_loader = build_probe_loader(
                features=train_features,
                targets=train_targets,
                batch_size=int(args.probe_batch_size),
                shuffle=True,
                pin_memory=pin_memory,
                generator=train_generator,
            )
            train_metrics = run_probe_epoch(
                model=model,
                loader=train_loader,
                device=device,
                criterion=criterion,
                is_train=True,
                optimizer=optimizer,
                num_classes=num_classes,
                desc=f"Train-{backbone_name}-lr{lr:g}-e{epoch}",
            )
            valid_metrics = run_probe_epoch(
                model=model,
                loader=valid_loader,
                device=device,
                criterion=criterion,
                is_train=False,
                optimizer=None,
                num_classes=num_classes,
                desc=f"Valid-{backbone_name}-lr{lr:g}-e{epoch}",
            )
            history_for_lr.append(
                {
                    "epoch": epoch,
                    "lr": lr,
                    "train_loss": float(train_metrics["loss"]),
                    "train_miou": float(train_metrics["miou"]),
                    "train_dice": float(train_metrics["dice"]),
                    "valid_loss": float(valid_metrics["loss"]),
                    "valid_miou": float(valid_metrics["miou"]),
                    "valid_dice": float(valid_metrics["dice"]),
                }
            )
            log(
                f"[{backbone_name}] lr={lr} epoch={epoch} | "
                f"train_loss={train_metrics['loss']:.6f} train_mIoU={train_metrics['miou']:.6f} "
                f"valid_loss={valid_metrics['loss']:.6f} valid_mIoU={valid_metrics['miou']:.6f} "
                f"valid_dice={valid_metrics['dice']:.6f}"
            )
            eta_tracker.finish_epoch(epoch=epoch, epoch_seconds=(time.time() - epoch_start))
            if full_run_tracker is not None:
                substep_remaining = eta_tracker.remaining_total_seconds()
                short_step_avg = full_run_tracker.average_completed_short_step_seconds(default_seconds=45.0)
                current_step_tail_overhead = max(short_step_avg, 60.0)
                future_short_steps = 4 * short_step_avg
                future_other_training_step = 0.0
                future_training_note = ""
                if future_training_estimator is not None:
                    estimate = future_training_estimator(substep_remaining)
                    if isinstance(estimate, tuple):
                        future_other_training_step = max(0.0, float(estimate[0]))
                        future_training_note = str(estimate[1]) if len(estimate) > 1 else ""
                    else:
                        future_other_training_step = max(0.0, float(estimate))
                elif backbone_name == "imagenet":
                    current_step_elapsed = full_run_tracker.current_step_elapsed_seconds()
                    future_other_training_step = max(
                        substep_remaining,
                        current_step_elapsed + substep_remaining,
                    )
                full_run_remaining = (
                    substep_remaining
                    + current_step_tail_overhead
                    + future_other_training_step
                    + future_short_steps
                )
                full_run_tracker.live_eta_from_remaining(
                    remaining_seconds=full_run_remaining,
                    note=(
                        f"backbone={backbone_name} | lr={lr} | epoch={epoch}/{int(args.max_epoch)} | "
                        f"substep_remaining~{format_duration(substep_remaining)} | "
                        f"future_training_est~{format_duration(future_other_training_step)}"
                        f"{' | ' + future_training_note if future_training_note else ''}"
                    ),
                )

            improved = valid_metrics["miou"] > (best_metric_for_lr + float(args.early_stopping_min_delta))
            if improved:
                best_metric_for_lr = float(valid_metrics["miou"])
                best_epoch_for_lr = epoch
                best_state_for_lr = clone_state_dict(model)
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            safe_torch_save(
                {
                    "signature": signature,
                    "stage": "epoch",
                    "current_lr_index": int(lr_index),
                    "current_lr": float(lr),
                    "current_epoch": int(epoch),
                    "model_state": clone_state_dict(model),
                    "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                    "best_state_for_lr": best_state_for_lr,
                    "best_metric_for_lr": float(best_metric_for_lr),
                    "best_epoch_for_lr": int(best_epoch_for_lr),
                    "epochs_without_improve": int(epochs_without_improve),
                    "history_for_lr": history_for_lr,
                    "lr_search_results": lr_search_results,
                    "best_lr": best_lr,
                    "best_epochs_trained": int(best_epochs_trained),
                    "best_history": best_history,
                    "best_state": best_state,
                    "best_valid_score": float(best_valid_score),
                    "probe_params": int(probe_params),
                    "eta_state": {
                        "completed_epochs": int(eta_tracker.completed_epochs),
                        "total_epoch_budget": int(eta_tracker.total_epoch_budget),
                        "elapsed_seconds": float(max(0.0, time.time() - eta_tracker.global_start)),
                    },
                    "eta_lr_state": {
                        "current_lr_epoch_times": list(map(float, eta_tracker.current_lr_epoch_times)),
                    },
                },
                resume_ckpt_path,
            )

            if epochs_without_improve >= int(args.early_stopping_patience):
                log(
                    f"[{backbone_name}] early stopping | lr={lr} "
                    f"patience={args.early_stopping_patience} best_epoch={best_epoch_for_lr}"
                )
                eta_tracker.skip_remaining_epochs(int(args.max_epoch) - epoch)
                break

        lr_search_results["candidates"][str(lr)] = {
            "best_valid_miou": float(best_metric_for_lr),
            "best_epoch": int(best_epoch_for_lr),
            "epochs_trained": int(len(history_for_lr)),
            "best_valid_dice": float(
                next(
                    item["valid_dice"]
                    for item in history_for_lr
                    if item["epoch"] == best_epoch_for_lr
                )
            )
            if best_epoch_for_lr > 0
            else float("nan"),
        }

        if select_best_candidate(
            current_score=float(best_metric_for_lr),
            current_lr=float(lr),
            best_score=best_valid_score,
            best_lr=best_lr,
        ):
            best_valid_score = float(best_metric_for_lr)
            best_lr = float(lr)
            best_epochs_trained = int(len(history_for_lr))
            best_history = list(history_for_lr)
            best_state = best_state_for_lr

        safe_torch_save(
            {
                "signature": signature,
                "stage": "between_candidates",
                "next_lr_index": int(lr_index + 1),
                "lr_search_results": lr_search_results,
                "best_lr": best_lr,
                "best_epochs_trained": int(best_epochs_trained),
                "best_history": best_history,
                "best_state": best_state,
                "best_valid_score": float(best_valid_score),
                "probe_params": int(probe_params),
                "eta_state": {
                    "completed_epochs": int(eta_tracker.completed_epochs),
                    "total_epoch_budget": int(eta_tracker.total_epoch_budget),
                    "elapsed_seconds": float(max(0.0, time.time() - eta_tracker.global_start)),
                },
            },
            resume_ckpt_path,
        )

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_lr is None or best_state is None:
        raise RuntimeError(f"Failed to train any probe candidate for {backbone_name}")

    best_model = instantiate_probe_model(
        in_channels=in_channels,
        output_size=output_size,
        device=device,
        num_classes=num_classes,
    )
    best_model.load_state_dict(best_state, strict=True)
    lr_search_results["best_lr"] = best_lr
    lr_search_results["best_valid_miou"] = best_valid_score
    lr_search_results["epochs_trained"] = best_epochs_trained
    lr_search_results["probe_params"] = probe_params

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(best_history, f, indent=2)
    log(f"Saved best history: {history_path}")

    safe_torch_save(
        {
            "signature": signature,
            "backbone_name": backbone_name,
            "best_lr": float(best_lr),
            "best_epochs_trained": int(best_epochs_trained),
            "best_history": best_history,
            "best_state": best_state,
            "best_valid_score": float(best_valid_score),
            "lr_search_results": lr_search_results,
            "probe_params": int(probe_params),
        },
        final_ckpt_path,
    )
    remove_file_if_exists(resume_ckpt_path)
    log(f"[{backbone_name}] Saved final probe checkpoint: {final_ckpt_path}")

    return best_model, best_lr, best_epochs_trained, best_history, lr_search_results


def evaluate_probe_split(
    model: nn.Module,
    backbone_name: str,
    split_name: str,
    features: torch.Tensor,
    targets: torch.Tensor,
    manifest: pd.DataFrame,
    best_lr: float,
    epochs_trained: int,
    class_names: List[str],
    device: torch.device,
    batch_size: int,
) -> Tuple[SplitMetrics, pd.DataFrame]:
    loader = build_probe_loader(
        features=features,
        targets=targets,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )
    criterion = nn.CrossEntropyLoss()
    model.eval()
    meter = SegmentationMetricsAccumulator(num_classes=len(class_names))
    rows: List[Dict[str, object]] = []
    offset = 0

    with torch.no_grad():
        for features_batch, targets_batch in tqdm(loader, desc=f"Eval-{backbone_name}-{split_name}"):
            features_batch = features_batch.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
            targets_batch = targets_batch.to(device=device, dtype=torch.long, non_blocking=(device.type == "cuda"))
            logits = model(features_batch)
            preds = logits.argmax(dim=1)
            meter.add(preds, targets_batch)
            preds_cpu = preds.cpu()
            targets_cpu = targets_batch.cpu()
            batch_size_actual = int(preds_cpu.shape[0])
            for idx in range(batch_size_actual):
                metrics = metrics_from_confusion(
                    confusion_from_tensors(
                        preds_cpu[idx],
                        targets_cpu[idx],
                        num_classes=len(class_names),
                    )
                )
                row = manifest.iloc[offset + idx]
                rows.append(
                    {
                        "image_id": int(row["image_id"]),
                        "split": split_name,
                        "image_path": row["image_path"],
                        "mask_path": row["mask_path"],
                        "pixel_acc": float(metrics["pixel_acc"]),
                        "miou": float(metrics["miou"]),
                        "dice": float(metrics["dice"]),
                        f"iou_{class_names[0]}": float(metrics["per_class_iou"][0]),
                        f"iou_{class_names[1]}": float(metrics["per_class_iou"][1]),
                        f"dice_{class_names[0]}": float(metrics["per_class_dice"][0]),
                        f"dice_{class_names[1]}": float(metrics["per_class_dice"][1]),
                    }
                )
            offset += batch_size_actual

    global_metrics = meter.compute()
    split_metrics = SplitMetrics(
        backbone_name=backbone_name,
        split=split_name,
        best_lr=float(best_lr),
        epochs_trained=int(epochs_trained),
        num_images=int(len(manifest)),
        pixel_acc=float(global_metrics["pixel_acc"]),
        miou=float(global_metrics["miou"]),
        dice=float(global_metrics["dice"]),
        per_class_iou=list(map(float, global_metrics["per_class_iou"])),
        per_class_dice=list(map(float, global_metrics["per_class_dice"])),
        probe_params=int(model.classifier.weight.numel() + model.classifier.bias.numel()),
    )
    return split_metrics, pd.DataFrame(rows)


def confusion_from_tensors(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> np.ndarray:
    pred_np = pred.detach().cpu().numpy().astype(np.int64).reshape(-1)
    target_np = target.detach().cpu().numpy().astype(np.int64).reshape(-1)
    valid = (target_np >= 0) & (target_np < num_classes)
    indices = num_classes * target_np[valid] + pred_np[valid]
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def colorize_mask(mask: np.ndarray, viz_map: List[List[int]]) -> np.ndarray:
    colors = np.asarray(viz_map, dtype=np.uint8)
    return colors[mask]


def overlay_mask_on_image(image_rgb: np.ndarray, mask: np.ndarray, viz_map: List[List[int]]) -> np.ndarray:
    color_mask = colorize_mask(mask, viz_map)
    overlay = image_rgb.copy().astype(np.float32)
    color_mask_f = color_mask.astype(np.float32)
    fg = mask != 0
    overlay[fg] = 0.65 * overlay[fg] + 0.35 * color_mask_f[fg]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def load_resized_image(image_path: str, img_size: Tuple[int, int]) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(image, (int(img_size[0]), int(img_size[1])), interpolation=cv2.INTER_LINEAR)


def collect_example_predictions(
    model: nn.Module,
    test_features: torch.Tensor,
    test_targets: torch.Tensor,
    test_manifest: pd.DataFrame,
    example_indices: Sequence[int],
    img_size: Tuple[int, int],
    viz_map: List[List[int]],
    device: torch.device,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    model.eval()
    with torch.no_grad():
        for idx in example_indices:
            feature = test_features[idx : idx + 1].to(device=device, dtype=torch.float32)
            logits = model(feature)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            target = test_targets[idx].cpu().numpy().astype(np.uint8)
            row = test_manifest.iloc[idx]
            image_rgb = load_resized_image(str(row["image_path"]), img_size=img_size)
            rows.append(
                {
                    "image_id": int(row["image_id"]),
                    "image_path": row["image_path"],
                    "input": image_rgb,
                    "gt_overlay": overlay_mask_on_image(image_rgb, target, viz_map),
                    "pred_overlay": overlay_mask_on_image(image_rgb, pred, viz_map),
                }
            )
    return rows


def save_bar_compare(summary_df: pd.DataFrame, output_path: Path) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    order = ["imagenet", "cag"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, metric, ylabel, title in [
        (axes[0], "miou", "mIoU", "Test mIoU"),
        (axes[1], "dice", "Dice", "Test Dice"),
    ]:
        sub = test_df.set_index("backbone_name").loc[order].reset_index()
        xs = np.arange(len(order))
        ys = sub[metric].to_numpy(dtype=float)
        colors = [PLOT_COLORS[name] for name in order]
        ax.bar(xs, ys, color=colors, width=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(["ImageNet", "CAG"])
        ax.set_ylim(0.0, min(1.0, max(ys) * 1.15))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.4f}", ha="center", va="bottom", fontsize=10)
    fig.suptitle("Segmentation Linear Probe Comparison", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_learning_curves(
    histories: Dict[str, List[Dict[str, object]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for backbone_name, history in histories.items():
        epochs = [item["epoch"] for item in history]
        valid_miou = [item["valid_miou"] for item in history]
        valid_dice = [item["valid_dice"] for item in history]
        color = PLOT_COLORS[backbone_name]
        label = "ImageNet" if backbone_name == "imagenet" else "CAG"
        axes[0].plot(epochs, valid_miou, label=label, color=color)
        axes[1].plot(epochs, valid_dice, label=label, color=color)
    axes[0].set_title("Validation mIoU")
    axes[1].set_title("Validation Dice")
    for ax, ylabel in zip(axes, ["mIoU", "Dice"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Segmentation Linear Probe Learning Curves", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_example_figure(
    imagenet_examples: List[Dict[str, object]],
    cag_examples: List[Dict[str, object]],
    output_path: Path,
) -> None:
    if len(imagenet_examples) != len(cag_examples):
        raise ValueError("ImageNet/CAG example count mismatch")
    num_rows = len(imagenet_examples)
    fig, axes = plt.subplots(num_rows, 4, figsize=(14, max(3 * num_rows, 4)))
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    col_titles = ["Input", "GT", "ImageNet Pred", "CAG Pred"]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title)
    for row_idx, (img_row, cag_row) in enumerate(zip(imagenet_examples, cag_examples)):
        axes[row_idx, 0].imshow(img_row["input"])
        axes[row_idx, 1].imshow(img_row["gt_overlay"])
        axes[row_idx, 2].imshow(img_row["pred_overlay"])
        axes[row_idx, 3].imshow(cag_row["pred_overlay"])
        stem = Path(str(img_row["image_path"])).stem
        axes[row_idx, 0].set_ylabel(stem, rotation=0, ha="right", va="center", fontsize=9)
        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
    fig.suptitle("Segmentation Linear Probe Examples", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def format_float(x: float) -> str:
    if np.isnan(x) or np.isinf(x):
        return "nan"
    return f"{x:.6f}"


def write_markdown_summary(
    output_path: Path,
    summary_df: pd.DataFrame,
    probe_params: int,
) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    row_i = test_df[test_df["backbone_name"] == "imagenet"].iloc[0]
    row_c = test_df[test_df["backbone_name"] == "cag"].iloc[0]

    delta_miou = float(row_c["miou"]) - float(row_i["miou"])
    delta_dice = float(row_c["dice"]) - float(row_i["dice"])
    if delta_miou > 0.0 and delta_dice > 0.0:
        interpretation = "dense structural transferability improves with CAG pretraining"
    elif max(float(row_i["miou"]), float(row_c["miou"])) < 0.5 and max(
        float(row_i["dice"]), float(row_c["dice"])
    ) < 0.5:
        interpretation = "dense information is not linearly decodable enough from frozen tokens"
    else:
        interpretation = "dense decodability advantage is inconclusive"

    lines = [
        "# Local Analysis 2 — Segmentation Linear Probe",
        "",
        "## Setup",
        "- Strict probe: last-layer normalized patch token + 1x1 conv + bilinear upsampling",
        "- Backbone is fully frozen",
        "- Input size: 640x640",
        "- Patch size: 16 (40x40 grid)",
        f"- Probe parameters: {probe_params}",
        "",
        "## Test Metrics",
        "",
        "| Backbone | mIoU | Dice | Pixel Acc |",
        "|---|---:|---:|---:|",
        f"| ImageNet | {format_float(float(row_i['miou']))} | {format_float(float(row_i['dice']))} | {format_float(float(row_i['pixel_acc']))} |",
        f"| CAG | {format_float(float(row_c['miou']))} | {format_float(float(row_c['dice']))} | {format_float(float(row_c['pixel_acc']))} |",
        "",
        "## Interpretation",
        f"- Delta mIoU (CAG - ImageNet): {delta_miou:.6f}",
        f"- Delta Dice (CAG - ImageNet): {delta_dice:.6f}",
        f"- {interpretation}",
        "- This analysis measures dense structural information that is linearly decodable from frozen patch tokens.",
        "",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
