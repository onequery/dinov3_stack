#!/usr/bin/env python3
"""
Global Analysis 4-3 — View Classification.

Experiment matrix:
1) ImageNet - Raw Frozen Nearest-Centroid
2) ImageNet - Frozen Strict Linear View Classification Probe
3) CAG - Raw Frozen Nearest-Centroid
4) CAG - Frozen Strict Linear View Classification Probe

Target:
- 9-way view classification derived from PositionerPrimaryAngle / PositionerSecondaryAngle
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, silhouette_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
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
TARGET_NAME = "view_9way"
PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}
VIEW_LABELS_9WAY = [
    "up_left",
    "up_center",
    "up_right",
    "center_left",
    "center_center",
    "center_right",
    "down_left",
    "down_center",
    "down_right",
]
VIEW_INDEX = {label: idx for idx, label in enumerate(VIEW_LABELS_9WAY)}
VIEW_COLORS = {
    "up_left": "#2166AC",
    "up_center": "#67A9CF",
    "up_right": "#D1E5F0",
    "center_left": "#1B7837",
    "center_center": "#7FBF7B",
    "center_right": "#D9F0D3",
    "down_left": "#B2182B",
    "down_center": "#EF8A62",
    "down_right": "#FDDBC7",
}
ALLOWED_SPLITS = ("train", "valid", "test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class ClassificationMetrics:
    mode: str
    target: str
    backbone_name: str
    split: str
    seed: int | None
    best_lr: float | None
    epochs_trained: int | None
    num_images: int
    accuracy: float
    macro_f1: float
    balanced_accuracy: float

    def to_row(self) -> Dict[str, object]:
        return {
            "mode": self.mode,
            "target": self.target,
            "backbone_name": self.backbone_name,
            "split": self.split,
            "seed": self.seed if self.seed is not None else "",
            "best_lr": self.best_lr if self.best_lr is not None else "",
            "epochs_trained": self.epochs_trained if self.epochs_trained is not None else "",
            "num_images": self.num_images,
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "balanced_accuracy": self.balanced_accuracy,
        }


@dataclass
class StepDef:
    name: str
    kind: str


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


class AnalysisRunTracker:
    def __init__(self, steps: Sequence[StepDef]):
        self.steps = list(steps)
        self.total_steps = max(1, len(self.steps))
        self.completed_steps = 0
        self.start_time = time.time()
        self.current_step_index = -1
        self.current_step_started_at = self.start_time
        self.short_step_durations: List[float] = []
        self.probe_step_durations: List[float] = []

    def start_step(self, index: int) -> None:
        self.current_step_index = int(index)
        self.current_step_started_at = time.time()
        current = self.current_step_index + 1
        log(f"[ETA][FULL-RUN] Step {current}/{self.total_steps} START | {self.steps[self.current_step_index].name}")

    def finish_step(self) -> None:
        if self.current_step_index < 0:
            return
        step = self.steps[self.current_step_index]
        duration = max(0.0, time.time() - self.current_step_started_at)
        if step.kind == "probe":
            self.probe_step_durations.append(duration)
        else:
            self.short_step_durations.append(duration)
        self.completed_steps = max(self.completed_steps, self.current_step_index + 1)
        elapsed = max(0.0, time.time() - self.start_time)
        avg_step = elapsed / max(1, self.completed_steps)
        remaining = max(0.0, avg_step * (self.total_steps - self.completed_steps))
        eta_dt = datetime.now() + timedelta(seconds=remaining)
        progress = 100.0 * self.completed_steps / self.total_steps
        log(
            f"[ETA][FULL-RUN] Step {self.completed_steps}/{self.total_steps} DONE | {step.name} | "
            f"progress={progress:.1f}% | elapsed={format_duration(elapsed)} | "
            f"remaining~{format_duration(remaining)} | eta={eta_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def current_step_name(self) -> str:
        if self.current_step_index < 0:
            return "unknown"
        return self.steps[self.current_step_index].name

    def current_step_elapsed_seconds(self) -> float:
        return max(0.0, time.time() - self.current_step_started_at)

    def remaining_probe_runs_after_current(self) -> int:
        if self.current_step_index < 0:
            return sum(1 for step in self.steps if step.kind == "probe")
        future = self.steps[self.current_step_index + 1 :]
        return sum(1 for step in future if step.kind == "probe")

    def remaining_short_steps_after_current(self) -> int:
        if self.current_step_index < 0:
            return sum(1 for step in self.steps if step.kind != "probe")
        future = self.steps[self.current_step_index + 1 :]
        return sum(1 for step in future if step.kind != "probe")

    def average_short_step_seconds(self, default_seconds: float = 30.0) -> float:
        if self.short_step_durations:
            return float(sum(self.short_step_durations) / len(self.short_step_durations))
        return float(default_seconds)

    def estimate_probe_run_seconds(self, fallback_current_total_seconds: float) -> float:
        if self.probe_step_durations:
            return float(sum(self.probe_step_durations) / len(self.probe_step_durations))
        return max(1.0, float(fallback_current_total_seconds))

    def log_live_eta(self, current_probe_remaining_seconds: float, note_fields: Dict[str, object]) -> None:
        elapsed = max(0.0, time.time() - self.start_time)
        current_elapsed = self.current_step_elapsed_seconds()
        current_total_estimate = current_elapsed + max(0.0, current_probe_remaining_seconds)
        future_probe_runs = self.remaining_probe_runs_after_current()
        avg_probe_run = self.estimate_probe_run_seconds(current_total_estimate)
        remaining_short_steps = self.remaining_short_steps_after_current()
        avg_short = self.average_short_step_seconds()
        remaining = (
            max(0.0, current_probe_remaining_seconds)
            + future_probe_runs * avg_probe_run
            + remaining_short_steps * avg_short
        )
        total_runtime = elapsed + remaining
        overall_progress = (elapsed / total_runtime) if total_runtime > 0.0 else 0.0
        eta_dt = datetime.now() + timedelta(seconds=remaining)
        fields = {
            "step": f"{self.current_step_index + 1}/{self.total_steps}",
            "current_step": self.current_step_name(),
            "overall_progress": f"{overall_progress * 100.0:.1f}%",
            "elapsed": format_duration(elapsed),
            "remaining": format_duration(remaining),
            "eta": eta_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "future_probe_runs": future_probe_runs,
            "avg_probe_run": format_duration(avg_probe_run),
            "remaining_short_steps": remaining_short_steps,
        }
        fields.update(note_fields)
        serialized = " | ".join(f"{k}={v}" for k, v in fields.items())
        log(f"[ETA][FULL-RUN][LIVE] {serialized}")


class ProbeEpochTracker:
    def __init__(
        self,
        run_tracker: AnalysisRunTracker,
        backbone_name: str,
        target: str,
        seed: int,
        lr_values: Sequence[float],
        max_epoch: int,
    ):
        self.run_tracker = run_tracker
        self.backbone_name = backbone_name
        self.target = target
        self.seed = int(seed)
        self.lr_values = [float(v) for v in lr_values]
        self.max_epoch = int(max_epoch)
        self.total_epoch_budget = max(1, len(self.lr_values) * self.max_epoch)
        self.completed_epochs = 0
        self.global_start = time.time()
        self.current_lr = None
        self.current_epoch_times: List[float] = []

    def start_lr(self, lr: float, lr_index: int) -> None:
        self.current_lr = float(lr)
        self.current_epoch_times = []
        log(
            f"[ETA][TRAIN-SUBSTEP][{self.backbone_name}] seed={self.seed} target={self.target} "
            f"LR candidate {lr_index}/{len(self.lr_values)} START | lr={self.current_lr}"
        )

    def finish_epoch(self, epoch: int, epoch_seconds: float) -> None:
        self.completed_epochs += 1
        self.current_epoch_times.append(float(epoch_seconds))
        avg_epoch_this_lr = sum(self.current_epoch_times) / max(1, len(self.current_epoch_times))
        remaining_this_lr = max(0.0, (self.max_epoch - int(epoch)) * avg_epoch_this_lr)
        total_elapsed = max(0.0, time.time() - self.global_start)
        avg_epoch_total = total_elapsed / max(1, self.completed_epochs)
        remaining_total = max(0.0, (self.total_epoch_budget - self.completed_epochs) * avg_epoch_total)
        eta_dt = datetime.now() + timedelta(seconds=remaining_total)
        log(
            f"[ETA][TRAIN-SUBSTEP][{self.backbone_name}] seed={self.seed} target={self.target} "
            f"lr={self.current_lr} epoch={epoch}/{self.max_epoch} | "
            f"epoch_time={format_duration(epoch_seconds)} | avg_epoch={format_duration(avg_epoch_this_lr)} | "
            f"remaining_this_lr~{format_duration(remaining_this_lr)} | "
            f"remaining_substep_total~{format_duration(remaining_total)} | "
            f"eta={eta_dt.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        self.run_tracker.log_live_eta(
            current_probe_remaining_seconds=remaining_total,
            note_fields={
                "backbone": self.backbone_name,
                "mode": "probe_linear",
                "target": self.target,
                "seed": self.seed,
                "lr": self.current_lr,
                "epoch": f"{epoch}/{self.max_epoch}",
                "substep_remaining": format_duration(remaining_total),
            },
        )

    def skip_remaining_epochs(self, skipped_epochs: int) -> None:
        skipped = max(0, int(skipped_epochs))
        if skipped > 0:
            self.total_epoch_budget = max(self.completed_epochs, self.total_epoch_budget - skipped)


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


class StrictLinearClassificationProbe(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.projector = nn.Linear(embed_dim, embed_dim, bias=True)
        self.classifier = nn.Linear(embed_dim, num_classes, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projector(x)
        return F.normalize(z, dim=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return z, logits


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def setup_console_and_file_logging(output_root: Path, log_file_arg: str | None, default_prefix: str) -> Tuple[Path, TextIO, TextIO, TextIO]:
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


def restore_console_logging(file_handle: TextIO, original_stdout: TextIO, original_stderr: TextIO) -> None:
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    file_handle.close()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int, strict_deterministic: bool = False) -> None:
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


def build_transform(resize_size: int, center_crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop((center_crop_size, center_crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def hash_records(records: Iterable[Sequence[object]]) -> str:
    hasher = hashlib.sha256()
    for row in records:
        line = "\t".join(str(v) for v in row)
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> str:
    return hash_records(df.loc[:, list(columns)].itertuples(index=False, name=None))


def make_feature_hash(features: torch.Tensor) -> str:
    arr = features.detach().cpu().float().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def clean_numeric(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def quantize_horizontal(primary: float) -> str:
    if primary < -10.0:
        return "left"
    if primary > 10.0:
        return "right"
    return "center"


def quantize_vertical(secondary: float) -> str:
    if secondary < -10.0:
        return "down"
    if secondary > 10.0:
        return "up"
    return "center"


def build_view_label(primary: float | None, secondary: float | None) -> tuple[str | None, str | None, str | None]:
    if primary is None or secondary is None:
        return None, None, None
    horizontal = quantize_horizontal(primary)
    vertical = quantize_vertical(secondary)
    return horizontal, vertical, f"{vertical}_{horizontal}"


def _relative_parts_from_image(image_root: Path, image_path: Path) -> tuple[str, str, str, str, Path]:
    rel = image_path.resolve().relative_to(image_root.resolve())
    if len(rel.parts) < 6:
        raise ValueError(f"Unexpected image path under dataset root: {image_path}")
    split, class_name, patient_id, study_id = rel.parts[:4]
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"Unexpected split in image path: {image_path}")
    return split, class_name, patient_id, study_id, rel


def _balanced_downsample_split(split_df: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    if max_images <= 0 or len(split_df) <= max_images:
        return split_df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups: Dict[str, List[int]] = {}
    for label, group in split_df.groupby("view_label_9way", sort=True):
        indices = group.index.to_numpy(copy=True)
        rng.shuffle(indices)
        groups[str(label)] = indices.tolist()
    ordered_labels = [label for label in VIEW_LABELS_9WAY if label in groups]
    selected: List[int] = []
    cursors = {label: 0 for label in ordered_labels}
    while len(selected) < max_images:
        advanced = False
        for label in ordered_labels:
            cursor = cursors[label]
            bucket = groups[label]
            if cursor < len(bucket) and len(selected) < max_images:
                selected.append(bucket[cursor])
                cursors[label] = cursor + 1
                advanced = True
        if not advanced:
            break
    return split_df.loc[selected].sort_values(["split", "class_name", "patient_id", "study_id", "img_path"]).reset_index(drop=True)


def build_view_manifest(image_root: Path, dcm_root: Path, max_images_per_split: int | None, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, object]] = []
    drop_rows: List[Dict[str, object]] = []
    image_paths = sorted([p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not image_paths:
        raise FileNotFoundError(f"No images found under {image_root}")

    for image_path in image_paths:
        split, class_name, patient_id, study_id, rel = _relative_parts_from_image(image_root, image_path)
        dcm_rel_path = rel.with_suffix(".dcm")
        dcm_path = dcm_root / dcm_rel_path
        if not dcm_path.exists():
            raise FileNotFoundError(f"Missing DICOM for image: {image_path} -> {dcm_path}")
        dcm = pydicom.dcmread(
            str(dcm_path),
            stop_before_pixels=True,
            specific_tags=["PositionerPrimaryAngle", "PositionerSecondaryAngle"],
            force=True,
        )
        primary = clean_numeric(getattr(dcm, "PositionerPrimaryAngle", None))
        secondary = clean_numeric(getattr(dcm, "PositionerSecondaryAngle", None))
        horizontal, vertical, view_label = build_view_label(primary, secondary)
        if view_label is None:
            drop_rows.append(
                {
                    "img_path": str(image_path.resolve()),
                    "split": split,
                    "class_name": class_name,
                    "patient_id": patient_id,
                    "study_id": study_id,
                    "dicom_rel_path": str(dcm_rel_path),
                    "drop_reason": "missing_view_angle",
                }
            )
            continue
        rows.append(
            {
                "img_path": str(image_path.resolve()),
                "split": split,
                "class_name": class_name,
                "patient_id": patient_id,
                "study_id": study_id,
                "dicom_rel_path": str(dcm_rel_path),
                "image_rel_path": str(rel),
                "PositionerPrimaryAngle": float(primary),
                "PositionerSecondaryAngle": float(secondary),
                "view_horizontal_10deg": horizontal,
                "view_vertical_10deg": vertical,
                "view_label_9way": view_label,
                "view_label_index": int(VIEW_INDEX[view_label]),
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("View manifest is empty after dropping missing-angle images.")
    manifest = manifest.sort_values(["split", "class_name", "patient_id", "study_id", "img_path"]).reset_index(drop=True)

    sampled_parts: List[pd.DataFrame] = []
    for split in ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split].copy()
        if split_df.empty:
            raise ValueError(f"No samples found for split={split}")
        if max_images_per_split is not None:
            split_df = _balanced_downsample_split(split_df, int(max_images_per_split), seed + hash(split) % 1000)
        sampled_parts.append(split_df)
    manifest = pd.concat(sampled_parts, axis=0, ignore_index=True)
    manifest = manifest.sort_values(["split", "class_name", "patient_id", "study_id", "img_path"]).reset_index(drop=True)
    manifest.insert(0, "image_id", np.arange(len(manifest), dtype=np.int64))
    drop_df = pd.DataFrame(drop_rows)
    return manifest, drop_df


def validate_view_manifest(manifest: pd.DataFrame, allow_missing_classes: bool) -> None:
    required_columns = [
        "image_id",
        "img_path",
        "split",
        "class_name",
        "patient_id",
        "study_id",
        "dicom_rel_path",
        "PositionerPrimaryAngle",
        "PositionerSecondaryAngle",
        "view_horizontal_10deg",
        "view_vertical_10deg",
        "view_label_9way",
        "view_label_index",
    ]
    missing = [c for c in required_columns if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    for split in ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split]
        if split_df.empty:
            raise ValueError(f"No rows for split={split}")
        present = set(split_df["view_label_9way"].astype(str).tolist())
        missing_classes = [label for label in VIEW_LABELS_9WAY if label not in present]
        if missing_classes and not allow_missing_classes:
            raise ValueError(f"Split {split} is missing view classes: {missing_classes}")


def split_manifest_map(manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for split in ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"No images found for split={split}")
        result[split] = split_df
    return result


def save_manifest(split_manifest: pd.DataFrame, output_path: Path) -> None:
    split_manifest.to_csv(output_path, index=False)


def save_manifest_audits(manifest: pd.DataFrame, drop_df: pd.DataFrame, output_root: Path) -> None:
    overview = (
        manifest.groupby(["split", "class_name"], sort=True)
        .size()
        .reset_index(name="num_images")
        .sort_values(["split", "class_name"])
        .reset_index(drop=True)
    )
    overview.to_csv(output_root / "summary_global_4_3_manifest_overview.csv", index=False)

    class_dist = (
        manifest.groupby(["split", "view_label_9way"], sort=True)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "view_label_9way"])
        .reset_index(drop=True)
    )
    class_dist.to_csv(output_root / "summary_global_4_3_view_class_counts.csv", index=False)

    if drop_df.empty:
        drop_summary = pd.DataFrame(columns=["split", "class_name", "drop_reason", "count"])
    else:
        drop_summary = (
            drop_df.groupby(["split", "class_name", "drop_reason"], sort=True)
            .size()
            .reset_index(name="count")
            .sort_values(["split", "class_name", "drop_reason"])
            .reset_index(drop=True)
        )
    drop_df.to_csv(output_root / "summary_global_4_3_drop_audit_rows.csv", index=False)
    drop_summary.to_csv(output_root / "summary_global_4_3_drop_audit.csv", index=False)


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
        if not torch.is_tensor(output[0]):
            raise ValueError("Backbone output first element is not a tensor.")
        tensor = output[0]
    else:
        raise ValueError(f"Unsupported backbone output type: {type(output)}")
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        return tensor[:, 0, :]
    raise ValueError(f"Unsupported output tensor shape: {tuple(tensor.shape)}")


def extract_or_load_features(
    manifest: pd.DataFrame,
    split_name: str,
    backbone_name: str,
    ckpt_path: str,
    args: argparse.Namespace,
    out_root: Path,
) -> torch.Tensor:
    features_path = out_root / f"features_{backbone_name}_{split_name}.pt"
    meta_path = out_root / f"features_{backbone_name}_{split_name}.meta.json"
    index_path = out_root / f"feature_index_{split_name}.csv"
    manifest_hash = hash_dataframe(manifest, ["image_id", "img_path"])
    expected_meta = {
        "manifest_hash": manifest_hash,
        "backbone_name": backbone_name,
        "split": split_name,
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "model_name": args.model_name,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "resize_size": int(args.resize_size),
        "center_crop_size": int(args.center_crop_size),
        "feature_source": "x_norm_clstoken",
    }
    if args.cache_features and features_path.exists() and meta_path.exists():
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_meta == expected_meta:
            features = torch.load(features_path, map_location="cpu")
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"Cached features are not a tensor: {features_path}")
            if features.ndim != 2 or features.shape[0] != len(manifest):
                raise ValueError(f"Cached features shape mismatch for {features_path}: {tuple(features.shape)}")
            log(f"Reusing feature cache for {backbone_name}/{split_name}: {features_path}")
            return features.float()

    requested_device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        log(f"CUDA requested but unavailable. Falling back to CPU for {backbone_name}/{split_name}.")
        requested_device = "cpu"

    transform = build_transform(args.resize_size, args.center_crop_size)
    dataset = ImagePathDataset(manifest["img_path"].tolist(), transform)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.feature_batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=str(requested_device).startswith("cuda"),
        drop_last=False,
    )

    model = Dinov3Backbone(weights=ckpt_path, model_name=args.model_name, repo_dir=args.repo_dir).to(requested_device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    feats_list: List[torch.Tensor] = []
    started = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Features-{backbone_name}-{split_name}"):
            batch = batch.to(requested_device, non_blocking=True)
            out = model(batch)
            feat = extract_global_representation(out)
            feat = F.normalize(feat.float(), dim=1)
            feats_list.append(feat.cpu())
    features = torch.cat(feats_list, dim=0).contiguous().float()
    elapsed = max(0.0, time.time() - started)
    log(
        f"Extracted features for {backbone_name}/{split_name} | n={features.shape[0]} d={features.shape[1]} "
        f"| elapsed={format_duration(elapsed)}"
    )

    if args.cache_features:
        torch.save(features, features_path)
        meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
        manifest.loc[:, ["image_id", "img_path"]].to_csv(index_path, index=False)
    return features


def l2_normalize_cpu(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1).cpu()


def labels_to_tensor(values: Sequence[str]) -> torch.Tensor:
    return torch.tensor([VIEW_INDEX[str(v)] for v in values], dtype=torch.long)


def present_label_indices(y_true: np.ndarray, y_pred: np.ndarray) -> List[int]:
    present = sorted(set(int(v) for v in y_true.tolist()) | set(int(v) for v in y_pred.tolist()))
    return present if present else list(range(len(VIEW_LABELS_9WAY)))


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mode: str,
    target: str,
    backbone_name: str,
    split_name: str,
    seed: int | None,
    best_lr: float | None,
    epochs_trained: int | None,
) -> ClassificationMetrics:
    labels = present_label_indices(y_true, y_pred)
    return ClassificationMetrics(
        mode=mode,
        target=target,
        backbone_name=backbone_name,
        split=split_name,
        seed=seed,
        best_lr=best_lr,
        epochs_trained=epochs_trained,
        num_images=int(len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )


def compute_inter_centroid_distance(embeddings: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    unique = sorted(set(int(v) for v in labels.tolist()))
    if len(unique) <= 1:
        return float("nan"), float("nan")
    centroids = []
    for label_idx in unique:
        centroid = embeddings[labels == label_idx].mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)
    centroids = np.stack(centroids, axis=0)
    sims = centroids @ centroids.T
    dists = 1.0 - sims
    iu = np.triu_indices_from(dists, k=1)
    pairwise = dists[iu]
    return float(pairwise.mean()), float(pairwise.min())


def _build_umap_coords(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(5, min(15, len(embeddings) - 1)),
        min_dist=0.2,
        metric="cosine",
        random_state=random_state,
    )
    return reducer.fit_transform(embeddings)


def save_view_umap(
    embedding_map: Dict[str, np.ndarray],
    test_manifest_map: Dict[str, pd.DataFrame],
    output_path: Path,
    seed: int,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4), squeeze=False)
    for col_idx, backbone_name in enumerate(["imagenet", "cag"]):
        ax = axes[0, col_idx]
        embedding = embedding_map[backbone_name]
        manifest = test_manifest_map[backbone_name]
        coords = _build_umap_coords(embedding, random_state=seed)
        labels = manifest["view_label_9way"].astype(str).to_numpy()
        for label in VIEW_LABELS_9WAY:
            mask = labels == label
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.78,
                linewidths=0,
                color=VIEW_COLORS[label],
                label=label,
            )
        ax.set_title(f"{title_prefix} | {backbone_name}")
        ax.grid(alpha=0.18)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_confusion_figure(cm: np.ndarray, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 7.3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(len(VIEW_LABELS_9WAY)))
    ax.set_yticks(np.arange(len(VIEW_LABELS_9WAY)))
    ax.set_xticklabels(VIEW_LABELS_9WAY, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(VIEW_LABELS_9WAY, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_nearest_centroids(train_features: torch.Tensor, train_labels: np.ndarray) -> Tuple[torch.Tensor, List[int]]:
    emb = l2_normalize_cpu(train_features)
    label_indices = sorted(set(int(v) for v in train_labels.tolist()))
    centroids: List[torch.Tensor] = []
    for label_idx in label_indices:
        mask = torch.from_numpy((train_labels == label_idx).astype(np.bool_))
        centroid = emb[mask].mean(dim=0)
        centroid = F.normalize(centroid.view(1, -1), dim=1).view(-1)
        centroids.append(centroid)
    return torch.stack(centroids, dim=0), label_indices


def predict_nearest_centroid(features: torch.Tensor, centroids: torch.Tensor, centroid_labels: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    emb = l2_normalize_cpu(features)
    sims = torch.matmul(emb, centroids.T).cpu().numpy()
    best = sims.argmax(axis=1)
    pred_indices = np.asarray([int(centroid_labels[idx]) for idx in best], dtype=np.int64)
    best_scores = sims[np.arange(len(best)), best]
    return pred_indices, best_scores.astype(np.float64)


def build_prediction_df(
    manifest: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    *,
    mode: str,
    backbone_name: str,
    split_name: str,
    seed: int | None,
) -> pd.DataFrame:
    out = manifest.loc[:, [
        "image_id",
        "img_path",
        "split",
        "class_name",
        "patient_id",
        "study_id",
        "dicom_rel_path",
        "view_label_9way",
        "view_label_index",
        "PositionerPrimaryAngle",
        "PositionerSecondaryAngle",
    ]].copy()
    out.insert(0, "mode", mode)
    out.insert(1, "backbone_name", backbone_name)
    out.insert(2, "seed", "" if seed is None else int(seed))
    out["true_label_index"] = y_true
    out["pred_label_index"] = y_pred
    out["pred_label"] = [VIEW_LABELS_9WAY[int(v)] for v in y_pred.tolist()]
    out["correct"] = (y_true == y_pred).astype(np.int64)
    out["pred_score"] = scores
    return out


def train_one_epoch_probe(
    model: StrictLinearClassificationProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    device: torch.device,
    epoch_seed: int,
) -> float:
    dataset = TensorDataset(features, labels)
    generator = torch.Generator()
    generator.manual_seed(int(epoch_seed))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=generator)
    model.train()
    losses: List[float] = []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device, non_blocking=True)
        batch_labels = batch_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(batch_features)
        loss = F.cross_entropy(logits, batch_labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_probe_embeddings_and_logits(
    model: StrictLinearClassificationProbe,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = TensorDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.eval()
    emb_list: List[torch.Tensor] = []
    logit_list: List[torch.Tensor] = []
    with torch.no_grad():
        for (batch_features,) in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            emb, logits = model(batch_features)
            emb_list.append(emb.cpu())
            logit_list.append(logits.cpu())
    return torch.cat(emb_list, dim=0).contiguous().float(), torch.cat(logit_list, dim=0).contiguous().float()


def make_probe_signature(
    seed: int,
    backbone_name: str,
    target: str,
    train_manifest_hash: str,
    valid_manifest_hash: str,
    test_manifest_hash: str,
    feature_hashes: Dict[str, str],
    args: argparse.Namespace,
    embed_dim: int,
) -> Dict[str, object]:
    return {
        "seed": int(seed),
        "backbone_name": backbone_name,
        "target": target,
        "train_manifest_hash": train_manifest_hash,
        "valid_manifest_hash": valid_manifest_hash,
        "test_manifest_hash": test_manifest_hash,
        "feature_hashes": feature_hashes,
        "lr_grid": [float(v) for v in args.probe_lr_grid],
        "max_epoch": int(args.probe_max_epoch),
        "patience": int(args.probe_patience),
        "min_delta": float(args.probe_min_delta),
        "probe_batch_size": int(args.probe_batch_size),
        "embed_dim": int(embed_dim),
        "num_classes": int(len(VIEW_LABELS_9WAY)),
        "selection_metric": "macro_f1",
        "strict_deterministic": bool(args.strict_deterministic),
    }


def train_probe_with_lr_search(
    backbone_name: str,
    target: str,
    seed: int,
    train_features: torch.Tensor,
    valid_features: torch.Tensor,
    train_labels: np.ndarray,
    valid_labels: np.ndarray,
    out_root: Path,
    args: argparse.Namespace,
    tracker: ProbeEpochTracker,
    feature_hashes: Dict[str, str],
    manifest_hashes: Dict[str, str],
) -> Tuple[StrictLinearClassificationProbe, Dict[str, object], List[Dict[str, object]]]:
    embed_dim = int(train_features.shape[1])
    device = torch.device(args.device if args.device and torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    train_targets = torch.from_numpy(train_labels.astype(np.int64))
    signature = make_probe_signature(
        seed=seed,
        backbone_name=backbone_name,
        target=target,
        train_manifest_hash=manifest_hashes["train"],
        valid_manifest_hash=manifest_hashes["valid"],
        test_manifest_hash=manifest_hashes["test"],
        feature_hashes=feature_hashes,
        args=args,
        embed_dim=embed_dim,
    )
    ckpt_path = out_root / f"probe_checkpoint_seed{seed}_{backbone_name}_{target}.pt"
    history_path = out_root / f"history_seed{seed}_{backbone_name}_{target}.json"

    if ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location="cpu")
        if payload.get("signature") == signature:
            model = StrictLinearClassificationProbe(embed_dim, len(VIEW_LABELS_9WAY))
            model.load_state_dict(payload["model_state_dict"], strict=True)
            log(f"Reusing final trained probe checkpoint: {ckpt_path}")
            return model, payload["summary"], payload.get("lr_search_rows", [])

    lr_search_rows: List[Dict[str, object]] = []
    best_summary: Dict[str, object] | None = None
    best_state_dict: Dict[str, torch.Tensor] | None = None

    for lr_index, lr in enumerate(args.probe_lr_grid):
        set_global_seed(seed, strict_deterministic=args.strict_deterministic)
        model = StrictLinearClassificationProbe(embed_dim, len(VIEW_LABELS_9WAY)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=args.probe_weight_decay)
        tracker.start_lr(float(lr), lr_index + 1)
        best_valid_metric = -float("inf")
        best_epoch = 0
        best_candidate_state: Dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history_rows: List[Dict[str, object]] = []

        for epoch in range(1, int(args.probe_max_epoch) + 1):
            epoch_started = time.time()
            epoch_seed = int(seed * 1000 + lr_index * 100 + epoch)
            train_loss = train_one_epoch_probe(
                model=model,
                features=train_features,
                labels=train_targets,
                optimizer=optimizer,
                batch_size=int(args.probe_batch_size),
                device=device,
                epoch_seed=epoch_seed,
            )
            valid_embeddings, valid_logits = evaluate_probe_embeddings_and_logits(model, valid_features, int(args.probe_batch_size), device)
            valid_pred = valid_logits.argmax(dim=1).cpu().numpy().astype(np.int64)
            valid_metrics = compute_classification_metrics(
                y_true=valid_labels,
                y_pred=valid_pred,
                mode="probe_linear",
                target=target,
                backbone_name=backbone_name,
                split_name="valid",
                seed=seed,
                best_lr=float(lr),
                epochs_trained=epoch,
            )
            history_rows.append(
                {
                    "epoch": epoch,
                    "lr": float(lr),
                    "train_loss": float(train_loss),
                    "valid_accuracy": float(valid_metrics.accuracy),
                    "valid_macro_f1": float(valid_metrics.macro_f1),
                    "valid_balanced_accuracy": float(valid_metrics.balanced_accuracy),
                }
            )
            current_score = float(valid_metrics.macro_f1)
            improved = best_candidate_state is None or current_score > (best_valid_metric + float(args.probe_min_delta))
            if improved:
                best_valid_metric = current_score
                best_epoch = int(epoch)
                best_candidate_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            log(
                f"[{backbone_name}] seed={seed} target={target} lr={lr} epoch={epoch} | "
                f"train_loss={train_loss:.6f} valid_acc={valid_metrics.accuracy:.6f} "
                f"valid_macro_f1={valid_metrics.macro_f1:.6f} valid_bal_acc={valid_metrics.balanced_accuracy:.6f}"
            )
            tracker.finish_epoch(epoch, max(0.0, time.time() - epoch_started))
            if epochs_without_improvement >= int(args.probe_patience):
                tracker.skip_remaining_epochs(int(args.probe_max_epoch) - epoch)
                log(
                    f"Early stopping | seed={seed} backbone={backbone_name} target={target} "
                    f"lr={lr} best_epoch={best_epoch} best_valid_macro_f1={best_valid_metric:.6f}"
                )
                break

        if best_candidate_state is None:
            raise RuntimeError(f"No best state captured for seed={seed}, backbone={backbone_name}, lr={lr}")
        candidate_summary = {
            "seed": int(seed),
            "backbone_name": backbone_name,
            "target": target,
            "lr": float(lr),
            "best_valid_macro_f1": float(best_valid_metric),
            "best_epoch": int(best_epoch),
            "epochs_trained": len(history_rows),
        }
        lr_search_rows.append(candidate_summary)
        if best_summary is None or float(candidate_summary["best_valid_macro_f1"]) > float(best_summary["best_valid_macro_f1"]):
            best_summary = candidate_summary
            best_state_dict = best_candidate_state

    if best_state_dict is None or best_summary is None:
        raise RuntimeError(f"Training failed to produce a valid probe for {backbone_name}/{target}/seed={seed}")

    final_model = StrictLinearClassificationProbe(embed_dim, len(VIEW_LABELS_9WAY))
    final_model.load_state_dict(best_state_dict, strict=True)
    torch.save(
        {
            "seed": int(seed),
            "backbone_name": backbone_name,
            "target": target,
            "signature": signature,
            "model_state_dict": best_state_dict,
            "summary": best_summary,
            "lr_search_rows": lr_search_rows,
        },
        ckpt_path,
    )
    history_path.write_text(json.dumps(lr_search_rows, indent=2), encoding="utf-8")
    return final_model, best_summary, lr_search_rows


def summarize_probe_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (target, backbone_name, split), group in raw_df.groupby(["target", "backbone_name", "split"], sort=True):
        row = {
            "mode": "probe_linear",
            "target": target,
            "backbone_name": backbone_name,
            "split": split,
            "num_seeds": int(group["seed"].nunique()),
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": float(group["accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": float(group["macro_f1"].std(ddof=1)) if len(group) > 1 else 0.0,
            "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
            "balanced_accuracy_std": float(group["balanced_accuracy"].std(ddof=1)) if len(group) > 1 else 0.0,
            "epochs_trained_mean": float(pd.to_numeric(group["epochs_trained"], errors="coerce").mean()),
            "best_lr_vote": str(group["best_lr"].astype(str).value_counts().sort_index().idxmax()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_raw_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (target, backbone_name, split), group in raw_df.groupby(["target", "backbone_name", "split"], sort=True):
        row = {
            "mode": "raw_frozen",
            "target": target,
            "backbone_name": backbone_name,
            "split": split,
            "num_seeds": 0,
            "accuracy_mean": float(group["accuracy"].mean()),
            "accuracy_std": 0.0,
            "macro_f1_mean": float(group["macro_f1"].mean()),
            "macro_f1_std": 0.0,
            "balanced_accuracy_mean": float(group["balanced_accuracy"].mean()),
            "balanced_accuracy_std": 0.0,
            "epochs_trained_mean": 0.0,
            "best_lr_vote": "",
        }
        rows.append(row)
    return pd.DataFrame(rows)


def save_metric_compare_figure(summary_df: pd.DataFrame, metric_prefix: str, ylabel: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), squeeze=False)
    modes = ["raw_frozen", "probe_linear"]
    for idx, split in enumerate(["valid", "test"]):
        ax = axes[0, idx]
        subset = summary_df.loc[summary_df["split"] == split].copy()
        x = np.arange(len(["imagenet", "cag"]))
        width = 0.35
        for mode_idx, mode in enumerate(modes):
            ys = []
            yerr = []
            for backbone in ["imagenet", "cag"]:
                row = subset.loc[(subset["mode"] == mode) & (subset["backbone_name"] == backbone)]
                if row.empty:
                    ys.append(np.nan)
                    yerr.append(0.0)
                else:
                    ys.append(float(row.iloc[0][f"{metric_prefix}_mean"]))
                    yerr.append(float(row.iloc[0][f"{metric_prefix}_std"]))
            offset = (mode_idx - 0.5) * width
            ax.bar(x + offset, ys, width=width, color=[PLOT_COLORS[b] for b in ["imagenet", "cag"]], alpha=0.40 if mode == "raw_frozen" else 0.85, label=mode if idx == 0 else None)
            ax.errorbar(x + offset, ys, yerr=yerr, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(["ImageNet", "CAG"])
        ax.set_ylabel(ylabel)
        ax.set_title(f"{split}")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown_summary(output_path: Path, summary_df: pd.DataFrame, run_meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Global Analysis 4-3: View Classification")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- image root: `{run_meta['image_root']}`")
    lines.append(f"- output root: `{run_meta['output_root']}`")
    lines.append(f"- feature source: `{run_meta['feature_source']}`")
    lines.append(f"- view taxonomy: `{run_meta['view_taxonomy']}`")
    lines.append(f"- probe seeds: `{', '.join(str(v) for v in run_meta['probe_seed_set'])}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| mode | split | backbone | accuracy | macro_f1 | balanced_accuracy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    display_df = summary_df.sort_values(["split", "mode", "backbone_name"]).reset_index(drop=True)
    for row in display_df.itertuples():
        lines.append(
            f"| {row.mode} | {row.split} | {row.backbone_name} | {row.accuracy_mean:.4f} | {row.macro_f1_mean:.4f} | {row.balanced_accuracy_mean:.4f} |"
        )
    lines.append("")
    lines.append("## Key Files")
    lines.append("")
    for key in [
        "aggregate_summary_path",
        "raw_summary_path",
        "probe_raw_summary_path",
        "view_distribution_summary_path",
        "fig_umap_raw",
        "fig_umap_probe",
    ]:
        if key in run_meta:
            lines.append(f"- `{run_meta[key]}`")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_steps(args: argparse.Namespace) -> List[StepDef]:
    steps: List[StepDef] = [StepDef("build view manifest", "short")]
    for backbone_name in ["imagenet", "cag"]:
        for split in ALLOWED_SPLITS:
            steps.append(StepDef(f"extract features | {backbone_name} | {split}", "short"))
    for backbone_name in ["imagenet", "cag"]:
        steps.append(StepDef(f"raw centroid evaluate | {backbone_name}", "short"))
    for seed in args.probe_seeds:
        for backbone_name in ["imagenet", "cag"]:
            steps.append(StepDef(f"probe train/eval | seed={seed} | {backbone_name}", "probe"))
    steps.extend(
        [
            StepDef("aggregate summaries", "short"),
            StepDef("save figures", "short"),
            StepDef("write markdown", "short"),
            StepDef("write run meta", "short"),
        ]
    )
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--dcm-root", default="input/stent_split_dcm_unique_view")
    parser.add_argument("--output-root", default="outputs/global_4_3_view_classification_unique_view")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--imagenet-ckpt", default=DEFAULT_IMAGENET_CKPT)
    parser.add_argument("--cag-ckpt", default=DEFAULT_CAG_CKPT)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-size", type=int, default=480)
    parser.add_argument("--center-crop-size", type=int, default=448)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--probe-lr-grid", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-min-delta", type=float, default=0.0)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    ensure_dir(out_root)
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        out_root,
        args.log_file,
        default_prefix="global_4_3_view_classification",
    )

    try:
        set_global_seed(args.seed, strict_deterministic=args.strict_deterministic)
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        image_root = Path(args.image_root).resolve()
        dcm_root = Path(args.dcm_root).resolve()
        steps = build_steps(args)
        run_tracker = AnalysisRunTracker(steps)
        step_idx = 0

        run_tracker.start_step(step_idx)
        manifest, drop_df = build_view_manifest(
            image_root=image_root,
            dcm_root=dcm_root,
            max_images_per_split=args.max_images_per_split,
            seed=args.seed,
        )
        validate_view_manifest(manifest, allow_missing_classes=args.max_images_per_split is not None)
        split_manifests = split_manifest_map(manifest)
        for split_name, split_df in split_manifests.items():
            save_manifest(split_df, out_root / f"image_manifest_{split_name}.csv")
        save_manifest_audits(manifest, drop_df, out_root)
        manifest_hashes = {
            split_name: hash_dataframe(
                split_df,
                [
                    "image_id",
                    "img_path",
                    "patient_id",
                    "study_id",
                    "dicom_rel_path",
                    "view_label_9way",
                    "view_label_index",
                ],
            )
            for split_name, split_df in split_manifests.items()
        }
        run_tracker.finish_step()
        step_idx += 1

        backbone_ckpts = {"imagenet": args.imagenet_ckpt, "cag": args.cag_ckpt}
        feature_store: Dict[str, Dict[str, torch.Tensor]] = {"imagenet": {}, "cag": {}}
        feature_hashes: Dict[str, Dict[str, str]] = {"imagenet": {}, "cag": {}}
        for backbone_name in ["imagenet", "cag"]:
            for split_name in ALLOWED_SPLITS:
                run_tracker.start_step(step_idx)
                features = extract_or_load_features(
                    manifest=split_manifests[split_name],
                    split_name=split_name,
                    backbone_name=backbone_name,
                    ckpt_path=backbone_ckpts[backbone_name],
                    args=args,
                    out_root=out_root,
                )
                feature_store[backbone_name][split_name] = features
                feature_hashes[backbone_name][split_name] = make_feature_hash(features)
                run_tracker.finish_step()
                step_idx += 1

        raw_metric_rows: List[Dict[str, object]] = []
        test_raw_embeddings: Dict[str, np.ndarray] = {}
        test_raw_manifest_map: Dict[str, pd.DataFrame] = {}
        geometry_rows: List[Dict[str, object]] = []

        train_labels = split_manifests["train"]["view_label_index"].to_numpy(dtype=np.int64)
        for backbone_name in ["imagenet", "cag"]:
            run_tracker.start_step(step_idx)
            centroids, centroid_labels = build_nearest_centroids(feature_store[backbone_name]["train"], train_labels)
            for split_name in ["valid", "test"]:
                split_df = split_manifests[split_name]
                y_true = split_df["view_label_index"].to_numpy(dtype=np.int64)
                y_pred, best_scores = predict_nearest_centroid(feature_store[backbone_name][split_name], centroids, centroid_labels)
                metrics = compute_classification_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    mode="raw_frozen",
                    target=TARGET_NAME,
                    backbone_name=backbone_name,
                    split_name=split_name,
                    seed=None,
                    best_lr=None,
                    epochs_trained=None,
                )
                raw_metric_rows.append(metrics.to_row())
                pred_df = build_prediction_df(
                    split_df,
                    y_true,
                    y_pred,
                    best_scores,
                    mode="raw_frozen",
                    backbone_name=backbone_name,
                    split_name=split_name,
                    seed=None,
                )
                pred_df.to_csv(out_root / f"per_image_{TARGET_NAME}_raw_{backbone_name}_{split_name}.csv", index=False)
                if split_name == "test":
                    emb = l2_normalize_cpu(feature_store[backbone_name][split_name]).numpy()
                    test_raw_embeddings[backbone_name] = emb
                    test_raw_manifest_map[backbone_name] = split_df.copy()
                    try:
                        silhouette = float(silhouette_score(emb, y_true, metric="cosine")) if len(np.unique(y_true)) > 1 else float("nan")
                    except Exception:
                        silhouette = float("nan")
                    centroid_mean, centroid_min = compute_inter_centroid_distance(emb, y_true)
                    geometry_rows.append(
                        {
                            "row_type": "embedding_geometry",
                            "mode": "raw_frozen",
                            "backbone_name": backbone_name,
                            "split": split_name,
                            "silhouette_score": silhouette,
                            "inter_centroid_distance_mean": centroid_mean,
                            "inter_centroid_distance_min": centroid_min,
                        }
                    )
                    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(VIEW_LABELS_9WAY))))
                    save_confusion_figure(cm, out_root / f"fig_global4_3_confusion_raw_{backbone_name}_{split_name}.png", f"raw | {backbone_name} | {split_name}")
            run_tracker.finish_step()
            step_idx += 1

        probe_metric_rows: List[Dict[str, object]] = []
        probe_lr_search_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        probe_embedding_seed11: Dict[str, np.ndarray] = {}
        probe_test_manifest_seed11: Dict[str, pd.DataFrame] = {}

        for seed in args.probe_seeds:
            for backbone_name in ["imagenet", "cag"]:
                probe_lr_search_results.setdefault(str(seed), {})
                run_tracker.start_step(step_idx)
                tracker = ProbeEpochTracker(
                    run_tracker=run_tracker,
                    backbone_name=backbone_name,
                    target=TARGET_NAME,
                    seed=seed,
                    lr_values=args.probe_lr_grid,
                    max_epoch=args.probe_max_epoch,
                )
                model, best_summary, lr_rows = train_probe_with_lr_search(
                    backbone_name=backbone_name,
                    target=TARGET_NAME,
                    seed=seed,
                    train_features=feature_store[backbone_name]["train"],
                    valid_features=feature_store[backbone_name]["valid"],
                    train_labels=split_manifests["train"]["view_label_index"].to_numpy(dtype=np.int64),
                    valid_labels=split_manifests["valid"]["view_label_index"].to_numpy(dtype=np.int64),
                    out_root=out_root,
                    args=args,
                    tracker=tracker,
                    feature_hashes=feature_hashes[backbone_name],
                    manifest_hashes=manifest_hashes,
                )
                probe_lr_search_results[str(seed)][backbone_name] = lr_rows
                device = torch.device(args.device if args.device and torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
                model = model.to(device)
                for split_name in ["valid", "test"]:
                    split_df = split_manifests[split_name]
                    y_true = split_df["view_label_index"].to_numpy(dtype=np.int64)
                    probe_embeddings, probe_logits = evaluate_probe_embeddings_and_logits(
                        model,
                        feature_store[backbone_name][split_name],
                        int(args.probe_batch_size),
                        device,
                    )
                    y_pred = probe_logits.argmax(dim=1).cpu().numpy().astype(np.int64)
                    scores = probe_logits.max(dim=1).values.cpu().numpy().astype(np.float64)
                    metrics = compute_classification_metrics(
                        y_true=y_true,
                        y_pred=y_pred,
                        mode="probe_linear",
                        target=TARGET_NAME,
                        backbone_name=backbone_name,
                        split_name=split_name,
                        seed=seed,
                        best_lr=float(best_summary["lr"]),
                        epochs_trained=int(best_summary["epochs_trained"]),
                    )
                    probe_metric_rows.append(metrics.to_row())
                    pred_df = build_prediction_df(
                        split_df,
                        y_true,
                        y_pred,
                        scores,
                        mode="probe_linear",
                        backbone_name=backbone_name,
                        split_name=split_name,
                        seed=seed,
                    )
                    pred_df.to_csv(out_root / f"per_image_{TARGET_NAME}_probe_seed{seed}_{backbone_name}_{split_name}.csv", index=False)
                    if split_name == "test" and int(seed) == 11:
                        probe_embedding_seed11[backbone_name] = probe_embeddings.numpy()
                        probe_test_manifest_seed11[backbone_name] = split_df.copy()
                        try:
                            silhouette = float(silhouette_score(probe_embeddings.numpy(), y_true, metric="cosine")) if len(np.unique(y_true)) > 1 else float("nan")
                        except Exception:
                            silhouette = float("nan")
                        centroid_mean, centroid_min = compute_inter_centroid_distance(probe_embeddings.numpy(), y_true)
                        geometry_rows.append(
                            {
                                "row_type": "embedding_geometry",
                                "mode": "probe_linear_seed11",
                                "backbone_name": backbone_name,
                                "split": split_name,
                                "silhouette_score": silhouette,
                                "inter_centroid_distance_mean": centroid_mean,
                                "inter_centroid_distance_min": centroid_min,
                            }
                        )
                    if split_name == "test":
                        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(VIEW_LABELS_9WAY))))
                        save_confusion_figure(cm, out_root / f"fig_global4_3_confusion_probe_seed{seed}_{backbone_name}_{split_name}.png", f"probe seed{seed} | {backbone_name} | {split_name}")
                run_tracker.finish_step()
                step_idx += 1

        run_tracker.start_step(step_idx)
        raw_summary_df = pd.DataFrame(raw_metric_rows)
        probe_raw_df = pd.DataFrame(probe_metric_rows)
        raw_aggregate_df = summarize_raw_metrics(raw_summary_df)
        probe_aggregate_df = summarize_probe_metrics(probe_raw_df)
        summary_df = pd.concat([raw_aggregate_df, probe_aggregate_df], axis=0, ignore_index=True)
        raw_summary_df.to_csv(out_root / "summary_global_4_3_view_classification_raw.csv", index=False)
        probe_raw_df.to_csv(out_root / "summary_global_4_3_view_classification_probe_raw.csv", index=False)
        summary_df.to_csv(out_root / "summary_global_4_3_view_classification.csv", index=False)
        (out_root / "probe_lr_search_results.json").write_text(json.dumps(probe_lr_search_results, indent=2), encoding="utf-8")

        distribution_rows: List[Dict[str, object]] = []
        for split_name in ALLOWED_SPLITS:
            split_df = split_manifests[split_name]
            counts = split_df["view_label_9way"].value_counts().to_dict()
            for label in VIEW_LABELS_9WAY:
                distribution_rows.append(
                    {
                        "row_type": "count",
                        "split": split_name,
                        "view_label_9way": label,
                        "count": int(counts.get(label, 0)),
                    }
                )
        distribution_rows.extend(geometry_rows)
        pd.DataFrame(distribution_rows).to_csv(out_root / "summary_global_4_3_view_class_distribution.csv", index=False)
        run_tracker.finish_step()
        step_idx += 1

        run_tracker.start_step(step_idx)
        save_metric_compare_figure(summary_df, "accuracy", "Accuracy", out_root / "fig_global4_3_accuracy_compare.png")
        save_metric_compare_figure(summary_df, "macro_f1", "Macro-F1", out_root / "fig_global4_3_macro_f1_compare.png")
        save_metric_compare_figure(summary_df, "balanced_accuracy", "Balanced Accuracy", out_root / "fig_global4_3_balanced_accuracy_compare.png")
        save_view_umap(test_raw_embeddings, test_raw_manifest_map, out_root / "fig_global4_3_umap_view_category_raw.png", seed=args.seed, title_prefix="raw")
        save_view_umap(probe_embedding_seed11, probe_test_manifest_seed11, out_root / "fig_global4_3_umap_view_category_probe.png", seed=args.seed, title_prefix="probe seed11")
        run_tracker.finish_step()
        step_idx += 1

        run_meta = {
            "image_root": str(image_root),
            "dcm_root": str(dcm_root),
            "output_root": str(out_root.resolve()),
            "log_path": str(log_path.resolve()),
            "probe_seed_set": [int(v) for v in args.probe_seeds],
            "feature_source": "x_norm_clstoken",
            "view_taxonomy": "10deg_center_band_9way",
            "aggregate_policy": "probe=mean+-std, raw=single-run",
            "selected_targets": [TARGET_NAME],
            "manifest_hashes": manifest_hashes,
            "feature_hashes": feature_hashes,
            "raw_summary_path": str((out_root / "summary_global_4_3_view_classification_raw.csv").resolve()),
            "probe_raw_summary_path": str((out_root / "summary_global_4_3_view_classification_probe_raw.csv").resolve()),
            "aggregate_summary_path": str((out_root / "summary_global_4_3_view_classification.csv").resolve()),
            "view_distribution_summary_path": str((out_root / "summary_global_4_3_view_class_distribution.csv").resolve()),
            "fig_umap_raw": str((out_root / "fig_global4_3_umap_view_category_raw.png").resolve()),
            "fig_umap_probe": str((out_root / "fig_global4_3_umap_view_category_probe.png").resolve()),
            "step_count": len(steps),
            "steps": [{"name": step.name, "kind": step.kind} for step in steps],
        }

        run_tracker.start_step(step_idx)
        write_markdown_summary(out_root / "analysis_global_4_3_view_classification.md", summary_df, run_meta)
        run_tracker.finish_step()
        step_idx += 1

        run_tracker.start_step(step_idx)
        (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        run_tracker.finish_step()
        step_idx += 1

        log("Global Analysis 4-3 completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
