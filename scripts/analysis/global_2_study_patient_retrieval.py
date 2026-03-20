#!/usr/bin/env python3
"""
Global Analysis 2 — Study-level / Patient-level Retrieval.

Experiment matrix:
1) ImageNet - Raw Frozen Retrieval
2) ImageNet - Frozen Strict Linear Retrieval Probe
3) CAG - Raw Frozen Retrieval
4) CAG - Frozen Strict Linear Retrieval Probe

Targets:
- patient-level retrieval
- study-level retrieval
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
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
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}


@dataclass
class RetrievalMetrics:
    mode: str
    target: str
    backbone_name: str
    split: str
    seed: int | None
    best_lr: float | None
    epochs_trained: int | None
    num_images: int
    queries_with_positive: int
    queries_with_no_positive: int
    map_score: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    median_first_positive_rank: float

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
            "queries_with_positive": self.queries_with_positive,
            "queries_with_no_positive": self.queries_with_no_positive,
            "mAP": self.map_score,
            "recall_at_1": self.recall_at_1,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "median_first_positive_rank": self.median_first_positive_rank,
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

    def log_live_eta(
        self,
        current_probe_remaining_seconds: float,
        note_fields: Dict[str, object],
    ) -> None:
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
            "eta": eta_dt.strftime('%Y-%m-%d %H:%M:%S'),
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

    def remaining_total_seconds(self) -> float:
        total_elapsed = max(0.0, time.time() - self.global_start)
        avg_epoch_total = total_elapsed / max(1, self.completed_epochs)
        return max(0.0, (self.total_epoch_budget - self.completed_epochs) * avg_epoch_total)


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


class StrictLinearRetrievalProbe(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        return F.normalize(z, dim=1)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = float(temperature)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()
        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
        mask = mask * logits_mask
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        return -mean_log_prob_pos.mean()


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
    default_prefix: str,
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


def restore_console_logging(
    file_handle: TextIO,
    original_stdout: TextIO,
    original_stderr: TextIO,
) -> None:
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
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def collect_image_paths(root_dir: str, exts: Tuple[str, ...]) -> List[str]:
    all_paths: List[str] = []
    for ext in exts:
        all_paths.extend(glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True))
    return sorted(set(all_paths))


def hash_records(records: Iterable[Sequence[object]]) -> str:
    hasher = hashlib.sha256()
    for row in records:
        line = "\t".join(str(v) for v in row)
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> str:
    return hash_records(df.loc[:, list(columns)].itertuples(index=False, name=None))


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
    max_images_per_split: int | None,
    seed: int,
) -> pd.DataFrame:
    image_paths = collect_image_paths(image_root, exts)
    if not image_paths:
        raise ValueError(f"No images found under: {image_root}")
    rows = []
    for path in image_paths:
        patient_id, study_id = parse_patient_study_from_path(path)
        split, class_name = parse_split_and_class(path, image_root)
        rows.append(
            {
                "img_path": path,
                "patient_id": patient_id,
                "study_id": study_id,
                "split": split,
                "class_name": class_name,
            }
        )
    manifest = pd.DataFrame(rows).sort_values("img_path").reset_index(drop=True)
    if max_images_per_split is not None and max_images_per_split > 0:
        rng = np.random.default_rng(int(seed))
        sampled = []
        for split_name, split_df in manifest.groupby("split", sort=False):
            if len(split_df) <= max_images_per_split:
                sampled.append(split_df)
                continue
            selected = np.sort(rng.choice(len(split_df), size=max_images_per_split, replace=False))
            sampled.append(split_df.iloc[selected])
        manifest = pd.concat(sampled, axis=0).sort_values("img_path").reset_index(drop=True)
    manifest.insert(0, "image_id", np.arange(len(manifest), dtype=np.int64))
    return manifest


def split_manifest_map(manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    split_to_df: Dict[str, pd.DataFrame] = {}
    for split_name in ["train", "valid", "test"]:
        split_df = manifest.loc[manifest["split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"No images found for split={split_name}")
        split_to_df[split_name] = split_df
    return split_to_df


def save_manifest(split_manifest: pd.DataFrame, output_path: Path) -> None:
    split_manifest.to_csv(output_path, index=False)


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
        with open(meta_path, "r", encoding="utf-8") as handle:
            cached_meta = json.load(handle)
        if cached_meta == expected_meta:
            features = torch.load(features_path, map_location="cpu")
            if not isinstance(features, torch.Tensor):
                raise ValueError(f"Cached features are not a tensor: {features_path}")
            if features.ndim != 2 or features.shape[0] != len(manifest):
                raise ValueError(
                    f"Cached features shape mismatch for {features_path}: {tuple(features.shape)}"
                )
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

    model = Dinov3Backbone(
        weights=ckpt_path,
        model_name=args.model_name,
        repo_dir=args.repo_dir,
    ).to(requested_device)
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
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(expected_meta, handle, indent=2)
        manifest.loc[:, ["image_id", "img_path"]].to_csv(index_path, index=False)
    return features


def build_label_mapping(train_labels: Sequence[str]) -> Dict[str, int]:
    unique = sorted(set(str(v) for v in train_labels))
    return {key: idx for idx, key in enumerate(unique)}


def labels_to_tensor(values: Sequence[str], mapping: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([mapping[str(v)] for v in values], dtype=torch.long)


def get_target_values(manifest: pd.DataFrame, target: str) -> np.ndarray:
    if target == "patient":
        return manifest["patient_id"].to_numpy(dtype=str)
    if target == "study":
        return manifest["study_id"].to_numpy(dtype=str)
    raise ValueError(f"Unsupported target: {target}")


def l2_normalize_cpu(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1).cpu()


def compute_retrieval_metrics(
    embeddings: torch.Tensor,
    labels: Sequence[str],
    manifest: pd.DataFrame,
    target: str,
    mode: str,
    backbone_name: str,
    split_name: str,
    seed: int | None,
    best_lr: float | None,
    epochs_trained: int | None,
    max_topk_log: int,
) -> Tuple[RetrievalMetrics, pd.DataFrame, np.ndarray]:
    emb = l2_normalize_cpu(embeddings)
    sims = torch.matmul(emb, emb.T).numpy().astype(np.float32)
    n = emb.shape[0]
    labels_arr = np.asarray(labels, dtype=str)
    k_list = [1, 5, 10]
    recall_hits = {k: [] for k in k_list}
    ap_list: List[float] = []
    first_positive_ranks: List[int] = []
    per_query_rows: List[Dict[str, object]] = []
    queries_with_no_positive = 0

    image_paths = manifest["img_path"].to_numpy(dtype=str)
    patient_ids = manifest["patient_id"].to_numpy(dtype=str)
    study_ids = manifest["study_id"].to_numpy(dtype=str)

    for i in range(n):
        query_label = labels_arr[i]
        positive_mask = labels_arr == query_label
        positive_mask[i] = False
        positive_indices = np.flatnonzero(positive_mask)
        score_vec = sims[i].copy()
        score_vec[i] = -1e9
        sorted_idx = np.argsort(-score_vec)

        for k in k_list:
            recall_hits[k].append(int(len(positive_indices) > 0 and np.isin(sorted_idx[:k], positive_indices).any()))

        if len(positive_indices) == 0:
            queries_with_no_positive += 1
            ap = float("nan")
            first_positive_rank = np.nan
        else:
            y_true = np.zeros(n, dtype=np.int32)
            y_true[positive_indices] = 1
            ap = float(average_precision_score(y_true, score_vec))
            ap_list.append(ap)
            first_positive_rank = np.nan
            for rank, idx in enumerate(sorted_idx.tolist(), start=1):
                if idx in positive_indices:
                    first_positive_rank = rank
                    first_positive_ranks.append(rank)
                    break

        nn_idx = sorted_idx[:max_topk_log]
        per_query_rows.append(
            {
                "mode": mode,
                "target": target,
                "backbone_name": backbone_name,
                "split": split_name,
                "seed": seed if seed is not None else "",
                "query_index": i,
                "query_path": image_paths[i],
                "query_patient_id": patient_ids[i],
                "query_study_id": study_ids[i],
                "query_label": query_label,
                "num_positives": int(len(positive_indices)),
                "AP": ap,
                "first_positive_rank": first_positive_rank,
                "R@1": recall_hits[1][-1],
                "R@5": recall_hits[5][-1],
                "R@10": recall_hits[10][-1],
                "top_paths": " | ".join(image_paths[nn_idx]),
                "top_scores": " | ".join(f"{float(score_vec[j]):.6f}" for j in nn_idx),
                "top_patient_ids": " | ".join(patient_ids[nn_idx]),
                "top_study_ids": " | ".join(study_ids[nn_idx]),
                "top_is_positive": " | ".join(str(int(labels_arr[j] == query_label)) for j in nn_idx),
            }
        )

    valid_ap = np.asarray(ap_list, dtype=np.float64)
    valid_ranks = np.asarray(first_positive_ranks, dtype=np.float64)
    queries_with_positive = int(n - queries_with_no_positive)
    metrics = RetrievalMetrics(
        mode=mode,
        target=target,
        backbone_name=backbone_name,
        split=split_name,
        seed=seed,
        best_lr=best_lr,
        epochs_trained=epochs_trained,
        num_images=int(n),
        queries_with_positive=queries_with_positive,
        queries_with_no_positive=queries_with_no_positive,
        map_score=float(np.mean(valid_ap)) if valid_ap.size else float("nan"),
        recall_at_1=float(np.mean(recall_hits[1])) if recall_hits[1] else float("nan"),
        recall_at_5=float(np.mean(recall_hits[5])) if recall_hits[5] else float("nan"),
        recall_at_10=float(np.mean(recall_hits[10])) if recall_hits[10] else float("nan"),
        median_first_positive_rank=float(np.median(valid_ranks)) if valid_ranks.size else float("nan"),
    )
    per_query_df = pd.DataFrame(per_query_rows)
    rank_values = np.asarray(first_positive_ranks, dtype=np.int32)
    return metrics, per_query_df, rank_values


def train_one_epoch_probe(
    model: StrictLinearRetrievalProbe,
    features: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    epoch_seed: int,
) -> float:
    model.train()
    num_items = int(features.shape[0])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(epoch_seed))
    perm = torch.randperm(num_items, generator=generator)
    losses: List[float] = []
    for start in range(0, num_items, batch_size):
        idx = perm[start : start + batch_size]
        x = features[idx].to(device, non_blocking=True)
        y = labels[idx].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        z = model(x)
        loss = criterion(z, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def evaluate_probe_model(
    model: StrictLinearRetrievalProbe,
    features: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), batch_size):
            x = features[start : start + batch_size].to(device, non_blocking=True)
            outputs.append(model(x).cpu())
    return torch.cat(outputs, dim=0)


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
        "strict_deterministic": bool(args.strict_deterministic),
    }


def train_probe_with_lr_search(
    backbone_name: str,
    target: str,
    seed: int,
    train_features: torch.Tensor,
    valid_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels_raw: Sequence[str],
    valid_labels_raw: Sequence[str],
    test_labels_raw: Sequence[str],
    valid_manifest: pd.DataFrame,
    out_root: Path,
    args: argparse.Namespace,
    run_tracker: AnalysisRunTracker,
    tracker: ProbeEpochTracker,
    feature_hashes: Dict[str, str],
    manifest_hashes: Dict[str, str],
) -> Tuple[StrictLinearRetrievalProbe, Dict[str, object], List[Dict[str, object]]]:
    del test_features, test_labels_raw  # used only after training; not needed for selection here.
    embed_dim = int(train_features.shape[1])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    label_mapping = build_label_mapping(train_labels_raw)
    train_labels = labels_to_tensor(train_labels_raw, label_mapping)
    job_signature = make_probe_signature(
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

    final_ckpt_path = out_root / f"probe_checkpoint_seed{seed}_{backbone_name}_{target}.pt"
    resume_ckpt_path = out_root / f"probe_resume_seed{seed}_{backbone_name}_{target}.pt"
    history_path = out_root / f"history_seed{seed}_{backbone_name}_{target}.json"

    if final_ckpt_path.exists():
        payload = torch.load(final_ckpt_path, map_location="cpu")
        if payload.get("signature") == job_signature:
            set_global_seed(seed, strict_deterministic=args.strict_deterministic)
            model = StrictLinearRetrievalProbe(embed_dim)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            log(f"Reusing final trained probe checkpoint: {final_ckpt_path}")
            return model, payload["summary"], payload.get("lr_search_rows", [])

    lr_search_rows: List[Dict[str, object]] = []
    completed_candidates: List[Dict[str, object]] = []
    best_summary: Dict[str, object] | None = None
    best_state_dict: Dict[str, torch.Tensor] | None = None
    resume_payload = None
    if resume_ckpt_path.exists():
        payload = torch.load(resume_ckpt_path, map_location="cpu")
        if payload.get("signature") == job_signature:
            resume_payload = payload
            lr_search_rows = list(payload.get("lr_search_rows", []))
            completed_candidates = list(payload.get("completed_candidates", []))
            best_summary = payload.get("best_summary")
            best_state_dict = payload.get("best_state_dict")
            log(
                f"Loaded resume checkpoint | seed={seed} backbone={backbone_name} target={target} | "
                f"lr_index={payload.get('current_lr_index', 0)} epoch={payload.get('current_epoch', 0)}"
            )

    candidate_start_index = 0
    if resume_payload is not None:
        candidate_start_index = int(resume_payload.get("current_lr_index", 0))

    criterion = SupConLoss(temperature=args.supcon_temperature)

    for lr_index, lr in enumerate(args.probe_lr_grid):
        if lr_index < candidate_start_index:
            continue

        resume_candidate = resume_payload if (resume_payload is not None and int(resume_payload.get("current_lr_index", -1)) == lr_index) else None
        if resume_candidate is None:
            set_global_seed(seed, strict_deterministic=args.strict_deterministic)
            model = StrictLinearRetrievalProbe(embed_dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=args.probe_weight_decay)
            start_epoch = 1
            history_rows: List[Dict[str, object]] = []
            best_valid_map = -float("inf")
            best_epoch = 0
            best_candidate_state = None
            epochs_without_improvement = 0
        else:
            set_global_seed(seed, strict_deterministic=args.strict_deterministic)
            model = StrictLinearRetrievalProbe(embed_dim).to(device)
            model.load_state_dict(resume_candidate["model_state_dict"], strict=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=args.probe_weight_decay)
            optimizer.load_state_dict(resume_candidate["optimizer_state_dict"])
            start_epoch = int(resume_candidate.get("current_epoch", 0)) + 1
            history_rows = list(resume_candidate.get("current_history", []))
            best_valid_map = float(resume_candidate.get("current_best_valid_map", -float("inf")))
            best_epoch = int(resume_candidate.get("current_best_epoch", 0))
            best_candidate_state = resume_candidate.get("current_best_state_dict")
            epochs_without_improvement = int(resume_candidate.get("current_epochs_without_improvement", 0))
            log(
                f"Resuming mid-candidate | seed={seed} backbone={backbone_name} target={target} "
                f"lr={lr} start_epoch={start_epoch}"
            )

        tracker.start_lr(float(lr), lr_index + 1)
        for epoch in range(start_epoch, int(args.probe_max_epoch) + 1):
            epoch_started = time.time()
            epoch_seed = int(seed * 1000 + lr_index * 100 + epoch)
            train_loss = train_one_epoch_probe(
                model=model,
                features=train_features,
                labels=train_labels,
                optimizer=optimizer,
                criterion=criterion,
                batch_size=int(args.probe_batch_size),
                device=device,
                epoch_seed=epoch_seed,
            )
            valid_embeddings = evaluate_probe_model(model, valid_features, int(args.probe_batch_size), device)
            valid_metrics, _, _ = compute_retrieval_metrics(
                embeddings=valid_embeddings,
                labels=valid_labels_raw,
                manifest=valid_manifest,
                target=target,
                mode="probe_linear",
                backbone_name=backbone_name,
                split_name="valid",
                seed=seed,
                best_lr=float(lr),
                epochs_trained=epoch,
                max_topk_log=args.max_topk_log,
            )
            history_rows.append(
                {
                    "epoch": epoch,
                    "lr": float(lr),
                    "train_loss": float(train_loss),
                    "valid_mAP": float(valid_metrics.map_score),
                    "valid_recall_at_1": float(valid_metrics.recall_at_1),
                    "valid_recall_at_5": float(valid_metrics.recall_at_5),
                    "valid_recall_at_10": float(valid_metrics.recall_at_10),
                }
            )
            current_valid_map = float(valid_metrics.map_score)
            current_score = current_valid_map if not math.isnan(current_valid_map) else -float("inf")
            best_score = best_valid_map if not math.isnan(best_valid_map) else -float("inf")
            improved = best_candidate_state is None or current_score > (best_score + float(args.probe_min_delta))
            if improved:
                best_valid_map = current_valid_map
                best_epoch = int(epoch)
                best_candidate_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            log(
                f"[{backbone_name}] seed={seed} target={target} lr={lr} epoch={epoch} | "
                f"train_loss={train_loss:.6f} valid_mAP={valid_metrics.map_score:.6f} "
                f"valid_R@1={valid_metrics.recall_at_1:.6f} valid_R@5={valid_metrics.recall_at_5:.6f}"
            )
            epoch_seconds = max(0.0, time.time() - epoch_started)
            tracker.finish_epoch(epoch, epoch_seconds)
            torch.save(
                {
                    "signature": job_signature,
                    "current_lr_index": lr_index,
                    "current_epoch": epoch,
                    "current_history": history_rows,
                    "current_best_valid_map": best_valid_map,
                    "current_best_epoch": best_epoch,
                    "current_best_state_dict": best_candidate_state,
                    "current_epochs_without_improvement": epochs_without_improvement,
                    "completed_candidates": completed_candidates,
                    "best_summary": best_summary,
                    "best_state_dict": best_state_dict,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_search_rows": lr_search_rows,
                },
                resume_ckpt_path,
            )
            if epochs_without_improvement >= int(args.probe_patience):
                tracker.skip_remaining_epochs(int(args.probe_max_epoch) - epoch)
                log(
                    f"Early stopping | seed={seed} backbone={backbone_name} target={target} "
                    f"lr={lr} best_epoch={best_epoch} best_valid_mAP={best_valid_map:.6f}"
                )
                break

        if best_candidate_state is None:
            raise RuntimeError(f"No best state captured for seed={seed}, backbone={backbone_name}, target={target}, lr={lr}")
        candidate_summary = {
            "seed": int(seed),
            "backbone_name": backbone_name,
            "target": target,
            "lr": float(lr),
            "best_valid_mAP": float(best_valid_map),
            "best_epoch": int(best_epoch),
            "epochs_trained": len(history_rows),
        }
        completed_candidates.append(candidate_summary)
        lr_search_rows.append(candidate_summary)

        if best_summary is None or float(best_valid_map) > float(best_summary["best_valid_mAP"]):
            best_summary = candidate_summary
            best_state_dict = best_candidate_state
        resume_payload = None

    if best_state_dict is None or best_summary is None:
        raise RuntimeError(f"Training failed to produce a valid probe for {backbone_name}/{target}/seed={seed}")

    final_model = StrictLinearRetrievalProbe(embed_dim)
    final_model.load_state_dict(best_state_dict, strict=True)
    final_model = final_model.to(device)
    torch.save(
        {
            "signature": job_signature,
            "model_state_dict": best_state_dict,
            "summary": best_summary,
            "lr_search_rows": lr_search_rows,
        },
        final_ckpt_path,
    )
    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(lr_search_rows, handle, indent=2)
    return final_model.cpu(), best_summary, lr_search_rows


def summarize_probe_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (target, backbone_name, split), group in raw_df.groupby(["target", "backbone_name", "split"], sort=True):
        best_lrs = group["best_lr"].astype(str)
        counts = best_lrs.value_counts().sort_index()
        row = {
            "mode": "probe_linear",
            "target": target,
            "backbone_name": backbone_name,
            "split": split,
            "num_seeds": int(group["seed"].nunique()),
            "mAP_mean": float(group["mAP"].mean()),
            "mAP_std": float(group["mAP"].std(ddof=1)) if len(group) > 1 else 0.0,
            "recall_at_1_mean": float(group["recall_at_1"].mean()),
            "recall_at_1_std": float(group["recall_at_1"].std(ddof=1)) if len(group) > 1 else 0.0,
            "recall_at_5_mean": float(group["recall_at_5"].mean()),
            "recall_at_5_std": float(group["recall_at_5"].std(ddof=1)) if len(group) > 1 else 0.0,
            "recall_at_10_mean": float(group["recall_at_10"].mean()),
            "recall_at_10_std": float(group["recall_at_10"].std(ddof=1)) if len(group) > 1 else 0.0,
            "median_first_positive_rank_mean": float(group["median_first_positive_rank"].mean()),
            "queries_with_positive_mean": float(group["queries_with_positive"].mean()),
            "queries_with_no_positive_mean": float(group["queries_with_no_positive"].mean()),
            "epochs_trained_mean": float(pd.to_numeric(group["epochs_trained"], errors="coerce").mean()),
            "epochs_trained_std": float(pd.to_numeric(group["epochs_trained"], errors="coerce").std(ddof=1)) if len(group) > 1 else 0.0,
            "best_lr_mode": counts.idxmax() if not counts.empty else "",
            "best_lr_counts": json.dumps({str(k): int(v) for k, v in counts.items()}),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_raw_metrics(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in raw_df.iterrows():
        rows.append(
            {
                "mode": "raw_frozen",
                "target": row["target"],
                "backbone_name": row["backbone_name"],
                "split": row["split"],
                "num_seeds": 1,
                "mAP_mean": float(row["mAP"]),
                "mAP_std": 0.0,
                "recall_at_1_mean": float(row["recall_at_1"]),
                "recall_at_1_std": 0.0,
                "recall_at_5_mean": float(row["recall_at_5"]),
                "recall_at_5_std": 0.0,
                "recall_at_10_mean": float(row["recall_at_10"]),
                "recall_at_10_std": 0.0,
                "median_first_positive_rank_mean": float(row["median_first_positive_rank"]),
                "queries_with_positive_mean": float(row["queries_with_positive"]),
                "queries_with_no_positive_mean": float(row["queries_with_no_positive"]),
                "epochs_trained_mean": float("nan"),
                "epochs_trained_std": float("nan"),
                "best_lr_mode": "",
                "best_lr_counts": "{}",
            }
        )
    return pd.DataFrame(rows)


def make_grouped_bar_positions(group_count: int, bar_count: int, width: float = 0.18) -> Tuple[np.ndarray, List[np.ndarray]]:
    centers = np.arange(group_count, dtype=np.float64)
    offsets = []
    start = -width * (bar_count - 1) / 2.0
    for idx in range(bar_count):
        offsets.append(centers + start + idx * width)
    return centers, offsets


def save_map_compare_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df[summary_df["split"] == "test"].copy()
    targets = ["patient", "study"]
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    centers, xs = make_grouped_bar_positions(len(targets), len(conditions), width=0.18)
    plt.figure(figsize=(10, 6))
    for idx, (backbone_name, mode, label) in enumerate(conditions):
        means = []
        errs = []
        for target in targets:
            row = plot_df[(plot_df["target"] == target) & (plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)]
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(row.iloc[0]["mAP_mean"]))
                errs.append(float(row.iloc[0]["mAP_std"]))
        color = PLOT_COLORS[backbone_name]
        alpha = 0.55 if mode == "raw_frozen" else 0.95
        plt.bar(xs[idx], means, width=0.18, color=color, alpha=alpha, label=label, yerr=errs, capsize=4)
    plt.xticks(centers, ["Patient", "Study"])
    plt.ylabel("mAP")
    plt.title("Global Analysis 2: Raw vs Probe Retrieval mAP")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_recall_compare_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df[summary_df["split"] == "test"].copy()
    targets = ["patient", "study"]
    ks = [1, 5, 10]
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, target in zip(axes, targets):
        centers, xs = make_grouped_bar_positions(len(ks), len(conditions), width=0.18)
        for idx, (backbone_name, mode, label) in enumerate(conditions):
            means = []
            errs = []
            row = plot_df[(plot_df["target"] == target) & (plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)]
            for k in ks:
                if row.empty:
                    means.append(np.nan)
                    errs.append(0.0)
                else:
                    means.append(float(row.iloc[0][f"recall_at_{k}_mean"]))
                    errs.append(float(row.iloc[0][f"recall_at_{k}_std"]))
            color = PLOT_COLORS[backbone_name]
            alpha = 0.55 if mode == "raw_frozen" else 0.95
            ax.bar(xs[idx], means, width=0.18, color=color, alpha=alpha, yerr=errs, capsize=4, label=label)
        ax.set_xticks(centers, ["R@1", "R@5", "R@10"])
        ax.set_title(f"{target.title()} Retrieval")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Recall@K")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Global Analysis 2: Recall@K Comparison", y=1.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_cdf(points: np.ndarray, max_rank: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros(max_rank, dtype=np.float64)
    xs = np.arange(1, max_rank + 1)
    return np.asarray([(points <= x).mean() for x in xs], dtype=np.float64)


def save_rank_cdf_figure(
    raw_rank_map: Dict[Tuple[str, str], np.ndarray],
    probe_rank_map: Dict[Tuple[str, str], List[np.ndarray]],
    output_path: Path,
) -> None:
    targets = ["patient", "study"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, target in zip(axes, targets):
        max_rank = 1
        for backbone_name in ["imagenet", "cag"]:
            raw_points = raw_rank_map.get((target, backbone_name), np.array([], dtype=np.int32))
            if raw_points.size:
                max_rank = max(max_rank, int(np.percentile(raw_points, 95)))
            for seed_points in probe_rank_map.get((target, backbone_name), []):
                if seed_points.size:
                    max_rank = max(max_rank, int(np.percentile(seed_points, 95)))
        xs = np.arange(1, max_rank + 1)
        for backbone_name in ["imagenet", "cag"]:
            color = PLOT_COLORS[backbone_name]
            raw_points = raw_rank_map.get((target, backbone_name), np.array([], dtype=np.int32))
            raw_cdf = build_cdf(raw_points, max_rank)
            ax.plot(xs, raw_cdf, color=color, linestyle="--", linewidth=2, label=f"{backbone_name.title()} Raw")
            probe_curves = [build_cdf(points, max_rank) for points in probe_rank_map.get((target, backbone_name), [])]
            if probe_curves:
                probe_arr = np.stack(probe_curves, axis=0)
                mean = probe_arr.mean(axis=0)
                std = probe_arr.std(axis=0, ddof=1) if probe_arr.shape[0] > 1 else np.zeros_like(mean)
                ax.plot(xs, mean, color=color, linewidth=2, label=f"{backbone_name.title()} Probe")
                ax.fill_between(xs, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), color=color, alpha=0.18)
        ax.set_title(f"{target.title()} Retrieval")
        ax.set_xlabel("First Positive Rank")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Query Ratio (CDF)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Global Analysis 2: First Positive Rank CDF", y=1.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_probe_minus_raw_delta_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df[summary_df["split"] == "test"].copy()
    targets = ["patient", "study"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, target in zip(axes, targets):
        labels = ["ImageNet", "CAG"]
        xs = np.arange(len(labels))
        means = []
        errs = []
        for backbone_name in ["imagenet", "cag"]:
            probe_row = plot_df[(plot_df["target"] == target) & (plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == "probe_linear")]
            raw_row = plot_df[(plot_df["target"] == target) & (plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == "raw_frozen")]
            if probe_row.empty or raw_row.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                delta = float(probe_row.iloc[0]["mAP_mean"]) - float(raw_row.iloc[0]["mAP_mean"])
                means.append(delta)
                errs.append(float(probe_row.iloc[0]["mAP_std"]))
        colors = [PLOT_COLORS["imagenet"], PLOT_COLORS["cag"]]
        bars = ax.bar(xs, means, color=colors, yerr=errs, capsize=4)
        ax.axhline(0.0, color="black", linewidth=1)
        for bar, value in zip(bars, means):
            if np.isfinite(value):
                ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:+.4f}", ha="center", va="bottom")
        ax.set_xticks(xs, labels)
        ax.set_title(f"{target.title()} Retrieval")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Probe mAP - Raw mAP")
    fig.suptitle("Global Analysis 2: Probe Improvement over Raw Retrieval", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    return f"{value:.6f}"


def write_markdown_summary(output_path: Path, summary_df: pd.DataFrame, run_meta: Dict[str, object]) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    lines = []
    lines.append("# Global Analysis 2: Study-level / Patient-level Retrieval")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Dataset: `{run_meta['image_root']}`")
    lines.append(f"- Seed set (probe only): `{', '.join(str(v) for v in run_meta['probe_seed_set'])}`")
    lines.append(f"- Targets: `patient`, `study`")
    lines.append(f"- Modes: `raw_frozen`, `probe_linear`")
    lines.append("")
    lines.append("## Test Summary")
    lines.append("")
    lines.append("| Mode | Target | Backbone | mAP | R@1 | R@5 | R@10 | Median First Positive Rank |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in test_df.sort_values(["target", "mode", "backbone_name"]).iterrows():
        lines.append(
            f"| {row['mode']} | {row['target']} | {row['backbone_name']} | "
            f"{format_float(float(row['mAP_mean']))} +/- {format_float(float(row['mAP_std']))} | "
            f"{format_float(float(row['recall_at_1_mean']))} +/- {format_float(float(row['recall_at_1_std']))} | "
            f"{format_float(float(row['recall_at_5_mean']))} +/- {format_float(float(row['recall_at_5_std']))} | "
            f"{format_float(float(row['recall_at_10_mean']))} +/- {format_float(float(row['recall_at_10_std']))} | "
            f"{format_float(float(row['median_first_positive_rank_mean']))} |"
        )
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `raw 약함 + probe 회복` -> raw CLS geometry weakness")
    lines.append("- `raw 약함 + probe도 약함` -> global decodability weakness")
    lines.append("- `study만 약함` -> finer-grained study coherence weakness")
    lines.append("- `patient도 study도 약함` -> broader global representation weakness")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_steps(args: argparse.Namespace) -> List[StepDef]:
    steps: List[StepDef] = [StepDef("Build split manifests", "short")]
    for backbone_name in ["imagenet", "cag"]:
        for split_name in ["train", "valid", "test"]:
            steps.append(StepDef(f"Extract {backbone_name} {split_name} CLS features", "short"))
    for backbone_name in ["imagenet", "cag"]:
        for target in ["patient", "study"]:
            for split_name in ["valid", "test"]:
                steps.append(StepDef(f"Run raw retrieval | backbone={backbone_name} target={target} split={split_name}", "short"))
    for seed in args.probe_seeds:
        for backbone_name in ["imagenet", "cag"]:
            for target in ["patient", "study"]:
                steps.append(StepDef(f"Run probe retrieval | seed={seed} backbone={backbone_name} target={target}", "probe"))
    steps.extend(
        [
            StepDef("Write summary CSVs", "short"),
            StepDef("Render figures", "short"),
            StepDef("Write markdown summary", "short"),
            StepDef("Write run metadata", "short"),
        ]
    )
    return steps


def make_feature_hash(features: torch.Tensor) -> str:
    arr = features.numpy().astype(np.float32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--output-root", default="outputs/global_2_study_patient_retrieval_unique_view")
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
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--probe-lr-grid", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-min-delta", type=float, default=0.0)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--max-topk-log", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    ensure_dir(out_root)
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        out_root,
        args.log_file,
        default_prefix="global_2_study_patient_retrieval",
    )

    try:
        set_global_seed(args.seed, strict_deterministic=args.strict_deterministic)
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        steps = build_steps(args)
        run_tracker = AnalysisRunTracker(steps)
        step_idx = 0

        run_tracker.start_step(step_idx)
        manifest = build_manifest(
            image_root=args.image_root,
            exts=("png", "jpg", "jpeg"),
            max_images_per_split=args.max_images_per_split,
            seed=args.seed,
        )
        split_manifests = split_manifest_map(manifest)
        for split_name, split_df in split_manifests.items():
            save_manifest(split_df, out_root / f"image_manifest_{split_name}.csv")
        manifest_hashes = {
            split_name: hash_dataframe(split_df, ["image_id", "img_path", "patient_id", "study_id"])
            for split_name, split_df in split_manifests.items()
        }
        run_tracker.finish_step()
        step_idx += 1

        backbone_ckpts = {"imagenet": args.imagenet_ckpt, "cag": args.cag_ckpt}
        feature_store: Dict[str, Dict[str, torch.Tensor]] = {"imagenet": {}, "cag": {}}
        feature_hashes: Dict[str, Dict[str, str]] = {"imagenet": {}, "cag": {}}
        for backbone_name in ["imagenet", "cag"]:
            for split_name in ["train", "valid", "test"]:
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
        raw_rank_map: Dict[Tuple[str, str], np.ndarray] = {}
        for backbone_name in ["imagenet", "cag"]:
            for target in ["patient", "study"]:
                for split_name in ["valid", "test"]:
                    run_tracker.start_step(step_idx)
                    split_df = split_manifests[split_name]
                    labels = get_target_values(split_df, target)
                    metrics, per_query_df, rank_values = compute_retrieval_metrics(
                        embeddings=feature_store[backbone_name][split_name],
                        labels=labels,
                        manifest=split_df,
                        target=target,
                        mode="raw_frozen",
                        backbone_name=backbone_name,
                        split_name=split_name,
                        seed=None,
                        best_lr=None,
                        epochs_trained=None,
                        max_topk_log=args.max_topk_log,
                    )
                    raw_metric_rows.append(metrics.to_row())
                    per_query_df.to_csv(
                        out_root / f"per_query_{target}_raw_{backbone_name}_{split_name}.csv",
                        index=False,
                    )
                    if split_name == "test":
                        raw_rank_map[(target, backbone_name)] = rank_values
                    run_tracker.finish_step()
                    step_idx += 1

        probe_metric_rows: List[Dict[str, object]] = []
        probe_lr_search_results: Dict[str, Dict[str, Dict[str, List[Dict[str, object]]]]] = {}
        probe_rank_map: Dict[Tuple[str, str], List[np.ndarray]] = {}

        for seed in args.probe_seeds:
            for backbone_name in ["imagenet", "cag"]:
                probe_lr_search_results.setdefault(str(seed), {}).setdefault(backbone_name, {})
                for target in ["patient", "study"]:
                    run_tracker.start_step(step_idx)
                    tracker = ProbeEpochTracker(
                        run_tracker=run_tracker,
                        backbone_name=backbone_name,
                        target=target,
                        seed=seed,
                        lr_values=args.probe_lr_grid,
                        max_epoch=args.probe_max_epoch,
                    )
                    split_label_values = {
                        split_name: get_target_values(split_manifests[split_name], target)
                        for split_name in ["train", "valid", "test"]
                    }
                    model, best_summary, lr_rows = train_probe_with_lr_search(
                        backbone_name=backbone_name,
                        target=target,
                        seed=seed,
                        train_features=feature_store[backbone_name]["train"],
                        valid_features=feature_store[backbone_name]["valid"],
                        test_features=feature_store[backbone_name]["test"],
                        train_labels_raw=split_label_values["train"],
                        valid_labels_raw=split_label_values["valid"],
                        test_labels_raw=split_label_values["test"],
                        valid_manifest=split_manifests["valid"],
                        out_root=out_root,
                        args=args,
                        run_tracker=run_tracker,
                        tracker=tracker,
                        feature_hashes=feature_hashes[backbone_name],
                        manifest_hashes=manifest_hashes,
                    )
                    probe_lr_search_results[str(seed)][backbone_name][target] = lr_rows
                    model = model.to(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
                    for split_name in ["valid", "test"]:
                        probe_embeddings = evaluate_probe_model(
                            model=model,
                            features=feature_store[backbone_name][split_name],
                            batch_size=args.probe_batch_size,
                            device=torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"),
                        )
                        metrics, per_query_df, rank_values = compute_retrieval_metrics(
                            embeddings=probe_embeddings,
                            labels=split_label_values[split_name],
                            manifest=split_manifests[split_name],
                            target=target,
                            mode="probe_linear",
                            backbone_name=backbone_name,
                            split_name=split_name,
                            seed=seed,
                            best_lr=float(best_summary["lr"]),
                            epochs_trained=int(best_summary["epochs_trained"]),
                            max_topk_log=args.max_topk_log,
                        )
                        probe_metric_rows.append(metrics.to_row())
                        per_query_df.to_csv(
                            out_root / f"per_query_{target}_probe_seed{seed}_{backbone_name}_{split_name}.csv",
                            index=False,
                        )
                        if split_name == "test":
                            probe_rank_map.setdefault((target, backbone_name), []).append(rank_values)
                    run_tracker.finish_step()
                    step_idx += 1

        run_tracker.start_step(step_idx)
        raw_summary_df = pd.DataFrame(raw_metric_rows)
        probe_raw_df = pd.DataFrame(probe_metric_rows)
        raw_aggregate_df = summarize_raw_metrics(raw_summary_df)
        probe_aggregate_df = summarize_probe_metrics(probe_raw_df)
        summary_df = pd.concat([raw_aggregate_df, probe_aggregate_df], axis=0, ignore_index=True)
        raw_summary_df.to_csv(out_root / "summary_global_2_retrieval_raw.csv", index=False)
        probe_raw_df.to_csv(out_root / "summary_global_2_retrieval_probe_raw.csv", index=False)
        summary_df.to_csv(out_root / "summary_global_2_retrieval.csv", index=False)
        with open(out_root / "probe_lr_search_results.json", "w", encoding="utf-8") as handle:
            json.dump(probe_lr_search_results, handle, indent=2)
        run_tracker.finish_step()
        step_idx += 1

        run_tracker.start_step(step_idx)
        save_map_compare_figure(summary_df, out_root / "fig_global2_map_compare.png")
        save_recall_compare_figure(summary_df, out_root / "fig_global2_recall_at_k_compare.png")
        save_rank_cdf_figure(raw_rank_map, probe_rank_map, out_root / "fig_global2_rank_cdf_compare.png")
        save_probe_minus_raw_delta_figure(summary_df, out_root / "fig_global2_probe_minus_raw_delta.png")
        run_tracker.finish_step()
        step_idx += 1

        run_meta = {
            "image_root": str(Path(args.image_root).resolve()),
            "output_root": str(out_root.resolve()),
            "log_path": str(log_path.resolve()),
            "probe_seed_set": [int(v) for v in args.probe_seeds],
            "feature_source": "x_norm_clstoken",
            "aggregate_policy": "probe=mean+-std, raw=single-run",
            "selected_targets": ["patient", "study"],
            "manifest_hashes": manifest_hashes,
            "feature_hashes": feature_hashes,
            "raw_summary_path": str((out_root / "summary_global_2_retrieval_raw.csv").resolve()),
            "probe_raw_summary_path": str((out_root / "summary_global_2_retrieval_probe_raw.csv").resolve()),
            "aggregate_summary_path": str((out_root / "summary_global_2_retrieval.csv").resolve()),
            "step_count": len(steps),
            "steps": [{"name": step.name, "kind": step.kind} for step in steps],
        }

        run_tracker.start_step(step_idx)
        write_markdown_summary(out_root / "analysis_global_2_study_patient_retrieval.md", summary_df, run_meta)
        run_tracker.finish_step()
        step_idx += 1

        run_tracker.start_step(step_idx)
        with open(out_root / "run_meta.json", "w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, indent=2)
        run_tracker.finish_step()
        step_idx += 1

        log("Global Analysis 2 completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
