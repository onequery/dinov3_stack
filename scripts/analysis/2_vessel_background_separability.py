#!/usr/bin/env python3
"""
Analysis 2 — Vessel vs Background Separability

Compare frozen patch-token representations between:
1) ImageNet-1K pretrained backbone
2) CAG contrast pretrained backbone
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import tempfile
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
import torch.nn.functional as F
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
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
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
PATIENT_RE = re.compile(r"(^|[\\/])(\d{8})(?=([\\/]|$))")


@dataclass
class ProbeMetrics:
    backbone_name: str
    split: str
    alpha: float
    n_vessel_patches: int
    n_background_patches: int
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    f1: float
    vessel_recall: float
    background_recall: float

    def to_row(self) -> Dict[str, float | int | str]:
        return {
            "backbone_name": self.backbone_name,
            "split": self.split,
            "alpha": self.alpha,
            "n_vessel_patches": self.n_vessel_patches,
            "n_background_patches": self.n_background_patches,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "vessel_recall": self.vessel_recall,
            "background_recall": self.background_recall,
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


class ImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], img_size: Tuple[int, int]):
        self.image_paths = list(image_paths)
        self.width = int(img_size[0])
        self.height = int(img_size[1])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, image_path


def image_collate_fn(batch: Sequence[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    images = torch.stack([item[0] for item in batch], dim=0)
    paths = [item[1] for item in batch]
    return images, paths


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
) -> Tuple[Path, TextIO, TextIO, TextIO]:
    if log_file_arg:
        log_path = Path(log_file_arg).expanduser()
        if not log_path.is_absolute():
            log_path = output_root / log_path
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = output_root / f"2_vessel_background_separability_{stamp}.log"

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
        line = "\t".join(str(v) for v in row)
        hasher.update(line.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> str:
    return hash_records(df.loc[:, list(columns)].itertuples(index=False, name=None))


def parse_patient_id(path: str) -> str:
    match = PATIENT_RE.search(path)
    return match.group(2) if match else ""


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
    shared_paths = sorted(set(image_map.keys()) & set(mask_map.keys()))
    if not shared_paths:
        raise ValueError(
            "No paired image/mask files found. "
            f"images_root={image_root}, masks_root={mask_root}"
        )
    return [image_map[path] for path in shared_paths], [mask_map[path] for path in shared_paths]


def load_seg_config(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Segmentation config must be a mapping.")
    classes = cfg.get("ALL_CLASSES", [])
    if classes != ["background", "coronary"]:
        log(f"[WARN] Unexpected ALL_CLASSES in seg config: {classes}")
    return cfg


def maybe_subsample_pairs(
    image_paths: List[str],
    mask_paths: List[str],
    max_images: int | None,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if max_images is None or max_images <= 0 or max_images >= len(image_paths):
        return image_paths, mask_paths
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(image_paths), size=max_images, replace=False))
    return [image_paths[i] for i in selected], [mask_paths[i] for i in selected]


def compute_patch_occupancy(mask_path: str, img_size: Tuple[int, int], patch_size: int) -> np.ndarray:
    width, height = int(img_size[0]), int(img_size[1])
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError("img-size must be divisible by patch-size")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    mask = (mask > 0).astype(np.float32)
    rows = height // patch_size
    cols = width // patch_size
    grid = (
        mask.reshape(rows, patch_size, cols, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(rows * cols, patch_size * patch_size)
    )
    return grid.mean(axis=1)


def sample_indices(indices: np.ndarray, cap: int | None, rng: np.random.Generator) -> np.ndarray:
    if cap is None or cap <= 0 or len(indices) <= cap:
        return np.sort(indices)
    return np.sort(rng.choice(indices, size=cap, replace=False))


def build_patch_rows_for_image(
    split_name: str,
    image_path: str,
    mask_path: str,
    img_size: Tuple[int, int],
    patch_size: int,
    vessel_frac_pos_thr: float,
    background_frac_neg_thr: float,
    subset_mode: str,
    max_pos: int | None,
    max_neg: int | None,
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    occupancies = compute_patch_occupancy(mask_path, img_size=img_size, patch_size=patch_size)
    pos_idx = np.where(occupancies >= vessel_frac_pos_thr)[0]
    neg_idx = np.where(occupancies <= background_frac_neg_thr)[0]

    if subset_mode == "balanced_sample":
        pos_idx = sample_indices(pos_idx, max_pos, rng)
        neg_idx = sample_indices(neg_idx, max_neg, rng)
    elif subset_mode != "all_eligible":
        raise ValueError(f"Unsupported subset_mode={subset_mode}")

    rows = img_size[1] // patch_size
    cols = img_size[0] // patch_size
    rel_image = image_path
    rel_mask = mask_path
    patient_id = parse_patient_id(image_path)

    out: List[Dict[str, object]] = []
    for patch_index in pos_idx.tolist():
        out.append(
            {
                "split": split_name,
                "image_path": rel_image,
                "mask_path": rel_mask,
                "patient_id": patient_id,
                "patch_index": int(patch_index),
                "patch_row": int(patch_index // cols),
                "patch_col": int(patch_index % cols),
                "vessel_fraction": float(occupancies[patch_index]),
                "patch_label": "vessel",
            }
        )
    for patch_index in neg_idx.tolist():
        out.append(
            {
                "split": split_name,
                "image_path": rel_image,
                "mask_path": rel_mask,
                "patient_id": patient_id,
                "patch_index": int(patch_index),
                "patch_row": int(patch_index // cols),
                "patch_col": int(patch_index % cols),
                "vessel_fraction": float(occupancies[patch_index]),
                "patch_label": "background",
            }
        )
    return out


def build_or_load_patch_manifest(
    subset_name: str,
    split_name: str,
    image_paths: List[str],
    mask_paths: List[str],
    img_size: Tuple[int, int],
    patch_size: int,
    vessel_frac_pos_thr: float,
    background_frac_neg_thr: float,
    out_root: Path,
    seed: int,
    subset_mode: str,
    max_pos: int | None,
    max_neg: int | None,
) -> pd.DataFrame:
    manifest_path = out_root / f"patch_manifest_{subset_name}.csv"
    meta_path = out_root / f"patch_manifest_{subset_name}.meta.json"
    pair_signature = hash_records(zip(image_paths, mask_paths))
    desired_meta = {
        "split": split_name,
        "pair_signature": pair_signature,
        "img_size": list(img_size),
        "patch_size": int(patch_size),
        "vessel_frac_pos_thr": float(vessel_frac_pos_thr),
        "background_frac_neg_thr": float(background_frac_neg_thr),
        "subset_mode": subset_mode,
        "max_pos": None if max_pos is None else int(max_pos),
        "max_neg": None if max_neg is None else int(max_neg),
        "seed": int(seed),
        "n_images": int(len(image_paths)),
    }

    if manifest_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        if saved_meta == desired_meta:
            log(f"Reusing patch manifest: {manifest_path}")
            df = pd.read_csv(manifest_path)
            required = {
                "patch_id",
                "split",
                "image_path",
                "mask_path",
                "patch_index",
                "patch_row",
                "patch_col",
                "vessel_fraction",
                "patch_label",
            }
            if required.issubset(df.columns):
                return df

    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    iterator = zip(image_paths, mask_paths)
    for image_path, mask_path in tqdm(list(iterator), desc=f"PatchManifest-{subset_name}"):
        rows.extend(
            build_patch_rows_for_image(
                split_name=split_name,
                image_path=image_path,
                mask_path=mask_path,
                img_size=img_size,
                patch_size=patch_size,
                vessel_frac_pos_thr=vessel_frac_pos_thr,
                background_frac_neg_thr=background_frac_neg_thr,
                subset_mode=subset_mode,
                max_pos=max_pos,
                max_neg=max_neg,
                rng=rng,
            )
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No eligible patches found for subset {subset_name}")
    df = df.sort_values(["image_path", "patch_row", "patch_col", "patch_label"]).reset_index(drop=True)
    df.insert(0, "patch_id", np.arange(len(df), dtype=np.int64))
    df.to_csv(manifest_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(desired_meta, f, indent=2)
    return df


def build_or_load_viz_manifest(
    test_manifest: pd.DataFrame,
    viz_samples_per_class: int,
    out_root: Path,
    seed: int,
) -> pd.DataFrame:
    manifest_path = out_root / "patch_manifest_test_viz.csv"
    meta_path = out_root / "patch_manifest_test_viz.meta.json"
    base_hash = hash_dataframe(
        test_manifest,
        columns=["patch_id", "image_path", "patch_row", "patch_col", "patch_label"],
    )
    desired_meta = {
        "source_hash": base_hash,
        "viz_samples_per_class": int(viz_samples_per_class),
        "seed": int(seed),
    }

    if manifest_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        if saved_meta == desired_meta:
            log(f"Reusing viz patch manifest: {manifest_path}")
            return pd.read_csv(manifest_path)

    rng = np.random.default_rng(seed)
    parts = []
    for label in ["vessel", "background"]:
        sub = test_manifest[test_manifest["patch_label"] == label].copy()
        n = min(len(sub), int(viz_samples_per_class))
        if n <= 0:
            raise ValueError(f"No patches available for label `{label}` in test manifest.")
        chosen = np.sort(rng.choice(len(sub), size=n, replace=False))
        parts.append(sub.iloc[chosen])
    viz_manifest = pd.concat(parts, axis=0).sort_values("patch_id").reset_index(drop=True)
    viz_manifest.to_csv(manifest_path, index=False)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(desired_meta, f, indent=2)
    return viz_manifest


def infer_eval_subset_name(manifest: pd.DataFrame) -> str:
    split_values = manifest["split"].astype(str).unique().tolist()
    split_name = split_values[0] if len(split_values) == 1 else "mixed"
    patch_count = int(len(manifest))
    if split_name == "test" and patch_count > 100_000:
        return "test_all"
    if split_name == "test":
        return "test_viz"
    return f"{split_name}_probe"


def extract_or_load_patch_features(
    manifest: pd.DataFrame,
    backbone_name: str,
    ckpt_path: str,
    args: argparse.Namespace,
    out_root: Path,
    subset_name: str,
) -> np.ndarray:
    features_path = out_root / f"features_{backbone_name}_{subset_name}.pt"
    meta_path = out_root / f"features_{backbone_name}_{subset_name}.meta.json"
    manifest_hash = hash_dataframe(
        manifest,
        columns=["patch_id", "image_path", "patch_index", "patch_label", "vessel_fraction"],
    )
    expected_meta = {
        "manifest_hash": manifest_hash,
        "backbone_name": backbone_name,
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "model_name": args.model_name,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "img_size": [int(args.img_size[0]), int(args.img_size[1])],
        "patch_size": int(args.patch_size),
    }

    if args.cache_features and features_path.exists() and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            saved_meta = json.load(f)
        if saved_meta == expected_meta:
            log(f"Reusing feature cache: {features_path}")
            cached = torch.load(features_path, map_location="cpu")
            if isinstance(cached, torch.Tensor):
                return cached.float().numpy()
            raise ValueError(f"Cached features must be torch.Tensor: {features_path}")

    requested_device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if str(requested_device).startswith("cuda") and not torch.cuda.is_available():
        log(f"CUDA requested but unavailable in this environment. Falling back to CPU for {backbone_name}.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    wrapper = Dinov3Backbone(
        weights=ckpt_path,
        model_name=args.model_name,
        repo_dir=args.repo_dir,
    )
    backbone = wrapper.backbone_model.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    image_paths = manifest["image_path"].drop_duplicates().tolist()
    dataset = ImageDataset(image_paths=image_paths, img_size=(int(args.img_size[0]), int(args.img_size[1])))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        collate_fn=image_collate_fn,
    )

    path_to_requests: Dict[str, List[Tuple[int, int]]] = {}
    for row_idx, row in manifest.iterrows():
        path_to_requests.setdefault(row["image_path"], []).append((row_idx, int(row["patch_index"])))

    features = np.zeros((len(manifest), backbone.norm.normalized_shape[0]), dtype=np.float32)
    log(
        f"Extracting patch features ({backbone_name}, {subset_name}) on device={device} | "
        f"n_images={len(image_paths):,} n_patches={len(manifest):,}"
    )
    with torch.no_grad():
        for images, batch_paths in tqdm(loader, desc=f"Extract-{backbone_name}-{subset_name}"):
            images = images.to(device, non_blocking=True)
            outputs = backbone.forward_features(images)
            patch_tokens = outputs["x_norm_patchtokens"]
            patch_tokens = F.normalize(patch_tokens, dim=2)
            patch_tokens = patch_tokens.cpu().numpy()
            for batch_idx, image_path in enumerate(batch_paths):
                requests = path_to_requests[image_path]
                patch_indices = [patch_index for _, patch_index in requests]
                selected = patch_tokens[batch_idx, patch_indices, :]
                row_indices = [row_index for row_index, _ in requests]
                features[np.asarray(row_indices, dtype=np.int64)] = selected

    if args.cache_features:
        torch.save(torch.from_numpy(features).half(), features_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(expected_meta, f, indent=2)
    return features


def labels_from_manifest(manifest: pd.DataFrame) -> np.ndarray:
    return (manifest["patch_label"].to_numpy() == "vessel").astype(np.int32)


def fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    valid_y: np.ndarray,
    alpha_grid: Sequence[float],
    seed: int,
) -> Tuple[object, float, Dict[float, float]]:
    best_model = None
    best_alpha = None
    best_auc = -np.inf
    alpha_to_auc: Dict[float, float] = {}

    for alpha in alpha_grid:
        model = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                alpha=float(alpha),
                random_state=seed,
                max_iter=2000,
                tol=1e-3,
            ),
        )
        model.fit(train_x, train_y)
        valid_prob = model.predict_proba(valid_x)[:, 1]
        valid_auc = float(roc_auc_score(valid_y, valid_prob))
        alpha_to_auc[float(alpha)] = valid_auc
        if valid_auc > best_auc:
            best_auc = valid_auc
            best_alpha = float(alpha)
            best_model = model

    if best_model is None or best_alpha is None:
        raise RuntimeError("Failed to fit linear probe.")
    return best_model, best_alpha, alpha_to_auc


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, alpha: float, backbone_name: str, split: str) -> ProbeMetrics:
    pred = (prob >= 0.5).astype(np.int32)
    return ProbeMetrics(
        backbone_name=backbone_name,
        split=split,
        alpha=float(alpha),
        n_vessel_patches=int((y_true == 1).sum()),
        n_background_patches=int((y_true == 0).sum()),
        roc_auc=float(roc_auc_score(y_true, prob)),
        pr_auc=float(average_precision_score(y_true, prob)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, pred)),
        f1=float(f1_score(y_true, pred)),
        vessel_recall=float(recall_score(y_true, pred, pos_label=1)),
        background_recall=float(recall_score(y_true, pred, pos_label=0)),
    )


def build_prediction_dataframe(manifest: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    pred = (prob >= 0.5).astype(np.int32)
    out = manifest.copy()
    out["true_label"] = labels_from_manifest(manifest)
    out["pred_prob_vessel"] = prob
    out["pred_label"] = pred
    return out


def reduce_to_2d(features: np.ndarray, seed: int) -> Tuple[np.ndarray, str]:
    n_samples, dim = features.shape
    n_components = min(50, dim, max(2, n_samples - 1))
    pca = PCA(n_components=n_components, random_state=seed)
    reduced = pca.fit_transform(features)
    try:
        numba_cache_dir = Path(tempfile.gettempdir()) / "dinov3_stack_numba_cache"
        ensure_dir(numba_cache_dir)
        os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
        import umap  # type: ignore

        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(reduced), "UMAP"
    except Exception as exc:
        log(f"[WARN] UMAP failed, falling back to TSNE: {exc}")
        perplexity = min(30.0, max(5.0, (n_samples - 1) / 3.0))
        reducer = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
        return reducer.fit_transform(reduced), "TSNE"


def save_umap_compare(
    viz_manifest: pd.DataFrame,
    features_imagenet: np.ndarray,
    features_cag: np.ndarray,
    output_path: Path,
    seed: int,
) -> str:
    y = labels_from_manifest(viz_manifest)
    coords_imagenet, method = reduce_to_2d(features_imagenet, seed=seed)
    coords_cag, method_cag = reduce_to_2d(features_cag, seed=seed)
    if method_cag != method:
        method = f"{method}/{method_cag}"

    labels = np.where(y == 1, "vessel", "background")
    color_map = {"vessel": "#DD8452", "background": "#4C72B0"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, coords, title in [
        (axes[0], coords_imagenet, "ImageNet"),
        (axes[1], coords_cag, "CAG"),
    ]:
        for label in ["background", "vessel"]:
            mask = labels == label
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=8,
                alpha=0.5,
                c=color_map[label],
                label=label,
                rasterized=True,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"Patch {method} Comparison", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return method


def save_roc_compare(
    y_true_imagenet: np.ndarray,
    prob_imagenet: np.ndarray,
    y_true_cag: np.ndarray,
    prob_cag: np.ndarray,
    output_path: Path,
) -> None:
    fpr_i, tpr_i, _ = roc_curve(y_true_imagenet, prob_imagenet)
    fpr_c, tpr_c, _ = roc_curve(y_true_cag, prob_cag)
    auc_i = roc_auc_score(y_true_imagenet, prob_imagenet)
    auc_c = roc_auc_score(y_true_cag, prob_cag)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr_i, tpr_i, label=f"ImageNet (AUC={auc_i:.4f})")
    plt.plot(fpr_c, tpr_c, label=f"CAG (AUC={auc_c:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Patch Vessel-vs-Background ROC")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_pr_compare(
    y_true_imagenet: np.ndarray,
    prob_imagenet: np.ndarray,
    y_true_cag: np.ndarray,
    prob_cag: np.ndarray,
    output_path: Path,
) -> None:
    p_i, r_i, _ = precision_recall_curve(y_true_imagenet, prob_imagenet)
    p_c, r_c, _ = precision_recall_curve(y_true_cag, prob_cag)
    ap_i = average_precision_score(y_true_imagenet, prob_imagenet)
    ap_c = average_precision_score(y_true_cag, prob_cag)

    plt.figure(figsize=(7, 6))
    plt.plot(r_i, p_i, label=f"ImageNet (AP={ap_i:.4f})")
    plt.plot(r_c, p_c, label=f"CAG (AP={ap_c:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Patch Vessel-vs-Background Precision-Recall")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_score_hist_compare(
    y_true_imagenet: np.ndarray,
    prob_imagenet: np.ndarray,
    y_true_cag: np.ndarray,
    prob_cag: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharex=True, sharey=True)
    for ax, y_true, prob, title in [
        (axes[0], y_true_imagenet, prob_imagenet, "ImageNet"),
        (axes[1], y_true_cag, prob_cag, "CAG"),
    ]:
        ax.hist(prob[y_true == 1], bins=80, density=True, alpha=0.6, label="vessel")
        ax.hist(prob[y_true == 0], bins=80, density=True, alpha=0.6, label="background")
        ax.set_title(title)
        ax.set_xlabel("Predicted vessel probability")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Patch Score Distribution", y=0.98)
    fig.tight_layout(rect=[0, 0.08, 1, 0.93])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def format_float(x: float) -> str:
    if np.isnan(x) or np.isinf(x):
        return "nan"
    return f"{x:.6f}"


def write_markdown_summary(
    output_path: Path,
    summary_df: pd.DataFrame,
    viz_method: str,
    prevalence_test: float,
) -> None:
    test_rows = summary_df[summary_df["split"] == "test"].copy()
    if set(test_rows["backbone_name"]) != {"imagenet", "cag"}:
        raise ValueError("Expected ImageNet and CAG test rows in summary.")
    row_i = test_rows[test_rows["backbone_name"] == "imagenet"].iloc[0]
    row_c = test_rows[test_rows["backbone_name"] == "cag"].iloc[0]

    delta_auc = float(row_c["roc_auc"]) - float(row_i["roc_auc"])
    delta_pr = float(row_c["pr_auc"]) - float(row_i["pr_auc"])
    if delta_auc > 0.0 and delta_pr > 0.0:
        interpretation = (
            "CAG backbone improves frozen patch-level vessel/background separability "
            "relative to ImageNet on the test split."
        )
    elif delta_auc < 0.0 and delta_pr < 0.0:
        interpretation = (
            "CAG backbone does not improve frozen patch-level vessel/background separability "
            "relative to ImageNet on the test split."
        )
    else:
        interpretation = (
            "The two backbones trade off depending on metric; no single winner is established "
            "from patch separability alone."
        )

    lines = [
        "# Analysis 2 — Vessel vs Background Separability",
        "",
        "## Setup",
        "- Patch token source: last-layer `x_norm_patchtokens`",
        "- Input size: 448x448",
        "- Patch size: 16 (28x28 grid)",
        "- Patch labeling: vessel if occupancy >= 0.10, background if occupancy <= 0.01",
        f"- Visualization method: {viz_method}",
        f"- Test positive prevalence (eligible vessel patches): {prevalence_test:.6f}",
        "",
        "## Test Metrics",
        "",
        "| Backbone | ROC-AUC | PR-AUC | Balanced Acc | F1 | Vessel Recall | Background Recall |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| ImageNet | {format_float(float(row_i['roc_auc']))} | "
            f"{format_float(float(row_i['pr_auc']))} | "
            f"{format_float(float(row_i['balanced_accuracy']))} | "
            f"{format_float(float(row_i['f1']))} | "
            f"{format_float(float(row_i['vessel_recall']))} | "
            f"{format_float(float(row_i['background_recall']))} |"
        ),
        (
            f"| CAG | {format_float(float(row_c['roc_auc']))} | "
            f"{format_float(float(row_c['pr_auc']))} | "
            f"{format_float(float(row_c['balanced_accuracy']))} | "
            f"{format_float(float(row_c['f1']))} | "
            f"{format_float(float(row_c['vessel_recall']))} | "
            f"{format_float(float(row_c['background_recall']))} |"
        ),
        "",
        "## Interpretation",
        f"- Delta ROC-AUC (CAG - ImageNet): {delta_auc:.6f}",
        f"- Delta PR-AUC (CAG - ImageNet): {delta_pr:.6f}",
        f"- {interpretation}",
        "- This analysis is local and foreground-focused, unlike Analysis 1 which was a global study-coherence proxy.",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analysis 2 — Vessel vs Background Separability")
    parser.add_argument("--train-images", default="input/MPXA-Seg/train_images")
    parser.add_argument("--train-masks", default="input/MPXA-Seg/train_labels")
    parser.add_argument("--valid-images", default="input/MPXA-Seg/valid_images")
    parser.add_argument("--valid-masks", default="input/MPXA-Seg/valid_labels")
    parser.add_argument("--test-images", default="input/MPXA-Seg/test_images")
    parser.add_argument("--test-masks", default="input/MPXA-Seg/test_labels")
    parser.add_argument("--seg-config", default="configs_segmentation/mpxa-seg.yaml")
    parser.add_argument("--imagenet-ckpt", default=DEFAULT_IMAGENET_CKPT)
    parser.add_argument("--cag-ckpt", default=DEFAULT_CAG_CKPT)
    parser.add_argument("--output-root", default="outputs/7_vessel_background_separability")
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    parser.add_argument("--img-size", nargs=2, type=int, default=[448, 448], help="width height")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--vessel-frac-pos-thr", type=float, default=0.10)
    parser.add_argument("--background-frac-neg-thr", type=float, default=0.01)
    parser.add_argument("--train-max-pos-per-image", type=int, default=32)
    parser.add_argument("--train-max-neg-per-image", type=int, default=32)
    parser.add_argument("--valid-max-pos-per-image", type=int, default=32)
    parser.add_argument("--valid-max-neg-per-image", type=int, default=32)
    parser.add_argument("--viz-samples-per-class", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
        cfg = load_seg_config(args.seg_config)
        log(f"Loaded seg config: {cfg}")
        tracker = ProgressTracker(total_steps=12)

        tracker.start_step("Load image/mask pairs")
        split_pairs = {}
        split_seed_offset = {"train": 0, "valid": 1, "test": 2}
        for split_name, image_root, mask_root in [
            ("train", args.train_images, args.train_masks),
            ("valid", args.valid_images, args.valid_masks),
            ("test", args.test_images, args.test_masks),
        ]:
            image_paths = collect_paths(image_root)
            mask_paths = collect_paths(mask_root)
            image_paths, mask_paths = pair_image_mask_paths(image_paths, mask_paths, image_root, mask_root)
            image_paths, mask_paths = maybe_subsample_pairs(
                image_paths,
                mask_paths,
                max_images=args.max_images_per_split,
                seed=args.seed + split_seed_offset[split_name],
            )
            split_pairs[split_name] = (image_paths, mask_paths)
            log(f"{split_name}: paired_images={len(image_paths):,}")
        tracker.finish_step("Load image/mask pairs")

        tracker.start_step("Build/load train probe manifest")
        train_manifest = build_or_load_patch_manifest(
            subset_name="train_probe",
            split_name="train",
            image_paths=split_pairs["train"][0],
            mask_paths=split_pairs["train"][1],
            img_size=(int(args.img_size[0]), int(args.img_size[1])),
            patch_size=int(args.patch_size),
            vessel_frac_pos_thr=float(args.vessel_frac_pos_thr),
            background_frac_neg_thr=float(args.background_frac_neg_thr),
            out_root=out_root,
            seed=args.seed,
            subset_mode="balanced_sample",
            max_pos=int(args.train_max_pos_per_image),
            max_neg=int(args.train_max_neg_per_image),
        )
        log(f"train_probe patches={len(train_manifest):,}")
        tracker.finish_step("Build/load train probe manifest")

        tracker.start_step("Build/load valid probe manifest")
        valid_manifest = build_or_load_patch_manifest(
            subset_name="valid_probe",
            split_name="valid",
            image_paths=split_pairs["valid"][0],
            mask_paths=split_pairs["valid"][1],
            img_size=(int(args.img_size[0]), int(args.img_size[1])),
            patch_size=int(args.patch_size),
            vessel_frac_pos_thr=float(args.vessel_frac_pos_thr),
            background_frac_neg_thr=float(args.background_frac_neg_thr),
            out_root=out_root,
            seed=args.seed + 101,
            subset_mode="balanced_sample",
            max_pos=int(args.valid_max_pos_per_image),
            max_neg=int(args.valid_max_neg_per_image),
        )
        log(f"valid_probe patches={len(valid_manifest):,}")
        tracker.finish_step("Build/load valid probe manifest")

        tracker.start_step("Build/load test manifests")
        test_manifest = build_or_load_patch_manifest(
            subset_name="test_all",
            split_name="test",
            image_paths=split_pairs["test"][0],
            mask_paths=split_pairs["test"][1],
            img_size=(int(args.img_size[0]), int(args.img_size[1])),
            patch_size=int(args.patch_size),
            vessel_frac_pos_thr=float(args.vessel_frac_pos_thr),
            background_frac_neg_thr=float(args.background_frac_neg_thr),
            out_root=out_root,
            seed=args.seed + 202,
            subset_mode="all_eligible",
            max_pos=None,
            max_neg=None,
        )
        test_viz_manifest = build_or_load_viz_manifest(
            test_manifest=test_manifest,
            viz_samples_per_class=int(args.viz_samples_per_class),
            out_root=out_root,
            seed=args.seed + 303,
        )
        log(f"test_all patches={len(test_manifest):,}")
        log(f"test_viz patches={len(test_viz_manifest):,}")
        tracker.finish_step("Build/load test manifests")

        ckpt_imagenet = str(Path(args.imagenet_ckpt).expanduser().resolve())
        ckpt_cag = str(Path(args.cag_ckpt).expanduser().resolve())

        tracker.start_step("Extract/load ImageNet features")
        features = {}
        for subset_name, manifest in [
            ("train_probe", train_manifest),
            ("valid_probe", valid_manifest),
            ("test_all", test_manifest),
            ("test_viz", test_viz_manifest),
        ]:
            features[f"imagenet_{subset_name}"] = extract_or_load_patch_features(
                manifest=manifest,
                backbone_name="imagenet",
                ckpt_path=ckpt_imagenet,
                args=args,
                out_root=out_root,
                subset_name=subset_name,
            )
        tracker.finish_step("Extract/load ImageNet features")

        tracker.start_step("Extract/load CAG features")
        for subset_name, manifest in [
            ("train_probe", train_manifest),
            ("valid_probe", valid_manifest),
            ("test_all", test_manifest),
            ("test_viz", test_viz_manifest),
        ]:
            features[f"cag_{subset_name}"] = extract_or_load_patch_features(
                manifest=manifest,
                backbone_name="cag",
                ckpt_path=ckpt_cag,
                args=args,
                out_root=out_root,
                subset_name=subset_name,
            )
        tracker.finish_step("Extract/load CAG features")

        tracker.start_step("Fit/select ImageNet linear probe")
        y_train = labels_from_manifest(train_manifest)
        y_valid = labels_from_manifest(valid_manifest)
        y_test = labels_from_manifest(test_manifest)
        probe_imagenet, alpha_imagenet, alpha_auc_imagenet = fit_probe(
            train_x=features["imagenet_train_probe"],
            train_y=y_train,
            valid_x=features["imagenet_valid_probe"],
            valid_y=y_valid,
            alpha_grid=[1e-5, 1e-4, 1e-3],
            seed=args.seed,
        )
        valid_prob_imagenet = probe_imagenet.predict_proba(features["imagenet_valid_probe"])[:, 1]
        test_prob_imagenet = probe_imagenet.predict_proba(features["imagenet_test_all"])[:, 1]
        tracker.finish_step("Fit/select ImageNet linear probe")

        tracker.start_step("Fit/select CAG linear probe")
        probe_cag, alpha_cag, alpha_auc_cag = fit_probe(
            train_x=features["cag_train_probe"],
            train_y=y_train,
            valid_x=features["cag_valid_probe"],
            valid_y=y_valid,
            alpha_grid=[1e-5, 1e-4, 1e-3],
            seed=args.seed,
        )
        valid_prob_cag = probe_cag.predict_proba(features["cag_valid_probe"])[:, 1]
        test_prob_cag = probe_cag.predict_proba(features["cag_test_all"])[:, 1]
        tracker.finish_step("Fit/select CAG linear probe")

        tracker.start_step("Save predictions and summary")
        metrics_rows = [
            compute_metrics(y_valid, valid_prob_imagenet, alpha_imagenet, "imagenet", "valid").to_row(),
            compute_metrics(y_test, test_prob_imagenet, alpha_imagenet, "imagenet", "test").to_row(),
            compute_metrics(y_valid, valid_prob_cag, alpha_cag, "cag", "valid").to_row(),
            compute_metrics(y_test, test_prob_cag, alpha_cag, "cag", "test").to_row(),
        ]
        summary_df = pd.DataFrame(metrics_rows)
        summary_path = out_root / "summary_vessel_background_separability.csv"
        summary_df.to_csv(summary_path, index=False)
        log(f"Saved summary: {summary_path}")

        pred_imagenet_path = out_root / "per_patch_predictions_imagenet_test.csv"
        pred_cag_path = out_root / "per_patch_predictions_cag_test.csv"
        build_prediction_dataframe(test_manifest, test_prob_imagenet).to_csv(pred_imagenet_path, index=False)
        build_prediction_dataframe(test_manifest, test_prob_cag).to_csv(pred_cag_path, index=False)
        log(f"Saved predictions: {pred_imagenet_path}")
        log(f"Saved predictions: {pred_cag_path}")

        alpha_path = out_root / "probe_alpha_search.json"
        with open(alpha_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "imagenet": {"best_alpha": alpha_imagenet, "valid_roc_auc": alpha_auc_imagenet},
                    "cag": {"best_alpha": alpha_cag, "valid_roc_auc": alpha_auc_cag},
                },
                f,
                indent=2,
            )
        log(f"Saved alpha search: {alpha_path}")
        tracker.finish_step("Save predictions and summary")

        tracker.start_step("Render figures")
        viz_method = save_umap_compare(
            viz_manifest=test_viz_manifest,
            features_imagenet=features["imagenet_test_viz"],
            features_cag=features["cag_test_viz"],
            output_path=out_root / "fig_patch_umap_compare.png",
            seed=args.seed,
        )
        save_roc_compare(
            y_true_imagenet=y_test,
            prob_imagenet=test_prob_imagenet,
            y_true_cag=y_test,
            prob_cag=test_prob_cag,
            output_path=out_root / "fig_probe_roc_compare.png",
        )
        save_pr_compare(
            y_true_imagenet=y_test,
            prob_imagenet=test_prob_imagenet,
            y_true_cag=y_test,
            prob_cag=test_prob_cag,
            output_path=out_root / "fig_probe_pr_compare.png",
        )
        save_score_hist_compare(
            y_true_imagenet=y_test,
            prob_imagenet=test_prob_imagenet,
            y_true_cag=y_test,
            prob_cag=test_prob_cag,
            output_path=out_root / "fig_score_hist_compare.png",
        )
        log(f"Rendered figures with embedding reducer: {viz_method}")
        tracker.finish_step("Render figures")

        tracker.start_step("Write markdown summary")
        prevalence_test = float((y_test == 1).mean())
        markdown_path = out_root / "analysis_vessel_background_separability.md"
        write_markdown_summary(
            output_path=markdown_path,
            summary_df=summary_df,
            viz_method=viz_method,
            prevalence_test=prevalence_test,
        )
        log(f"Saved markdown summary: {markdown_path}")
        tracker.finish_step("Write markdown summary")

        tracker.start_step("Save run metadata")
        run_meta_path = out_root / "run_meta.json"
        with open(run_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "args": vars(args),
                    "img_size": list(map(int, args.img_size)),
                    "train_probe_hash": hash_dataframe(train_manifest, ["patch_id", "image_path", "patch_index", "patch_label"]),
                    "valid_probe_hash": hash_dataframe(valid_manifest, ["patch_id", "image_path", "patch_index", "patch_label"]),
                    "test_all_hash": hash_dataframe(test_manifest, ["patch_id", "image_path", "patch_index", "patch_label"]),
                    "test_viz_hash": hash_dataframe(test_viz_manifest, ["patch_id", "image_path", "patch_index", "patch_label"]),
                },
                f,
                indent=2,
            )
        log(f"Saved run meta: {run_meta_path}")
        tracker.finish_step("Save run metadata")

        log(f"Done. Log file: {log_path}")
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
