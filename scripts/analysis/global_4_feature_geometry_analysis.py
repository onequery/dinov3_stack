#!/usr/bin/env python3
"""
Global Analysis 4 — Feature Geometry Analysis.

This analysis is fully offline. It reuses Global Analysis 2 artifacts:
- test manifest
- cached test features
- trained probe checkpoints

It measures pair geometry, target-conditioned margin geometry, cluster geometry,
and spectral/isotropy geometry for the 4-way comparison:
- ImageNet Raw Frozen
- ImageNet Probe
- CAG Raw Frozen
- CAG Probe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import davies_bouldin_score, silhouette_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}
CONDITION_ORDER = [
    ("imagenet", "raw_frozen", "ImageNet Raw"),
    ("imagenet", "probe_linear", "ImageNet Probe"),
    ("cag", "raw_frozen", "CAG Raw"),
    ("cag", "probe_linear", "CAG Probe"),
]
PAIR_GROUP_ORDER = [
    "same_study_near_view",
    "same_study_mid_view",
    "same_study_far_view",
    "same_patient_cross_study",
    "different_patient_near_view",
    "different_patient_mid_view",
    "different_patient_far_view",
]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]


@dataclass(frozen=True)
class JobDef:
    kind: str
    mode: str | None = None
    backbone_name: str | None = None
    target: str | None = None
    seed: int | None = None
    name: str | None = None

    @property
    def display_name(self) -> str:
        if self.kind == "postprocess":
            return self.name or "postprocess"
        seed_text = "raw" if self.seed is None else f"seed{self.seed}"
        return f"{self.mode}/{self.backbone_name}/{self.target}/{seed_text}"


@dataclass
class PairMeta:
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_group_masks: Dict[str, np.ndarray]
    same_patient_all_mask: np.ndarray
    same_study_mask: np.ndarray
    same_patient_cross_study_mask: np.ndarray


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


class StrictLinearRetrievalProbe(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        return F.normalize(z, dim=1)


class FullRunTracker:
    def __init__(self, jobs: Sequence[JobDef]):
        self.jobs = list(jobs)
        self.start_time = time.time()
        self.current_job_index = -1
        self.current_job_started_at = self.start_time
        self.completed_job_durations: List[float] = []

    def start_job(self, index: int) -> None:
        self.current_job_index = int(index)
        self.current_job_started_at = time.time()
        job = self.jobs[self.current_job_index]
        log(f"[ETA][JOB] START | job={self.current_job_index + 1}/{len(self.jobs)} | {job.display_name}")
        self.log_live_eta(job_progress=0.0, phase_name="start")

    def update_phase(self, phase_name: str, phase_index: int, phase_total: int) -> None:
        progress = 0.0 if phase_total <= 0 else float(phase_index) / float(phase_total)
        job = self.jobs[self.current_job_index]
        log(
            f"[ETA][JOB] phase={phase_name} | "
            f"job={self.current_job_index + 1}/{len(self.jobs)} | "
            f"job_progress={progress * 100.0:.1f}% | {job.display_name}"
        )
        self.log_live_eta(job_progress=progress, phase_name=phase_name)

    def finish_job(self) -> None:
        duration = max(0.0, time.time() - self.current_job_started_at)
        self.completed_job_durations.append(duration)
        job = self.jobs[self.current_job_index]
        log(
            f"[ETA][JOB] DONE | job={self.current_job_index + 1}/{len(self.jobs)} | "
            f"{job.display_name} | duration={format_duration(duration)}"
        )
        self.log_live_eta(job_progress=1.0, phase_name="done")

    def log_live_eta(self, job_progress: float, phase_name: str) -> None:
        elapsed = max(0.0, time.time() - self.start_time)
        current_elapsed = max(0.0, time.time() - self.current_job_started_at)
        completed_jobs = len(self.completed_job_durations)
        future_jobs = max(0, len(self.jobs) - completed_jobs - (1 if self.current_job_index >= 0 else 0))

        current_total_estimate = current_elapsed / max(job_progress, 0.1)
        if self.completed_job_durations:
            avg_job = float(sum(self.completed_job_durations) / len(self.completed_job_durations))
        else:
            avg_job = max(1.0, current_total_estimate)
        current_remaining = max(0.0, current_total_estimate - current_elapsed)
        remaining = current_remaining + future_jobs * avg_job
        total_estimate = elapsed + remaining
        overall_progress = 0.0 if total_estimate <= 0.0 else elapsed / total_estimate
        eta_dt = datetime.now() + timedelta(seconds=remaining)
        job = self.jobs[self.current_job_index] if self.current_job_index >= 0 else None
        fields = {
            "current_job": job.display_name if job is not None else "setup",
            "phase": phase_name,
            "job_progress": f"{job_progress * 100.0:.1f}%",
            "completed_jobs": completed_jobs,
            "future_jobs": future_jobs,
            "avg_job_time": format_duration(avg_job),
            "overall_progress": f"{overall_progress * 100.0:.1f}%",
            "elapsed": format_duration(elapsed),
            "remaining": format_duration(remaining),
            "eta": eta_dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
        if job is not None:
            fields.update(
                {
                    "backbone": job.backbone_name or "-",
                    "mode": job.mode or "-",
                    "target": job.target or "-",
                    "seed": "raw" if job.seed is None else job.seed,
                }
            )
        serialized = " | ".join(f"{k}={v}" for k, v in fields.items())
        log(f"[ETA][FULL-RUN][LIVE] {serialized}")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:04.1f}s"
    total = int(round(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    arr = features.detach().cpu().numpy().astype(np.float32, copy=False)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def read_view_angles(dcm_path: Path) -> Tuple[float, float]:
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
    primary = ds.get("PositionerPrimaryAngle", None)
    secondary = ds.get("PositionerSecondaryAngle", None)
    if primary is None or secondary is None:
        raise ValueError(f"Missing angle tag: {dcm_path}")
    return float(primary), float(secondary)


def build_angle_manifest(
    global2_root: Path,
    image_root: Path,
    dcm_root: Path,
    output_root: Path,
) -> pd.DataFrame:
    base_manifest_path = global2_root / "image_manifest_test.csv"
    base_manifest = load_csv(base_manifest_path)
    manifest_hash = hash_dataframe(
        base_manifest,
        ["image_id", "img_path", "patient_id", "study_id", "split", "class_name"],
    )
    cache_path = output_root / "image_manifest_test_with_angles.csv"
    meta_path = output_root / "image_manifest_test_with_angles.meta.json"
    expected_meta = {
        "source_manifest_hash": manifest_hash,
        "image_root": str(image_root.resolve()),
        "dcm_root": str(dcm_root.resolve()),
    }
    if cache_path.exists() and meta_path.exists():
        cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if cached_meta == expected_meta:
            return pd.read_csv(cache_path)

    rows: List[Dict[str, object]] = []
    for _, row in base_manifest.iterrows():
        img_path = Path(str(row["img_path"])).resolve()
        rel = img_path.relative_to(image_root.resolve())
        dcm_rel = rel.with_suffix(".dcm")
        dcm_path = dcm_root.resolve() / dcm_rel
        if not dcm_path.exists():
            raise FileNotFoundError(f"Failed to map PNG to DICOM: {img_path} -> {dcm_path}")
        primary, secondary = read_view_angles(dcm_path)
        rows.append(
            {
                **row.to_dict(),
                "dcm_path": str(dcm_path),
                "dcm_rel_path": str(dcm_rel),
                "primary_angle": primary,
                "secondary_angle": secondary,
            }
        )
    out_df = pd.DataFrame(rows)
    if out_df["primary_angle"].isna().any() or out_df["secondary_angle"].isna().any():
        raise RuntimeError("Angle metadata contains NaNs after join.")
    out_df.to_csv(cache_path, index=False)
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return out_df


def maybe_subsample_manifest(manifest: pd.DataFrame, max_images: int | None, seed: int) -> pd.DataFrame:
    if max_images is None or max_images <= 0 or len(manifest) <= max_images:
        return manifest.copy().reset_index(drop=True)
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(manifest), size=max_images, replace=False))
    return manifest.iloc[selected].sort_values("img_path").reset_index(drop=True)


def load_test_features(global2_root: Path, backbone_name: str) -> Tuple[torch.Tensor, Dict[str, object]]:
    features_path = global2_root / f"features_{backbone_name}_test.pt"
    meta_path = global2_root / f"features_{backbone_name}_test.meta.json"
    features = torch.load(features_path, map_location="cpu")
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"Unexpected feature object: {type(features)}")
    if features.ndim != 2:
        raise ValueError(f"Unexpected feature shape: {tuple(features.shape)}")
    if features.shape[0] != len(load_csv(global2_root / "image_manifest_test.csv")):
        raise ValueError("Feature count does not match full test manifest length.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return features.float().contiguous(), meta


def l2_normalize_cpu(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1).cpu()


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def apply_probe_checkpoint(
    global2_root: Path,
    backbone_name: str,
    target: str,
    seed: int,
    features: torch.Tensor,
    manifest_hash: str,
    feature_hash: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    ckpt_path = global2_root / f"probe_checkpoint_seed{seed}_{backbone_name}_{target}.pt"
    payload = torch.load(ckpt_path, map_location="cpu")
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        raise ValueError(f"Missing signature in checkpoint: {ckpt_path}")
    if signature.get("seed") != int(seed) or signature.get("backbone_name") != backbone_name or signature.get("target") != target:
        raise ValueError(f"Checkpoint signature mismatch: {ckpt_path}")
    if signature.get("test_manifest_hash") != manifest_hash:
        raise ValueError(f"Checkpoint test manifest hash mismatch: {ckpt_path}")
    feature_hashes = signature.get("feature_hashes", {})
    if feature_hashes.get("test") != feature_hash:
        raise ValueError(f"Checkpoint test feature hash mismatch: {ckpt_path}")
    model = StrictLinearRetrievalProbe(int(features.shape[1]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            x = features[start : start + int(batch_size)].to(device, non_blocking=True)
            outputs.append(model(x).cpu())
    embeddings = torch.cat(outputs, dim=0).contiguous().float()
    return embeddings, payload.get("summary", {})


def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    emb = l2_normalize_cpu(embeddings)
    return torch.matmul(emb, emb.T).numpy().astype(np.float32)


def build_pair_meta(manifest: pd.DataFrame, near_thresh: float, mid_thresh: float) -> PairMeta:
    patient_ids = manifest["patient_id"].to_numpy(dtype=str)
    study_ids = manifest["study_id"].to_numpy(dtype=str)
    prim = manifest["primary_angle"].to_numpy(dtype=np.float32)
    sec = manifest["secondary_angle"].to_numpy(dtype=np.float32)

    pair_i, pair_j = np.triu_indices(len(manifest), k=1)
    same_patient = patient_ids[pair_i] == patient_ids[pair_j]
    same_study = study_ids[pair_i] == study_ids[pair_j]
    delta_view = np.sqrt((prim[pair_i] - prim[pair_j]) ** 2 + (sec[pair_i] - sec[pair_j]) ** 2)
    near = delta_view <= near_thresh
    mid = (delta_view > near_thresh) & (delta_view <= mid_thresh)
    far = delta_view > mid_thresh

    pair_group_masks = {
        "same_study_near_view": same_study & near,
        "same_study_mid_view": same_study & mid,
        "same_study_far_view": same_study & far,
        "same_patient_cross_study": same_patient & (~same_study),
        "different_patient_near_view": (~same_patient) & near,
        "different_patient_mid_view": (~same_patient) & mid,
        "different_patient_far_view": (~same_patient) & far,
    }
    return PairMeta(
        pair_i=pair_i,
        pair_j=pair_j,
        pair_group_masks=pair_group_masks,
        same_patient_all_mask=same_patient,
        same_study_mask=same_study,
        same_patient_cross_study_mask=same_patient & (~same_study),
    )


def compute_pair_geometry_summary(
    sims: np.ndarray,
    pair_meta: PairMeta,
    mode: str,
    backbone_name: str,
    target: str,
    seed: int | None,
) -> pd.DataFrame:
    pair_cos = sims[pair_meta.pair_i, pair_meta.pair_j]
    rows: List[Dict[str, object]] = []
    for pair_group in PAIR_GROUP_ORDER:
        mask = pair_meta.pair_group_masks[pair_group]
        values = pair_cos[mask]
        distances = 1.0 - values
        rows.append(
            {
                "mode": mode,
                "target": target,
                "backbone_name": backbone_name,
                "pair_group": pair_group,
                "seed": "" if seed is None else int(seed),
                "num_pairs": int(values.size),
                "cosine_mean": float(values.mean()) if values.size else float("nan"),
                "cosine_median": float(np.median(values)) if values.size else float("nan"),
                "distance_mean": float(distances.mean()) if values.size else float("nan"),
                "distance_median": float(np.median(distances)) if values.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def compute_margin_geometry_summary(
    sims: np.ndarray,
    manifest: pd.DataFrame,
    mode: str,
    backbone_name: str,
    target: str,
    seed: int | None,
) -> pd.DataFrame:
    patient_ids = manifest["patient_id"].to_numpy(dtype=str)
    study_ids = manifest["study_id"].to_numpy(dtype=str)
    best_positive_scores: List[float] = []
    hardest_negative_scores: List[float] = []
    margins: List[float] = []
    negative_wins: List[float] = []
    same_patient_cross_study_negative_scores: List[float] = []

    for i in range(len(manifest)):
        score_vec = sims[i].copy()
        score_vec[i] = -1e9
        same_patient = patient_ids == patient_ids[i]
        same_study = study_ids == study_ids[i]
        if target == "patient":
            positive_mask = same_patient.copy()
            positive_mask[i] = False
            negative_mask = ~same_patient
        elif target == "study":
            positive_mask = same_study.copy()
            positive_mask[i] = False
            negative_mask = ~same_study
            cross_study_same_patient = same_patient & (~same_study)
            cross_study_same_patient[i] = False
            if cross_study_same_patient.any():
                same_patient_cross_study_negative_scores.append(float(score_vec[cross_study_same_patient].max()))
        else:
            raise ValueError(target)

        if not positive_mask.any() or not negative_mask.any():
            continue
        best_positive = float(score_vec[positive_mask].max())
        hardest_negative = float(score_vec[negative_mask].max())
        best_positive_scores.append(best_positive)
        hardest_negative_scores.append(hardest_negative)
        margins.append(best_positive - hardest_negative)
        negative_wins.append(float(hardest_negative > best_positive))

    row = {
        "mode": mode,
        "target": target,
        "backbone_name": backbone_name,
        "seed": "" if seed is None else int(seed),
        "num_queries_with_positive": int(len(best_positive_scores)),
        "best_positive_similarity_mean": float(np.mean(best_positive_scores)) if best_positive_scores else float("nan"),
        "hardest_negative_similarity_mean": float(np.mean(hardest_negative_scores)) if hardest_negative_scores else float("nan"),
        "margin_mean": float(np.mean(margins)) if margins else float("nan"),
        "margin_median": float(np.median(margins)) if margins else float("nan"),
        "negative_win_rate": float(np.mean(negative_wins)) if negative_wins else float("nan"),
        "same_patient_cross_study_negative_similarity_mean": (
            float(np.mean(same_patient_cross_study_negative_scores)) if same_patient_cross_study_negative_scores else float("nan")
        ),
        "queries_with_same_patient_cross_study_negative": int(len(same_patient_cross_study_negative_scores)),
    }
    return pd.DataFrame([row])


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return x / denom


def compute_cluster_geometry_rows(
    embeddings: torch.Tensor,
    manifest: pd.DataFrame,
    mode: str,
    backbone_name: str,
    target: str,
    seed: int | None,
) -> pd.DataFrame:
    emb = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    emb = _normalize_rows(emb)
    rows: List[Dict[str, object]] = []

    for label_level in ["patient", "study"]:
        labels = manifest[f"{label_level}_id"].to_numpy(dtype=str)
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        cluster_sizes = np.bincount(inverse).astype(np.int64)
        num_clusters = int(unique_labels.size)
        centroids = np.zeros((num_clusters, emb.shape[1]), dtype=np.float32)
        intra_distances: List[float] = []
        for cluster_index in range(num_clusters):
            member_mask = inverse == cluster_index
            member_emb = emb[member_mask]
            centroid = member_emb.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            centroids[cluster_index] = centroid
            sims_to_centroid = member_emb @ centroid
            intra_distances.extend((1.0 - sims_to_centroid).tolist())

        if num_clusters >= 2:
            centroid_sims = centroids @ centroids.T
            centroid_dists = 1.0 - centroid_sims
            np.fill_diagonal(centroid_dists, np.inf)
            nearest_inter = centroid_dists.min(axis=1)
            nearest_inter_centroid_mean = float(np.mean(nearest_inter))
        else:
            nearest_inter_centroid_mean = float("nan")

        intra_to_centroid_mean = float(np.mean(intra_distances)) if intra_distances else float("nan")
        if math.isfinite(intra_to_centroid_mean) and intra_to_centroid_mean > 0.0 and math.isfinite(nearest_inter_centroid_mean):
            separation_ratio = float(nearest_inter_centroid_mean / intra_to_centroid_mean)
        else:
            separation_ratio = float("nan")

        try:
            silhouette = float(silhouette_score(emb, labels, metric="euclidean")) if num_clusters >= 2 else float("nan")
        except Exception:
            silhouette = float("nan")
        try:
            davies_bouldin = float(davies_bouldin_score(emb, labels)) if num_clusters >= 2 else float("nan")
        except Exception:
            davies_bouldin = float("nan")

        rows.append(
            {
                "mode": mode,
                "target": target,
                "backbone_name": backbone_name,
                "label_level": label_level,
                "seed": "" if seed is None else int(seed),
                "num_clusters": num_clusters,
                "mean_cluster_size": float(cluster_sizes.mean()) if cluster_sizes.size else float("nan"),
                "intra_to_centroid_mean": intra_to_centroid_mean,
                "nearest_inter_centroid_mean": nearest_inter_centroid_mean,
                "separation_ratio": separation_ratio,
                "silhouette_score": silhouette,
                "davies_bouldin_score": davies_bouldin,
            }
        )
    return pd.DataFrame(rows)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def compute_uniformity_from_similarity_matrix(sims: np.ndarray) -> float:
    upper_i, upper_j = np.triu_indices_from(sims, k=1)
    if upper_i.size == 0:
        return float("nan")
    cos = sims[upper_i, upper_j]
    sq_dist = np.clip(2.0 - 2.0 * cos, 0.0, None)
    return float(np.log(np.exp(-2.0 * sq_dist).mean()))


def compute_spectral_geometry_summary(
    embeddings: torch.Tensor,
    sims: np.ndarray,
    pair_meta: PairMeta,
    mode: str,
    backbone_name: str,
    target: str,
    seed: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    emb = embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
    norms = np.linalg.norm(emb, axis=1)
    centered = emb - emb.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov).astype(np.float64)
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = np.sort(eigvals)[::-1]
    eig_sum = float(eigvals.sum())
    if eig_sum <= 0.0:
        eig_share = np.zeros_like(eigvals)
    else:
        eig_share = eigvals / eig_sum
    positive_share = eig_share[eig_share > 0]
    if positive_share.size:
        effective_rank = float(np.exp(-(positive_share * np.log(positive_share)).sum()))
    else:
        effective_rank = float("nan")
    eig_sq_sum = float((eigvals ** 2).sum())
    participation_ratio = float((eig_sum ** 2) / eig_sq_sum) if eig_sq_sum > 0 else float("nan")
    anisotropy_ratio = float(eigvals[0] / (eig_sum / max(1, len(eigvals)))) if eig_sum > 0 else float("nan")
    lambda1_share = float(eig_share[0]) if eig_share.size else float("nan")

    pair_cos = sims[pair_meta.pair_i, pair_meta.pair_j]
    same_patient_all = pair_cos[pair_meta.same_patient_all_mask]
    same_patient_cross = pair_cos[pair_meta.same_patient_cross_study_mask]
    same_study = pair_cos[pair_meta.same_study_mask]

    summary_row = {
        "mode": mode,
        "target": target,
        "backbone_name": backbone_name,
        "seed": "" if seed is None else int(seed),
        "embed_dim": int(emb.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(ddof=0)),
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "anisotropy_ratio": anisotropy_ratio,
        "lambda1_share": lambda1_share,
        "uniformity": compute_uniformity_from_similarity_matrix(sims),
        "alignment_same_patient_all": _safe_mean(same_patient_all),
        "alignment_same_patient_cross_study": _safe_mean(same_patient_cross),
        "alignment_same_study": _safe_mean(same_study),
    }

    topk = min(32, eig_share.size)
    spectrum_rows = [
        {
            "mode": mode,
            "target": target,
            "backbone_name": backbone_name,
            "seed": "" if seed is None else int(seed),
            "component_index": idx + 1,
            "eigenvalue_share": float(eig_share[idx]),
        }
        for idx in range(topk)
    ]
    return pd.DataFrame([summary_row]), pd.DataFrame(spectrum_rows)


def aggregate_rows(
    raw_df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
    stable_count_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in raw_df.groupby(list(group_cols), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {group_cols[i]: keys[i] for i in range(len(group_cols))}
        if "seed" in group.columns:
            seed_series = pd.to_numeric(group["seed"], errors="coerce")
            row["num_seeds"] = int(seed_series.nunique())
        else:
            row["num_seeds"] = 1
        for count_col in stable_count_cols:
            vals = pd.to_numeric(group[count_col], errors="coerce")
            row[count_col] = int(round(float(vals.mean()))) if vals.notna().any() else 0
        for col in metric_cols:
            vals = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def condition_positions(group_count: int, n_conditions: int, width: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    centers = np.arange(group_count, dtype=np.float64)
    start = -width * (n_conditions - 1) / 2.0
    offsets = [centers + start + idx * width for idx in range(n_conditions)]
    return centers, offsets


def _condition_label(backbone_name: str, mode: str) -> str:
    for b, m, label in CONDITION_ORDER:
        if b == backbone_name and m == mode:
            return label
    return f"{backbone_name}-{mode}"


def save_pair_similarity_probe_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    probe_df = summary_df[summary_df["mode"] == "probe_linear"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
    for ax, target in zip(axes, ["patient", "study"]):
        sub_df = probe_df[probe_df["target"] == target]
        x = np.arange(len(PAIR_GROUP_ORDER), dtype=np.float64)
        for backbone_name in ["imagenet", "cag"]:
            means = []
            stds = []
            for pair_group in PAIR_GROUP_ORDER:
                row = sub_df[(sub_df["backbone_name"] == backbone_name) & (sub_df["pair_group"] == pair_group)]
                means.append(float(row.iloc[0]["cosine_mean_mean"]))
                stds.append(float(row.iloc[0]["cosine_mean_std"]))
            means_arr = np.array(means, dtype=np.float64)
            stds_arr = np.array(stds, dtype=np.float64)
            ax.plot(x, means_arr, marker="o", linewidth=2.0, color=PLOT_COLORS[backbone_name], label=backbone_name)
            ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr, color=PLOT_COLORS[backbone_name], alpha=0.18)
        ax.set_xticks(x, [g.replace("_", "\n") for g in PAIR_GROUP_ORDER])
        ax.set_title(f"Probe Pair Geometry | {target.title()} head")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean Cosine Similarity")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_margin_compare_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["margin_mean", "negative_win_rate"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    width = 0.18
    for row_idx, target in enumerate(["patient", "study"]):
        target_df = summary_df[summary_df["target"] == target]
        centers, offsets = condition_positions(1, len(CONDITION_ORDER), width)
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            vals = []
            errs = []
            for backbone_name, mode, _ in CONDITION_ORDER:
                row = target_df[(target_df["backbone_name"] == backbone_name) & (target_df["mode"] == mode)]
                vals.append(float(row.iloc[0][f"{metric}_mean"]))
                errs.append(float(row.iloc[0][f"{metric}_std"]))
            for idx, (backbone_name, mode, label) in enumerate(CONDITION_ORDER):
                ax.bar(
                    offsets[idx],
                    [vals[idx]],
                    width=width,
                    color=PLOT_COLORS[backbone_name],
                    alpha=0.55 if mode == "raw_frozen" else 0.95,
                    yerr=[errs[idx]],
                    capsize=3,
                    label=label if row_idx == 0 and col_idx == 0 else None,
                )
            ax.set_xticks(centers, [target.title()])
            ax.set_title(f"{target.title()} | {metric}")
            ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_cluster_metrics_compare_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["intra_to_centroid_mean", "nearest_inter_centroid_mean", "separation_ratio", "silhouette_score"]
    combo_order = [
        ("patient", "patient", "p-head\np-cluster"),
        ("patient", "study", "p-head\ns-cluster"),
        ("study", "patient", "s-head\np-cluster"),
        ("study", "study", "s-head\ns-cluster"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))
    width = 0.18
    for ax, metric in zip(axes.flatten(), metrics):
        centers, offsets = condition_positions(len(combo_order), len(CONDITION_ORDER), width)
        for idx, (backbone_name, mode, label) in enumerate(CONDITION_ORDER):
            vals = []
            errs = []
            for target, label_level, _ in combo_order:
                row = summary_df[
                    (summary_df["target"] == target)
                    & (summary_df["label_level"] == label_level)
                    & (summary_df["backbone_name"] == backbone_name)
                    & (summary_df["mode"] == mode)
                ]
                vals.append(float(row.iloc[0][f"{metric}_mean"]))
                errs.append(float(row.iloc[0][f"{metric}_std"]))
            ax.bar(
                offsets[idx],
                vals,
                width=width,
                color=PLOT_COLORS[backbone_name],
                alpha=0.55 if mode == "raw_frozen" else 0.95,
                yerr=errs,
                capsize=3,
                label=label if metric == metrics[0] else None,
            )
        ax.set_xticks(centers, [label for _, _, label in combo_order])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heatmap(ax: plt.Axes, values: np.ndarray, row_labels: Sequence[str], col_labels: Sequence[str], title: str) -> None:
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_title(title)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            val = values[r, c]
            text = "nan" if not math.isfinite(float(val)) else f"{val:.3f}"
            ax.text(c, r, text, ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_spectral_compare_figure(
    spectrum_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    scalar_metrics = ["effective_rank", "anisotropy_ratio", "lambda1_share", "uniformity"]
    row_labels = [_condition_label(b, m) for b, m, _ in CONDITION_ORDER]
    col_labels = scalar_metrics
    for row_idx, target in enumerate(["patient", "study"]):
        ax_line = axes[row_idx, 0]
        target_spec = spectrum_df[spectrum_df["target"] == target]
        for backbone_name, mode, label in CONDITION_ORDER:
            row = target_spec[(target_spec["backbone_name"] == backbone_name) & (target_spec["mode"] == mode)].sort_values("component_index")
            x = row["component_index"].to_numpy(dtype=np.int32)
            y = row["eigenvalue_share_mean"].to_numpy(dtype=np.float64)
            yerr = row["eigenvalue_share_std"].to_numpy(dtype=np.float64)
            ax_line.plot(x, y, marker="o", linewidth=2.0, color=PLOT_COLORS[backbone_name], alpha=0.55 if mode == "raw_frozen" else 0.95, label=label)
            ax_line.fill_between(x, y - yerr, y + yerr, color=PLOT_COLORS[backbone_name], alpha=0.12 if mode == "probe_linear" else 0.06)
        ax_line.set_title(f"{target.title()} head | top-32 eigenspectrum")
        ax_line.set_xlabel("Component index")
        ax_line.set_ylabel("Eigenvalue share")
        ax_line.grid(axis="y", alpha=0.25)

        target_summary = summary_df[summary_df["target"] == target]
        heat = np.zeros((len(CONDITION_ORDER), len(scalar_metrics)), dtype=np.float64)
        for r, (backbone_name, mode, _) in enumerate(CONDITION_ORDER):
            row = target_summary[(target_summary["backbone_name"] == backbone_name) & (target_summary["mode"] == mode)]
            for c, metric in enumerate(scalar_metrics):
                heat[r, c] = float(row.iloc[0][f"{metric}_mean"])
        _heatmap(axes[row_idx, 1], heat, row_labels, col_labels, f"{target.title()} head | spectral scalars")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_umap_patients(manifest: pd.DataFrame, max_patients: int) -> List[str]:
    grouped = (
        manifest.groupby("patient_id")
        .agg(num_images=("image_id", "size"), num_studies=("study_id", "nunique"))
        .reset_index()
    )
    grouped = grouped[grouped["num_studies"] >= 2].sort_values(["num_images", "patient_id"], ascending=[False, True])
    return grouped["patient_id"].astype(str).head(max_patients).tolist()


def save_umap_probe_figure(
    image_root: Path,
    out_root: Path,
    global2_root: Path,
    manifest: pd.DataFrame,
    selected_patients: Sequence[str],
    device: torch.device,
    probe_batch_size: int,
    probe_seeds: Sequence[int],
    feature_store: Dict[str, torch.Tensor],
    feature_hashes: Dict[str, str],
    global2_test_manifest_hash: str,
    output_path: Path,
    seed: int,
) -> None:
    import umap

    if not selected_patients:
        log("Skipping UMAP figure: no >=2-study patients in active manifest.")
        return
    patient_cmap = plt.get_cmap("tab10")
    selected_set = set(selected_patients)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    chosen_seed = int(probe_seeds[0])
    umap_rows: List[Dict[str, object]] = []

    for row_idx, backbone_name in enumerate(["imagenet", "cag"]):
        for col_idx, target in enumerate(["patient", "study"]):
            ax = axes[row_idx, col_idx]
            full_embeddings, _ = apply_probe_checkpoint(
                global2_root=global2_root,
                backbone_name=backbone_name,
                target=target,
                seed=chosen_seed,
                features=feature_store[backbone_name],
                manifest_hash=global2_test_manifest_hash,
                feature_hash=feature_hashes[backbone_name],
                device=device,
                batch_size=probe_batch_size,
            )
            active_indices = manifest["image_id"].astype(int).to_numpy(dtype=np.int64)
            emb = l2_normalize_cpu(full_embeddings)[active_indices].numpy().astype(np.float32, copy=False)
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=max(5, min(15, len(manifest) - 1)),
                min_dist=0.2,
                metric="cosine",
                random_state=seed,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*", category=UserWarning)
                coords = reducer.fit_transform(emb)
            ax.scatter(coords[:, 0], coords[:, 1], s=14, c="#D0D0D0", alpha=0.35, linewidths=0)
            for patient_idx, patient_id in enumerate(selected_patients):
                patient_df = manifest[manifest["patient_id"].astype(str) == patient_id].copy()
                study_ids = sorted(patient_df["study_id"].astype(str).unique())
                study_marker_map = {study_id: MARKERS[idx % len(MARKERS)] for idx, study_id in enumerate(study_ids)}
                patient_mask = manifest["patient_id"].astype(str) == patient_id
                for study_id in study_ids:
                    mask = patient_mask & (manifest["study_id"].astype(str) == study_id)
                    pts = coords[mask.to_numpy(dtype=bool)]
                    if len(pts) == 0:
                        continue
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        s=42,
                        c=[patient_cmap(patient_idx % patient_cmap.N)],
                        marker=study_marker_map[study_id],
                        edgecolors="black",
                        linewidths=0.35,
                        label=f"P{patient_id}/S{study_id}" if (row_idx == 0 and col_idx == 0) else None,
                    )
                for idx_local, point in zip(patient_df.index.tolist(), coords[patient_mask.to_numpy(dtype=bool)]):
                    umap_rows.append(
                        {
                            "backbone_name": backbone_name,
                            "target": target,
                            "seed": chosen_seed,
                            "image_id": int(manifest.loc[idx_local, "image_id"]),
                            "img_path": str(manifest.loc[idx_local, "img_path"]),
                            "patient_id": str(manifest.loc[idx_local, "patient_id"]),
                            "study_id": str(manifest.loc[idx_local, "study_id"]),
                            "umap_x": float(point[0]),
                            "umap_y": float(point[1]),
                        }
                    )
            ax.set_title(f"{backbone_name} | {target} head | probe seed {chosen_seed}")
            ax.grid(alpha=0.18)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncols=min(4, len(handles)), bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if umap_rows:
        pd.DataFrame(umap_rows).to_csv(out_root / "umap_probe_seed11_coordinates.csv", index=False)


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    return f"{value:.6f}"


def _lookup_row(df: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for key, value in filters.items():
        mask &= df[key] == value
    sub = df[mask]
    if sub.empty:
        raise KeyError(filters)
    return sub.iloc[0]


def write_markdown_summary(
    output_path: Path,
    pair_df: pd.DataFrame,
    margin_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    spectral_df: pd.DataFrame,
    run_meta: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("# Global Analysis 4: Feature Geometry Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Global Analysis 2 root: `{run_meta['global2_root']}`")
    lines.append(f"- DICOM root: `{run_meta['dcm_root']}`")
    lines.append(f"- Probe seed set: `{', '.join(str(v) for v in run_meta['probe_seed_set'])}`")
    lines.append(
        f"- View bins: near <= {run_meta['near_view_threshold_deg']}, "
        f"mid <= {run_meta['mid_view_threshold_deg']}, far > {run_meta['mid_view_threshold_deg']}"
    )
    lines.append(f"- Headline split: `{run_meta['headline_split']}`")
    lines.append("")

    lines.append("## Headline Geometry Signals (Probe)")
    lines.append("")
    for target in ["patient", "study"]:
        same_patient_cross = {
            backbone: _lookup_row(pair_df, mode="probe_linear", target=target, backbone_name=backbone, pair_group="same_patient_cross_study")
            for backbone in ["imagenet", "cag"]
        }
        diff_mid = {
            backbone: _lookup_row(pair_df, mode="probe_linear", target=target, backbone_name=backbone, pair_group="different_patient_mid_view")
            for backbone in ["imagenet", "cag"]
        }
        margin_rows = {
            backbone: _lookup_row(margin_df, mode="probe_linear", target=target, backbone_name=backbone)
            for backbone in ["imagenet", "cag"]
        }
        lines.append(f"### {target.title()} head")
        lines.append("")
        lines.append(
            f"- `same_patient_cross_study` cosine mean: "
            f"ImageNet `{format_float(float(same_patient_cross['imagenet']['cosine_mean_mean']))}`, "
            f"CAG `{format_float(float(same_patient_cross['cag']['cosine_mean_mean']))}`"
        )
        lines.append(
            f"- `different_patient_mid_view` cosine mean: "
            f"ImageNet `{format_float(float(diff_mid['imagenet']['cosine_mean_mean']))}`, "
            f"CAG `{format_float(float(diff_mid['cag']['cosine_mean_mean']))}`"
        )
        lines.append(
            f"- margin mean: "
            f"ImageNet `{format_float(float(margin_rows['imagenet']['margin_mean_mean']))}`, "
            f"CAG `{format_float(float(margin_rows['cag']['margin_mean_mean']))}`"
        )
        lines.append(
            f"- negative win rate: "
            f"ImageNet `{format_float(float(margin_rows['imagenet']['negative_win_rate_mean']))}`, "
            f"CAG `{format_float(float(margin_rows['cag']['negative_win_rate_mean']))}`"
        )
        if target == "study":
            lines.append(
                f"- `same_patient_cross_study` negative similarity mean: "
                f"ImageNet `{format_float(float(margin_rows['imagenet']['same_patient_cross_study_negative_similarity_mean_mean']))}`, "
                f"CAG `{format_float(float(margin_rows['cag']['same_patient_cross_study_negative_similarity_mean_mean']))}`"
            )
        lines.append("")

    lines.append("## Cluster Geometry (Probe)")
    lines.append("")
    lines.append("| Head Target | Label Level | Backbone | Separation Ratio | Silhouette | Davies-Bouldin |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    probe_cluster = cluster_df[cluster_df["mode"] == "probe_linear"].sort_values(["target", "label_level", "backbone_name"])
    for _, row in probe_cluster.iterrows():
        lines.append(
            f"| {row['target']} | {row['label_level']} | {row['backbone_name']} | "
            f"{format_float(float(row['separation_ratio_mean']))} +/- {format_float(float(row['separation_ratio_std']))} | "
            f"{format_float(float(row['silhouette_score_mean']))} +/- {format_float(float(row['silhouette_score_std']))} | "
            f"{format_float(float(row['davies_bouldin_score_mean']))} +/- {format_float(float(row['davies_bouldin_score_std']))} |"
        )
    lines.append("")

    lines.append("## Spectral Geometry")
    lines.append("")
    lines.append("| Head Target | Backbone | Mode | Effective Rank | Anisotropy | Lambda1 Share | Uniformity |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    spectral_rows = spectral_df.sort_values(["target", "backbone_name", "mode"])
    for _, row in spectral_rows.iterrows():
        lines.append(
            f"| {row['target']} | {row['backbone_name']} | {row['mode']} | "
            f"{format_float(float(row['effective_rank_mean']))} +/- {format_float(float(row['effective_rank_std']))} | "
            f"{format_float(float(row['anisotropy_ratio_mean']))} +/- {format_float(float(row['anisotropy_ratio_std']))} | "
            f"{format_float(float(row['lambda1_share_mean']))} +/- {format_float(float(row['lambda1_share_std']))} | "
            f"{format_float(float(row['uniformity_mean']))} +/- {format_float(float(row['uniformity_std']))} |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- Check whether CAG probe embeddings raise `same_patient_cross_study` similarity while also lowering `study`-target margin.")
    lines.append("- Check whether CAG probe embeddings keep `different_patient_near/mid` similarity higher, which would indicate weaker fine-grained separation.")
    lines.append("- Check whether patient/study cluster separation ratios and silhouette scores are lower for CAG, indicating under-separation or overlap.")
    lines.append("- Check whether effective rank is lower or anisotropy is higher for CAG, indicating a less retrieval-friendly global geometry.")
    lines.append("- This analysis is evidence-only. It does not recommend a develop direction yet.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_jobs(probe_seeds: Sequence[int]) -> List[JobDef]:
    jobs: List[JobDef] = []
    for backbone_name in ["imagenet", "cag"]:
        for target in ["patient", "study"]:
            jobs.append(JobDef(kind="analysis", mode="raw_frozen", backbone_name=backbone_name, target=target, seed=None))
    for seed in probe_seeds:
        for backbone_name in ["imagenet", "cag"]:
            for target in ["patient", "study"]:
                jobs.append(JobDef(kind="analysis", mode="probe_linear", backbone_name=backbone_name, target=target, seed=int(seed)))
    jobs.append(JobDef(kind="postprocess", name="aggregate_and_render_figures"))
    jobs.append(JobDef(kind="postprocess", name="write_markdown_and_metadata"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global2-root", default="outputs/global_2_study_patient_retrieval_unique_view")
    parser.add_argument("--image-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--dcm-root", default="input/stent_split_dcm_unique_view")
    parser.add_argument("--output-root", default="outputs/global_4_feature_geometry_analysis_unique_view")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--near-view-threshold-deg", type=float, default=5.0)
    parser.add_argument("--mid-view-threshold-deg", type=float, default=20.0)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    ensure_dir(out_root)
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        out_root,
        args.log_file,
        default_prefix="global_4_feature_geometry_analysis",
    )

    try:
        set_global_seed(int(args.seed))
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        if args.near_view_threshold_deg <= 0.0 or args.mid_view_threshold_deg <= args.near_view_threshold_deg:
            raise ValueError("Expected positive ascending view bins: near < mid.")

        global2_root = Path(args.global2_root).resolve()
        image_root = Path(args.image_root).resolve()
        dcm_root = Path(args.dcm_root).resolve()
        device = resolve_device(args.device)

        full_manifest = build_angle_manifest(global2_root, image_root, dcm_root, out_root)
        manifest = maybe_subsample_manifest(full_manifest, args.max_images, int(args.seed))
        manifest.to_csv(out_root / "image_manifest_test_with_angles_active.csv", index=False)
        full_manifest_hash = hash_dataframe(
            full_manifest,
            [
                "image_id",
                "img_path",
                "patient_id",
                "study_id",
                "class_name",
                "primary_angle",
                "secondary_angle",
            ],
        )
        global2_test_manifest_hash = hash_dataframe(
            full_manifest,
            ["image_id", "img_path", "patient_id", "study_id"],
        )
        active_manifest_hash = hash_dataframe(
            manifest,
            [
                "image_id",
                "img_path",
                "patient_id",
                "study_id",
                "class_name",
                "primary_angle",
                "secondary_angle",
            ],
        )
        active_indices = manifest["image_id"].astype(int).to_numpy(dtype=np.int64)
        pair_meta = build_pair_meta(manifest, float(args.near_view_threshold_deg), float(args.mid_view_threshold_deg))

        feature_store: Dict[str, torch.Tensor] = {}
        feature_meta_store: Dict[str, Dict[str, object]] = {}
        feature_hashes: Dict[str, str] = {}
        for backbone_name in ["imagenet", "cag"]:
            features, meta = load_test_features(global2_root, backbone_name)
            feature_store[backbone_name] = features
            feature_meta_store[backbone_name] = meta
            feature_hashes[backbone_name] = make_feature_hash(features)
            log(
                f"Loaded test features | backbone={backbone_name} | n={features.shape[0]} d={features.shape[1]} "
                f"| feature_hash={feature_hashes[backbone_name][:12]}"
            )

        jobs = build_jobs(args.probe_seeds)
        tracker = FullRunTracker(jobs)

        pair_raw_tables: List[pd.DataFrame] = []
        margin_raw_tables: List[pd.DataFrame] = []
        cluster_raw_tables: List[pd.DataFrame] = []
        spectral_raw_tables: List[pd.DataFrame] = []
        spectrum_raw_tables: List[pd.DataFrame] = []

        for job_index, job in enumerate(jobs[:16]):
            tracker.start_job(job_index)
            if job.mode == "raw_frozen":
                embeddings = l2_normalize_cpu(feature_store[job.backbone_name])[active_indices]
                tracker.update_phase("load_embeddings", 1, 6)
            else:
                full_embeddings, _ = apply_probe_checkpoint(
                    global2_root=global2_root,
                    backbone_name=job.backbone_name,
                    target=job.target,
                    seed=int(job.seed),
                    features=feature_store[job.backbone_name],
                    manifest_hash=global2_test_manifest_hash,
                    feature_hash=feature_hashes[job.backbone_name],
                    device=device,
                    batch_size=int(args.probe_batch_size),
                )
                embeddings = l2_normalize_cpu(full_embeddings)[active_indices]
                tracker.update_phase("load_embeddings", 1, 6)

            sims = compute_similarity_matrix(embeddings)
            tracker.update_phase("pair_geometry", 2, 6)
            pair_raw_tables.append(
                compute_pair_geometry_summary(
                    sims=sims,
                    pair_meta=pair_meta,
                    mode=job.mode,
                    backbone_name=job.backbone_name,
                    target=job.target,
                    seed=job.seed,
                )
            )
            tracker.update_phase("margin_geometry", 3, 6)
            margin_raw_tables.append(
                compute_margin_geometry_summary(
                    sims=sims,
                    manifest=manifest,
                    mode=job.mode,
                    backbone_name=job.backbone_name,
                    target=job.target,
                    seed=job.seed,
                )
            )
            tracker.update_phase("cluster_geometry", 4, 6)
            cluster_raw_tables.append(
                compute_cluster_geometry_rows(
                    embeddings=embeddings,
                    manifest=manifest,
                    mode=job.mode,
                    backbone_name=job.backbone_name,
                    target=job.target,
                    seed=job.seed,
                )
            )
            tracker.update_phase("spectral_geometry", 5, 6)
            spectral_summary, spectrum_rows = compute_spectral_geometry_summary(
                embeddings=embeddings,
                sims=sims,
                pair_meta=pair_meta,
                mode=job.mode,
                backbone_name=job.backbone_name,
                target=job.target,
                seed=job.seed,
            )
            spectral_raw_tables.append(spectral_summary)
            spectrum_raw_tables.append(spectrum_rows)
            tracker.update_phase("write_raw_rows", 6, 6)
            tracker.finish_job()

        tracker.start_job(16)
        pair_raw_df = pd.concat(pair_raw_tables, axis=0, ignore_index=True)
        margin_raw_df = pd.concat(margin_raw_tables, axis=0, ignore_index=True)
        cluster_raw_df = pd.concat(cluster_raw_tables, axis=0, ignore_index=True)
        spectral_raw_df = pd.concat(spectral_raw_tables, axis=0, ignore_index=True)
        spectrum_raw_df = pd.concat(spectrum_raw_tables, axis=0, ignore_index=True)

        pair_summary_df = aggregate_rows(
            raw_df=pair_raw_df,
            group_cols=["mode", "target", "backbone_name", "pair_group"],
            metric_cols=["cosine_mean", "cosine_median", "distance_mean", "distance_median"],
            stable_count_cols=["num_pairs"],
        )
        margin_summary_df = aggregate_rows(
            raw_df=margin_raw_df,
            group_cols=["mode", "target", "backbone_name"],
            metric_cols=[
                "best_positive_similarity_mean",
                "hardest_negative_similarity_mean",
                "margin_mean",
                "margin_median",
                "negative_win_rate",
                "same_patient_cross_study_negative_similarity_mean",
            ],
            stable_count_cols=["num_queries_with_positive", "queries_with_same_patient_cross_study_negative"],
        )
        cluster_summary_df = aggregate_rows(
            raw_df=cluster_raw_df,
            group_cols=["mode", "target", "backbone_name", "label_level"],
            metric_cols=[
                "mean_cluster_size",
                "intra_to_centroid_mean",
                "nearest_inter_centroid_mean",
                "separation_ratio",
                "silhouette_score",
                "davies_bouldin_score",
            ],
            stable_count_cols=["num_clusters"],
        )
        spectral_summary_df = aggregate_rows(
            raw_df=spectral_raw_df,
            group_cols=["mode", "target", "backbone_name"],
            metric_cols=[
                "embed_dim",
                "norm_mean",
                "norm_std",
                "effective_rank",
                "participation_ratio",
                "anisotropy_ratio",
                "lambda1_share",
                "uniformity",
                "alignment_same_patient_all",
                "alignment_same_patient_cross_study",
                "alignment_same_study",
            ],
            stable_count_cols=[],
        )
        spectrum_summary_df = aggregate_rows(
            raw_df=spectrum_raw_df,
            group_cols=["mode", "target", "backbone_name", "component_index"],
            metric_cols=["eigenvalue_share"],
            stable_count_cols=[],
        )

        pair_raw_df.to_csv(out_root / "summary_global_4_pair_geometry_raw.csv", index=False)
        pair_summary_df.to_csv(out_root / "summary_global_4_pair_geometry.csv", index=False)
        margin_raw_df.to_csv(out_root / "summary_global_4_margin_geometry_raw.csv", index=False)
        margin_summary_df.to_csv(out_root / "summary_global_4_margin_geometry.csv", index=False)
        cluster_raw_df.to_csv(out_root / "summary_global_4_cluster_geometry_raw.csv", index=False)
        cluster_summary_df.to_csv(out_root / "summary_global_4_cluster_geometry.csv", index=False)
        spectral_raw_df.to_csv(out_root / "summary_global_4_spectral_geometry_raw.csv", index=False)
        spectral_summary_df.to_csv(out_root / "summary_global_4_spectral_geometry.csv", index=False)
        spectrum_raw_df.to_csv(out_root / "summary_global_4_eigenspectrum_raw.csv", index=False)
        spectrum_summary_df.to_csv(out_root / "summary_global_4_eigenspectrum.csv", index=False)

        save_pair_similarity_probe_figure(pair_summary_df, out_root / "fig_global4_pair_similarity_probe.png")
        save_margin_compare_figure(margin_summary_df, out_root / "fig_global4_margin_compare.png")
        save_cluster_metrics_compare_figure(cluster_summary_df, out_root / "fig_global4_cluster_metrics_compare.png")
        save_spectral_compare_figure(
            spectrum_df=spectrum_summary_df,
            summary_df=spectral_summary_df,
            output_path=out_root / "fig_global4_spectral_compare.png",
        )
        if not args.skip_umap:
            selected_patients = select_umap_patients(manifest, max_patients=8)
            save_umap_probe_figure(
                image_root=image_root,
                out_root=out_root,
                global2_root=global2_root,
                manifest=manifest,
                selected_patients=selected_patients,
                device=device,
                probe_batch_size=int(args.probe_batch_size),
                probe_seeds=[int(v) for v in args.probe_seeds],
                feature_store=feature_store,
                feature_hashes=feature_hashes,
                global2_test_manifest_hash=global2_test_manifest_hash,
                output_path=out_root / "fig_global4_umap_probe.png",
                seed=int(args.seed),
            )
        tracker.update_phase("aggregate_and_render", 1, 1)
        tracker.finish_job()

        tracker.start_job(17)
        run_meta = {
            "global2_root": str(global2_root),
            "image_root": str(image_root),
            "dcm_root": str(dcm_root),
            "output_root": str(out_root.resolve()),
            "log_path": str(log_path.resolve()),
            "probe_seed_set": [int(v) for v in args.probe_seeds],
            "feature_source": "x_norm_clstoken",
            "headline_split": "test",
            "near_view_threshold_deg": float(args.near_view_threshold_deg),
            "mid_view_threshold_deg": float(args.mid_view_threshold_deg),
            "skip_umap": bool(args.skip_umap),
            "full_manifest_hash": full_manifest_hash,
            "global2_test_manifest_hash": global2_test_manifest_hash,
            "active_manifest_hash": active_manifest_hash,
            "feature_hashes": feature_hashes,
            "job_count": len(jobs),
            "jobs": [
                {
                    "kind": job.kind,
                    "mode": job.mode if job.mode is not None else "-",
                    "backbone_name": job.backbone_name if job.backbone_name is not None else "-",
                    "target": job.target if job.target is not None else "-",
                    "seed": job.seed if job.seed is not None else "raw",
                    "name": job.name if job.name is not None else "-",
                }
                for job in jobs
            ],
            "pair_raw_summary_path": str((out_root / "summary_global_4_pair_geometry_raw.csv").resolve()),
            "pair_summary_path": str((out_root / "summary_global_4_pair_geometry.csv").resolve()),
            "margin_raw_summary_path": str((out_root / "summary_global_4_margin_geometry_raw.csv").resolve()),
            "margin_summary_path": str((out_root / "summary_global_4_margin_geometry.csv").resolve()),
            "cluster_raw_summary_path": str((out_root / "summary_global_4_cluster_geometry_raw.csv").resolve()),
            "cluster_summary_path": str((out_root / "summary_global_4_cluster_geometry.csv").resolve()),
            "spectral_raw_summary_path": str((out_root / "summary_global_4_spectral_geometry_raw.csv").resolve()),
            "spectral_summary_path": str((out_root / "summary_global_4_spectral_geometry.csv").resolve()),
            "eigenspectrum_raw_summary_path": str((out_root / "summary_global_4_eigenspectrum_raw.csv").resolve()),
            "eigenspectrum_summary_path": str((out_root / "summary_global_4_eigenspectrum.csv").resolve()),
        }
        write_markdown_summary(
            output_path=out_root / "analysis_global_4_feature_geometry.md",
            pair_df=pair_summary_df,
            margin_df=margin_summary_df,
            cluster_df=cluster_summary_df,
            spectral_df=spectral_summary_df,
            run_meta=run_meta,
        )
        (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        tracker.update_phase("write_markdown_and_metadata", 1, 1)
        tracker.finish_job()
        log("Global Analysis 4 completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
