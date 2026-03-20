#!/usr/bin/env python3
"""
Global Analysis 3 — Hard Positive / Hard Negative Analysis with View-Angle Metadata.

This analysis is fully offline. It reuses Global Analysis 2 artifacts:
- test manifest
- cached test features
- trained probe checkpoints

It joins DICOM view-angle metadata from the unique-view DICOM root and then
localizes which pair subtypes are responsible for the observed global weakness.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}
CONDITION_ORDER = [
    ("imagenet", "raw_frozen", "ImageNet Raw"),
    ("imagenet", "probe_linear", "ImageNet Probe"),
    ("cag", "raw_frozen", "CAG Raw"),
    ("cag", "probe_linear", "CAG Probe"),
]


@dataclass(frozen=True)
class JobDef:
    mode: str
    backbone_name: str
    target: str
    seed: int | None

    @property
    def display_name(self) -> str:
        seed_text = "raw" if self.seed is None else f"seed{self.seed}"
        return f"{self.mode}/{self.backbone_name}/{self.target}/{seed_text}"


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
                    "backbone": job.backbone_name,
                    "mode": job.mode,
                    "target": job.target,
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


def get_positive_subtypes(target: str) -> List[str]:
    if target == "patient":
        return [
            "same_study_near_view",
            "same_study_mid_view",
            "same_study_far_view",
            "cross_study",
        ]
    if target == "study":
        return [
            "same_study_near_view",
            "same_study_mid_view",
            "same_study_far_view",
        ]
    raise ValueError(target)


def get_negative_subtypes(target: str) -> List[str]:
    if target == "patient":
        return [
            "different_patient_near_view",
            "different_patient_mid_view",
            "different_patient_far_view",
        ]
    if target == "study":
        return [
            "same_patient_cross_study",
            "different_patient_near_view",
            "different_patient_mid_view",
            "different_patient_far_view",
        ]
    raise ValueError(target)


def build_subtype_masks(
    target: str,
    query_index: int,
    patient_ids: np.ndarray,
    study_ids: np.ndarray,
    delta_view: np.ndarray,
    near_thresh: float,
    mid_thresh: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n = len(patient_ids)
    not_self = np.ones(n, dtype=bool)
    not_self[query_index] = False
    same_patient = patient_ids == patient_ids[query_index]
    same_study = study_ids == study_ids[query_index]
    same_study_other = same_study & not_self
    same_patient_other = same_patient & not_self
    near_mask = delta_view <= near_thresh
    mid_mask = (delta_view > near_thresh) & (delta_view <= mid_thresh)
    far_mask = delta_view > mid_thresh

    if target == "patient":
        overall_positive = same_patient_other
        positive_masks = {
            "same_study_near_view": same_study_other & near_mask,
            "same_study_mid_view": same_study_other & mid_mask,
            "same_study_far_view": same_study_other & far_mask,
            "cross_study": same_patient_other & (~same_study),
        }
        diff_patient = ~same_patient
        negative_masks = {
            "different_patient_near_view": diff_patient & near_mask,
            "different_patient_mid_view": diff_patient & mid_mask,
            "different_patient_far_view": diff_patient & far_mask,
        }
        return overall_positive, positive_masks, negative_masks

    if target == "study":
        overall_positive = same_study_other
        positive_masks = {
            "same_study_near_view": same_study_other & near_mask,
            "same_study_mid_view": same_study_other & mid_mask,
            "same_study_far_view": same_study_other & far_mask,
        }
        diff_patient = ~same_patient
        negative_masks = {
            "same_patient_cross_study": same_patient_other & (~same_study),
            "different_patient_near_view": diff_patient & near_mask,
            "different_patient_mid_view": diff_patient & mid_mask,
            "different_patient_far_view": diff_patient & far_mask,
        }
        return overall_positive, positive_masks, negative_masks

    raise ValueError(f"Unsupported target: {target}")


def compute_hard_pair_tables(
    sims: np.ndarray,
    manifest: pd.DataFrame,
    mode: str,
    backbone_name: str,
    target: str,
    seed: int | None,
    near_thresh: float,
    mid_thresh: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(manifest)
    image_paths = manifest["img_path"].to_numpy(dtype=str)
    patient_ids = manifest["patient_id"].to_numpy(dtype=str)
    study_ids = manifest["study_id"].to_numpy(dtype=str)
    class_names = manifest["class_name"].to_numpy(dtype=str)
    prim = manifest["primary_angle"].to_numpy(dtype=np.float32)
    sec = manifest["secondary_angle"].to_numpy(dtype=np.float32)

    positive_rows: List[Dict[str, object]] = []
    negative_rows: List[Dict[str, object]] = []
    positive_subtypes = get_positive_subtypes(target)
    negative_subtypes = get_negative_subtypes(target)

    for i in range(n):
        score_vec = sims[i].copy()
        score_vec[i] = -1e9
        sorted_idx = np.argsort(-score_vec)
        ranks = np.empty(n, dtype=np.int32)
        ranks[sorted_idx] = np.arange(1, n + 1, dtype=np.int32)
        ranks[i] = 0

        delta_view = np.sqrt((prim - prim[i]) ** 2 + (sec - sec[i]) ** 2)
        overall_positive_mask, positive_masks, negative_masks = build_subtype_masks(
            target=target,
            query_index=i,
            patient_ids=patient_ids,
            study_ids=study_ids,
            delta_view=delta_view,
            near_thresh=near_thresh,
            mid_thresh=mid_thresh,
        )

        overall_positive_indices = np.flatnonzero(overall_positive_mask)
        if len(overall_positive_indices) > 0:
            first_positive_rank = int(ranks[overall_positive_indices].min())
        else:
            first_positive_rank = math.nan

        for subtype in positive_subtypes:
            candidate_indices = np.flatnonzero(positive_masks[subtype])
            if len(candidate_indices) == 0:
                continue
            best_local = int(candidate_indices[np.argmax(score_vec[candidate_indices])])
            best_rank = int(ranks[best_local])
            best_score = float(score_vec[best_local])
            positive_rows.append(
                {
                    "mode": mode,
                    "target": target,
                    "backbone_name": backbone_name,
                    "seed": "" if seed is None else int(seed),
                    "query_index": i,
                    "query_path": image_paths[i],
                    "query_patient_id": patient_ids[i],
                    "query_study_id": study_ids[i],
                    "query_class_name": class_names[i],
                    "query_primary_angle": float(prim[i]),
                    "query_secondary_angle": float(sec[i]),
                    "subtype": subtype,
                    "candidate_index": best_local,
                    "candidate_path": image_paths[best_local],
                    "candidate_patient_id": patient_ids[best_local],
                    "candidate_study_id": study_ids[best_local],
                    "candidate_class_name": class_names[best_local],
                    "candidate_primary_angle": float(prim[best_local]),
                    "candidate_secondary_angle": float(sec[best_local]),
                    "delta_view": float(delta_view[best_local]),
                    "rank": best_rank,
                    "similarity": best_score,
                    "first_positive_rank": first_positive_rank,
                    "recall_at_1": int(best_rank <= 1),
                    "recall_at_5": int(best_rank <= 5),
                    "recall_at_10": int(best_rank <= 10),
                }
            )

        for subtype in negative_subtypes:
            candidate_indices = np.flatnonzero(negative_masks[subtype])
            if len(candidate_indices) == 0:
                continue
            best_local = int(candidate_indices[np.argmax(score_vec[candidate_indices])])
            best_rank = int(ranks[best_local])
            best_score = float(score_vec[best_local])
            pre_positive_intrusion = (
                int(best_rank < first_positive_rank)
                if isinstance(first_positive_rank, (int, np.integer)) and not math.isnan(float(first_positive_rank))
                else math.nan
            )
            negative_rows.append(
                {
                    "mode": mode,
                    "target": target,
                    "backbone_name": backbone_name,
                    "seed": "" if seed is None else int(seed),
                    "query_index": i,
                    "query_path": image_paths[i],
                    "query_patient_id": patient_ids[i],
                    "query_study_id": study_ids[i],
                    "query_class_name": class_names[i],
                    "query_primary_angle": float(prim[i]),
                    "query_secondary_angle": float(sec[i]),
                    "subtype": subtype,
                    "candidate_index": best_local,
                    "candidate_path": image_paths[best_local],
                    "candidate_patient_id": patient_ids[best_local],
                    "candidate_study_id": study_ids[best_local],
                    "candidate_class_name": class_names[best_local],
                    "candidate_primary_angle": float(prim[best_local]),
                    "candidate_secondary_angle": float(sec[best_local]),
                    "delta_view": float(delta_view[best_local]),
                    "rank": best_rank,
                    "similarity": best_score,
                    "first_positive_rank": first_positive_rank,
                    "intrusion_at_1": int(best_rank <= 1),
                    "intrusion_at_5": int(best_rank <= 5),
                    "intrusion_at_10": int(best_rank <= 10),
                    "pre_positive_intrusion": pre_positive_intrusion,
                }
            )

    return pd.DataFrame(positive_rows), pd.DataFrame(negative_rows)


def summarize_positive_raw(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["mode", "target", "backbone_name", "subtype", "seed"]
    for keys, group in df.groupby(group_cols, sort=True):
        mode, target, backbone_name, subtype, seed = keys
        rows.append(
            {
                "mode": mode,
                "target": target,
                "backbone_name": backbone_name,
                "subtype": subtype,
                "seed": seed,
                "num_queries_with_subtype": int(len(group)),
                "best_positive_rank_median": float(group["rank"].median()),
                "best_positive_rank_mean": float(group["rank"].mean()),
                "best_positive_score_mean": float(group["similarity"].mean()),
                "subtype_recall_at_1": float(group["recall_at_1"].mean()),
                "subtype_recall_at_5": float(group["recall_at_5"].mean()),
                "subtype_recall_at_10": float(group["recall_at_10"].mean()),
            }
        )
    return pd.DataFrame(rows)


def summarize_negative_raw(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["mode", "target", "backbone_name", "subtype", "seed"]
    for keys, group in df.groupby(group_cols, sort=True):
        mode, target, backbone_name, subtype, seed = keys
        pre_positive = pd.to_numeric(group["pre_positive_intrusion"], errors="coerce")
        rows.append(
            {
                "mode": mode,
                "target": target,
                "backbone_name": backbone_name,
                "subtype": subtype,
                "seed": seed,
                "num_queries_with_subtype": int(len(group)),
                "hardest_negative_rank_median": float(group["rank"].median()),
                "hardest_negative_rank_mean": float(group["rank"].mean()),
                "hardest_negative_score_mean": float(group["similarity"].mean()),
                "intrusion_at_1": float(group["intrusion_at_1"].mean()),
                "intrusion_at_5": float(group["intrusion_at_5"].mean()),
                "intrusion_at_10": float(group["intrusion_at_10"].mean()),
                "pre_positive_intrusion_rate": float(pre_positive.mean()) if pre_positive.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_probe_and_raw(raw_df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["mode", "target", "backbone_name", "subtype"]
    for keys, group in raw_df.groupby(group_cols, sort=True):
        mode, target, backbone_name, subtype = keys
        if mode == "raw_frozen":
            base = group.iloc[0]
            row = {
                "mode": mode,
                "target": target,
                "backbone_name": backbone_name,
                "subtype": subtype,
                "num_seeds": 1,
                "num_queries_with_subtype": int(base["num_queries_with_subtype"]),
            }
            for col in metric_cols:
                row[f"{col}_mean"] = float(base[col])
                row[f"{col}_std"] = 0.0
            rows.append(row)
            continue
        row = {
            "mode": mode,
            "target": target,
            "backbone_name": backbone_name,
            "subtype": subtype,
            "num_seeds": int(group["seed"].replace("", np.nan).nunique()),
            "num_queries_with_subtype": int(round(float(group["num_queries_with_subtype"].mean()))),
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def select_examples(df: pd.DataFrame, harder_is_higher_rank: bool, limit_per_group: int) -> pd.DataFrame:
    out_frames: List[pd.DataFrame] = []
    group_cols = ["mode", "target", "backbone_name", "seed", "subtype"]
    for _, group in df.groupby(group_cols, sort=True):
        sort_cols = ["rank", "similarity"]
        ascending = [not harder_is_higher_rank, harder_is_higher_rank]
        out_frames.append(group.sort_values(sort_cols, ascending=ascending).head(limit_per_group))
    if not out_frames:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(out_frames, axis=0, ignore_index=True)


def _condition_label(backbone_name: str, mode: str) -> str:
    for b_name, mode_name, label in CONDITION_ORDER:
        if b_name == backbone_name and mode_name == mode:
            return label
    return f"{backbone_name}-{mode}"


def make_group_positions(group_count: int, bar_count: int, width: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    centers = np.arange(group_count, dtype=np.float64)
    offsets = []
    start = -width * (bar_count - 1) / 2.0
    for idx in range(bar_count):
        offsets.append(centers + start + idx * width)
    return centers, offsets


def save_positive_rank_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    targets = ["patient", "study"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    width = 0.18
    for ax, target in zip(axes, targets):
        subtypes = get_positive_subtypes(target)
        centers, offsets = make_group_positions(len(subtypes), len(CONDITION_ORDER), width)
        for idx, (backbone_name, mode, label) in enumerate(CONDITION_ORDER):
            means = []
            errs = []
            for subtype in subtypes:
                row = summary_df[
                    (summary_df["target"] == target)
                    & (summary_df["backbone_name"] == backbone_name)
                    & (summary_df["mode"] == mode)
                    & (summary_df["subtype"] == subtype)
                ]
                means.append(float(row.iloc[0]["best_positive_rank_median_mean"]) if not row.empty else np.nan)
                errs.append(float(row.iloc[0]["best_positive_rank_median_std"]) if not row.empty else 0.0)
            color = PLOT_COLORS[backbone_name]
            alpha = 0.55 if mode == "raw_frozen" else 0.95
            ax.bar(offsets[idx], means, width=width, color=color, alpha=alpha, yerr=errs, capsize=3, label=label)
        ax.set_xticks(centers, [s.replace("_", "\n") for s in subtypes])
        ax.set_title(f"{target.title()} Positives")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Median Best Positive Rank")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Global Analysis 3: Positive Subtype Rank", y=1.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_positive_recall_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    targets = ["patient", "study"]
    metrics = [("subtype_recall_at_1_mean", "R@1"), ("subtype_recall_at_5_mean", "R@5"), ("subtype_recall_at_10_mean", "R@10")]
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(18, 8), sharey="row")
    width = 0.18
    for row_idx, target in enumerate(targets):
        subtypes = get_positive_subtypes(target)
        centers, offsets = make_group_positions(len(subtypes), len(CONDITION_ORDER), width)
        for col_idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for idx, (backbone_name, mode, label) in enumerate(CONDITION_ORDER):
                means = []
                errs = []
                for subtype in subtypes:
                    row = summary_df[
                        (summary_df["target"] == target)
                        & (summary_df["backbone_name"] == backbone_name)
                        & (summary_df["mode"] == mode)
                        & (summary_df["subtype"] == subtype)
                    ]
                    means.append(float(row.iloc[0][metric_col]) if not row.empty else np.nan)
                    errs.append(float(row.iloc[0][metric_col.replace("_mean", "_std")]) if not row.empty else 0.0)
                color = PLOT_COLORS[backbone_name]
                alpha = 0.55 if mode == "raw_frozen" else 0.95
                ax.bar(offsets[idx], means, width=width, color=color, alpha=alpha, yerr=errs, capsize=3)
            ax.set_xticks(centers, [s.replace("_", "\n") for s in subtypes])
            ax.set_title(f"{target.title()} {metric_label}")
            ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Positive Recall")
    axes[1, 0].set_ylabel("Positive Recall")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PLOT_COLORS[b], alpha=0.55 if m == "raw_frozen" else 0.95) for b, m, _ in CONDITION_ORDER]
    labels = [label for _, _, label in CONDITION_ORDER]
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_negative_intrusion_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    targets = ["patient", "study"]
    metrics = [
        ("intrusion_at_1_mean", "Intr@1"),
        ("intrusion_at_5_mean", "Intr@5"),
        ("intrusion_at_10_mean", "Intr@10"),
        ("pre_positive_intrusion_rate_mean", "Pre-Pos"),
    ]
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(22, 8), sharey="row")
    width = 0.18
    for row_idx, target in enumerate(targets):
        subtypes = get_negative_subtypes(target)
        centers, offsets = make_group_positions(len(subtypes), len(CONDITION_ORDER), width)
        for col_idx, (metric_col, metric_label) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for idx, (backbone_name, mode, _) in enumerate(CONDITION_ORDER):
                means = []
                errs = []
                for subtype in subtypes:
                    row = summary_df[
                        (summary_df["target"] == target)
                        & (summary_df["backbone_name"] == backbone_name)
                        & (summary_df["mode"] == mode)
                        & (summary_df["subtype"] == subtype)
                    ]
                    means.append(float(row.iloc[0][metric_col]) if not row.empty else np.nan)
                    errs.append(float(row.iloc[0][metric_col.replace("_mean", "_std")]) if not row.empty else 0.0)
                color = PLOT_COLORS[backbone_name]
                alpha = 0.55 if mode == "raw_frozen" else 0.95
                ax.bar(offsets[idx], means, width=width, color=color, alpha=alpha, yerr=errs, capsize=3)
            ax.set_xticks(centers, [s.replace("_", "\n") for s in subtypes])
            ax.set_title(f"{target.title()} {metric_label}")
            ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Negative Difficulty")
    axes[1, 0].set_ylabel("Negative Difficulty")
    handles = [plt.Rectangle((0, 0), 1, 1, color=PLOT_COLORS[b], alpha=0.55 if m == "raw_frozen" else 0.95) for b, m, _ in CONDITION_ORDER]
    labels = [label for _, _, label in CONDITION_ORDER]
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_delta_figure(
    positive_summary_df: pd.DataFrame,
    negative_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    targets = ["patient", "study"]
    modes = ["raw_frozen", "probe_linear"]
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey="row")
    width = 0.34

    for row_idx, target in enumerate(targets):
        positive_subtypes = get_positive_subtypes(target)
        centers, offsets = make_group_positions(len(positive_subtypes), len(modes), width)
        ax = axes[row_idx, 0]
        for idx, mode in enumerate(modes):
            deltas = []
            for subtype in positive_subtypes:
                img_row = positive_summary_df[
                    (positive_summary_df["target"] == target)
                    & (positive_summary_df["backbone_name"] == "imagenet")
                    & (positive_summary_df["mode"] == mode)
                    & (positive_summary_df["subtype"] == subtype)
                ]
                cag_row = positive_summary_df[
                    (positive_summary_df["target"] == target)
                    & (positive_summary_df["backbone_name"] == "cag")
                    & (positive_summary_df["mode"] == mode)
                    & (positive_summary_df["subtype"] == subtype)
                ]
                if img_row.empty or cag_row.empty:
                    deltas.append(np.nan)
                else:
                    deltas.append(float(cag_row.iloc[0]["subtype_recall_at_10_mean"]) - float(img_row.iloc[0]["subtype_recall_at_10_mean"]))
            ax.bar(offsets[idx], deltas, width=width, color="#8C8C8C" if mode == "raw_frozen" else "#2F2F2F", alpha=0.85, label="Raw" if mode == "raw_frozen" else "Probe")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(centers, [s.replace("_", "\n") for s in positive_subtypes])
        ax.set_title(f"{target.title()} Positive | CAG - ImageNet R@10")
        ax.grid(axis="y", alpha=0.25)

        negative_subtypes = get_negative_subtypes(target)
        centers, offsets = make_group_positions(len(negative_subtypes), len(modes), width)
        ax = axes[row_idx, 1]
        for idx, mode in enumerate(modes):
            deltas = []
            for subtype in negative_subtypes:
                img_row = negative_summary_df[
                    (negative_summary_df["target"] == target)
                    & (negative_summary_df["backbone_name"] == "imagenet")
                    & (negative_summary_df["mode"] == mode)
                    & (negative_summary_df["subtype"] == subtype)
                ]
                cag_row = negative_summary_df[
                    (negative_summary_df["target"] == target)
                    & (negative_summary_df["backbone_name"] == "cag")
                    & (negative_summary_df["mode"] == mode)
                    & (negative_summary_df["subtype"] == subtype)
                ]
                if img_row.empty or cag_row.empty:
                    deltas.append(np.nan)
                else:
                    deltas.append(float(cag_row.iloc[0]["intrusion_at_10_mean"]) - float(img_row.iloc[0]["intrusion_at_10_mean"]))
            ax.bar(offsets[idx], deltas, width=width, color="#8C8C8C" if mode == "raw_frozen" else "#2F2F2F", alpha=0.85, label="Raw" if mode == "raw_frozen" else "Probe")
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(centers, [s.replace("_", "\n") for s in negative_subtypes])
        ax.set_title(f"{target.title()} Negative | CAG - ImageNet Intr@10")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    return f"{value:.6f}"


def _worst_positive_subtype(summary_df: pd.DataFrame, target: str, mode: str) -> Tuple[str, float]:
    rows = []
    for subtype in get_positive_subtypes(target):
        img_row = summary_df[(summary_df["target"] == target) & (summary_df["mode"] == mode) & (summary_df["backbone_name"] == "imagenet") & (summary_df["subtype"] == subtype)]
        cag_row = summary_df[(summary_df["target"] == target) & (summary_df["mode"] == mode) & (summary_df["backbone_name"] == "cag") & (summary_df["subtype"] == subtype)]
        if img_row.empty or cag_row.empty:
            continue
        delta = float(cag_row.iloc[0]["subtype_recall_at_10_mean"]) - float(img_row.iloc[0]["subtype_recall_at_10_mean"])
        rows.append((subtype, delta))
    return min(rows, key=lambda x: x[1]) if rows else ("none", float("nan"))


def _worst_negative_subtype(summary_df: pd.DataFrame, target: str, mode: str) -> Tuple[str, float]:
    rows = []
    for subtype in get_negative_subtypes(target):
        img_row = summary_df[(summary_df["target"] == target) & (summary_df["mode"] == mode) & (summary_df["backbone_name"] == "imagenet") & (summary_df["subtype"] == subtype)]
        cag_row = summary_df[(summary_df["target"] == target) & (summary_df["mode"] == mode) & (summary_df["backbone_name"] == "cag") & (summary_df["subtype"] == subtype)]
        if img_row.empty or cag_row.empty:
            continue
        delta = float(cag_row.iloc[0]["intrusion_at_10_mean"]) - float(img_row.iloc[0]["intrusion_at_10_mean"])
        rows.append((subtype, delta))
    return max(rows, key=lambda x: x[1]) if rows else ("none", float("nan"))


def write_markdown_summary(
    output_path: Path,
    positive_summary_df: pd.DataFrame,
    negative_summary_df: pd.DataFrame,
    run_meta: Dict[str, object],
) -> None:
    lines: List[str] = []
    lines.append("# Global Analysis 3: Hard Positive / Hard Negative Analysis")
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
    lines.append("")

    for target in ["patient", "study"]:
        lines.append(f"## {target.title()} Retrieval")
        lines.append("")
        lines.append("### Positive Subtypes")
        lines.append("")
        lines.append("| Mode | Backbone | Subtype | Median Best Rank | Best Score | R@1 | R@5 | R@10 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        pos_rows = positive_summary_df[positive_summary_df["target"] == target].sort_values(["mode", "backbone_name", "subtype"])
        for _, row in pos_rows.iterrows():
            lines.append(
                f"| {row['mode']} | {row['backbone_name']} | {row['subtype']} | "
                f"{format_float(float(row['best_positive_rank_median_mean']))} +/- {format_float(float(row['best_positive_rank_median_std']))} | "
                f"{format_float(float(row['best_positive_score_mean_mean']))} +/- {format_float(float(row['best_positive_score_mean_std']))} | "
                f"{format_float(float(row['subtype_recall_at_1_mean']))} +/- {format_float(float(row['subtype_recall_at_1_std']))} | "
                f"{format_float(float(row['subtype_recall_at_5_mean']))} +/- {format_float(float(row['subtype_recall_at_5_std']))} | "
                f"{format_float(float(row['subtype_recall_at_10_mean']))} +/- {format_float(float(row['subtype_recall_at_10_std']))} |"
            )
        lines.append("")
        lines.append("### Negative Subtypes")
        lines.append("")
        lines.append("| Mode | Backbone | Subtype | Median Hardest Rank | Hardest Score | Intr@1 | Intr@5 | Intr@10 | Pre-Positive Intrusion |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        neg_rows = negative_summary_df[negative_summary_df["target"] == target].sort_values(["mode", "backbone_name", "subtype"])
        for _, row in neg_rows.iterrows():
            lines.append(
                f"| {row['mode']} | {row['backbone_name']} | {row['subtype']} | "
                f"{format_float(float(row['hardest_negative_rank_median_mean']))} +/- {format_float(float(row['hardest_negative_rank_median_std']))} | "
                f"{format_float(float(row['hardest_negative_score_mean_mean']))} +/- {format_float(float(row['hardest_negative_score_mean_std']))} | "
                f"{format_float(float(row['intrusion_at_1_mean']))} +/- {format_float(float(row['intrusion_at_1_std']))} | "
                f"{format_float(float(row['intrusion_at_5_mean']))} +/- {format_float(float(row['intrusion_at_5_std']))} | "
                f"{format_float(float(row['intrusion_at_10_mean']))} +/- {format_float(float(row['intrusion_at_10_std']))} | "
                f"{format_float(float(row['pre_positive_intrusion_rate_mean']))} +/- {format_float(float(row['pre_positive_intrusion_rate_std']))} |"
            )
        lines.append("")
        lines.append("### Failure Localization")
        lines.append("")
        for mode in ["raw_frozen", "probe_linear"]:
            worst_pos, worst_pos_delta = _worst_positive_subtype(positive_summary_df, target, mode)
            worst_neg, worst_neg_delta = _worst_negative_subtype(negative_summary_df, target, mode)
            lines.append(
                f"- `{mode}` worst positive subtype by CAG-ImageNet `R@10` delta: "
                f"`{worst_pos}` ({format_float(worst_pos_delta)})"
            )
            lines.append(
                f"- `{mode}` worst negative subtype by CAG-ImageNet `Intr@10` delta: "
                f"`{worst_neg}` ({format_float(worst_neg_delta)})"
            )
        lines.append("")

    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `cross_study` positive weakness -> temporal / study-level invariance issue")
    lines.append("- `same_study_far_view` positive weakness -> cross-view invariance issue")
    lines.append("- `different_patient_near_view` negative weakness -> visually similar negative discrimination issue")
    lines.append("- `same_patient_cross_study` negative weakness in study retrieval -> study-boundary anchoring issue")
    lines.append("- subtype-localized gap absent -> move next to geometry / anchoring diagnosis")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_jobs(probe_seeds: Sequence[int]) -> List[JobDef]:
    jobs: List[JobDef] = []
    for backbone_name in ["imagenet", "cag"]:
        for target in ["patient", "study"]:
            jobs.append(JobDef("raw_frozen", backbone_name, target, None))
    for seed in probe_seeds:
        for backbone_name in ["imagenet", "cag"]:
            for target in ["patient", "study"]:
                jobs.append(JobDef("probe_linear", backbone_name, target, int(seed)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global2-root", default="outputs/global_2_study_patient_retrieval_unique_view")
    parser.add_argument("--image-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--dcm-root", default="input/stent_split_dcm_unique_view")
    parser.add_argument("--output-root", default="outputs/global_3_hard_positive_negative_analysis_unique_view")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--view-bins", type=float, nargs=2, default=[5.0, 20.0])
    parser.add_argument("--num-examples-per-subtype", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    ensure_dir(out_root)
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        out_root,
        args.log_file,
        default_prefix="global_3_hard_positive_negative_analysis",
    )

    try:
        set_global_seed(int(args.seed))
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        near_thresh = float(args.view_bins[0])
        mid_thresh = float(args.view_bins[1])
        if near_thresh <= 0.0 or mid_thresh <= near_thresh:
            raise ValueError("Expected positive ascending view bins: near < mid.")

        global2_root = Path(args.global2_root).resolve()
        image_root = Path(args.image_root).resolve()
        dcm_root = Path(args.dcm_root).resolve()

        full_manifest = build_angle_manifest(global2_root, image_root, dcm_root, out_root)
        manifest = maybe_subsample_manifest(full_manifest, args.max_images, int(args.seed))
        manifest.to_csv(out_root / "image_manifest_test_with_angles_active.csv", index=False)
        active_indices = manifest["image_id"].astype(int).to_numpy(dtype=np.int64)
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
        manifest_hash = hash_dataframe(
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

        device = resolve_device(args.device)
        jobs = build_jobs(args.probe_seeds)
        tracker = FullRunTracker(jobs)

        positive_pair_tables: List[pd.DataFrame] = []
        negative_pair_tables: List[pd.DataFrame] = []
        positive_raw_rows: List[pd.DataFrame] = []
        negative_raw_rows: List[pd.DataFrame] = []

        for job_index, job in enumerate(jobs):
            tracker.start_job(job_index)

            if job.mode == "raw_frozen":
                embeddings = l2_normalize_cpu(feature_store[job.backbone_name])[active_indices]
                tracker.update_phase("load_embeddings", 1, 4)
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
                tracker.update_phase("load_embeddings", 1, 4)

            sims = compute_similarity_matrix(embeddings)
            tracker.update_phase("build_similarity_matrix", 2, 4)

            positive_df, negative_df = compute_hard_pair_tables(
                sims=sims,
                manifest=manifest,
                mode=job.mode,
                backbone_name=job.backbone_name,
                target=job.target,
                seed=job.seed,
                near_thresh=near_thresh,
                mid_thresh=mid_thresh,
            )
            if positive_df.empty or negative_df.empty:
                raise RuntimeError(f"Empty hard-pair table for job: {job.display_name}")
            tracker.update_phase("compute_subtype_metrics", 3, 4)

            positive_pair_tables.append(positive_df)
            negative_pair_tables.append(negative_df)
            positive_raw_rows.append(summarize_positive_raw(positive_df))
            negative_raw_rows.append(summarize_negative_raw(negative_df))
            tracker.update_phase("collect_job_outputs", 4, 4)
            tracker.finish_job()

        all_positive_pairs = pd.concat(positive_pair_tables, axis=0, ignore_index=True)
        all_negative_pairs = pd.concat(negative_pair_tables, axis=0, ignore_index=True)
        positive_raw_summary = pd.concat(positive_raw_rows, axis=0, ignore_index=True)
        negative_raw_summary = pd.concat(negative_raw_rows, axis=0, ignore_index=True)

        positive_summary = aggregate_probe_and_raw(
            positive_raw_summary,
            [
                "best_positive_rank_median",
                "best_positive_rank_mean",
                "best_positive_score_mean",
                "subtype_recall_at_1",
                "subtype_recall_at_5",
                "subtype_recall_at_10",
            ],
        )
        negative_summary = aggregate_probe_and_raw(
            negative_raw_summary,
            [
                "hardest_negative_rank_median",
                "hardest_negative_rank_mean",
                "hardest_negative_score_mean",
                "intrusion_at_1",
                "intrusion_at_5",
                "intrusion_at_10",
                "pre_positive_intrusion_rate",
            ],
        )

        positive_raw_summary.to_csv(out_root / "summary_global_3_hard_positive_raw.csv", index=False)
        negative_raw_summary.to_csv(out_root / "summary_global_3_hard_negative_raw.csv", index=False)
        positive_summary.to_csv(out_root / "summary_global_3_hard_positive.csv", index=False)
        negative_summary.to_csv(out_root / "summary_global_3_hard_negative.csv", index=False)

        hard_positive_examples = select_examples(
            all_positive_pairs,
            harder_is_higher_rank=True,
            limit_per_group=int(args.num_examples_per_subtype),
        )
        hard_negative_examples = select_examples(
            all_negative_pairs,
            harder_is_higher_rank=False,
            limit_per_group=int(args.num_examples_per_subtype),
        )
        hard_positive_examples.to_csv(out_root / "hard_positive_examples.csv", index=False)
        hard_negative_examples.to_csv(out_root / "hard_negative_examples.csv", index=False)

        save_positive_rank_figure(positive_summary, out_root / "fig_global3_positive_subtype_rank.png")
        save_positive_recall_figure(positive_summary, out_root / "fig_global3_positive_subtype_recall_at_k.png")
        save_negative_intrusion_figure(negative_summary, out_root / "fig_global3_negative_subtype_intrusion.png")
        save_delta_figure(
            positive_summary_df=positive_summary,
            negative_summary_df=negative_summary,
            output_path=out_root / "fig_global3_cag_minus_imagenet_subtype_delta.png",
        )

        run_meta = {
            "global2_root": str(global2_root),
            "image_root": str(image_root),
            "dcm_root": str(dcm_root),
            "output_root": str(out_root.resolve()),
            "log_path": str(log_path.resolve()),
            "probe_seed_set": [int(v) for v in args.probe_seeds],
            "feature_source": "x_norm_clstoken",
            "headline_split": "test",
            "near_view_threshold_deg": near_thresh,
            "mid_view_threshold_deg": mid_thresh,
            "num_examples_per_subtype": int(args.num_examples_per_subtype),
            "full_manifest_hash": full_manifest_hash,
            "global2_test_manifest_hash": global2_test_manifest_hash,
            "active_manifest_hash": manifest_hash,
            "feature_hashes": feature_hashes,
            "job_count": len(jobs),
            "jobs": [
                {
                    "mode": job.mode,
                    "backbone_name": job.backbone_name,
                    "target": job.target,
                    "seed": job.seed if job.seed is not None else "raw",
                }
                for job in jobs
            ],
            "positive_raw_summary_path": str((out_root / "summary_global_3_hard_positive_raw.csv").resolve()),
            "negative_raw_summary_path": str((out_root / "summary_global_3_hard_negative_raw.csv").resolve()),
            "positive_summary_path": str((out_root / "summary_global_3_hard_positive.csv").resolve()),
            "negative_summary_path": str((out_root / "summary_global_3_hard_negative.csv").resolve()),
        }
        write_markdown_summary(
            output_path=out_root / "analysis_global_3_hard_positive_negative.md",
            positive_summary_df=positive_summary,
            negative_summary_df=negative_summary,
            run_meta=run_meta,
        )
        (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        log("Global Analysis 3 completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
