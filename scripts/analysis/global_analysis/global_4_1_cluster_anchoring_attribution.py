#!/usr/bin/env python3
"""
Global Analysis 4-1 — Cluster Anchoring Attribution Analysis.

This analysis reuses Global Analysis 2 probe checkpoints and test features to find
which nuisance modalities anchor the probe embedding space more strongly than
patient/study identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import cv2
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}
BACKBONES = ["imagenet", "cag"]
TARGETS = ["patient", "study"]
KNN_K_DEFAULT = [5, 10, 20]
KMEANS_K_DEFAULT = [8, 12, 16]
REFERENCE_FIELDS = ["patient_id", "study_id"]
DTYPE_ORDER = {"categorical": 0, "continuous": 1}
FIELD_NAME_RE = re.compile(r"[^0-9A-Za-z]+")
NUMERIC_VRS = {"DS", "FD", "FL", "IS", "SL", "SS", "SV", "UL", "US", "UV"}
DATE_VRS = {"DA"}
TIME_VRS = {"TM"}
DATETIME_VRS = {"DT"}
STRING_VRS = {
    "AE",
    "AS",
    "AT",
    "CS",
    "LO",
    "LT",
    "PN",
    "SH",
    "ST",
    "UC",
    "UI",
    "UR",
    "UT",
}
MIN_CATEGORICAL_COVERAGE = 0.50
MAX_CATEGORICAL_UNIQUE = 200
MAX_CATEGORICAL_DOMINANT_SHARE = 0.995
MIN_CONTINUOUS_COVERAGE = 0.50
MIN_CONTINUOUS_UNIQUE = 5
MAX_STRING_LENGTH = 160


@dataclass(frozen=True)
class JobDef:
    kind: str
    backbone_name: str | None = None
    target: str | None = None
    seed: int | None = None
    name: str | None = None

    @property
    def display_name(self) -> str:
        if self.kind == "postprocess":
            return self.name or "postprocess"
        return f"probe/{self.backbone_name}/{self.target}/seed{self.seed}"


@dataclass(frozen=True)
class FieldSpec:
    field_name: str
    field_type: str
    field_group: str
    source: str


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
            f"[ETA][JOB] phase={phase_name} | job={self.current_job_index + 1}/{len(self.jobs)} | "
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
                    "target": job.target or "-",
                    "seed": job.seed,
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


def restore_console_logging(file_handle: TextIO, original_stdout: TextIO, original_stderr: TextIO) -> None:
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
    for record in records:
        text = "\t".join(str(value) for value in record)
        hasher.update(text.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> str:
    records = df.loc[:, list(columns)].itertuples(index=False, name=None)
    return hash_records(records)


def make_feature_hash(features: torch.Tensor) -> str:
    array = features.detach().cpu().numpy().astype(np.float32, copy=False)
    return hashlib.sha256(array.tobytes()).hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


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
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return features.float().contiguous(), meta


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_arg)
    return torch.device("cpu")


def l2_normalize_cpu(features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features.float(), dim=1).cpu()


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


def quantize_angle(value: float | int | None, step: float = 10.0) -> str | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return f"{int(round(float(value) / step) * step):+d}"


def sanitize_field_name(name: str) -> str:
    text = FIELD_NAME_RE.sub("", str(name))
    if not text:
        text = "UnnamedField"
    if text[0].isdigit():
        text = f"Field{text}"
    return text


def register_field(field_specs: Dict[str, FieldSpec], field_name: str, field_type: str, field_group: str, source: str) -> None:
    spec = FieldSpec(field_name=field_name, field_type=field_type, field_group=field_group, source=source)
    existing = field_specs.get(field_name)
    if existing is None:
        field_specs[field_name] = spec
        return
    if existing != spec:
        raise ValueError(f"Field spec conflict for {field_name}: {existing} vs {spec}")


def clean_categorical(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) > MAX_STRING_LENGTH:
        text = text[:MAX_STRING_LENGTH]
    if text == "" or text.lower() == "none" or text.lower() == "nan":
        return None
    return text


def clean_numeric(value: object) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def maybe_iterable_value(value: object) -> List[object] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
        try:
            seq = list(value)
        except TypeError:
            return None
        return seq
    return None


def normalize_dicom_keyword(elem: pydicom.dataelem.DataElement) -> str:
    keyword = elem.keyword or sanitize_field_name(elem.name)
    return sanitize_field_name(keyword)


def parse_dicom_date(text: str) -> Tuple[int, int, int] | None:
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if len(digits) < 8:
        return None
    year = int(digits[:4])
    month = int(digits[4:6])
    day = int(digits[6:8])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year, month, day


def parse_dicom_time(text: str) -> Tuple[int, int, int] | None:
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if len(digits) < 2:
        return None
    hour = int(digits[:2])
    minute = int(digits[2:4]) if len(digits) >= 4 else 0
    second = int(digits[4:6]) if len(digits) >= 6 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return hour, minute, second


def add_temporal_derivatives(
    values: Dict[str, object],
    field_specs: Dict[str, FieldSpec],
    keyword: str,
    vr: str,
    raw_text: str,
) -> None:
    if vr in DATE_VRS or vr in DATETIME_VRS:
        parsed_date = parse_dicom_date(raw_text)
        if parsed_date is not None:
            year, month, day = parsed_date
            derived = {
                f"{keyword}Year": str(year),
                f"{keyword}Month": f"{month:02d}",
                f"{keyword}Day": f"{day:02d}",
                f"{keyword}YearMonth": f"{year:04d}-{month:02d}",
            }
            for name, val in derived.items():
                values[name] = val
                register_field(field_specs, name, "categorical", "derived", f"{keyword}_temporal")
    if vr in TIME_VRS or vr in DATETIME_VRS:
        parsed_time = parse_dicom_time(raw_text)
        if parsed_time is not None:
            hour, minute, _second = parsed_time
            minute10 = int(minute // 10) * 10
            derived = {
                f"{keyword}Hour": f"{hour:02d}",
                f"{keyword}Minute10Bin": f"{minute10:02d}",
            }
            for name, val in derived.items():
                values[name] = val
                register_field(field_specs, name, "categorical", "derived", f"{keyword}_temporal")


def add_numeric_multivalue_derivatives(
    values: Dict[str, object],
    field_specs: Dict[str, FieldSpec],
    keyword: str,
    numeric_values: Sequence[float],
) -> None:
    arr = np.asarray(list(numeric_values), dtype=np.float64)
    if arr.size == 0:
        return
    derived = {
        f"{keyword}MultiMean": float(arr.mean()),
        f"{keyword}MultiMin": float(arr.min()),
        f"{keyword}MultiMax": float(arr.max()),
        f"{keyword}MultiSpan": float(arr.max() - arr.min()),
        f"{keyword}MultiCount": float(arr.size),
    }
    for name, val in derived.items():
        values[name] = val
        register_field(field_specs, name, "continuous", "derived", f"{keyword}_multivalue")


def extract_dicom_scalar_fields(dcm_path: Path) -> Tuple[Dict[str, object], Dict[str, FieldSpec]]:
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    values: Dict[str, object] = {}
    field_specs: Dict[str, FieldSpec] = {}
    for elem in ds:
        if elem.VR == "SQ" or elem.tag.is_private:
            continue
        keyword = normalize_dicom_keyword(elem)
        if keyword in {"PixelData"}:
            continue
        raw_value = elem.value
        if isinstance(raw_value, (bytes, bytearray)):
            continue
        seq = maybe_iterable_value(raw_value)
        if seq is not None and len(seq) == 0:
            continue
        if seq is not None and not isinstance(raw_value, (str, bytes, bytearray)):
            numeric_values = [clean_numeric(v) for v in seq]
            if all(v is not None for v in numeric_values):
                joined = clean_categorical("\\".join(str(v) for v in seq))
                if joined is not None:
                    values[keyword] = joined
                    register_field(field_specs, keyword, "categorical", "metadata", "dicom_scalar")
                add_numeric_multivalue_derivatives(values, field_specs, keyword, [float(v) for v in numeric_values if v is not None])
                continue
            joined = clean_categorical("\\".join(str(v) for v in seq))
            if joined is not None:
                values[keyword] = joined
                register_field(field_specs, keyword, "categorical", "metadata", "dicom_scalar")
            continue

        if elem.VR in NUMERIC_VRS:
            numeric = clean_numeric(raw_value)
            if numeric is None:
                continue
            values[keyword] = numeric
            register_field(field_specs, keyword, "continuous", "metadata", "dicom_scalar")
            continue

        cat = clean_categorical(raw_value)
        if cat is None:
            continue
        values[keyword] = cat
        register_field(field_specs, keyword, "categorical", "metadata", "dicom_scalar")
        if elem.VR in DATE_VRS | TIME_VRS | DATETIME_VRS:
            add_temporal_derivatives(values, field_specs, keyword, elem.VR, cat)

    return values, field_specs


def compute_entropy_from_image(img: np.ndarray) -> float:
    hist = np.bincount(img.reshape(-1), minlength=256).astype(np.float64)
    prob = hist / max(1.0, hist.sum())
    prob = prob[prob > 0]
    if prob.size == 0:
        return float("nan")
    return float(-(prob * np.log2(prob)).sum())


def compute_image_attributes(img_path: Path) -> Dict[str, float]:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_f = img.astype(np.float32)
    lap = cv2.Laplacian(img_f, cv2.CV_32F)
    gx = cv2.Sobel(img_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    canny = cv2.Canny(img, 100, 200)
    mean = float(img_f.mean())
    std = float(img_f.std(ddof=0))
    p10 = float(np.percentile(img_f, 10.0))
    p90 = float(np.percentile(img_f, 90.0))
    p01 = float(np.percentile(img_f, 1.0))
    p05 = float(np.percentile(img_f, 5.0))
    p25 = float(np.percentile(img_f, 25.0))
    p50 = float(np.percentile(img_f, 50.0))
    p75 = float(np.percentile(img_f, 75.0))
    p95 = float(np.percentile(img_f, 95.0))
    p99 = float(np.percentile(img_f, 99.0))
    centered = img_f - mean
    if std > 0.0:
        skewness = float(np.mean((centered / std) ** 3))
        kurtosis = float(np.mean((centered / std) ** 4) - 3.0)
    else:
        skewness = 0.0
        kurtosis = 0.0
    return {
        "intensity_mean": mean,
        "intensity_std": std,
        "intensity_min": float(img_f.min()),
        "intensity_max": float(img_f.max()),
        "intensity_median": p50,
        "intensity_p01": p01,
        "intensity_p05": p05,
        "intensity_p10": p10,
        "intensity_p25": p25,
        "intensity_p90": p90,
        "intensity_p95": p95,
        "intensity_p99": p99,
        "intensity_p75": p75,
        "dynamic_range_p90_p10": float(p90 - p10),
        "dynamic_range_p95_p05": float(p95 - p05),
        "dynamic_range_p99_p01": float(p99 - p01),
        "dynamic_range_max_min": float(img_f.max() - img_f.min()),
        "entropy": compute_entropy_from_image(img),
        "laplacian_variance": float(lap.var()),
        "laplacian_abs_mean": float(np.abs(lap).mean()),
        "gradient_mean": float(grad_mag.mean()),
        "gradient_std": float(grad_mag.std(ddof=0)),
        "edge_density_canny": float((canny > 0).mean()),
        "intensity_skewness": skewness,
        "intensity_kurtosis": kurtosis,
    }


def build_anchor_manifest(
    global2_root: Path,
    image_root: Path,
    dcm_root: Path,
    output_root: Path,
    max_images: int | None,
    seed: int,
    field_catalog_name: str = "summary_global_4_1_field_catalog.csv",
    manifest_name: str = "test_manifest_with_anchor_features.csv",
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    base_manifest = load_csv(global2_root / "image_manifest_test.csv")
    full_manifest_hash = hash_dataframe(base_manifest, ["image_id", "img_path", "patient_id", "study_id"])
    manifest = maybe_subsample_manifest(base_manifest, max_images, seed)
    rows: List[Dict[str, object]] = []
    field_specs: Dict[str, FieldSpec] = {}
    for _, row in manifest.iterrows():
        img_path = Path(str(row["img_path"])).resolve()
        rel = img_path.relative_to(image_root.resolve())
        dcm_path = dcm_root.resolve() / rel.with_suffix(".dcm")
        if not dcm_path.exists():
            raise FileNotFoundError(f"Missing DICOM for image: {img_path}")
        meta, dicom_specs = extract_dicom_scalar_fields(dcm_path)
        for spec in dicom_specs.values():
            register_field(field_specs, spec.field_name, spec.field_type, spec.field_group, spec.source)
        image_stats = compute_image_attributes(img_path)
        for image_field in image_stats:
            register_field(field_specs, image_field, "continuous", "image", "image_stats")
        primary = clean_numeric(meta.get("PositionerPrimaryAngle"))
        secondary = clean_numeric(meta.get("PositionerSecondaryAngle"))
        derived_values = {
            "primary_angle_bin_10deg": quantize_angle(primary, 10.0),
            "secondary_angle_bin_10deg": quantize_angle(secondary, 10.0),
            "view_bin_2d_10deg": None
            if primary is None or secondary is None
            else f"({quantize_angle(primary, 10.0)},{quantize_angle(secondary, 10.0)})",
            "primary_angle_bin_20deg": quantize_angle(primary, 20.0),
            "secondary_angle_bin_20deg": quantize_angle(secondary, 20.0),
            "view_bin_2d_20deg": None
            if primary is None or secondary is None
            else f"({quantize_angle(primary, 20.0)},{quantize_angle(secondary, 20.0)})",
            "primary_angle_abs": None if primary is None else abs(float(primary)),
            "secondary_angle_abs": None if secondary is None else abs(float(secondary)),
        }
        for name in [
            "primary_angle_bin_10deg",
            "secondary_angle_bin_10deg",
            "view_bin_2d_10deg",
            "primary_angle_bin_20deg",
            "secondary_angle_bin_20deg",
            "view_bin_2d_20deg",
        ]:
            register_field(field_specs, name, "categorical", "derived", "angle_derived")
        for name in ["primary_angle_abs", "secondary_angle_abs"]:
            register_field(field_specs, name, "continuous", "derived", "angle_derived")
        combined = {
            "image_id": int(row["image_id"]),
            "img_path": str(img_path),
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "class_name": str(row.get("class_name", "")),
            **meta,
            **derived_values,
            **image_stats,
        }
        rows.append(combined)
    out_df = pd.DataFrame(rows)
    if out_df["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image_id detected in anchor manifest.")
    out_df = out_df.sort_values("image_id").reset_index(drop=True)
    out_df.to_csv(output_root / manifest_name, index=False)
    field_specs_df = pd.DataFrame([spec.__dict__ for spec in field_specs.values()]).sort_values(
        ["field_group", "field_type", "field_name"]
    ).reset_index(drop=True)
    field_specs_df.to_csv(output_root / field_catalog_name, index=False)
    return out_df, full_manifest_hash, field_specs_df


def audit_fields(manifest: pd.DataFrame, field_specs_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    rows: List[Dict[str, object]] = []
    usable_categorical: List[str] = []
    usable_continuous: List[str] = []
    n = len(manifest)
    for spec in field_specs_df.itertuples():
        field_name = str(spec.field_name)
        source = str(spec.field_group)
        field_type = str(spec.field_type)
        if field_name not in manifest.columns:
            continue
        if field_name in REFERENCE_FIELDS:
            continue
        if field_type == "categorical":
            series = manifest[field_name].astype(object)
            valid = series.notna() & (series.astype(str) != "")
            coverage = float(valid.mean()) if n else float("nan")
            valid_values = series[valid].astype(str)
            unique_count = int(valid_values.nunique()) if valid.any() else 0
            dominant_share = float(valid_values.value_counts(normalize=True).iloc[0]) if unique_count else float("nan")
            usable = (
                coverage >= MIN_CATEGORICAL_COVERAGE
                and 2 <= unique_count <= MAX_CATEGORICAL_UNIQUE
                and (not math.isfinite(dominant_share) or dominant_share < MAX_CATEGORICAL_DOMINANT_SHARE)
            )
            if usable:
                usable_categorical.append(field_name)
            rows.append(
                {
                    "field_name": field_name,
                    "field_type": "categorical",
                    "field_group": source,
                    "source": str(spec.source),
                    "coverage": coverage,
                    "unique_count": unique_count,
                    "dominant_share": dominant_share,
                    "std": "",
                    "usable": int(usable),
                    "drop_reason": "" if usable else _categorical_drop_reason(coverage, unique_count, dominant_share),
                }
            )
            continue

        values = pd.to_numeric(manifest[field_name], errors="coerce")
        valid = values.notna()
        coverage = float(valid.mean()) if n else float("nan")
        unique_count = int(values[valid].nunique()) if valid.any() else 0
        std = float(values[valid].std(ddof=0)) if valid.any() else float("nan")
        usable = coverage >= MIN_CONTINUOUS_COVERAGE and unique_count >= MIN_CONTINUOUS_UNIQUE and math.isfinite(std) and std > 0.0
        if usable:
            usable_continuous.append(field_name)
        rows.append(
            {
                "field_name": field_name,
                "field_type": "continuous",
                "field_group": source,
                "source": str(spec.source),
                "coverage": coverage,
                "unique_count": unique_count,
                "dominant_share": "",
                "std": std,
                "usable": int(usable),
                "drop_reason": "" if usable else _continuous_drop_reason(coverage, unique_count, std),
            }
        )
    audit_df = pd.DataFrame(rows).sort_values(["field_type", "field_group", "field_name"]).reset_index(drop=True)
    return audit_df, usable_categorical, usable_continuous


def _categorical_drop_reason(coverage: float, unique_count: int, dominant_share: float) -> str:
    reasons: List[str] = []
    if not math.isfinite(coverage) or coverage < MIN_CATEGORICAL_COVERAGE:
        reasons.append(f"coverage<{MIN_CATEGORICAL_COVERAGE:.2f}")
    if unique_count < 2:
        reasons.append("unique<2")
    if unique_count > MAX_CATEGORICAL_UNIQUE:
        reasons.append(f"unique>{MAX_CATEGORICAL_UNIQUE}")
    if math.isfinite(dominant_share) and dominant_share >= MAX_CATEGORICAL_DOMINANT_SHARE:
        reasons.append(f"dominant>={MAX_CATEGORICAL_DOMINANT_SHARE:.3f}")
    return ",".join(reasons) if reasons else "unknown"


def _continuous_drop_reason(coverage: float, unique_count: int, std: float) -> str:
    reasons: List[str] = []
    if not math.isfinite(coverage) or coverage < MIN_CONTINUOUS_COVERAGE:
        reasons.append(f"coverage<{MIN_CONTINUOUS_COVERAGE:.2f}")
    if unique_count < MIN_CONTINUOUS_UNIQUE:
        reasons.append(f"unique<{MIN_CONTINUOUS_UNIQUE}")
    if not math.isfinite(std) or std <= 0.0:
        reasons.append("std<=0")
    return ",".join(reasons) if reasons else "unknown"


def compute_similarity_matrix(embeddings: torch.Tensor) -> np.ndarray:
    emb = l2_normalize_cpu(embeddings)
    return torch.matmul(emb, emb.T).numpy().astype(np.float32)


def build_topk_neighbors(sims: np.ndarray, max_k: int) -> np.ndarray:
    score = sims.copy()
    np.fill_diagonal(score, -np.inf)
    sorted_idx = np.argsort(-score, axis=1)
    return sorted_idx[:, :max_k]


def base_match_rate_from_labels(labels: np.ndarray, valid_mask: np.ndarray) -> float:
    valid_labels = labels[valid_mask]
    n = int(valid_labels.shape[0])
    if n < 2:
        return float("nan")
    _, counts = np.unique(valid_labels, return_counts=True)
    numerator = float(np.sum(counts * np.maximum(counts - 1, 0)))
    denominator = float(n * max(1, n - 1))
    return numerator / denominator if denominator > 0 else float("nan")


def compute_categorical_neighborhood_rows(
    topk_neighbors: np.ndarray,
    manifest: pd.DataFrame,
    field_names: Sequence[str],
    backbone_name: str,
    target: str,
    seed: int,
    k_values: Sequence[int],
    field_group_map: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for field_name in list(field_names) + REFERENCE_FIELDS:
        labels = manifest[field_name].astype(str).to_numpy(dtype=str)
        valid = manifest[field_name].notna().to_numpy(dtype=bool)
        base_match_rate = base_match_rate_from_labels(labels, valid)
        for k in k_values:
            per_query_rates: List[float] = []
            for idx in range(len(manifest)):
                if not valid[idx]:
                    continue
                nbr_idx = topk_neighbors[idx, :k]
                nbr_valid = valid[nbr_idx]
                if not nbr_valid.any():
                    continue
                matches = (labels[nbr_idx] == labels[idx]) & nbr_valid
                per_query_rates.append(float(matches[nbr_valid].mean()))
            neighbor_match_rate = float(np.mean(per_query_rates)) if per_query_rates else float("nan")
            purity_uplift = (
                float(neighbor_match_rate / base_match_rate)
                if math.isfinite(neighbor_match_rate) and math.isfinite(base_match_rate) and base_match_rate > 0.0
                else float("nan")
            )
            rows.append(
                {
                    "backbone_name": backbone_name,
                    "target": target,
                    "seed": int(seed),
                    "field_name": field_name,
                    "field_type": "categorical",
                    "field_group": field_group_map[field_name],
                    "k": int(k),
                    "num_valid_queries": int(len(per_query_rates)),
                    "base_match_rate": base_match_rate,
                    "neighbor_match_rate": neighbor_match_rate,
                    "purity_uplift": purity_uplift,
                }
            )
    return pd.DataFrame(rows)


def compute_continuous_neighborhood_rows(
    topk_neighbors: np.ndarray,
    manifest: pd.DataFrame,
    field_names: Sequence[str],
    backbone_name: str,
    target: str,
    seed: int,
    k_values: Sequence[int],
    field_group_map: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for field_name in field_names:
        values = pd.to_numeric(manifest[field_name], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(values)
        valid_values = values[valid]
        if valid_values.size < 2:
            global_abs_diff_mean = float("nan")
        else:
            diff = np.abs(valid_values[:, None] - valid_values[None, :])
            upper_i, upper_j = np.triu_indices(diff.shape[0], k=1)
            global_abs_diff_mean = float(diff[upper_i, upper_j].mean()) if upper_i.size else float("nan")
        for k in k_values:
            per_query_diffs: List[float] = []
            for idx in range(len(manifest)):
                if not valid[idx]:
                    continue
                nbr_idx = topk_neighbors[idx, :k]
                nbr_valid = valid[nbr_idx]
                if not nbr_valid.any():
                    continue
                diffs = np.abs(values[nbr_idx[nbr_valid]] - values[idx])
                per_query_diffs.append(float(diffs.mean()))
            neighbor_abs_diff_mean = float(np.mean(per_query_diffs)) if per_query_diffs else float("nan")
            neighbor_consistency = (
                float(1.0 - (neighbor_abs_diff_mean / global_abs_diff_mean))
                if math.isfinite(global_abs_diff_mean) and global_abs_diff_mean > 0.0 and math.isfinite(neighbor_abs_diff_mean)
                else float("nan")
            )
            rows.append(
                {
                    "backbone_name": backbone_name,
                    "target": target,
                    "seed": int(seed),
                    "field_name": field_name,
                    "field_type": "continuous",
                    "field_group": field_group_map[field_name],
                    "k": int(k),
                    "num_valid_queries": int(len(per_query_diffs)),
                    "global_abs_diff_mean": global_abs_diff_mean,
                    "neighbor_abs_diff_mean": neighbor_abs_diff_mean,
                    "neighbor_consistency": neighbor_consistency,
                }
            )
    return pd.DataFrame(rows)


def eta_squared_by_cluster(values: np.ndarray, cluster_labels: np.ndarray) -> float:
    valid = np.isfinite(values)
    values = values[valid]
    cluster_labels = cluster_labels[valid]
    if values.size < 2 or np.unique(cluster_labels).size < 2:
        return float("nan")
    grand_mean = float(values.mean())
    ss_total = float(((values - grand_mean) ** 2).sum())
    if ss_total <= 0.0:
        return float("nan")
    ss_between = 0.0
    for cluster_id in np.unique(cluster_labels):
        cluster_values = values[cluster_labels == cluster_id]
        if cluster_values.size == 0:
            continue
        ss_between += float(cluster_values.size) * float((cluster_values.mean() - grand_mean) ** 2)
    return float(ss_between / ss_total)


def compute_categorical_cluster_rows(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    field_names: Sequence[str],
    backbone_name: str,
    target: str,
    seed: int,
    cluster_k_values: Sequence[int],
    field_group_map: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for kmeans_k in cluster_k_values:
        if len(embeddings) < kmeans_k:
            continue
        labels = KMeans(n_clusters=int(kmeans_k), n_init=20, random_state=int(seed)).fit_predict(embeddings)
        for field_name in list(field_names) + REFERENCE_FIELDS:
            field_values = manifest[field_name].astype(str)
            valid = manifest[field_name].notna().to_numpy(dtype=bool)
            valid_labels = labels[valid]
            valid_values = field_values[valid].to_numpy(dtype=str)
            if valid_values.size < 2 or np.unique(valid_values).size < 2 or np.unique(valid_labels).size < 2:
                nmi = float("nan")
            else:
                nmi = float(normalized_mutual_info_score(valid_labels, valid_values))
            rows.append(
                {
                    "backbone_name": backbone_name,
                    "target": target,
                    "seed": int(seed),
                    "field_name": field_name,
                    "field_type": "categorical",
                    "field_group": field_group_map[field_name],
                    "kmeans_k": int(kmeans_k),
                    "num_valid_samples": int(valid.sum()),
                    "nmi": nmi,
                }
            )
    return pd.DataFrame(rows)


def compute_continuous_cluster_rows(
    embeddings: np.ndarray,
    manifest: pd.DataFrame,
    field_names: Sequence[str],
    backbone_name: str,
    target: str,
    seed: int,
    cluster_k_values: Sequence[int],
    field_group_map: Dict[str, str],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for kmeans_k in cluster_k_values:
        if len(embeddings) < kmeans_k:
            continue
        labels = KMeans(n_clusters=int(kmeans_k), n_init=20, random_state=int(seed)).fit_predict(embeddings)
        for field_name in field_names:
            values = pd.to_numeric(manifest[field_name], errors="coerce").to_numpy(dtype=np.float64)
            eta_sq = eta_squared_by_cluster(values, labels)
            rows.append(
                {
                    "backbone_name": backbone_name,
                    "target": target,
                    "seed": int(seed),
                    "field_name": field_name,
                    "field_type": "continuous",
                    "field_group": field_group_map[field_name],
                    "kmeans_k": int(kmeans_k),
                    "num_valid_samples": int(np.isfinite(values).sum()),
                    "eta_squared": eta_sq,
                }
            )
    return pd.DataFrame(rows)


def zscore_series(values: pd.Series) -> pd.Series:
    arr = values.to_numpy(dtype=np.float64)
    finite = np.isfinite(arr)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    if finite.sum() == 0:
        return pd.Series(out, index=values.index)
    mean = float(arr[finite].mean())
    std = float(arr[finite].std(ddof=0))
    if std <= 0.0 or not math.isfinite(std):
        out[finite] = 0.0
    else:
        out[finite] = (arr[finite] - mean) / std
    return pd.Series(out, index=values.index)


def compute_combined_anchor_rows(
    neighborhood_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    field_type: str,
    backbone_name: str,
    target: str,
    seed: int,
) -> pd.DataFrame:
    if field_type == "categorical":
        local_df = (
            neighborhood_df.groupby(["field_name", "field_group"], as_index=False)
            .agg(local_metric=("purity_uplift", "mean"))
        )
        global_df = (
            cluster_df.groupby(["field_name", "field_group"], as_index=False)
            .agg(global_metric=("nmi", "mean"))
        )
    elif field_type == "continuous":
        local_df = (
            neighborhood_df.groupby(["field_name", "field_group"], as_index=False)
            .agg(local_metric=("neighbor_consistency", "mean"))
        )
        global_df = (
            cluster_df.groupby(["field_name", "field_group"], as_index=False)
            .agg(global_metric=("eta_squared", "mean"))
        )
    else:
        raise ValueError(field_type)

    merged = local_df.merge(global_df, on=["field_name", "field_group"], how="outer")
    merged["local_z"] = zscore_series(merged["local_metric"])
    merged["global_z"] = zscore_series(merged["global_metric"])
    merged["combined_anchor_score"] = merged["local_z"] + merged["global_z"]
    nuisance_mask = merged["field_group"] != "reference"
    nuisance = merged[nuisance_mask].sort_values("combined_anchor_score", ascending=False).reset_index(drop=True)
    rank_map = {field: rank + 1 for rank, field in enumerate(nuisance["field_name"].tolist())}
    merged["rank_within_type"] = merged["field_name"].map(rank_map)
    merged["backbone_name"] = backbone_name
    merged["target"] = target
    merged["seed"] = int(seed)
    merged["field_type"] = field_type
    return merged[
        [
            "backbone_name",
            "target",
            "seed",
            "field_name",
            "field_type",
            "field_group",
            "local_metric",
            "global_metric",
            "local_z",
            "global_z",
            "combined_anchor_score",
            "rank_within_type",
        ]
    ].sort_values(["field_group", "rank_within_type", "field_name"], na_position="last")


def aggregate_rows(raw_df: pd.DataFrame, group_cols: Sequence[str], metric_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in raw_df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row["num_seeds"] = int(pd.to_numeric(group["seed"], errors="coerce").nunique()) if "seed" in group.columns else 0
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean()) if values.notna().any() else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    if not rows:
        columns = list(group_cols) + ["num_seeds"]
        for metric in metric_cols:
            columns.extend([f"{metric}_mean", f"{metric}_std"])
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)


def assign_anchor_ranks(anchor_df: pd.DataFrame) -> pd.DataFrame:
    if anchor_df.empty:
        out = anchor_df.copy()
        out["rank_within_type"] = np.nan
        return out
    out_frames: list[pd.DataFrame] = []
    for (backbone_name, target, field_type), group in anchor_df.groupby(["backbone_name", "target", "field_type"], sort=False):
        ranked = group.copy()
        ranked["rank_within_type"] = np.nan
        nuisance_mask = ranked["field_group"] != "reference"
        nuisance = ranked[nuisance_mask].sort_values(["combined_anchor_score_mean", "field_name"], ascending=[False, True]).reset_index(drop=True)
        rank_map = {field: rank + 1 for rank, field in enumerate(nuisance["field_name"].tolist())}
        ranked.loc[nuisance_mask, "rank_within_type"] = ranked.loc[nuisance_mask, "field_name"].map(rank_map)
        out_frames.append(ranked)
    return pd.concat(out_frames, ignore_index=True)


def save_field_audit_figure(audit_df: pd.DataFrame, output_path: Path) -> None:
    max_rows = max(
        int((audit_df["field_type"] == "categorical").sum()),
        int((audit_df["field_type"] == "continuous").sum()),
        1,
    )
    fig_height = max(6.0, min(0.25 * max_rows + 2.0, 30.0))
    fig, axes = plt.subplots(1, 2, figsize=(16, fig_height))
    for ax, field_type in zip(axes, ["categorical", "continuous"]):
        df = audit_df[audit_df["field_type"] == field_type].copy().sort_values(["usable", "coverage", "field_name"], ascending=[False, False, True])
        y = np.arange(len(df))
        ax.barh(y, df["coverage"].astype(float).to_numpy(dtype=np.float64), color=np.where(df["usable"].astype(int).to_numpy(dtype=np.int32) > 0, "#4C72B0", "#C44E52"))
        ax.set_yticks(y, df["field_name"].tolist())
        ax.set_xlim(0, 1.02)
        ax.set_title(f"{field_type.title()} field coverage")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _heatmap_color(
    ax: plt.Axes,
    values: np.ndarray,
    y_labels: Sequence[str],
    title: str,
    cmap: str,
    tick_fontsize_override: int | None = None,
    value_fontsize_override: int | None = None,
) -> matplotlib.image.AxesImage:
    if values.size == 0:
        values = np.zeros((1, 1), dtype=np.float64)
        y_labels = ["(none)"]
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    label_count = len(y_labels)
    if tick_fontsize_override is not None:
        tick_fontsize = tick_fontsize_override
    elif label_count <= 15:
        tick_fontsize = 8
    elif label_count <= 30:
        tick_fontsize = 7
    elif label_count <= 50:
        tick_fontsize = 6
    else:
        tick_fontsize = 5
    ax.set_yticks(np.arange(len(y_labels)), y_labels, fontsize=tick_fontsize)
    ax.set_xticks([0], ["score"])
    ax.set_title(title)
    for r in range(values.shape[0]):
        val = values[r, 0]
        rgba = im.cmap(im.norm(val))
        luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        color = "white" if luminance < 0.45 else "black"
        if value_fontsize_override is not None:
            value_fontsize = value_fontsize_override
        else:
            value_fontsize = 8 if label_count <= 20 else 7 if label_count <= 40 else 6
        ax.text(0, r, f"{val:.3f}", ha="center", va="center", color=color, fontsize=value_fontsize)
    return im


def save_anchor_rank_figure(
    anchor_df: pd.DataFrame,
    field_type: str,
    output_path: Path,
    analysis_title: str,
    top_n: int | None = 10,
) -> None:
    max_rows = 1
    for target in TARGETS:
        for backbone_name in BACKBONES:
            subset = anchor_df[
                (anchor_df["field_type"] == field_type)
                & (anchor_df["field_group"] != "reference")
                & (anchor_df["target"] == target)
                & (anchor_df["backbone_name"] == backbone_name)
            ].sort_values("combined_anchor_score_mean", ascending=False)
            if top_n is not None:
                subset = subset.head(top_n)
            max_rows = max(max_rows, len(subset))
    fig_height = max(10.0, min(28.0, 4.0 + 0.23 * max_rows))
    fig = plt.figure(figsize=(15, fig_height))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.42, hspace=0.28)
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )
    cax = fig.add_subplot(gs[:, 2])
    im = None
    for row_idx, target in enumerate(TARGETS):
        for col_idx, backbone_name in enumerate(BACKBONES):
            ax = axes[row_idx, col_idx]
            subset = anchor_df[
                (anchor_df["field_type"] == field_type)
                & (anchor_df["field_group"] != "reference")
                & (anchor_df["target"] == target)
                & (anchor_df["backbone_name"] == backbone_name)
            ].sort_values("combined_anchor_score_mean", ascending=False)
            if top_n is not None:
                subset = subset.head(top_n)
            values = subset[["combined_anchor_score_mean"]].to_numpy(dtype=np.float64)
            labels = subset["field_name"].tolist()
            im = _heatmap_color(ax, values, labels, f"{backbone_name} | {target}", "magma")
    if im is not None:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Combined anchor score")
    else:
        cax.axis("off")
    title_prefix = "Top" if top_n is not None else "All"
    fig.suptitle(f"{analysis_title}: {title_prefix} {field_type.title()} Anchors", y=0.995)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_anchor_rank_target_figure(
    anchor_df: pd.DataFrame,
    field_type: str,
    target: str,
    output_path: Path,
    analysis_title: str,
    top_n: int | None = None,
) -> None:
    subsets: list[pd.DataFrame] = []
    max_rows = 1
    for backbone_name in BACKBONES:
        subset = anchor_df[
            (anchor_df["field_type"] == field_type)
            & (anchor_df["field_group"] != "reference")
            & (anchor_df["target"] == target)
            & (anchor_df["backbone_name"] == backbone_name)
        ].sort_values("combined_anchor_score_mean", ascending=False)
        if top_n is not None:
            subset = subset.head(top_n)
        subsets.append(subset)
        max_rows = max(max_rows, len(subset))
    fig_height = max(12.0, min(32.0, 5.0 + 0.28 * max_rows))
    fig = plt.figure(figsize=(16, fig_height))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.055], wspace=0.40)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])
    im = None
    tick_fontsize = 10 if max_rows <= 35 else 9 if max_rows <= 55 else 8 if max_rows <= 75 else 7
    value_fontsize = 9 if max_rows <= 45 else 8
    for ax, backbone_name, subset in zip(axes, BACKBONES, subsets):
        values = subset[["combined_anchor_score_mean"]].to_numpy(dtype=np.float64)
        labels = subset["field_name"].tolist()
        im = _heatmap_color(
            ax,
            values,
            labels,
            f"{backbone_name} | {target}",
            "magma",
            tick_fontsize_override=tick_fontsize,
            value_fontsize_override=value_fontsize,
        )
    if im is not None:
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Combined anchor score")
    else:
        cax.axis("off")
    title_prefix = "Top" if top_n is not None else "All"
    fig.suptitle(f"{analysis_title}: {title_prefix} {field_type.title()} Anchors | {target.title()} Head", y=0.995)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _build_umap_coords(embeddings: np.ndarray, random_state: int) -> np.ndarray:
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(5, min(15, len(embeddings) - 1)),
        min_dist=0.2,
        metric="cosine",
        random_state=random_state,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden.*", category=UserWarning)
        return reducer.fit_transform(embeddings)


def save_umap_overlay_top_categorical(
    aggregate_anchor_df: pd.DataFrame,
    embedding_map: Dict[Tuple[str, str], np.ndarray],
    manifest: pd.DataFrame,
    output_path: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for row_idx, target in enumerate(TARGETS):
        for col_idx, backbone_name in enumerate(BACKBONES):
            ax = axes[row_idx, col_idx]
            subset = aggregate_anchor_df[
                (aggregate_anchor_df["field_type"] == "categorical")
                & (aggregate_anchor_df["field_group"] != "reference")
                & (aggregate_anchor_df["target"] == target)
                & (aggregate_anchor_df["backbone_name"] == backbone_name)
            ].sort_values("combined_anchor_score_mean", ascending=False)
            if subset.empty:
                ax.set_title(f"{backbone_name} | {target} | no categorical anchor")
                ax.axis("off")
                continue
            field_name = str(subset.iloc[0]["field_name"])
            coords = _build_umap_coords(embedding_map[(backbone_name, target)], random_state=seed)
            series = manifest[field_name].astype(str).fillna("<missing>")
            top_values = series.value_counts().head(8).index.tolist()
            color_values = series.where(series.isin(top_values), other="Other")
            palette = plt.get_cmap("tab10")
            unique_values = list(dict.fromkeys(top_values + (["Other"] if (color_values == "Other").any() else [])))
            for idx_value, value in enumerate(unique_values):
                mask = color_values.to_numpy(dtype=str) == value
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=18,
                    color=palette(idx_value % palette.N) if value != "Other" else "#D0D0D0",
                    alpha=0.75 if value != "Other" else 0.4,
                    label=value,
                    linewidths=0,
                )
            ax.set_title(f"{backbone_name} | {target} | {field_name}")
            ax.grid(alpha=0.18)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_umap_overlay_top_continuous(
    aggregate_anchor_df: pd.DataFrame,
    embedding_map: Dict[Tuple[str, str], np.ndarray],
    manifest: pd.DataFrame,
    output_path: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for row_idx, target in enumerate(TARGETS):
        for col_idx, backbone_name in enumerate(BACKBONES):
            ax = axes[row_idx, col_idx]
            subset = aggregate_anchor_df[
                (aggregate_anchor_df["field_type"] == "continuous")
                & (aggregate_anchor_df["target"] == target)
                & (aggregate_anchor_df["backbone_name"] == backbone_name)
            ].sort_values("combined_anchor_score_mean", ascending=False)
            if subset.empty:
                ax.set_title(f"{backbone_name} | {target} | no continuous anchor")
                ax.axis("off")
                continue
            field_name = str(subset.iloc[0]["field_name"])
            coords = _build_umap_coords(embedding_map[(backbone_name, target)], random_state=seed)
            values = pd.to_numeric(manifest[field_name], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(values)
            sc = ax.scatter(coords[~valid, 0], coords[~valid, 1], s=14, c="#D0D0D0", alpha=0.25, linewidths=0)
            sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=16, c=values[valid], cmap="viridis", alpha=0.8, linewidths=0)
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{backbone_name} | {target} | {field_name}")
            ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _safe_field_filename(field_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(field_name)).strip("._")
    return safe or "field"


def save_umap_overlay_field_categorical(
    embedding: np.ndarray,
    manifest: pd.DataFrame,
    field_name: str,
    output_path: Path,
    seed: int,
) -> None:
    coords = _build_umap_coords(embedding, random_state=seed)
    series = manifest[field_name].astype(str).fillna("<missing>")
    top_values = series.value_counts().head(8).index.tolist()
    color_values = series.where(series.isin(top_values), other="Other")
    palette = plt.get_cmap("tab10")
    unique_values = list(dict.fromkeys(top_values + (["Other"] if (color_values == "Other").any() else [])))
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.6))
    for idx_value, value in enumerate(unique_values):
        mask = color_values.to_numpy(dtype=str) == value
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            color=palette(idx_value % palette.N) if value != "Other" else "#D0D0D0",
            alpha=0.78 if value != "Other" else 0.35,
            label=value,
            linewidths=0,
        )
    ax.set_title(field_name)
    ax.grid(alpha=0.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_umap_overlay_field_continuous(
    embedding: np.ndarray,
    manifest: pd.DataFrame,
    field_name: str,
    output_path: Path,
    seed: int,
) -> None:
    coords = _build_umap_coords(embedding, random_state=seed)
    values = pd.to_numeric(manifest[field_name], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(values)
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 6.4))
    ax.scatter(coords[~valid, 0], coords[~valid, 1], s=14, c="#D0D0D0", alpha=0.22, linewidths=0)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1], s=16, c=values[valid], cmap="viridis", alpha=0.82, linewidths=0)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(field_name)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_umap_overlays_for_nonnegative_fields(
    aggregate_anchor_df: pd.DataFrame,
    embedding_map: Dict[Tuple[str, str], np.ndarray],
    manifest: pd.DataFrame,
    output_root: Path,
    seed: int,
) -> pd.DataFrame:
    ensure_dir(output_root)
    rows: list[dict[str, object]] = []
    for field_type in ["categorical", "continuous"]:
        for target in TARGETS:
            for backbone_name in BACKBONES:
                subset = aggregate_anchor_df[
                    (aggregate_anchor_df["field_type"] == field_type)
                    & (aggregate_anchor_df["target"] == target)
                    & (aggregate_anchor_df["backbone_name"] == backbone_name)
                    & (aggregate_anchor_df["combined_anchor_score_mean"] >= 0)
                ].sort_values(["combined_anchor_score_mean", "field_name"], ascending=[False, True])
                if field_type == "categorical":
                    subset = subset[subset["field_group"] != "reference"]
                export_dir = output_root / field_type / target / backbone_name
                ensure_dir(export_dir)
                embedding = embedding_map[(backbone_name, target)]
                for row in subset.itertuples():
                    filename = f"{int(row.rank_within_type):03d}_{_safe_field_filename(str(row.field_name))}.png" if pd.notna(row.rank_within_type) else f"{_safe_field_filename(str(row.field_name))}.png"
                    output_path = export_dir / filename
                    if field_type == "categorical":
                        save_umap_overlay_field_categorical(embedding, manifest, str(row.field_name), output_path, seed=seed)
                    else:
                        save_umap_overlay_field_continuous(embedding, manifest, str(row.field_name), output_path, seed=seed)
                    rows.append(
                        {
                            "field_type": field_type,
                            "target": target,
                            "backbone_name": backbone_name,
                            "field_name": str(row.field_name),
                            "field_group": str(row.field_group),
                            "rank_within_type": float(row.rank_within_type) if pd.notna(row.rank_within_type) else np.nan,
                            "combined_anchor_score_mean": float(row.combined_anchor_score_mean),
                            "combined_anchor_score_std": float(row.combined_anchor_score_std),
                            "output_path": str(output_path.resolve()),
                        }
                    )
    export_df = pd.DataFrame(rows)
    export_df.to_csv(output_root / "index_score_ge_0.csv", index=False)
    return export_df


def save_patient_vs_nuisance_compare(
    anchor_df: pd.DataFrame,
    output_path: Path,
    analysis_title: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for row_idx, target in enumerate(TARGETS):
        for col_idx, backbone_name in enumerate(BACKBONES):
            ax = axes[row_idx, col_idx]
            subset = anchor_df[(anchor_df["target"] == target) & (anchor_df["backbone_name"] == backbone_name)]
            patient_row = subset[subset["field_name"] == "patient_id"].iloc[0]
            study_row = subset[subset["field_name"] == "study_id"].iloc[0]
            top_cat = subset[(subset["field_type"] == "categorical") & (subset["field_group"] != "reference")].sort_values("combined_anchor_score_mean", ascending=False).iloc[0]
            top_cont = subset[subset["field_type"] == "continuous"].sort_values("combined_anchor_score_mean", ascending=False).iloc[0]
            labels = ["patient_id", "study_id", str(top_cat["field_name"]), str(top_cont["field_name"])]
            values = [
                float(patient_row["combined_anchor_score_mean"]),
                float(study_row["combined_anchor_score_mean"]),
                float(top_cat["combined_anchor_score_mean"]),
                float(top_cont["combined_anchor_score_mean"]),
            ]
            colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B2"]
            ax.barh(np.arange(len(labels)), values, color=colors)
            ax.set_yticks(np.arange(len(labels)), labels)
            ax.invert_yaxis()
            ax.set_title(f"{backbone_name} | {target}")
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle(f"{analysis_title}: Patient/Study identity vs top nuisance anchors", y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _cluster_purity(field_values: np.ndarray, cluster_labels: np.ndarray, cluster_id: int) -> Tuple[str, float, np.ndarray]:
    mask = cluster_labels == cluster_id
    values = field_values[mask]
    unique, counts = np.unique(values, return_counts=True)
    if unique.size == 0:
        return "", float("nan"), mask
    best_idx = int(np.argmax(counts))
    purity = float(counts[best_idx] / max(1, counts.sum()))
    return str(unique[best_idx]), purity, mask


def save_anchor_examples(
    anchor_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    embedding_map_seed11: Dict[Tuple[str, str], np.ndarray],
    manifest: pd.DataFrame,
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    top_fields = (
        anchor_df[(anchor_df["field_type"] == "categorical") & (anchor_df["field_group"] != "reference")]
        .groupby("field_name", as_index=False)
        .agg(best_score=("combined_anchor_score_mean", "max"))
        .sort_values("best_score", ascending=False)
        .head(2)
    )
    summary_rows: List[Dict[str, object]] = []
    for _, field_row in top_fields.iterrows():
        field_name = str(field_row["field_name"])
        best_condition = anchor_df[
            (anchor_df["field_name"] == field_name)
            & (anchor_df["field_type"] == "categorical")
            & (anchor_df["field_group"] != "reference")
        ].sort_values("combined_anchor_score_mean", ascending=False).iloc[0]
        backbone_name = str(best_condition["backbone_name"])
        target = str(best_condition["target"])
        k_choice = (
            cluster_df[
                (cluster_df["field_name"] == field_name)
                & (cluster_df["backbone_name"] == backbone_name)
                & (cluster_df["target"] == target)
            ]
            .sort_values("nmi_mean", ascending=False)
            .iloc[0]["kmeans_k"]
        )
        embeddings = embedding_map_seed11[(backbone_name, target)]
        cluster_labels = KMeans(n_clusters=int(k_choice), n_init=20, random_state=11).fit_predict(embeddings)
        field_values = manifest[field_name].astype(str).fillna("<missing>").to_numpy(dtype=str)
        purity_rows: List[Tuple[int, str, float, np.ndarray]] = []
        for cluster_id in np.unique(cluster_labels):
            dominant_value, purity, mask = _cluster_purity(field_values, cluster_labels, int(cluster_id))
            purity_rows.append((int(cluster_id), dominant_value, purity, mask))
        purity_rows.sort(key=lambda item: item[2], reverse=True)
        chosen = purity_rows[:3]

        thumb_w, thumb_h = 160, 160
        padding = 10
        cols = 4
        rows_n = len(chosen)
        canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * padding, rows_n * (thumb_h + 28) + (rows_n + 1) * padding + 24), color=(250, 250, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text((padding, 6), f"{field_name} | {backbone_name} | {target} | K={int(k_choice)}", fill=(0, 0, 0))
        for row_idx, (cluster_id, dominant_value, purity, mask) in enumerate(chosen):
            member_idx = np.flatnonzero(mask)[:cols]
            y0 = 28 + padding + row_idx * (thumb_h + 28 + padding)
            draw.text((padding, y0), f"cluster {cluster_id} | dominant={dominant_value} | purity={purity:.3f}", fill=(20, 20, 20))
            for col_idx, sample_idx in enumerate(member_idx):
                img_path = Path(str(manifest.iloc[int(sample_idx)]["img_path"]))
                img = Image.open(img_path).convert("L")
                img = ImageOps.fit(img, (thumb_w - 6, thumb_h - 6), method=Image.Resampling.BILINEAR)
                thumb = Image.new("RGB", (thumb_w, thumb_h), color="white")
                thumb.paste(Image.merge("RGB", (img, img, img)), (3, 3))
                thumb_draw = ImageDraw.Draw(thumb)
                thumb_draw.rectangle((0, 0, thumb_w - 1, thumb_h - 1), outline=(180, 180, 180), width=1)
                x0 = padding + col_idx * (thumb_w + padding)
                canvas.paste(thumb, (x0, y0 + 18))
            summary_rows.append(
                {
                    "field_name": field_name,
                    "backbone_name": backbone_name,
                    "target": target,
                    "kmeans_k": int(k_choice),
                    "cluster_id": cluster_id,
                    "dominant_value": dominant_value,
                    "purity": purity,
                }
            )
        canvas.save(output_dir / f"{field_name}_{backbone_name}_{target}.png")
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)


def _format_float(value: float) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.6f}"


def write_markdown(
    output_path: Path,
    audit_df: pd.DataFrame,
    field_specs_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    field_audit_path: Path,
    field_catalog_path: Path,
    analysis_title: str,
) -> None:
    lines: List[str] = []
    lines.append(f"# {analysis_title}: Cluster Anchoring Attribution Analysis")
    lines.append("")
    usable = audit_df[audit_df["usable"] == 1]
    dropped = audit_df[audit_df["usable"] == 0]
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- discovered categorical fields: {int((field_specs_df['field_type'] == 'categorical').sum())}")
    lines.append(f"- discovered continuous fields: {int((field_specs_df['field_type'] == 'continuous').sum())}")
    lines.append(f"- usable categorical fields: {', '.join(usable[usable['field_type'] == 'categorical']['field_name'].tolist())}")
    lines.append(f"- usable continuous fields: {', '.join(usable[usable['field_type'] == 'continuous']['field_name'].tolist())}")
    lines.append(f"- field audit CSV: `{field_audit_path}`")
    lines.append(f"- field catalog CSV: `{field_catalog_path}`")
    lines.append("")
    lines.append("## Top Anchors (Probe)")
    lines.append("")
    for target in TARGETS:
        lines.append(f"### {target.title()} head")
        lines.append("")
        for backbone_name in BACKBONES:
            subset = anchor_df[(anchor_df['target'] == target) & (anchor_df['backbone_name'] == backbone_name)]
            top_cat = subset[(subset['field_type'] == 'categorical') & (subset['field_group'] != 'reference')].sort_values('combined_anchor_score_mean', ascending=False).head(3)
            top_cont = subset[subset['field_type'] == 'continuous'].sort_values('combined_anchor_score_mean', ascending=False).head(3)
            lines.append(f"- {backbone_name}")
            cat_text = ", ".join(f"`{row.field_name}` ({_format_float(row.combined_anchor_score_mean)})" for row in top_cat.itertuples())
            cont_text = ", ".join(f"`{row.field_name}` ({_format_float(row.combined_anchor_score_mean)})" for row in top_cont.itertuples())
            lines.append(f"  - top categorical anchors: {cat_text}")
            lines.append(f"  - top continuous anchors: {cont_text}")
        lines.append("")
    lines.append("## Dropped Fields")
    lines.append("")
    lines.append("| Field | Type | Reason |")
    lines.append("| --- | --- | --- |")
    for row in dropped.sort_values(["field_type", "field_name"]).itertuples():
        lines.append(f"| {row.field_name} | {row.field_type} | {row.drop_reason} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If angle-derived fields dominate, the embedding space is view-anchored.")
    lines.append("- If device/protocol fields dominate, the embedding space is device/protocol anchored.")
    lines.append("- If brightness/entropy/sharpness fields dominate, the embedding space is image-appearance anchored.")
    lines.append("- Compare nuisance anchors against `patient_id` and `study_id` scores before concluding that patient/study identity is not the primary organizer.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_jobs(probe_seeds: Sequence[int]) -> List[JobDef]:
    jobs: List[JobDef] = []
    for seed in probe_seeds:
        for backbone_name in BACKBONES:
            for target in TARGETS:
                jobs.append(JobDef(kind="analysis", backbone_name=backbone_name, target=target, seed=int(seed)))
    jobs.append(JobDef(kind="postprocess", name="aggregate_and_render"))
    jobs.append(JobDef(kind="postprocess", name="write_markdown_and_meta"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global2-root", default="outputs/global_2_study_patient_retrieval_unique_view")
    parser.add_argument("--image-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--dcm-root", default="input/stent_split_dcm_unique_view")
    parser.add_argument("--output-root", default="outputs/global_4_1_cluster_anchoring_attribution_unique_view")
    parser.add_argument("--analysis-title", default="Global Analysis 4-1")
    parser.add_argument("--summary-prefix", default="summary_global_4_1")
    parser.add_argument("--figure-prefix", default="fig_global4_1")
    parser.add_argument("--markdown-name", default="analysis_global_4_1_cluster_anchoring_attribution.md")
    parser.add_argument("--log-prefix", default="global_4_1_cluster_anchoring_attribution")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--targets", nargs="+", default=["patient", "study"])
    parser.add_argument("--knn-k", type=int, nargs="+", default=KNN_K_DEFAULT)
    parser.add_argument("--cluster-k", type=int, nargs="+", default=KMEANS_K_DEFAULT)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    ensure_dir(out_root)
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        out_root,
        args.log_file,
        default_prefix=str(args.log_prefix),
    )

    try:
        set_global_seed(int(args.seed))
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        global2_root = Path(args.global2_root).resolve()
        image_root = Path(args.image_root).resolve()
        dcm_root = Path(args.dcm_root).resolve()
        device = resolve_device(args.device)
        targets = [str(t) for t in args.targets]
        knn_k_values = [int(v) for v in args.knn_k]
        cluster_k_values = [int(v) for v in args.cluster_k]
        if sorted(targets) != sorted(TARGETS):
            raise ValueError(f"Expected targets {TARGETS}, got {targets}")

        summary_name = lambda suffix: f"{args.summary_prefix}_{suffix}.csv"
        figure_name = lambda suffix: f"{args.figure_prefix}_{suffix}.png"

        manifest, full_manifest_hash, field_specs_df = build_anchor_manifest(
            global2_root=global2_root,
            image_root=image_root,
            dcm_root=dcm_root,
            output_root=out_root,
            max_images=args.max_images,
            seed=int(args.seed),
            field_catalog_name=summary_name("field_catalog"),
        )
        active_manifest_hash = hash_dataframe(manifest, ["image_id", "img_path", "patient_id", "study_id"])
        audit_df, usable_categorical, usable_continuous = audit_fields(manifest, field_specs_df)
        audit_df.to_csv(out_root / summary_name("field_audit"), index=False)
        save_field_audit_figure(audit_df, out_root / figure_name("field_audit"))
        log(
            f"Discovered fields | categorical={int((field_specs_df['field_type'] == 'categorical').sum())} "
            f"continuous={int((field_specs_df['field_type'] == 'continuous').sum())}"
        )
        log(f"Usable categorical fields: {usable_categorical}")
        log(f"Usable continuous fields: {usable_continuous}")

        field_group_map = dict(zip(field_specs_df["field_name"].tolist(), field_specs_df["field_group"].tolist()))
        for ref_name in REFERENCE_FIELDS:
            field_group_map[ref_name] = "reference"

        full_features: Dict[str, torch.Tensor] = {}
        feature_hashes: Dict[str, str] = {}
        global2_meta: Dict[str, Dict[str, object]] = {}
        for backbone_name in BACKBONES:
            features, meta = load_test_features(global2_root, backbone_name)
            full_features[backbone_name] = features
            feature_hashes[backbone_name] = make_feature_hash(features)
            global2_meta[backbone_name] = meta
            log(
                f"Loaded test features | backbone={backbone_name} | shape={tuple(features.shape)} | "
                f"feature_hash={feature_hashes[backbone_name][:12]}"
            )
        active_indices = manifest["image_id"].astype(int).to_numpy(dtype=np.int64)

        jobs = build_jobs(args.probe_seeds)
        tracker = FullRunTracker(jobs)

        neighborhood_categorical_raw_tables: List[pd.DataFrame] = []
        neighborhood_continuous_raw_tables: List[pd.DataFrame] = []
        cluster_categorical_raw_tables: List[pd.DataFrame] = []
        cluster_continuous_raw_tables: List[pd.DataFrame] = []
        anchor_rank_raw_tables: List[pd.DataFrame] = []
        seed11_embedding_map: Dict[Tuple[str, str], np.ndarray] = {}

        job_index = 0
        for seed in args.probe_seeds:
            for backbone_name in BACKBONES:
                for target in TARGETS:
                    tracker.start_job(job_index)
                    tracker.update_phase("load_probe_embedding", 1, 6)
                    full_probe_embeddings, summary = apply_probe_checkpoint(
                        global2_root=global2_root,
                        backbone_name=backbone_name,
                        target=target,
                        seed=int(seed),
                        features=full_features[backbone_name],
                        manifest_hash=full_manifest_hash,
                        feature_hash=feature_hashes[backbone_name],
                        device=device,
                        batch_size=int(args.probe_batch_size),
                    )
                    active_probe_embeddings = full_probe_embeddings[active_indices].contiguous()
                    active_np = active_probe_embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
                    if int(seed) == 11:
                        seed11_embedding_map[(backbone_name, target)] = active_np

                    tracker.update_phase("build_anchor_manifest", 2, 6)
                    sims = compute_similarity_matrix(active_probe_embeddings)
                    topk_neighbors = build_topk_neighbors(sims, max(knn_k_values))

                    tracker.update_phase("neighborhood_scores", 3, 6)
                    neighborhood_categorical_raw_tables.append(
                        compute_categorical_neighborhood_rows(
                            topk_neighbors=topk_neighbors,
                            manifest=manifest,
                            field_names=usable_categorical,
                            backbone_name=backbone_name,
                            target=target,
                            seed=int(seed),
                            k_values=knn_k_values,
                            field_group_map=field_group_map,
                        )
                    )
                    neighborhood_continuous_raw_tables.append(
                        compute_continuous_neighborhood_rows(
                            topk_neighbors=topk_neighbors,
                            manifest=manifest,
                            field_names=usable_continuous,
                            backbone_name=backbone_name,
                            target=target,
                            seed=int(seed),
                            k_values=knn_k_values,
                            field_group_map=field_group_map,
                        )
                    )

                    tracker.update_phase("cluster_scores", 4, 6)
                    cluster_categorical_raw_tables.append(
                        compute_categorical_cluster_rows(
                            embeddings=active_np,
                            manifest=manifest,
                            field_names=usable_categorical,
                            backbone_name=backbone_name,
                            target=target,
                            seed=int(seed),
                            cluster_k_values=cluster_k_values,
                            field_group_map=field_group_map,
                        )
                    )
                    cluster_continuous_raw_tables.append(
                        compute_continuous_cluster_rows(
                            embeddings=active_np,
                            manifest=manifest,
                            field_names=usable_continuous,
                            backbone_name=backbone_name,
                            target=target,
                            seed=int(seed),
                            cluster_k_values=cluster_k_values,
                            field_group_map=field_group_map,
                        )
                    )

                    tracker.update_phase("rank_attributes", 5, 6)
                    neighborhood_cat_job = neighborhood_categorical_raw_tables[-1]
                    neighborhood_cont_job = neighborhood_continuous_raw_tables[-1]
                    cluster_cat_job = cluster_categorical_raw_tables[-1]
                    cluster_cont_job = cluster_continuous_raw_tables[-1]
                    anchor_rank_raw_tables.append(
                        compute_combined_anchor_rows(
                            neighborhood_df=neighborhood_cat_job,
                            cluster_df=cluster_cat_job,
                            field_type="categorical",
                            backbone_name=backbone_name,
                            target=target,
                            seed=int(seed),
                        )
                    )
                    anchor_rank_raw_tables.append(
                        compute_combined_anchor_rows(
                            neighborhood_df=neighborhood_cont_job,
                            cluster_df=cluster_cont_job,
                            field_type="continuous",
                            backbone_name=backbone_name,
                            target=target,
                            seed=int(seed),
                        )
                    )

                    tracker.update_phase("write_raw_rows", 6, 6)
                    log(
                        f"Finished analysis job | backbone={backbone_name} target={target} seed={seed} | "
                        f"usable_cat={len(usable_categorical)} usable_cont={len(usable_continuous)}"
                    )
                    tracker.finish_job()
                    job_index += 1

        tracker.start_job(job_index)
        tracker.update_phase("aggregate", 1, 2)
        neighborhood_categorical_raw_df = pd.concat(neighborhood_categorical_raw_tables, axis=0, ignore_index=True)
        neighborhood_continuous_raw_df = pd.concat(neighborhood_continuous_raw_tables, axis=0, ignore_index=True)
        cluster_categorical_raw_df = pd.concat(cluster_categorical_raw_tables, axis=0, ignore_index=True)
        cluster_continuous_raw_df = pd.concat(cluster_continuous_raw_tables, axis=0, ignore_index=True)
        anchor_rank_raw_df = pd.concat(anchor_rank_raw_tables, axis=0, ignore_index=True)

        neighborhood_categorical_df = aggregate_rows(
            neighborhood_categorical_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "k"],
            metric_cols=["num_valid_queries", "base_match_rate", "neighbor_match_rate", "purity_uplift"],
        )
        neighborhood_continuous_df = aggregate_rows(
            neighborhood_continuous_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "k"],
            metric_cols=["num_valid_queries", "global_abs_diff_mean", "neighbor_abs_diff_mean", "neighbor_consistency"],
        )
        cluster_categorical_df = aggregate_rows(
            cluster_categorical_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "kmeans_k"],
            metric_cols=["num_valid_samples", "nmi"],
        )
        cluster_continuous_df = aggregate_rows(
            cluster_continuous_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "kmeans_k"],
            metric_cols=["num_valid_samples", "eta_squared"],
        )
        anchor_rank_df = aggregate_rows(
            anchor_rank_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group"],
            metric_cols=["local_metric", "global_metric", "local_z", "global_z", "combined_anchor_score"],
        )
        anchor_rank_df = assign_anchor_ranks(anchor_rank_df)

        neighborhood_categorical_raw_df.to_csv(out_root / summary_name("neighborhood_categorical_raw"), index=False)
        neighborhood_continuous_raw_df.to_csv(out_root / summary_name("neighborhood_continuous_raw"), index=False)
        cluster_categorical_raw_df.to_csv(out_root / summary_name("cluster_categorical_raw"), index=False)
        cluster_continuous_raw_df.to_csv(out_root / summary_name("cluster_continuous_raw"), index=False)
        anchor_rank_raw_df.to_csv(out_root / summary_name("anchor_rank_raw"), index=False)

        neighborhood_categorical_df.to_csv(out_root / summary_name("neighborhood_categorical"), index=False)
        neighborhood_continuous_df.to_csv(out_root / summary_name("neighborhood_continuous"), index=False)
        cluster_categorical_df.to_csv(out_root / summary_name("cluster_categorical"), index=False)
        cluster_continuous_df.to_csv(out_root / summary_name("cluster_continuous"), index=False)
        anchor_rank_df.to_csv(out_root / summary_name("anchor_rank"), index=False)

        save_anchor_rank_figure(anchor_rank_df, "categorical", out_root / figure_name("anchor_rank_categorical"), args.analysis_title, top_n=None)
        save_anchor_rank_target_figure(anchor_rank_df, "categorical", "patient", out_root / figure_name("anchor_rank_categorical_patient"), args.analysis_title, top_n=None)
        save_anchor_rank_target_figure(anchor_rank_df, "categorical", "study", out_root / figure_name("anchor_rank_categorical_study"), args.analysis_title, top_n=None)
        save_anchor_rank_figure(anchor_rank_df, "continuous", out_root / figure_name("anchor_rank_continuous"), args.analysis_title)
        save_anchor_rank_target_figure(anchor_rank_df, "continuous", "patient", out_root / figure_name("anchor_rank_continuous_patient"), args.analysis_title, top_n=None)
        save_anchor_rank_target_figure(anchor_rank_df, "continuous", "study", out_root / figure_name("anchor_rank_continuous_study"), args.analysis_title, top_n=None)
        save_umap_overlay_top_categorical(
            aggregate_anchor_df=anchor_rank_df,
            embedding_map=seed11_embedding_map,
            manifest=manifest,
            output_path=out_root / figure_name("umap_overlay_top_categorical"),
            seed=int(args.seed),
        )
        save_umap_overlay_top_continuous(
            aggregate_anchor_df=anchor_rank_df,
            embedding_map=seed11_embedding_map,
            manifest=manifest,
            output_path=out_root / figure_name("umap_overlay_top_continuous"),
            seed=int(args.seed),
        )
        export_umap_overlays_for_nonnegative_fields(
            aggregate_anchor_df=anchor_rank_df,
            embedding_map=seed11_embedding_map,
            manifest=manifest,
            output_root=out_root / "umap_overlays_score_ge_0",
            seed=int(args.seed),
        )
        save_patient_vs_nuisance_compare(anchor_rank_df, out_root / figure_name("patient_vs_nuisance_compare"), args.analysis_title)
        save_anchor_examples(
            anchor_df=anchor_rank_df,
            cluster_df=cluster_categorical_df,
            embedding_map_seed11=seed11_embedding_map,
            manifest=manifest,
            output_dir=out_root / "anchor_examples",
        )
        tracker.finish_job()
        job_index += 1

        tracker.start_job(job_index)
        tracker.update_phase("write_markdown", 1, 2)
        write_markdown(
            output_path=out_root / args.markdown_name,
            audit_df=audit_df,
            field_specs_df=field_specs_df,
            anchor_df=anchor_rank_df,
            field_audit_path=out_root / summary_name("field_audit"),
            field_catalog_path=out_root / summary_name("field_catalog"),
            analysis_title=args.analysis_title,
        )
        run_meta = {
            "global2_root": str(global2_root),
            "image_root": str(image_root),
            "dcm_root": str(dcm_root),
            "output_root": str(out_root.resolve()),
            "probe_seeds": [int(v) for v in args.probe_seeds],
            "targets": targets,
            "knn_k": knn_k_values,
            "cluster_k": cluster_k_values,
            "max_images": args.max_images,
            "seed": int(args.seed),
            "device": str(device),
            "full_test_manifest_hash": full_manifest_hash,
            "active_manifest_hash": active_manifest_hash,
            "usable_categorical_fields": usable_categorical,
            "usable_continuous_fields": usable_continuous,
            "field_catalog_path": str((out_root / summary_name("field_catalog")).resolve()),
            "num_discovered_fields": int(len(field_specs_df)),
            "feature_hashes": feature_hashes,
            "log_path": str(log_path.resolve()),
        }
        (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        tracker.finish_job()

        log(f"{args.analysis_title} completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
