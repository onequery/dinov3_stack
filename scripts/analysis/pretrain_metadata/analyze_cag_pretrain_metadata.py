#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import heapq
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import cv2
import numpy as np
import pandas as pd
import pydicom


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
FRAME_TYPES = ["curr_frame", "prev_1_frame", "prev_2_frame", "next_1_frame", "next_2_frame"]
NUMERIC_VRS = {"DS", "FD", "FL", "IS", "SL", "SS", "SV", "UL", "US", "UV"}
DATE_VRS = {"DA"}
TIME_VRS = {"TM"}
DATETIME_VRS = {"DT"}
MAX_STRING_LENGTH = 160
FIELD_NAME_RE = re.compile(r"[^0-9A-Za-z]+")
FILENAME_RE = re.compile(r"^(?P<patient_id>.+?)_(?P<study_date>\d{8})_(?P<dicom_id>[^_]+)_(?P<frame_idx>\d+)\.png$")
ID_COLUMNS = [
    "split",
    "frame_type",
    "image_path",
    "image_rel_path",
    "patient_id",
    "study_date",
    "dicom_id",
    "frame_idx",
    "dicom_path",
    "status",
    "error_message",
]


@dataclass(frozen=True)
class FieldSpec:
    field_name: str
    original_name: str
    field_type: str
    field_group: str
    source: str
    is_derived: int


@dataclass(frozen=True)
class SubtaskDef:
    split: str
    frame_type: str
    image_dir: Path
    image_paths: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.split}__{self.frame_type}"

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


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


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


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


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Empty._"
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def setup_console_and_file_logging(output_root: Path, default_prefix: str) -> tuple[Path, TextIO, TextIO, TextIO]:
    ensure_dir(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_root / f"{default_prefix}_{stamp}.log"
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


def sanitize_field_name(name: str) -> str:
    text = FIELD_NAME_RE.sub("", str(name))
    if not text:
        text = "UnnamedField"
    if text[0].isdigit():
        text = f"Field{text}"
    return text


def register_field(
    field_specs: Dict[str, FieldSpec],
    field_name: str,
    original_name: str,
    field_type: str,
    field_group: str,
    source: str,
    is_derived: bool,
) -> None:
    spec = FieldSpec(
        field_name=field_name,
        original_name=original_name,
        field_type=field_type,
        field_group=field_group,
        source=source,
        is_derived=int(bool(is_derived)),
    )
    merge_field_spec(field_specs, spec)


def reconcile_field_specs(existing: FieldSpec | None, candidate: FieldSpec) -> FieldSpec:
    if existing is None or existing == candidate:
        return candidate
    compatible_identity = (
        existing.field_name == candidate.field_name
        and existing.field_group == candidate.field_group
        and existing.source == candidate.source
        and existing.is_derived == candidate.is_derived
    )
    if compatible_identity and {existing.field_type, candidate.field_type} == {"continuous", "categorical"}:
        original_name = existing.original_name if existing.original_name == candidate.original_name else existing.original_name or candidate.original_name
        return FieldSpec(
            field_name=existing.field_name,
            original_name=original_name,
            field_type="categorical",
            field_group=existing.field_group,
            source=existing.source,
            is_derived=existing.is_derived,
        )
    raise ValueError(f"Field spec conflict for {candidate.field_name}: {existing} vs {candidate}")


def merge_field_spec(field_specs: Dict[str, FieldSpec], spec: FieldSpec) -> None:
    field_specs[spec.field_name] = reconcile_field_specs(field_specs.get(spec.field_name), spec)


def clean_categorical(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) > MAX_STRING_LENGTH:
        text = text[:MAX_STRING_LENGTH]
    if text == "" or text.lower() in {"none", "nan"}:
        return None
    return text


def clean_numeric(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except Exception:
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def maybe_iterable_value(value: object) -> List[object] | None:
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return list(value)
        except TypeError:
            return None
    return None


def normalize_dicom_keyword(elem: pydicom.dataelem.DataElement) -> str:
    keyword = elem.keyword or sanitize_field_name(elem.name)
    return sanitize_field_name(keyword)


def parse_dicom_date(text: str) -> tuple[int, int, int] | None:
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if len(digits) < 8:
        return None
    year = int(digits[:4])
    month = int(digits[4:6])
    day = int(digits[6:8])
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return year, month, day


def parse_dicom_time(text: str) -> tuple[int, int, int] | None:
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
            for name, val in {
                f"{keyword}Year": str(year),
                f"{keyword}Month": f"{month:02d}",
                f"{keyword}Day": f"{day:02d}",
                f"{keyword}YearMonth": f"{year:04d}-{month:02d}",
            }.items():
                values[name] = val
                register_field(field_specs, name, keyword, "categorical", "derived", "dicom_temporal", True)
    if vr in TIME_VRS or vr in DATETIME_VRS:
        parsed_time = parse_dicom_time(raw_text)
        if parsed_time is not None:
            hour, minute, _second = parsed_time
            minute10 = (minute // 10) * 10
            for name, val in {
                f"{keyword}Hour": f"{hour:02d}",
                f"{keyword}Minute10Bin": f"{minute10:02d}",
            }.items():
                values[name] = val
                register_field(field_specs, name, keyword, "categorical", "derived", "dicom_temporal", True)


def add_numeric_multivalue_derivatives(
    values: Dict[str, object],
    field_specs: Dict[str, FieldSpec],
    keyword: str,
    numeric_values: Sequence[float],
) -> None:
    arr = np.asarray(list(numeric_values), dtype=np.float64)
    if arr.size == 0:
        return
    for name, val in {
        f"{keyword}MultiMean": float(arr.mean()),
        f"{keyword}MultiMin": float(arr.min()),
        f"{keyword}MultiMax": float(arr.max()),
        f"{keyword}MultiSpan": float(arr.max() - arr.min()),
        f"{keyword}MultiCount": float(arr.size),
    }.items():
        values[name] = val
        register_field(field_specs, name, keyword, "continuous", "derived", "dicom_multivalue", True)


def extract_dicom_scalar_fields(dcm_path: Path) -> tuple[Dict[str, object], Dict[str, FieldSpec]]:
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
    values: Dict[str, object] = {}
    field_specs: Dict[str, FieldSpec] = {}
    for elem in ds:
        if elem.VR == "SQ" or elem.tag.is_private:
            continue
        keyword = normalize_dicom_keyword(elem)
        if keyword == "PixelData":
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
                    register_field(field_specs, keyword, elem.keyword or elem.name, "categorical", "metadata", "dicom_scalar", False)
                add_numeric_multivalue_derivatives(values, field_specs, keyword, [float(v) for v in numeric_values if v is not None])
                continue
            joined = clean_categorical("\\".join(str(v) for v in seq))
            if joined is not None:
                values[keyword] = joined
                register_field(field_specs, keyword, elem.keyword or elem.name, "categorical", "metadata", "dicom_scalar", False)
            continue
        if elem.VR in NUMERIC_VRS:
            numeric = clean_numeric(raw_value)
            if numeric is None:
                continue
            values[keyword] = numeric
            register_field(field_specs, keyword, elem.keyword or elem.name, "continuous", "metadata", "dicom_scalar", False)
            continue
        cat = clean_categorical(raw_value)
        if cat is None:
            continue
        values[keyword] = cat
        register_field(field_specs, keyword, elem.keyword or elem.name, "categorical", "metadata", "dicom_scalar", False)
        if elem.VR in DATE_VRS | TIME_VRS | DATETIME_VRS:
            add_temporal_derivatives(values, field_specs, keyword, elem.VR, cat)
    return values, field_specs


def compute_basic_image_attributes(img_path: Path) -> tuple[Dict[str, float], Dict[str, FieldSpec]]:
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_f = img.astype(np.float32)
    values = {
        "width": float(int(img.shape[1])),
        "height": float(int(img.shape[0])),
        "intensity_mean": float(img_f.mean()),
        "intensity_std": float(img_f.std(ddof=0)),
        "intensity_min": float(img_f.min()),
        "intensity_max": float(img_f.max()),
        "intensity_p01": float(np.percentile(img_f, 1.0)),
        "intensity_p50": float(np.percentile(img_f, 50.0)),
        "intensity_p99": float(np.percentile(img_f, 99.0)),
    }
    specs: Dict[str, FieldSpec] = {}
    for name in values:
        register_field(specs, name, name, "continuous", "image", "image_stats", True)
    return values, specs


def parse_image_filename(name: str) -> dict[str, str] | None:
    match = FILENAME_RE.match(name)
    if not match:
        return None
    return match.groupdict()


def ordered_columns(columns: Iterable[str]) -> list[str]:
    others = sorted(col for col in columns if col not in ID_COLUMNS)
    return [col for col in ID_COLUMNS if col in columns] + others


def get_enumeration_cache_paths(output_root: Path) -> tuple[Path, Path]:
    cache_dir = output_root / "enumeration_cache"
    meta_path = cache_dir / "meta.json"
    return cache_dir, meta_path


def try_load_enumeration_cache(
    image_root: Path,
    output_root: Path,
    splits: Sequence[str],
    max_images_per_subtask: int,
) -> list[SubtaskDef] | None:
    cache_dir, meta_path = get_enumeration_cache_paths(output_root)
    if not meta_path.exists():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log(f"ENUMERATE CACHE MISS | reason=meta_read_error | detail={type(exc).__name__}: {exc}")
        return None
    expected = {
        "version": 1,
        "image_root": str(image_root.resolve()),
        "splits": list(splits),
        "frame_types": list(FRAME_TYPES),
        "max_images_per_subtask": int(max_images_per_subtask),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            log(f"ENUMERATE CACHE MISS | reason=config_mismatch | key={key}")
            return None
    subtasks_payload = payload.get("subtasks", [])
    subtasks: list[SubtaskDef] = []
    total_expected = len(subtasks_payload)
    load_start = time.time()
    for idx, item in enumerate(subtasks_payload, start=1):
        split = item["split"]
        frame_type = item["frame_type"]
        image_dir = Path(item["image_dir"])
        manifest_file = cache_dir / item["manifest_file"]
        if not manifest_file.exists():
            log(f"ENUMERATE CACHE MISS | reason=missing_manifest | split={split} frame_type={frame_type}")
            return None
        names = manifest_file.read_text(encoding="utf-8").splitlines()
        image_paths = tuple(image_dir / name for name in names if name)
        subtasks.append(SubtaskDef(split=split, frame_type=frame_type, image_dir=image_dir, image_paths=image_paths))
        elapsed = time.time() - load_start
        remaining = estimate_remaining(elapsed, idx, total_expected)
        log(
            f"[ENUMERATE] CACHE | split={split} frame_type={frame_type} | images={len(image_paths)} | "
            f"subtasks_done={idx}/{total_expected} | elapsed={format_duration(elapsed)} | "
            f"remaining={format_duration(remaining) if math.isfinite(remaining) else 'n/a'}"
        )
    log(f"ENUMERATE CACHE HIT | subtasks={len(subtasks)} | total_images={sum(s.image_count for s in subtasks)}")
    return subtasks


def save_enumeration_cache(
    output_root: Path,
    image_root: Path,
    splits: Sequence[str],
    max_images_per_subtask: int,
    subtasks: Sequence[SubtaskDef],
) -> None:
    cache_dir, meta_path = get_enumeration_cache_paths(output_root)
    ensure_dir(cache_dir)
    payload_subtasks: list[dict[str, object]] = []
    for subtask in subtasks:
        manifest_name = f"{subtask.key}.txt"
        manifest_path = cache_dir / manifest_name
        manifest_path.write_text("\n".join(path.name for path in subtask.image_paths) + "\n", encoding="utf-8")
        payload_subtasks.append(
            {
                "split": subtask.split,
                "frame_type": subtask.frame_type,
                "image_dir": str(subtask.image_dir),
                "image_count": int(subtask.image_count),
                "manifest_file": manifest_name,
            }
        )
    atomic_write_json(
        meta_path,
        {
            "version": 1,
            "image_root": str(image_root.resolve()),
            "splits": list(splits),
            "frame_types": list(FRAME_TYPES),
            "max_images_per_subtask": int(max_images_per_subtask),
            "subtasks": payload_subtasks,
        },
    )


def build_subtasks(
    image_root: Path,
    splits: Sequence[str],
    max_images_per_subtask: int,
    output_root: Path,
    refresh_enumeration_cache: bool,
) -> list[SubtaskDef]:
    if not refresh_enumeration_cache:
        cached = try_load_enumeration_cache(image_root, output_root, splits, max_images_per_subtask)
        if cached is not None:
            return cached

    subtasks: list[SubtaskDef] = []
    total_expected = len(splits) * len(FRAME_TYPES)
    enumerate_start = time.time()
    subtasks_done = 0
    for split in splits:
        for frame_type in FRAME_TYPES:
            image_dir = image_root / split / frame_type
            log(f"[ENUMERATE] START | split={split} frame_type={frame_type} | subtask={subtasks_done + 1}/{total_expected}")
            if not image_dir.exists():
                subtasks_done += 1
                remaining = estimate_remaining(time.time() - enumerate_start, subtasks_done, total_expected)
                log(
                    f"[ENUMERATE] SKIP | split={split} frame_type={frame_type} | reason=missing_dir | "
                    f"subtasks_done={subtasks_done}/{total_expected} | remaining={format_duration(remaining) if math.isfinite(remaining) else 'n/a'}"
                )
                continue
            if max_images_per_subtask > 0:
                selected: list[Path] = []
                for path_entry in image_dir.iterdir():
                    if not path_entry.is_file() or path_entry.suffix.lower() != ".png":
                        continue
                    selected.append(path_entry)
                    if len(selected) >= max_images_per_subtask:
                        break
                image_paths = tuple(selected)
            else:
                png_iter = (p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
                image_paths = tuple(sorted(png_iter, key=lambda p: p.name))
            subtasks.append(SubtaskDef(split=split, frame_type=frame_type, image_dir=image_dir, image_paths=image_paths))
            subtasks_done += 1
            elapsed = time.time() - enumerate_start
            remaining = estimate_remaining(elapsed, subtasks_done, total_expected)
            log(
                f"[ENUMERATE] DONE | split={split} frame_type={frame_type} | images={len(image_paths)} | "
                f"subtasks_done={subtasks_done}/{total_expected} | elapsed={format_duration(elapsed)} | "
                f"remaining={format_duration(remaining) if math.isfinite(remaining) else 'n/a'}"
            )
    save_enumeration_cache(output_root, image_root, splits, max_images_per_subtask, subtasks)
    return subtasks


def estimate_remaining(elapsed_seconds: float, done: int, total: int) -> float:
    if total <= 0 or done <= 0:
        return float("nan")
    avg = elapsed_seconds / max(done, 1)
    return max(0.0, avg * (total - done))


def log_total_progress(start_time: float, done: int, total: int) -> None:
    elapsed = max(0.0, time.time() - start_time)
    remaining = estimate_remaining(elapsed, done, total)
    progress = 0.0 if total <= 0 else (100.0 * done / total)
    eta = datetime.now() + timedelta(seconds=0.0 if not math.isfinite(remaining) else remaining)
    remaining_text = "n/a" if not math.isfinite(remaining) else format_duration(remaining)
    log(
        f"[TOTAL] done={done}/{total} | progress={progress:.1f}% | "
        f"elapsed={format_duration(elapsed)} | remaining={remaining_text} | eta={eta.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def log_subtask_progress(subtask: SubtaskDef, subtask_start: float, done: int, total: int) -> None:
    elapsed = max(0.0, time.time() - subtask_start)
    remaining = estimate_remaining(elapsed, done, total)
    remaining_text = "n/a" if not math.isfinite(remaining) else format_duration(remaining)
    log(
        f"[SUBTASK] split={subtask.split} frame_type={subtask.frame_type} done={done}/{total} | "
        f"elapsed={format_duration(elapsed)} | remaining={remaining_text}"
    )


def atomic_write_json(path: Path, payload: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def save_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def process_single_image(args: tuple[Path, Path, str, str, Path]) -> tuple[dict[str, object], list[dict[str, object]]]:
    img_path, image_root, split, frame_type, dicom_root = args
    row: dict[str, object] = {
        "split": split,
        "frame_type": frame_type,
        "image_path": str(img_path.resolve()),
        "image_rel_path": str(img_path.resolve().relative_to(image_root.resolve())),
        "patient_id": "",
        "study_date": "",
        "dicom_id": "",
        "frame_idx": "",
        "dicom_path": "",
        "status": "ok",
        "error_message": "",
    }
    field_specs: Dict[str, FieldSpec] = {}
    image_stats, image_specs = compute_basic_image_attributes(img_path)
    for spec in image_specs.values():
        merge_field_spec(field_specs, spec)
    row.update(image_stats)

    parsed = parse_image_filename(img_path.name)
    if parsed is None:
        row["status"] = "bad_filename"
        row["error_message"] = f"Unrecognized filename pattern: {img_path.name}"
        return row, [asdict(spec) for spec in field_specs.values()]

    row["patient_id"] = parsed["patient_id"]
    row["study_date"] = parsed["study_date"]
    row["dicom_id"] = parsed["dicom_id"]
    row["frame_idx"] = parsed["frame_idx"]
    dcm_path = dicom_root / parsed["patient_id"] / parsed["study_date"] / "XA" / f"{parsed['dicom_id']}.dcm"
    row["dicom_path"] = str(dcm_path)

    if not dcm_path.exists():
        row["status"] = "missing_dicom"
        row["error_message"] = f"Missing DICOM: {dcm_path}"
        return row, [asdict(spec) for spec in field_specs.values()]

    try:
        meta, dicom_specs = extract_dicom_scalar_fields(dcm_path)
    except Exception as exc:
        row["status"] = "dicom_read_error"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        return row, [asdict(spec) for spec in field_specs.values()]

    for spec in dicom_specs.values():
        merge_field_spec(field_specs, spec)
    row.update(meta)
    return row, [asdict(spec) for spec in field_specs.values()]


def extract_one_subtask(
    subtask: SubtaskDef,
    image_root: Path,
    dicom_root: Path,
    per_image_dir: Path,
    num_workers: int,
    log_every: int,
    total_start: float,
    total_done_ref: list[int],
) -> tuple[Path, Path]:
    csv_path = per_image_dir / f"{subtask.key}.csv"
    spec_path = per_image_dir / f"{subtask.key}.field_specs.json"
    rows: List[dict[str, object]] = []
    merged_specs: Dict[str, FieldSpec] = {}
    subtask_total = subtask.image_count
    subtask_done = 0
    subtask_start = time.time()
    log(f"[SUBTASK] START | split={subtask.split} frame_type={subtask.frame_type} total={subtask_total}")

    jobs = [(img_path, image_root, subtask.split, subtask.frame_type, dicom_root) for img_path in subtask.image_paths]
    with cf.ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        for row, specs in executor.map(process_single_image, jobs):
            rows.append(row)
            for raw_spec in specs:
                spec = FieldSpec(**raw_spec)
                merge_field_spec(merged_specs, spec)
            subtask_done += 1
            total_done_ref[0] += 1
            if subtask_done == subtask_total or (log_every > 0 and subtask_done % log_every == 0):
                log_subtask_progress(subtask, subtask_start, subtask_done, subtask_total)
                log_total_progress(total_start, total_done_ref[0], total_done_ref[1])

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.loc[:, ordered_columns(df.columns)]
        df = df.sort_values(["image_rel_path"]).reset_index(drop=True)
    save_dataframe_atomic(df, csv_path)
    atomic_write_json(spec_path, [asdict(spec) for spec in sorted(merged_specs.values(), key=lambda x: x.field_name)])
    log(
        f"[SUBTASK] DONE | split={subtask.split} frame_type={subtask.frame_type} rows={len(df)} | "
        f"elapsed={format_duration(time.time() - subtask_start)}"
    )
    return csv_path, spec_path


def load_field_specs_from_json(path: Path) -> Dict[str, FieldSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: Dict[str, FieldSpec] = {}
    for raw in payload:
        spec = FieldSpec(**raw)
        merge_field_spec(out, spec)
    return out


def summarize_counts(manifest: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    empty_columns = group_cols + [
        "image_count",
        "ok_count",
        "bad_filename_count",
        "missing_dicom_count",
        "dicom_read_error_count",
        "unique_patient_count",
        "unique_study_count",
        "unique_dicom_count",
    ]
    if manifest.empty:
        return pd.DataFrame(columns=empty_columns)
    rows: list[dict[str, object]] = []
    for keys, group in manifest.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "image_count": int(len(group)),
                "ok_count": int((group["status"] == "ok").sum()),
                "bad_filename_count": int((group["status"] == "bad_filename").sum()),
                "missing_dicom_count": int((group["status"] == "missing_dicom").sum()),
                "dicom_read_error_count": int((group["status"] == "dicom_read_error").sum()),
                "unique_patient_count": int(group["patient_id"].replace("", pd.NA).dropna().nunique()),
                "unique_study_count": int(group["study_date"].replace("", pd.NA).dropna().nunique()),
                "unique_dicom_count": int(group["dicom_id"].replace("", pd.NA).dropna().nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_numeric(manifest: pd.DataFrame, numeric_fields: list[str], group_cols: list[str]) -> pd.DataFrame:
    empty_columns = group_cols + [
        "field_name",
        "count",
        "missing_count",
        "missing_ratio",
        "mean",
        "std",
        "min",
        "max",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
    ]
    if manifest.empty or not numeric_fields:
        return pd.DataFrame(columns=empty_columns)
    rows: list[dict[str, object]] = []
    for keys, group in manifest.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_key = dict(zip(group_cols, keys))
        for field in numeric_fields:
            if field not in group.columns:
                continue
            values = pd.to_numeric(group[field], errors="coerce")
            present = values.dropna()
            total = int(len(values))
            count = int(present.shape[0])
            missing = int(total - count)
            row = {
                **group_key,
                "field_name": field,
                "count": count,
                "missing_count": missing,
                "missing_ratio": float(missing / total) if total > 0 else float("nan"),
            }
            if count == 0:
                row.update({k: float("nan") for k in ["mean", "std", "min", "max", "p01", "p05", "p25", "p50", "p75", "p95", "p99"]})
            else:
                row.update(
                    {
                        "mean": float(present.mean()),
                        "std": float(present.std(ddof=0)),
                        "min": float(present.min()),
                        "max": float(present.max()),
                        "p01": float(present.quantile(0.01)),
                        "p05": float(present.quantile(0.05)),
                        "p25": float(present.quantile(0.25)),
                        "p50": float(present.quantile(0.50)),
                        "p75": float(present.quantile(0.75)),
                        "p95": float(present.quantile(0.95)),
                        "p99": float(present.quantile(0.99)),
                    }
                )
            rows.append(row)
    sort_cols = group_cols + ["field_name"]
    return pd.DataFrame(rows).sort_values(sort_cols).reset_index(drop=True)


def summarize_categorical(manifest: pd.DataFrame, categorical_fields: list[str], group_cols: list[str]) -> pd.DataFrame:
    empty_columns = group_cols + ["field_name", "value", "count", "ratio"]
    if manifest.empty or not categorical_fields:
        return pd.DataFrame(columns=empty_columns)
    rows: list[dict[str, object]] = []
    for keys, group in manifest.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_key = dict(zip(group_cols, keys))
        total = int(len(group))
        for field in categorical_fields:
            if field not in group.columns:
                continue
            series = group[field].astype(object)
            valid = series.notna() & (series.astype(str) != "")
            value_counts = series[valid].astype(str).value_counts(dropna=False)
            for value, count in value_counts.items():
                rows.append(
                    {
                        **group_key,
                        "field_name": field,
                        "value": str(value),
                        "count": int(count),
                        "ratio": float(count / total) if total > 0 else float("nan"),
                    }
                )
    sort_cols = group_cols + ["field_name", "count", "value"]
    return pd.DataFrame(rows).sort_values(sort_cols, ascending=[True] * len(group_cols) + [True, False, True]).reset_index(drop=True)


def build_field_catalog(manifest: pd.DataFrame, field_specs: Dict[str, FieldSpec]) -> pd.DataFrame:
    if not field_specs:
        return pd.DataFrame(columns=[
            "field_name",
            "original_name",
            "field_type",
            "field_group",
            "source",
            "is_derived",
            "present_count",
            "missing_count",
            "missing_ratio",
        ])
    rows: list[dict[str, object]] = []
    total = len(manifest)
    for spec in sorted(field_specs.values(), key=lambda x: (x.field_group, x.field_type, x.field_name)):
        series = manifest[spec.field_name] if spec.field_name in manifest.columns else pd.Series(dtype=object)
        present_mask = series.notna()
        if spec.field_type == "categorical":
            present_mask = present_mask & (series.astype(str) != "")
        present_count = int(present_mask.sum()) if len(series) else 0
        missing_count = int(total - present_count)
        rows.append(
            {
                **asdict(spec),
                "present_count": present_count,
                "missing_count": missing_count,
                "missing_ratio": float(missing_count / total) if total > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_status_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["split", "frame_type", "status", "count"])
    parts: list[pd.DataFrame] = []
    for group_cols in [["split", "frame_type"], ["split"], []]:
        if group_cols:
            grouped = manifest.groupby(group_cols + ["status"], dropna=False).size().reset_index(name="count")
        else:
            grouped = manifest.groupby(["status"], dropna=False).size().reset_index(name="count")
            grouped.insert(0, "frame_type", "__all__")
            grouped.insert(0, "split", "__all__")
            parts.append(grouped)
            continue
        if "split" not in grouped.columns:
            grouped.insert(0, "split", "__all__")
        if "frame_type" not in grouped.columns:
            grouped.insert(1, "frame_type", "__all__")
        parts.append(grouped)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["split", "frame_type", "status"]).reset_index(drop=True)


def build_missingness_summary(manifest: pd.DataFrame, field_specs: Dict[str, FieldSpec]) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["field_name", "field_type", "present_count", "missing_count", "missing_ratio"])
    rows: list[dict[str, object]] = []
    total = len(manifest)
    spec_by_field = field_specs
    for field in sorted(col for col in manifest.columns if col != "error_message"):
        series = manifest[field]
        present_mask = series.notna()
        field_type = spec_by_field.get(field).field_type if field in spec_by_field else "identifier"
        if field_type == "categorical" or series.dtype == object:
            present_mask = present_mask & (series.astype(str) != "")
        present_count = int(present_mask.sum())
        missing_count = int(total - present_count)
        rows.append(
            {
                "field_name": field,
                "field_type": field_type,
                "present_count": present_count,
                "missing_count": missing_count,
                "missing_ratio": float(missing_count / total) if total > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["field_type", "missing_ratio", "field_name"], ascending=[True, False, True]).reset_index(drop=True)


def render_markdown_report(
    output_root: Path,
    manifest: pd.DataFrame,
    field_catalog: pd.DataFrame,
    counts_by_split: pd.DataFrame,
    counts_by_split_frame_type: pd.DataFrame,
    status_counts: pd.DataFrame,
    missingness: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    ok_manifest = manifest[manifest["status"] == "ok"].copy()
    lines: list[str] = []
    lines.append("# CAG Pretraining Dataset Metadata Stats")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append(f"- image_root: `{args.image_root}`")
    lines.append(f"- dicom_root: `{args.dicom_root}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- splits: `{', '.join(args.splits)}`")
    lines.append(f"- num_workers: `{args.num_workers}`")
    lines.append(f"- log_every: `{args.log_every}`")
    lines.append(f"- resume: `{bool(args.resume)}`")
    lines.append(f"- max_images_per_subtask: `{args.max_images_per_subtask}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- total_images: `{len(manifest)}`")
    lines.append(f"- ok_images: `{int((manifest['status'] == 'ok').sum())}`")
    lines.append(f"- failed_images: `{int((manifest['status'] != 'ok').sum())}`")
    lines.append(f"- unique_patients_ok: `{ok_manifest['patient_id'].replace('', pd.NA).dropna().nunique()}`")
    lines.append(f"- unique_studies_ok: `{ok_manifest['study_date'].replace('', pd.NA).dropna().nunique()}`")
    lines.append(f"- unique_dicoms_ok: `{ok_manifest['dicom_id'].replace('', pd.NA).dropna().nunique()}`")
    lines.append("")
    lines.append("## Field Inventory")
    lines.append("")
    lines.append(f"- total_fields: `{len(field_catalog)}`")
    lines.append(f"- categorical_fields: `{int((field_catalog['field_type'] == 'categorical').sum())}`")
    lines.append(f"- continuous_fields: `{int((field_catalog['field_type'] == 'continuous').sum())}`")
    lines.append(f"- metadata_fields: `{int((field_catalog['field_group'] == 'metadata').sum())}`")
    lines.append(f"- derived_fields: `{int((field_catalog['field_group'] == 'derived').sum())}`")
    lines.append(f"- image_fields: `{int((field_catalog['field_group'] == 'image').sum())}`")
    lines.append("")
    lines.append("## Counts By Split")
    lines.append("")
    lines.append(dataframe_to_markdown(counts_by_split))
    lines.append("")
    lines.append("## Counts By Split And Frame Type")
    lines.append("")
    lines.append(dataframe_to_markdown(counts_by_split_frame_type))
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    lines.append(dataframe_to_markdown(status_counts))
    lines.append("")
    lines.append("## Highest Missingness Fields")
    lines.append("")
    lines.append(dataframe_to_markdown(missingness.head(30)))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `per_image/metadata_manifest_all.csv`")
    lines.append("- `per_image/field_catalog.csv`")
    lines.append("- `summary_counts_by_split.csv`")
    lines.append("- `summary_counts_by_split_frame_type.csv`")
    lines.append("- `summary_numeric_by_split.csv`")
    lines.append("- `summary_numeric_by_split_frame_type.csv`")
    lines.append("- `summary_categorical_all_values_by_split.csv`")
    lines.append("- `summary_categorical_all_values_by_split_frame_type.csv`")
    lines.append("- `summary_status_counts.csv`")
    lines.append("- `summary_missingness_by_field.csv`")
    lines.append("- `summary_dicom_resolution_failures.csv`")
    lines.append("- `enumeration_cache/meta.json`")
    lines.append("")
    (output_root / "analysis_cag_pretrain_metadata.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def combine_subtask_outputs(subtasks: Sequence[SubtaskDef], per_image_dir: Path) -> tuple[pd.DataFrame, Dict[str, FieldSpec]]:
    dfs: list[pd.DataFrame] = []
    field_specs: Dict[str, FieldSpec] = {}
    for subtask in subtasks:
        csv_path = per_image_dir / f"{subtask.key}.csv"
        spec_path = per_image_dir / f"{subtask.key}.field_specs.json"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        dfs.append(pd.read_csv(csv_path, low_memory=False))
        for _field_name, spec in load_field_specs_from_json(spec_path).items():
            merge_field_spec(field_specs, spec)
    manifest = pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame(columns=ID_COLUMNS)
    if not manifest.empty:
        manifest = manifest.loc[:, ordered_columns(manifest.columns)]
        manifest = manifest.sort_values(["split", "frame_type", "image_rel_path"]).reset_index(drop=True)
    return manifest, field_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CAG pretraining dataset metadata statistics.")
    parser.add_argument("--image-root", type=Path, default=Path("/mnt/nas/snubhcvc/project/cag_fm/pretrain/datasets/images"))
    parser.add_argument("--dicom-root", type=Path, default=Path("/mnt/nas/snubhcvc/raw/cpacs"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/cag_pretrain_metadata_stats"))
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-images-per-subtask", type=int, default=0)
    parser.add_argument("--refresh-enumeration-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    per_image_dir = output_root / "per_image"
    ensure_dir(per_image_dir)
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(output_root, "analyze_cag_pretrain_metadata")
    try:
        log(
            "Arguments: "
            + json.dumps(
                {
                    "image_root": str(args.image_root),
                    "dicom_root": str(args.dicom_root),
                    "output_root": str(output_root),
                    "splits": list(args.splits),
                    "num_workers": int(args.num_workers),
                    "log_every": int(args.log_every),
                    "resume": bool(args.resume),
                    "max_images_per_subtask": int(args.max_images_per_subtask),
                    "refresh_enumeration_cache": bool(args.refresh_enumeration_cache),
                    "log_path": str(log_path),
                },
                indent=2,
            )
        )

        stage_start = time.time()
        log("ENUMERATE START")
        subtasks = build_subtasks(args.image_root.resolve(), list(args.splits), int(args.max_images_per_subtask), output_root, bool(args.refresh_enumeration_cache))
        total_images = sum(subtask.image_count for subtask in subtasks)
        log(f"ENUMERATE DONE | subtasks={len(subtasks)} | total_images={total_images} | elapsed={format_duration(time.time() - stage_start)}")

        stage_start = time.time()
        log("EXTRACT START")
        total_done_ref = [0, total_images]
        for subtask in subtasks:
            csv_path = per_image_dir / f"{subtask.key}.csv"
            spec_path = per_image_dir / f"{subtask.key}.field_specs.json"
            if args.resume and csv_path.exists() and spec_path.exists():
                total_done_ref[0] += subtask.image_count
                log(f"[SUBTASK] SKIP | split={subtask.split} frame_type={subtask.frame_type} rows={subtask.image_count} | reason=resume")
                log_total_progress(stage_start, total_done_ref[0], total_done_ref[1])
                continue
            extract_one_subtask(
                subtask=subtask,
                image_root=args.image_root.resolve(),
                dicom_root=args.dicom_root.resolve(),
                per_image_dir=per_image_dir,
                num_workers=int(args.num_workers),
                log_every=int(args.log_every),
                total_start=stage_start,
                total_done_ref=total_done_ref,
            )
        log(f"EXTRACT DONE | elapsed={format_duration(time.time() - stage_start)}")

        stage_start = time.time()
        log("SUMMARIZE START")
        manifest, field_specs = combine_subtask_outputs(subtasks, per_image_dir)
        save_dataframe_atomic(manifest, per_image_dir / "metadata_manifest_all.csv")

        field_catalog = build_field_catalog(manifest, field_specs)
        save_dataframe_atomic(field_catalog, per_image_dir / "field_catalog.csv")

        counts_by_split = summarize_counts(manifest, ["split"])
        save_dataframe_atomic(counts_by_split, output_root / "summary_counts_by_split.csv")

        counts_by_split_frame_type = summarize_counts(manifest, ["split", "frame_type"])
        save_dataframe_atomic(counts_by_split_frame_type, output_root / "summary_counts_by_split_frame_type.csv")

        numeric_fields = sorted(spec.field_name for spec in field_specs.values() if spec.field_type == "continuous" and spec.field_name in manifest.columns)
        categorical_fields = sorted(spec.field_name for spec in field_specs.values() if spec.field_type == "categorical" and spec.field_name in manifest.columns)

        numeric_by_split = summarize_numeric(manifest, numeric_fields, ["split"])
        save_dataframe_atomic(numeric_by_split, output_root / "summary_numeric_by_split.csv")

        numeric_by_split_frame_type = summarize_numeric(manifest, numeric_fields, ["split", "frame_type"])
        save_dataframe_atomic(numeric_by_split_frame_type, output_root / "summary_numeric_by_split_frame_type.csv")

        categorical_by_split = summarize_categorical(manifest, categorical_fields, ["split"])
        save_dataframe_atomic(categorical_by_split, output_root / "summary_categorical_all_values_by_split.csv")

        categorical_by_split_frame_type = summarize_categorical(manifest, categorical_fields, ["split", "frame_type"])
        save_dataframe_atomic(categorical_by_split_frame_type, output_root / "summary_categorical_all_values_by_split_frame_type.csv")

        status_counts = build_status_summary(manifest)
        save_dataframe_atomic(status_counts, output_root / "summary_status_counts.csv")

        missingness = build_missingness_summary(manifest, field_specs)
        save_dataframe_atomic(missingness, output_root / "summary_missingness_by_field.csv")

        failures = manifest[manifest["status"] != "ok"].copy()
        if not failures.empty:
            failures = failures.loc[:, [col for col in ID_COLUMNS if col in failures.columns]]
        save_dataframe_atomic(failures, output_root / "summary_dicom_resolution_failures.csv")
        log(f"SUMMARIZE DONE | elapsed={format_duration(time.time() - stage_start)}")

        stage_start = time.time()
        log("REPORT START")
        render_markdown_report(
            output_root=output_root,
            manifest=manifest,
            field_catalog=field_catalog,
            counts_by_split=counts_by_split,
            counts_by_split_frame_type=counts_by_split_frame_type,
            status_counts=status_counts,
            missingness=missingness,
            args=args,
        )
        log(f"REPORT DONE | elapsed={format_duration(time.time() - stage_start)}")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
