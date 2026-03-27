#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pydicom

REPO_ROOT = Path(__file__).resolve().parents[3]
TARGET_MODEL_NAME_NORMALIZED = "P H I L I P S INTEGRIS H"
PNG_EXT = ".png"
DCM_EXT = ".dcm"
STENT_LABELS = ["stent", "no_stent"]
ID_COLUMNS = [
    "split",
    "stent_label",
    "patient_id",
    "study_date",
    "series_dir",
    "dicom_id",
    "image_rel_path",
    "dicom_rel_path",
    "status",
    "manufacturer_model_name_raw",
    "manufacturer_model_name_normalized",
    "output_image_path",
    "output_dicom_path",
    "error_message",
]


@dataclass(frozen=True)
class SubtaskDef:
    split: str
    stent_label: str
    image_dir: Path
    image_paths: tuple[Path, ...]

    @property
    def key(self) -> str:
        return f"{self.split}__{self.stent_label}"

    @property
    def image_count(self) -> int:
        return len(self.image_paths)


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:04.1f}s"
    total = int(round(seconds))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def estimate_remaining(elapsed_seconds: float, done: int, total: int) -> float:
    if total <= 0 or done <= 0:
        return float("nan")
    avg = elapsed_seconds / max(done, 1)
    return max(0.0, avg * (total - done))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def setup_logging(output_root: Path) -> tuple[Path, object, object, object]:
    ensure_dir(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_root / f"build_global_5_philips_integris_h_unique_view_subset_{stamp}.log"
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_out = sys.stdout
    orig_err = sys.stderr
    sys.stdout = TeeStream(orig_out, fh)
    sys.stderr = TeeStream(orig_err, fh)
    log(f"Console output is mirrored to log file: {log_path}")
    return log_path, fh, orig_out, orig_err


def restore_logging(fh: object, orig_out: object, orig_err: object) -> None:
    sys.stdout = orig_out
    sys.stderr = orig_err
    fh.close()


def normalize_manufacturer_model_name(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).upper().split())


def parse_image_rel_path(rel_path: Path) -> dict[str, str] | None:
    parts = rel_path.parts
    if len(parts) != 6:
        return None
    split, stent_label, patient_id, study_date, series_dir, image_name = parts
    if split not in {"train", "valid", "test"}:
        return None
    if stent_label not in set(STENT_LABELS):
        return None
    if series_dir != "XA":
        return None
    if Path(image_name).suffix.lower() != PNG_EXT:
        return None
    return {
        "split": split,
        "stent_label": stent_label,
        "patient_id": patient_id,
        "study_date": study_date,
        "series_dir": series_dir,
        "dicom_id": Path(image_name).stem,
    }


def build_subtasks(image_root: Path) -> list[SubtaskDef]:
    subtasks: list[SubtaskDef] = []
    total_expected = 3 * len(STENT_LABELS)
    enumerate_start = time.time()
    done = 0
    for split in ["train", "valid", "test"]:
        for stent_label in STENT_LABELS:
            image_dir = image_root / split / stent_label
            log(f"[ENUMERATE] START | split={split} stent_label={stent_label} | subtask={done + 1}/{total_expected}")
            if not image_dir.exists():
                done += 1
                remaining = estimate_remaining(time.time() - enumerate_start, done, total_expected)
                log(
                    f"[ENUMERATE] SKIP | split={split} stent_label={stent_label} | reason=missing_dir | "
                    f"subtasks_done={done}/{total_expected} | remaining={format_duration(remaining)}"
                )
                continue
            image_paths = tuple(sorted((p for p in image_dir.rglob(f"*{PNG_EXT}") if p.is_file()), key=lambda p: str(p.relative_to(image_dir))))
            subtasks.append(SubtaskDef(split=split, stent_label=stent_label, image_dir=image_dir, image_paths=image_paths))
            done += 1
            elapsed = time.time() - enumerate_start
            remaining = estimate_remaining(elapsed, done, total_expected)
            log(
                f"[ENUMERATE] DONE | split={split} stent_label={stent_label} | images={len(image_paths)} | "
                f"subtasks_done={done}/{total_expected} | elapsed={format_duration(elapsed)} | remaining={format_duration(remaining)}"
            )
    return subtasks


def log_total_progress(start_time: float, done: int, total: int) -> None:
    elapsed = max(0.0, time.time() - start_time)
    remaining = estimate_remaining(elapsed, done, total)
    progress = 0.0 if total <= 0 else (100.0 * done / total)
    eta = datetime.now() + timedelta(seconds=0.0 if not math.isfinite(remaining) else remaining)
    log(
        f"[TOTAL] done={done}/{total} | progress={progress:.1f}% | elapsed={format_duration(elapsed)} | "
        f"remaining={format_duration(remaining)} | eta={eta.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def read_manufacturer_model_name(dcm_path: Path, cache: dict[Path, tuple[str, str]]) -> tuple[str, str]:
    cached = cache.get(dcm_path)
    if cached is not None:
        return cached
    ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, specific_tags=["ManufacturerModelName"])
    raw = str(getattr(ds, "ManufacturerModelName", "") or "")
    normalized = normalize_manufacturer_model_name(raw)
    cache[dcm_path] = (raw, normalized)
    return raw, normalized


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


def ordered_columns(columns: Iterable[str]) -> list[str]:
    others = sorted(col for col in columns if col not in ID_COLUMNS)
    return [col for col in ID_COLUMNS if col in columns] + others


def summarize_selected(manifest: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    empty_columns = group_cols + [
        "image_count",
        "unique_patient_count",
        "unique_study_count",
        "unique_dicom_count",
    ]
    selected = manifest[manifest["status"] == "selected"].copy()
    if selected.empty:
        return pd.DataFrame(columns=empty_columns)
    rows: list[dict[str, object]] = []
    for keys, group in selected.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "image_count": int(len(group)),
                "unique_patient_count": int(group["patient_id"].replace("", pd.NA).dropna().nunique()),
                "unique_study_count": int(group["study_date"].replace("", pd.NA).dropna().nunique()),
                "unique_dicom_count": int(group["dicom_rel_path"].replace("", pd.NA).dropna().nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_status(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["split", "stent_label", "status", "count"])
    parts: list[pd.DataFrame] = []
    for group_cols in [["split", "stent_label"], ["split"], []]:
        if group_cols:
            grouped = manifest.groupby(group_cols + ["status"], dropna=False).size().reset_index(name="count")
        else:
            grouped = manifest.groupby(["status"], dropna=False).size().reset_index(name="count")
            grouped.insert(0, "stent_label", "__all__")
            grouped.insert(0, "split", "__all__")
            parts.append(grouped)
            continue
        if "split" not in grouped.columns:
            grouped.insert(0, "split", "__all__")
        if "stent_label" not in grouped.columns:
            grouped.insert(1, "stent_label", "__all__")
        parts.append(grouped)
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["split", "stent_label", "status"]).reset_index(drop=True)


def render_markdown_report(output_root: Path, manifest: pd.DataFrame, counts_by_split: pd.DataFrame, counts_by_split_stent_label: pd.DataFrame, status_counts: pd.DataFrame, args: argparse.Namespace) -> None:
    selected = manifest[manifest["status"] == "selected"].copy()
    lines: list[str] = []
    lines.append("# Global Analysis 5 Single-Device Subset")
    lines.append("")
    lines.append("## Run Config")
    lines.append("")
    lines.append(f"- image_root: `{args.image_root}`")
    lines.append(f"- dicom_root: `{args.dicom_root}`")
    lines.append(f"- output_root: `{output_root}`")
    lines.append(f"- target_model_name_normalized: `{TARGET_MODEL_NAME_NORMALIZED}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- source_images: `{len(manifest)}`")
    lines.append(f"- selected_images: `{int((manifest['status'] == 'selected').sum())}`")
    lines.append(f"- selected_patients: `{selected['patient_id'].replace('', pd.NA).dropna().nunique()}`")
    lines.append(f"- selected_studies: `{selected['study_date'].replace('', pd.NA).dropna().nunique()}`")
    lines.append(f"- selected_dicoms: `{selected['dicom_rel_path'].replace('', pd.NA).dropna().nunique()}`")
    lines.append("")
    lines.append("## Selected Counts By Split")
    lines.append("")
    lines.append(dataframe_to_markdown(counts_by_split))
    lines.append("")
    lines.append("## Selected Counts By Split And Stent Label")
    lines.append("")
    lines.append(dataframe_to_markdown(counts_by_split_stent_label))
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    lines.append(dataframe_to_markdown(status_counts))
    lines.append("")
    (output_root / "analysis_global_5_philips_integris_h_unique_view_subset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a single-device subset of Stent-Contrast unique-view for Global Analysis 5.")
    parser.add_argument("--image-root", type=Path, default=REPO_ROOT / "input/Stent-Contrast-unique-view")
    parser.add_argument("--dicom-root", type=Path, default=REPO_ROOT / "input/stent_split_dcm_unique_view")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "input/global_analysis_5_per_device_patient_retrieval/philips_integris_h_unique_view_subset")
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    images_out = output_root / "images"
    dicoms_out = output_root / "dicoms"
    log_path, fh, orig_out, orig_err = setup_logging(output_root)
    try:
        log(
            "Arguments: "
            + json.dumps(
                {
                    "image_root": str(args.image_root),
                    "dicom_root": str(args.dicom_root),
                    "output_root": str(output_root),
                    "log_every": int(args.log_every),
                    "overwrite": bool(args.overwrite),
                    "target_model_name_normalized": TARGET_MODEL_NAME_NORMALIZED,
                    "log_path": str(log_path),
                },
                indent=2,
            )
        )

        ensure_clean_dir(images_out, overwrite=bool(args.overwrite))
        ensure_clean_dir(dicoms_out, overwrite=bool(args.overwrite))

        enum_start = time.time()
        log("ENUMERATE START")
        subtasks = build_subtasks(args.image_root.resolve())
        total_images = int(sum(subtask.image_count for subtask in subtasks))
        log(f"ENUMERATE DONE | subtasks={len(subtasks)} | total_images={total_images} | elapsed={format_duration(time.time() - enum_start)}")

        build_start = time.time()
        log("BUILD START")
        rows: list[dict[str, object]] = []
        manufacturer_cache: dict[Path, tuple[str, str]] = {}
        done = 0
        selected_count = 0
        status_counter: Counter[str] = Counter()
        for subtask in subtasks:
            subtask_start = time.time()
            subtask_done = 0
            log(f"[SUBTASK] START | split={subtask.split} stent_label={subtask.stent_label} total={subtask.image_count}")
            for img_path in subtask.image_paths:
                rel_path = img_path.resolve().relative_to(args.image_root.resolve())
                parsed = parse_image_rel_path(rel_path)
                row = {
                    "split": subtask.split,
                    "stent_label": subtask.stent_label,
                    "patient_id": "",
                    "study_date": "",
                    "series_dir": "",
                    "dicom_id": "",
                    "image_rel_path": str(rel_path),
                    "dicom_rel_path": "",
                    "status": "selected",
                    "manufacturer_model_name_raw": "",
                    "manufacturer_model_name_normalized": "",
                    "output_image_path": "",
                    "output_dicom_path": "",
                    "error_message": "",
                }
                if parsed is None:
                    row["status"] = "bad_image_rel_path"
                    row["error_message"] = f"Unrecognized relative path pattern: {rel_path}"
                else:
                    row.update(
                        {
                            "patient_id": parsed["patient_id"],
                            "study_date": parsed["study_date"],
                            "series_dir": parsed["series_dir"],
                            "dicom_id": parsed["dicom_id"],
                        }
                    )
                    dicom_rel = rel_path.with_suffix(DCM_EXT)
                    row["dicom_rel_path"] = str(dicom_rel)
                    dcm_path = (args.dicom_root.resolve() / dicom_rel)
                    if not dcm_path.exists():
                        row["status"] = "missing_dicom"
                        row["error_message"] = f"Missing DICOM: {dcm_path}"
                    else:
                        try:
                            raw_name, normalized_name = read_manufacturer_model_name(dcm_path, manufacturer_cache)
                        except Exception as exc:
                            row["status"] = "dicom_read_error"
                            row["error_message"] = f"{type(exc).__name__}: {exc}"
                        else:
                            row["manufacturer_model_name_raw"] = raw_name
                            row["manufacturer_model_name_normalized"] = normalized_name
                            if normalized_name != TARGET_MODEL_NAME_NORMALIZED:
                                row["status"] = "model_mismatch"
                            else:
                                out_img = images_out / rel_path
                                out_dcm = dicoms_out / dicom_rel
                                ensure_dir(out_img.parent)
                                ensure_dir(out_dcm.parent)
                                shutil.copy2(img_path, out_img)
                                shutil.copy2(dcm_path, out_dcm)
                                row["output_image_path"] = str(out_img.resolve())
                                row["output_dicom_path"] = str(out_dcm.resolve())
                                selected_count += 1
                rows.append(row)
                status_counter[row["status"]] += 1
                done += 1
                subtask_done += 1
                if done == total_images or (args.log_every > 0 and done % args.log_every == 0):
                    elapsed = time.time() - build_start
                    remaining = estimate_remaining(elapsed, done, total_images)
                    total_eta = datetime.now() + timedelta(seconds=0.0 if not math.isfinite(remaining) else remaining)
                    sub_elapsed = time.time() - subtask_start
                    sub_remaining = estimate_remaining(sub_elapsed, subtask_done, subtask.image_count)
                    log(
                        f"[SUBTASK] split={subtask.split} stent_label={subtask.stent_label} done={subtask_done}/{subtask.image_count} | "
                        f"elapsed={format_duration(sub_elapsed)} | remaining={format_duration(sub_remaining)}"
                    )
                    log(
                        f"[TOTAL] done={done}/{total_images} | progress={(100.0 * done / total_images):.1f}% | "
                        f"selected={selected_count} | elapsed={format_duration(elapsed)} | remaining={format_duration(remaining)} | "
                        f"eta={total_eta.strftime('%Y-%m-%d %H:%M:%S')}"
                    )
            log(
                f"[SUBTASK] DONE | split={subtask.split} stent_label={subtask.stent_label} | selected_so_far={selected_count} | "
                f"elapsed={format_duration(time.time() - subtask_start)}"
            )
        log(f"BUILD DONE | total_images={total_images} | selected={selected_count} | elapsed={format_duration(time.time() - build_start)}")

        manifest = pd.DataFrame(rows)
        if not manifest.empty:
            manifest = manifest.loc[:, ordered_columns(manifest.columns)]
            manifest = manifest.sort_values(["split", "stent_label", "image_rel_path"]).reset_index(drop=True)
        selected_manifest = manifest[manifest["status"] == "selected"].copy()
        counts_by_split = summarize_selected(manifest, ["split"])
        counts_by_split_stent_label = summarize_selected(manifest, ["split", "stent_label"])
        status_counts = summarize_status(manifest)

        manifest.to_csv(output_root / "selection_audit.csv", index=False)
        selected_manifest.to_csv(output_root / "manifest_selected.csv", index=False)
        counts_by_split.to_csv(output_root / "summary_selected_counts_by_split.csv", index=False)
        counts_by_split_stent_label.to_csv(output_root / "summary_selected_counts_by_split_stent_label.csv", index=False)
        status_counts.to_csv(output_root / "summary_selection_status.csv", index=False)
        (output_root / "build_metadata.json").write_text(
            json.dumps(
                {
                    "image_root": str(args.image_root.resolve()),
                    "dicom_root": str(args.dicom_root.resolve()),
                    "output_root": str(output_root),
                    "target_model_name_normalized": TARGET_MODEL_NAME_NORMALIZED,
                    "source_image_count": int(total_images),
                    "selected_image_count": int(selected_count),
                    "status_counts": dict(status_counter),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        render_markdown_report(output_root, manifest, counts_by_split, counts_by_split_stent_label, status_counts, args)
    finally:
        restore_logging(fh, orig_out, orig_err)


if __name__ == "__main__":
    main()
