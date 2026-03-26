#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TextIO

import pandas as pd

import analyze_cag_pretrain_metadata as common


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
STENT_LABELS = ["stent", "no_stent"]
ID_COLUMNS = [
    "split",
    "stent_label",
    "image_path",
    "image_rel_path",
    "patient_id",
    "study_date",
    "series_dir",
    "dicom_id",
    "dicom_path",
    "status",
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


def log(message: str) -> None:
    common.log(message)


def format_duration(seconds: float) -> str:
    return common.format_duration(seconds)


def ensure_dir(path: Path) -> None:
    common.ensure_dir(path)


def ordered_columns(columns: Iterable[str]) -> list[str]:
    others = sorted(col for col in columns if col not in ID_COLUMNS)
    return [col for col in ID_COLUMNS if col in columns] + others


def setup_console_and_file_logging(output_root: Path, default_prefix: str) -> tuple[Path, TextIO, TextIO, TextIO]:
    return common.setup_console_and_file_logging(output_root, default_prefix)


def restore_console_logging(file_handle: TextIO, original_stdout: TextIO, original_stderr: TextIO) -> None:
    common.restore_console_logging(file_handle, original_stdout, original_stderr)


def estimate_remaining(elapsed_seconds: float, done: int, total: int) -> float:
    return common.estimate_remaining(elapsed_seconds, done, total)


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
        f"[SUBTASK] split={subtask.split} stent_label={subtask.stent_label} done={done}/{total} | "
        f"elapsed={format_duration(elapsed)} | remaining={remaining_text}"
    )


def atomic_write_json(path: Path, payload: object) -> None:
    common.atomic_write_json(path, payload)


def save_dataframe_atomic(df: pd.DataFrame, path: Path) -> None:
    common.save_dataframe_atomic(df, path)


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
    if not image_name.lower().endswith(".png"):
        return None
    return {
        "split": split,
        "stent_label": stent_label,
        "patient_id": patient_id,
        "study_date": study_date,
        "series_dir": series_dir,
        "dicom_id": Path(image_name).stem,
    }


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
        "stent_labels": list(STENT_LABELS),
        "max_images_per_subtask": int(max_images_per_subtask),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            log(f"ENUMERATE CACHE MISS | reason=config_mismatch | key={key}")
            return None
    subtasks_payload = payload.get("subtasks", [])
    subtasks: list[SubtaskDef] = []
    load_start = time.time()
    total_expected = len(subtasks_payload)
    for idx, item in enumerate(subtasks_payload, start=1):
        split = item["split"]
        stent_label = item["stent_label"]
        image_dir = Path(item["image_dir"])
        manifest_file = cache_dir / item["manifest_file"]
        if not manifest_file.exists():
            log(f"ENUMERATE CACHE MISS | reason=missing_manifest | split={split} stent_label={stent_label}")
            return None
        names = manifest_file.read_text(encoding="utf-8").splitlines()
        image_paths = tuple(image_dir / name for name in names if name)
        subtasks.append(SubtaskDef(split=split, stent_label=stent_label, image_dir=image_dir, image_paths=image_paths))
        elapsed = time.time() - load_start
        remaining = estimate_remaining(elapsed, idx, total_expected)
        log(
            f"[ENUMERATE] CACHE | split={split} stent_label={stent_label} | images={len(image_paths)} | "
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
        manifest_path.write_text(
            "\n".join(str(path.relative_to(subtask.image_dir)) for path in subtask.image_paths) + "\n",
            encoding="utf-8",
        )
        payload_subtasks.append(
            {
                "split": subtask.split,
                "stent_label": subtask.stent_label,
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
            "stent_labels": list(STENT_LABELS),
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
    total_expected = len(splits) * len(STENT_LABELS)
    enumerate_start = time.time()
    subtasks_done = 0
    for split in splits:
        for stent_label in STENT_LABELS:
            image_dir = image_root / split / stent_label
            log(f"[ENUMERATE] START | split={split} stent_label={stent_label} | subtask={subtasks_done + 1}/{total_expected}")
            if not image_dir.exists():
                subtasks_done += 1
                remaining = estimate_remaining(time.time() - enumerate_start, subtasks_done, total_expected)
                log(
                    f"[ENUMERATE] SKIP | split={split} stent_label={stent_label} | reason=missing_dir | "
                    f"subtasks_done={subtasks_done}/{total_expected} | remaining={format_duration(remaining) if math.isfinite(remaining) else 'n/a'}"
                )
                continue
            png_iter = (p for p in image_dir.rglob("*.png") if p.is_file())
            if max_images_per_subtask > 0:
                selected: list[Path] = []
                for path_entry in png_iter:
                    selected.append(path_entry)
                    if len(selected) >= max_images_per_subtask:
                        break
                image_paths = tuple(selected)
            else:
                image_paths = tuple(sorted(png_iter, key=lambda p: str(p.relative_to(image_dir))))
            subtasks.append(SubtaskDef(split=split, stent_label=stent_label, image_dir=image_dir, image_paths=image_paths))
            subtasks_done += 1
            elapsed = time.time() - enumerate_start
            remaining = estimate_remaining(elapsed, subtasks_done, total_expected)
            log(
                f"[ENUMERATE] DONE | split={split} stent_label={stent_label} | images={len(image_paths)} | "
                f"subtasks_done={subtasks_done}/{total_expected} | elapsed={format_duration(elapsed)} | "
                f"remaining={format_duration(remaining) if math.isfinite(remaining) else 'n/a'}"
            )
    save_enumeration_cache(output_root, image_root, splits, max_images_per_subtask, subtasks)
    return subtasks


def process_single_image(args: tuple[Path, Path, str, str, Path]) -> tuple[dict[str, object], list[dict[str, object]]]:
    img_path, image_root, split, stent_label, dicom_root = args
    row: dict[str, object] = {
        "split": split,
        "stent_label": stent_label,
        "image_path": str(img_path.resolve()),
        "image_rel_path": str(img_path.resolve().relative_to(image_root.resolve())),
        "patient_id": "",
        "study_date": "",
        "series_dir": "",
        "dicom_id": "",
        "dicom_path": "",
        "status": "ok",
        "error_message": "",
    }
    field_specs: Dict[str, common.FieldSpec] = {}
    image_stats, image_specs = common.compute_basic_image_attributes(img_path)
    for spec in image_specs.values():
        common.merge_field_spec(field_specs, spec)
    row.update(image_stats)

    rel_path = img_path.resolve().relative_to(image_root.resolve())
    parsed = parse_image_rel_path(rel_path)
    if parsed is None:
        row["status"] = "bad_image_rel_path"
        row["error_message"] = f"Unrecognized relative path pattern: {rel_path}"
        return row, [asdict(spec) for spec in field_specs.values()]

    row["patient_id"] = parsed["patient_id"]
    row["study_date"] = parsed["study_date"]
    row["series_dir"] = parsed["series_dir"]
    row["dicom_id"] = parsed["dicom_id"]
    dcm_path = dicom_root / split / stent_label / parsed["patient_id"] / parsed["study_date"] / parsed["series_dir"] / f"{parsed['dicom_id']}.dcm"
    row["dicom_path"] = str(dcm_path.resolve())

    if not dcm_path.exists():
        row["status"] = "missing_dicom"
        row["error_message"] = f"Missing DICOM: {dcm_path}"
        return row, [asdict(spec) for spec in field_specs.values()]

    try:
        meta, dicom_specs = common.extract_dicom_scalar_fields(dcm_path)
    except Exception as exc:
        row["status"] = "dicom_read_error"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        return row, [asdict(spec) for spec in field_specs.values()]

    for spec in dicom_specs.values():
        common.merge_field_spec(field_specs, spec)
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
    merged_specs: Dict[str, common.FieldSpec] = {}
    subtask_total = subtask.image_count
    subtask_done = 0
    subtask_start = time.time()
    log(f"[SUBTASK] START | split={subtask.split} stent_label={subtask.stent_label} total={subtask_total}")

    jobs = [(img_path, image_root, subtask.split, subtask.stent_label, dicom_root) for img_path in subtask.image_paths]
    with cf.ThreadPoolExecutor(max_workers=max(1, int(num_workers))) as executor:
        for row, specs in executor.map(process_single_image, jobs):
            rows.append(row)
            for raw_spec in specs:
                spec = common.FieldSpec(**raw_spec)
                common.merge_field_spec(merged_specs, spec)
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
        f"[SUBTASK] DONE | split={subtask.split} stent_label={subtask.stent_label} rows={len(df)} | "
        f"elapsed={format_duration(time.time() - subtask_start)}"
    )
    return csv_path, spec_path


def combine_subtask_outputs(subtasks: Sequence[SubtaskDef], per_image_dir: Path) -> tuple[pd.DataFrame, Dict[str, common.FieldSpec]]:
    dfs: list[pd.DataFrame] = []
    field_specs: Dict[str, common.FieldSpec] = {}
    for subtask in subtasks:
        csv_path = per_image_dir / f"{subtask.key}.csv"
        spec_path = per_image_dir / f"{subtask.key}.field_specs.json"
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        dfs.append(pd.read_csv(csv_path, low_memory=False))
        for _field_name, spec in common.load_field_specs_from_json(spec_path).items():
            common.merge_field_spec(field_specs, spec)
    manifest = pd.concat(dfs, ignore_index=True, sort=False) if dfs else pd.DataFrame(columns=ID_COLUMNS)
    if not manifest.empty:
        manifest = manifest.loc[:, ordered_columns(manifest.columns)]
        manifest = manifest.sort_values(["split", "stent_label", "image_rel_path"]).reset_index(drop=True)
    return manifest, field_specs


def summarize_counts(manifest: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    empty_columns = group_cols + [
        "image_count",
        "ok_count",
        "bad_image_rel_path_count",
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
                "bad_image_rel_path_count": int((group["status"] == "bad_image_rel_path").sum()),
                "missing_dicom_count": int((group["status"] == "missing_dicom").sum()),
                "dicom_read_error_count": int((group["status"] == "dicom_read_error").sum()),
                "unique_patient_count": int(group["patient_id"].replace("", pd.NA).dropna().nunique()),
                "unique_study_count": int(group["study_date"].replace("", pd.NA).dropna().nunique()),
                "unique_dicom_count": int(group["dicom_id"].replace("", pd.NA).dropna().nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def build_status_summary(manifest: pd.DataFrame) -> pd.DataFrame:
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


def render_markdown_report(
    output_root: Path,
    manifest: pd.DataFrame,
    field_catalog: pd.DataFrame,
    counts_by_split: pd.DataFrame,
    counts_by_split_stent_label: pd.DataFrame,
    status_counts: pd.DataFrame,
    missingness: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    ok_manifest = manifest[manifest["status"] == "ok"].copy()
    lines: list[str] = []
    lines.append("# Stent-Contrast Unique-View Metadata Stats")
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
    lines.append(common.dataframe_to_markdown(counts_by_split))
    lines.append("")
    lines.append("## Counts By Split And Stent Label")
    lines.append("")
    lines.append(common.dataframe_to_markdown(counts_by_split_stent_label))
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    lines.append(common.dataframe_to_markdown(status_counts))
    lines.append("")
    lines.append("## Highest Missingness Fields")
    lines.append("")
    lines.append(common.dataframe_to_markdown(missingness.head(30)))
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `per_image/metadata_manifest_all.csv`")
    lines.append("- `per_image/field_catalog.csv`")
    lines.append("- `summary_counts_by_split.csv`")
    lines.append("- `summary_counts_by_split_stent_label.csv`")
    lines.append("- `summary_numeric_by_split.csv`")
    lines.append("- `summary_numeric_by_split_stent_label.csv`")
    lines.append("- `summary_categorical_all_values_by_split.csv`")
    lines.append("- `summary_categorical_all_values_by_split_stent_label.csv`")
    lines.append("- `summary_status_counts.csv`")
    lines.append("- `summary_missingness_by_field.csv`")
    lines.append("- `summary_dicom_resolution_failures.csv`")
    lines.append("- `enumeration_cache/meta.json`")
    lines.append("")
    (output_root / "analysis_stent_contrast_unique_view_metadata.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Stent-Contrast unique-view dataset metadata statistics.")
    parser.add_argument("--image-root", type=Path, default=Path("input/Stent-Contrast-unique-view"))
    parser.add_argument("--dicom-root", type=Path, default=Path("input/stent_split_dcm_unique_view"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stent_contrast_unique_view_metadata_stats"))
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
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
    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(output_root, "analyze_stent_contrast_unique_view_metadata")
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
        subtasks = build_subtasks(
            image_root=args.image_root.resolve(),
            splits=list(args.splits),
            max_images_per_subtask=int(args.max_images_per_subtask),
            output_root=output_root,
            refresh_enumeration_cache=bool(args.refresh_enumeration_cache),
        )
        total_images = int(sum(subtask.image_count for subtask in subtasks))
        log(f"ENUMERATE DONE | subtasks={len(subtasks)} | total_images={total_images} | elapsed={format_duration(time.time() - stage_start)}")

        stage_start = time.time()
        log("EXTRACT START")
        total_done_ref = [0, total_images]
        completed = 0
        for subtask in subtasks:
            csv_path = per_image_dir / f"{subtask.key}.csv"
            spec_path = per_image_dir / f"{subtask.key}.field_specs.json"
            if args.resume and csv_path.exists() and spec_path.exists():
                existing_df = pd.read_csv(csv_path, low_memory=False)
                row_count = int(len(existing_df))
                total_done_ref[0] += row_count
                completed += 1
                log(
                    f"[SUBTASK] RESUME | split={subtask.split} stent_label={subtask.stent_label} | "
                    f"rows={row_count} | completed={completed}/{len(subtasks)}"
                )
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
            completed += 1
        log(f"EXTRACT DONE | subtasks={len(subtasks)} | elapsed={format_duration(time.time() - stage_start)}")

        stage_start = time.time()
        log("SUMMARIZE START")
        manifest, field_specs = combine_subtask_outputs(subtasks, per_image_dir)
        save_dataframe_atomic(manifest, per_image_dir / "metadata_manifest_all.csv")
        field_catalog = common.build_field_catalog(manifest, field_specs)
        save_dataframe_atomic(field_catalog, per_image_dir / "field_catalog.csv")

        numeric_fields = [spec.field_name for spec in field_specs.values() if spec.field_type == "continuous"]
        categorical_fields = [spec.field_name for spec in field_specs.values() if spec.field_type == "categorical"]

        counts_by_split = summarize_counts(manifest, ["split"])
        counts_by_split_stent_label = summarize_counts(manifest, ["split", "stent_label"])
        numeric_by_split = common.summarize_numeric(manifest, numeric_fields, ["split"])
        numeric_by_split_stent_label = common.summarize_numeric(manifest, numeric_fields, ["split", "stent_label"])
        categorical_by_split = common.summarize_categorical(manifest, categorical_fields, ["split"])
        categorical_by_split_stent_label = common.summarize_categorical(manifest, categorical_fields, ["split", "stent_label"])
        status_counts = build_status_summary(manifest)
        missingness = common.build_missingness_summary(manifest, field_specs)
        resolution_failures = manifest.loc[manifest["status"] != "ok", ID_COLUMNS].copy() if not manifest.empty else pd.DataFrame(columns=ID_COLUMNS)

        save_dataframe_atomic(counts_by_split, output_root / "summary_counts_by_split.csv")
        save_dataframe_atomic(counts_by_split_stent_label, output_root / "summary_counts_by_split_stent_label.csv")
        save_dataframe_atomic(numeric_by_split, output_root / "summary_numeric_by_split.csv")
        save_dataframe_atomic(numeric_by_split_stent_label, output_root / "summary_numeric_by_split_stent_label.csv")
        save_dataframe_atomic(categorical_by_split, output_root / "summary_categorical_all_values_by_split.csv")
        save_dataframe_atomic(categorical_by_split_stent_label, output_root / "summary_categorical_all_values_by_split_stent_label.csv")
        save_dataframe_atomic(status_counts, output_root / "summary_status_counts.csv")
        save_dataframe_atomic(missingness, output_root / "summary_missingness_by_field.csv")
        save_dataframe_atomic(resolution_failures, output_root / "summary_dicom_resolution_failures.csv")
        log(f"SUMMARIZE DONE | elapsed={format_duration(time.time() - stage_start)}")

        stage_start = time.time()
        log("REPORT START")
        render_markdown_report(
            output_root=output_root,
            manifest=manifest,
            field_catalog=field_catalog,
            counts_by_split=counts_by_split,
            counts_by_split_stent_label=counts_by_split_stent_label,
            status_counts=status_counts,
            missingness=missingness,
            args=args,
        )
        log(f"REPORT DONE | elapsed={format_duration(time.time() - stage_start)}")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
