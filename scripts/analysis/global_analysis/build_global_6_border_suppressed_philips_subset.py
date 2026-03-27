#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

PREPROCESS_DIR = REPO_ROOT / "scripts/exp/fm_improve_exp1-input_policy/preprocess"
if str(PREPROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PREPROCESS_DIR))

from border_suppression_common import (  # noqa: E402
    DEFAULT_BLUR_SIGMA,
    DEFAULT_CENTER_X,
    DEFAULT_CENTER_Y,
    DEFAULT_FEATHER_WIDTH,
    DEFAULT_RADIUS,
    apply_border_suppression,
    make_three_panel,
    overlay_mask_boundary,
    safe_name_from_rel_path,
)
from input_standardization_common import read_grayscale, write_grayscale  # noqa: E402


TARGET_IMAGE_SHAPE = (512, 512)
DEFAULT_BASELINE_ROOT = REPO_ROOT / "input/global_analysis_6_patient_retrieval_border_suppressed_philips/baseline_philips_unique_view_subset"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "input/global_analysis_6_patient_retrieval_border_suppressed_philips/border_suppressed_philips_unique_view_subset"
PREVIEW_LIMIT = 3


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


def setup_logging(output_root: Path, log_prefix: str):
    ensure_dir(output_root)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_root / f"{log_prefix}_{stamp}.log"
    fh = open(log_path, "a", encoding="utf-8", buffering=1)
    orig_out = sys.stdout
    orig_err = sys.stderr
    sys.stdout = TeeStream(orig_out, fh)
    sys.stderr = TeeStream(orig_err, fh)
    log(f"Console output is mirrored to log file: {log_path}")
    return log_path, fh, orig_out, orig_err


def restore_logging(fh, orig_out, orig_err) -> None:
    sys.stdout = orig_out
    sys.stderr = orig_err
    fh.close()


def summarize_selected(manifest: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    empty_columns = group_cols + [
        "image_count",
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
                "unique_patient_count": int(group["patient_id"].replace("", pd.NA).dropna().nunique()),
                "unique_study_count": int(group["study_date"].replace("", pd.NA).dropna().nunique()),
                "unique_dicom_count": int(group["dicom_rel_path"].replace("", pd.NA).dropna().nunique()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the border-suppressed PHILIPS subset for Global Analysis 6.")
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--analysis-title", type=str, default="Global Analysis 6 Border-Suppressed PHILIPS Subset")
    parser.add_argument("--markdown-name", type=str, default="analysis_global_6_border_suppressed_philips_unique_view_subset.md")
    parser.add_argument("--log-prefix", type=str, default="build_global_6_border_suppressed_philips_unique_view_subset")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--center-x", type=int, default=DEFAULT_CENTER_X)
    parser.add_argument("--center-y", type=int, default=DEFAULT_CENTER_Y)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA)
    parser.add_argument("--feather-width", type=int, default=DEFAULT_FEATHER_WIDTH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_root = args.baseline_root.resolve()
    output_root = args.output_root.resolve()
    images_out = output_root / "images"
    dicoms_out = output_root / "dicoms"
    preview_root = output_root / "qa_previews"

    if not (baseline_root / "manifest_selected.csv").exists():
        raise FileNotFoundError(f"Missing baseline manifest: {baseline_root / 'manifest_selected.csv'}")

    log_path, fh, orig_out, orig_err = setup_logging(output_root, args.log_prefix)
    try:
        ensure_clean_dir(images_out, overwrite=bool(args.overwrite))
        ensure_clean_dir(dicoms_out, overwrite=bool(args.overwrite))
        ensure_clean_dir(preview_root, overwrite=bool(args.overwrite))

        baseline_manifest = pd.read_csv(baseline_root / "manifest_selected.csv")
        baseline_manifest = baseline_manifest.sort_values(["split", "stent_label", "image_rel_path"]).reset_index(drop=True)
        total_images = int(len(baseline_manifest))
        if total_images == 0:
            raise ValueError("Baseline manifest_selected.csv is empty.")

        log(
            "Arguments: "
            + json.dumps(
                {
                    "baseline_root": str(baseline_root),
                    "output_root": str(output_root),
                    "analysis_title": str(args.analysis_title),
                    "log_every": int(args.log_every),
                    "overwrite": bool(args.overwrite),
                    "center_x": int(args.center_x),
                    "center_y": int(args.center_y),
                    "radius": int(args.radius),
                    "blur_sigma": float(args.blur_sigma),
                    "feather_width": int(args.feather_width),
                    "log_path": str(log_path),
                },
                indent=2,
            )
        )

        build_start = time.time()
        log("BUILD START")
        rows: list[dict[str, object]] = []
        preview_records: list[dict[str, object]] = []
        status_counter: Counter[str] = Counter()
        done = 0
        preview_done = 0

        baseline_images_root = baseline_root / "images"
        baseline_dicoms_root = baseline_root / "dicoms"

        for row in baseline_manifest.to_dict(orient="records"):
            rel_path = Path(str(row["image_rel_path"]))
            dicom_rel = Path(str(row["dicom_rel_path"]))
            source_img = baseline_images_root / rel_path
            source_dcm = baseline_dicoms_root / dicom_rel
            out_img = images_out / rel_path
            out_dcm = dicoms_out / dicom_rel
            ensure_dir(out_img.parent)
            ensure_dir(out_dcm.parent)

            image = read_grayscale(source_img)
            applied = False
            if tuple(image.shape) == TARGET_IMAGE_SHAPE:
                suppressed, mask, _background, _alpha = apply_border_suppression(
                    image,
                    center_x=int(args.center_x),
                    center_y=int(args.center_y),
                    radius=int(args.radius),
                    blur_sigma=float(args.blur_sigma),
                    feather_width=int(args.feather_width),
                )
                write_grayscale(out_img, suppressed)
                applied = True
                status = "suppressed"
                if preview_done < PREVIEW_LIMIT:
                    overlay = overlay_mask_boundary(image, mask)
                    panel = make_three_panel(image, suppressed, overlay)
                    safe_name = safe_name_from_rel_path(rel_path.parts)
                    panel_path = preview_root / f"{safe_name}__panel.png"
                    import cv2

                    cv2.imwrite(str(panel_path), panel)
                    preview_records.append(
                        {
                            "rel_path": str(rel_path),
                            "panel_path": str(panel_path.resolve()),
                            "status": status,
                        }
                    )
                    preview_done += 1
            else:
                shutil.copy2(source_img, out_img)
                status = "copied_shape_mismatch"
            shutil.copy2(source_dcm, out_dcm)

            rows.append(
                {
                    "split": row["split"],
                    "stent_label": row["stent_label"],
                    "patient_id": row["patient_id"],
                    "study_date": row["study_date"],
                    "series_dir": row["series_dir"],
                    "dicom_id": row["dicom_id"],
                    "image_rel_path": str(rel_path),
                    "dicom_rel_path": str(dicom_rel),
                    "source_image_path": str(source_img.resolve()),
                    "source_dicom_path": str(source_dcm.resolve()),
                    "status": status,
                    "suppression_applied": int(applied),
                    "image_shape": f"{image.shape[0]}x{image.shape[1]}",
                    "output_image_path": str(out_img.resolve()),
                    "output_dicom_path": str(out_dcm.resolve()),
                    "error_message": "",
                }
            )
            status_counter[status] += 1
            done += 1
            if done == total_images or (args.log_every > 0 and done % int(args.log_every) == 0):
                elapsed = time.time() - build_start
                remaining = estimate_remaining(elapsed, done, total_images)
                eta = datetime.now() + timedelta(seconds=0.0 if not math.isfinite(remaining) else remaining)
                log(
                    f"[SUBTASK] variant=border_suppressed_philips done={done}/{total_images} | "
                    f"elapsed={format_duration(elapsed)} | remaining={format_duration(remaining)}"
                )
                log(
                    f"[TOTAL] done={done}/{total_images} | progress={(100.0 * done / total_images):.1f}% | "
                    f"applied={int(status_counter['suppressed'])} | copied_shape_mismatch={int(status_counter['copied_shape_mismatch'])} | "
                    f"elapsed={format_duration(elapsed)} | remaining={format_duration(remaining)} | eta={eta.strftime('%Y-%m-%d %H:%M:%S')}"
                )
        log(
            f"BUILD DONE | total_images={total_images} | applied={int(status_counter['suppressed'])} | "
            f"copied_shape_mismatch={int(status_counter['copied_shape_mismatch'])} | elapsed={format_duration(time.time() - build_start)}"
        )

        manifest = pd.DataFrame(rows).sort_values(["split", "stent_label", "image_rel_path"]).reset_index(drop=True)
        counts_by_split = summarize_selected(manifest, ["split"])
        counts_by_split_stent_label = summarize_selected(manifest, ["split", "stent_label"])
        preview_df = pd.DataFrame(preview_records)

        manifest.to_csv(output_root / "selection_audit.csv", index=False)
        manifest.to_csv(output_root / "manifest_selected.csv", index=False)
        counts_by_split.to_csv(output_root / "summary_selected_counts_by_split.csv", index=False)
        counts_by_split_stent_label.to_csv(output_root / "summary_selected_counts_by_split_stent_label.csv", index=False)
        preview_df.to_csv(output_root / "summary_qa_previews.csv", index=False)

        build_meta = {
            "baseline_root": str(baseline_root),
            "output_root": str(output_root),
            "analysis_title": str(args.analysis_title),
            "source_image_count": int(total_images),
            "status_counts": dict(status_counter),
            "suppression_parameters": {
                "center_x": int(args.center_x),
                "center_y": int(args.center_y),
                "radius": int(args.radius),
                "blur_sigma": float(args.blur_sigma),
                "feather_width": int(args.feather_width),
            },
            "target_image_shape": list(TARGET_IMAGE_SHAPE),
            "qa_preview_count": int(len(preview_records)),
        }
        (output_root / "build_metadata.json").write_text(json.dumps(build_meta, indent=2), encoding="utf-8")

        lines: list[str] = []
        lines.append(f"# {args.analysis_title}")
        lines.append("")
        lines.append("## Run Config")
        lines.append("")
        lines.append(f"- baseline_root: `{baseline_root}`")
        lines.append(f"- output_root: `{output_root}`")
        lines.append(f"- suppression params: `(center_x={args.center_x}, center_y={args.center_y}, radius={args.radius}, blur_sigma={args.blur_sigma}, feather_width={args.feather_width})`")
        lines.append("")
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- source_images: `{total_images}`")
        lines.append(f"- suppression_applied: `{int(status_counter['suppressed'])}`")
        lines.append(f"- copied_shape_mismatch: `{int(status_counter['copied_shape_mismatch'])}`")
        lines.append("")
        lines.append("## Selected Counts By Split")
        lines.append("")
        lines.append(dataframe_to_markdown(counts_by_split))
        lines.append("")
        lines.append("## Selected Counts By Split And Stent Label")
        lines.append("")
        lines.append(dataframe_to_markdown(counts_by_split_stent_label))
        lines.append("")
        lines.append("## QA Preview Panels")
        lines.append("")
        if preview_df.empty:
            lines.append("_Empty._")
        else:
            lines.append(dataframe_to_markdown(preview_df))
        lines.append("")
        (output_root / args.markdown_name).write_text("\n".join(lines), encoding="utf-8")
    finally:
        restore_logging(fh, orig_out, orig_err)


if __name__ == "__main__":
    main()
