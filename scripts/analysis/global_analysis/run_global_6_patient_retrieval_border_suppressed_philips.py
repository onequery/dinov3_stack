#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

DEFAULT_INPUT_ROOT = REPO_ROOT / "input/global_analysis_6_patient_retrieval_border_suppressed_philips"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/analysis2_rep_analysis/global_6_patient_retrieval_border_suppressed_philips"
DEFAULT_SOURCE_IMAGE_ROOT = REPO_ROOT / "input/Stent-Contrast-unique-view"
DEFAULT_SOURCE_DCM_ROOT = REPO_ROOT / "input/stent_split_dcm_unique_view"
DEFAULT_IMAGENET_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
DEFAULT_CAG_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
EXPECTED_COUNTS = {"train": 582, "valid": 201, "test": 265}

STAGE_SPECS = [
    ("build_baseline", "build or validate GA6 baseline PHILIPS subset"),
    ("build_suppressed", "build or validate GA6 border-suppressed PHILIPS subset"),
    ("run_baseline", "run baseline single-device retrieval pipeline"),
    ("run_suppressed", "run border-suppressed single-device retrieval pipeline"),
    ("summarize_retrieval", "generate GA6 patient retrieval comparison report"),
    ("summarize_anchoring", "generate GA6 anchoring comparison report"),
]


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
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_clear(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        shutil.rmtree(path)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    cache_root = REPO_ROOT / "outputs/.runtime_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    env.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def setup_console_and_file_logging(output_root: Path, log_file_arg: str | None, default_prefix: str):
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


def estimate_overall_remaining_seconds(
    *,
    started_at: float,
    completed_stages: int,
    total_stages: int,
    current_stage_started_at: float | None,
) -> tuple[float, float | None]:
    now = time.time()
    total_elapsed = now - started_at
    if total_stages <= 0:
        return total_elapsed, 0.0
    if current_stage_started_at is None:
        if completed_stages <= 0:
            return total_elapsed, None
        avg_completed = total_elapsed / completed_stages
        remaining = avg_completed * max(0, total_stages - completed_stages)
        return total_elapsed, remaining
    current_stage_elapsed = max(0.0, now - current_stage_started_at)
    if completed_stages <= 0:
        remaining = current_stage_elapsed * max(0, total_stages - 1)
        return total_elapsed, remaining
    elapsed_before_current = max(0.0, current_stage_started_at - started_at)
    avg_completed = elapsed_before_current / completed_stages
    estimated_current_total = max(avg_completed, current_stage_elapsed)
    remaining_current = max(0.0, estimated_current_total - current_stage_elapsed)
    remaining_future = avg_completed * max(0, total_stages - completed_stages - 1)
    return total_elapsed, remaining_current + remaining_future


def log_overall_eta(
    *,
    label: str,
    started_at: float,
    completed_stages: int,
    total_stages: int,
    current_stage_started_at: float | None,
) -> None:
    total_elapsed, remaining_seconds = estimate_overall_remaining_seconds(
        started_at=started_at,
        completed_stages=completed_stages,
        total_stages=total_stages,
        current_stage_started_at=current_stage_started_at,
    )
    remaining_text = format_duration(remaining_seconds)
    eta_text = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
        if remaining_seconds is not None
        else "estimating"
    )
    log(
        f"[EXP-ETA] current={label} | stages_done={completed_stages}/{total_stages} "
        f"| stages_remaining={max(0, total_stages - completed_stages)} "
        f"| total_elapsed={format_duration(total_elapsed)} | remaining_est={remaining_text} | eta={eta_text}"
    )


def run_command(
    cmd: list[str],
    env: dict[str, str],
    *,
    label: str,
    started_at: float,
    completed_stages: int,
    total_stages: int,
    current_stage_started_at: float,
) -> None:
    log("CMD: " + " ".join(cmd))
    with subprocess.Popen(
        cmd,
        env=env,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if "[ETA]" in line:
                log_overall_eta(
                    label=label,
                    started_at=started_at,
                    completed_stages=completed_stages,
                    total_stages=total_stages,
                    current_stage_started_at=current_stage_started_at,
                )
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def validate_counts(summary_csv: Path, expected: dict[str, int]) -> None:
    df = pd.read_csv(summary_csv)
    actual = {str(row["split"]): int(row["image_count"]) for _, row in df.iterrows()}
    if actual != expected:
        raise ValueError(f"Subset counts mismatch for {summary_csv}: expected {expected}, got {actual}")


def write_subset_note(input_root: Path, baseline_root: Path, variant_root: Path) -> None:
    def count_lines(root: Path) -> list[str]:
        df = pd.read_csv(root / "summary_selected_counts_by_split.csv")
        return [f"  - {row['split']}: {int(row['image_count'])}" for _, row in df.sort_values('split').iterrows()]

    lines = [
        "# Global Analysis 6: Patient Retrieval with Border-Suppressed PHILIPS Subsets",
        "",
        f"- frame root: `{input_root}`",
        f"- baseline subset: `{baseline_root}`",
        f"- border-suppressed subset: `{variant_root}`",
        "",
        "## Counts",
        "",
        "- baseline_philips:",
        *count_lines(baseline_root),
        "- border_suppressed_philips:",
        *count_lines(variant_root),
        "",
    ]
    (input_root / "analysis_global_6_patient_retrieval_border_suppressed_philips_subsets.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--source-image-root", default=str(DEFAULT_SOURCE_IMAGE_ROOT))
    parser.add_argument("--source-dcm-root", default=str(DEFAULT_SOURCE_DCM_ROOT))
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
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--probe-lr-grid", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-min-delta", type=float, default=0.0)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--center-x", type=int, default=256)
    parser.add_argument("--center-y", type=int, default=256)
    parser.add_argument("--radius", type=int, default=332)
    parser.add_argument("--blur-sigma", type=float, default=25.0)
    parser.add_argument("--feather-width", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    baseline_input_root = input_root / "baseline_philips_unique_view_subset"
    variant_input_root = input_root / "border_suppressed_philips_unique_view_subset"
    baseline_output_root = output_root / "baseline_philips"
    variant_output_root = output_root / "border_suppressed_philips"
    reports_root = output_root / "reports"
    anchoring_reports_root = output_root / "anchoring_reports"

    maybe_clear(output_root, overwrite=bool(args.overwrite))
    ensure_dir(input_root)
    ensure_dir(output_root)
    ensure_dir(reports_root)
    ensure_dir(anchoring_reports_root)

    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        output_root,
        args.log_file,
        default_prefix="run_global_6_patient_retrieval_border_suppressed_philips",
    )
    try:
        env = build_env()
        started_at = time.time()
        completed_stages = 0
        total_stages = len(STAGE_SPECS)
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")

        build_baseline_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/global_analysis/build_global_5_single_device_unique_view_subset.py"),
            "--image-root",
            str(Path(args.source_image_root).resolve()),
            "--dicom-root",
            str(Path(args.source_dcm_root).resolve()),
            "--output-root",
            str(baseline_input_root),
            "--target-model-name",
            "P H I L I P S INTEGRIS H",
            "--analysis-title",
            "Global Analysis 6 Baseline PHILIPS Subset",
            "--markdown-name",
            "analysis_global_6_baseline_philips_unique_view_subset.md",
            "--log-prefix",
            "build_global_6_baseline_philips_unique_view_subset",
            "--log-every",
            "200",
        ]
        if args.overwrite:
            build_baseline_cmd.append("--overwrite")

        build_variant_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/global_analysis/build_global_6_border_suppressed_philips_subset.py"),
            "--baseline-root",
            str(baseline_input_root),
            "--output-root",
            str(variant_input_root),
            "--analysis-title",
            "Global Analysis 6 Border-Suppressed PHILIPS Subset",
            "--markdown-name",
            "analysis_global_6_border_suppressed_philips_unique_view_subset.md",
            "--log-prefix",
            "build_global_6_border_suppressed_philips_unique_view_subset",
            "--log-every",
            "100",
            "--center-x",
            str(args.center_x),
            "--center-y",
            str(args.center_y),
            "--radius",
            str(args.radius),
            "--blur-sigma",
            str(args.blur_sigma),
            "--feather-width",
            str(args.feather_width),
        ]
        if args.overwrite:
            build_variant_cmd.append("--overwrite")

        def make_ga5_subrun_cmd(image_root: Path, dcm_root: Path, arm_output_root: Path, analysis_title: str) -> list[str]:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/analysis/global_analysis/run_global_5_single_device_patient_retrieval.py"),
                "--image-root",
                str(image_root),
                "--dcm-root",
                str(dcm_root),
                "--retrieval-root",
                str(arm_output_root / "retrieval_benchmark"),
                "--output-root",
                str(arm_output_root),
                "--analysis-title",
                analysis_title,
                "--analysis-name",
                "global_6_patient_retrieval_border_suppressed_philips",
                "--summary-prefix",
                "summary_global_6",
                "--figure-prefix",
                "fig_global6",
                "--patient-markdown-name",
                "analysis_global_6_patient_retrieval.md",
                "--anchoring-markdown-name",
                "analysis_global_6_cluster_anchoring_attribution.md",
                "--anchoring-log-prefix",
                "global_6_cluster_anchoring_attribution",
                "--imagenet-ckpt",
                str(args.imagenet_ckpt),
                "--cag-ckpt",
                str(args.cag_ckpt),
                "--model-name",
                str(args.model_name),
                "--repo-dir",
                str(args.repo_dir),
                "--device",
                str(args.device),
                "--resize-size",
                str(args.resize_size),
                "--center-crop-size",
                str(args.center_crop_size),
                "--feature-batch-size",
                str(args.feature_batch_size),
                "--probe-batch-size",
                str(args.probe_batch_size),
                "--num-workers",
                str(args.num_workers),
                "--probe-max-epoch",
                str(args.probe_max_epoch),
                "--probe-patience",
                str(args.probe_patience),
                "--probe-min-delta",
                str(args.probe_min_delta),
                "--probe-weight-decay",
                str(args.probe_weight_decay),
                "--supcon-temperature",
                str(args.supcon_temperature),
                "--seed",
                str(args.seed),
                "--probe-seeds",
                *[str(v) for v in args.probe_seeds],
                "--probe-lr-grid",
                *[str(v) for v in args.probe_lr_grid],
            ]
            if args.max_images_per_split is not None and int(args.max_images_per_split) > 0:
                cmd.extend(["--max-images-per-split", str(args.max_images_per_split)])
            if args.strict_deterministic:
                cmd.append("--strict-deterministic")
            if args.cache_features:
                cmd.append("--cache-features")
            if args.overwrite:
                cmd.append("--overwrite")
            return cmd

        run_baseline_cmd = make_ga5_subrun_cmd(
            baseline_input_root / "images",
            baseline_input_root / "dicoms",
            baseline_output_root,
            "Global Analysis 6 | Baseline PHILIPS",
        )
        run_variant_cmd = make_ga5_subrun_cmd(
            variant_input_root / "images",
            variant_input_root / "dicoms",
            variant_output_root,
            "Global Analysis 6 | Border-Suppressed PHILIPS",
        )

        summarize_retrieval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/global_analysis/summarize_global_6_patient_retrieval.py"),
            "--baseline-root",
            str(baseline_output_root),
            "--variant-root",
            str(variant_output_root),
            "--output-root",
            str(reports_root),
            "--baseline-arm-name",
            "baseline_philips",
            "--variant-arm-name",
            "border_suppressed_philips",
            "--analysis-title",
            "Global Analysis 6",
        ]

        summarize_anchor_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/global_analysis/summarize_global_6_anchor_comparison.py"),
            "--baseline-root",
            str(baseline_output_root),
            "--variant-root",
            str(variant_output_root),
            "--output-root",
            str(anchoring_reports_root),
            "--baseline-arm-name",
            "baseline_philips",
            "--variant-arm-name",
            "border_suppressed_philips",
            "--analysis-title",
            "Global Analysis 6",
        ]

        stage_cmds = {
            "build_baseline": build_baseline_cmd,
            "build_suppressed": build_variant_cmd,
            "run_baseline": run_baseline_cmd,
            "run_suppressed": run_variant_cmd,
            "summarize_retrieval": summarize_retrieval_cmd,
            "summarize_anchoring": summarize_anchor_cmd,
        }

        for stage_index, (stage_key, stage_label) in enumerate(STAGE_SPECS, start=1):
            stage_started_at = time.time()
            log(f"START {stage_index}/{total_stages} | {stage_label}")
            run_command(
                stage_cmds[stage_key],
                env,
                label=stage_label,
                started_at=started_at,
                completed_stages=completed_stages,
                total_stages=total_stages,
                current_stage_started_at=stage_started_at,
            )

            if stage_key == "build_baseline":
                validate_counts(baseline_input_root / "summary_selected_counts_by_split.csv", EXPECTED_COUNTS)
            elif stage_key == "build_suppressed":
                validate_counts(variant_input_root / "summary_selected_counts_by_split.csv", EXPECTED_COUNTS)
                write_subset_note(input_root, baseline_input_root, variant_input_root)

            completed_stages += 1
            total_elapsed, remaining_seconds = estimate_overall_remaining_seconds(
                started_at=started_at,
                completed_stages=completed_stages,
                total_stages=total_stages,
                current_stage_started_at=None,
            )
            remaining_text = format_duration(remaining_seconds)
            eta_text = (
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
                if remaining_seconds is not None
                else "estimating"
            )
            log(
                f"DONE {stage_index}/{total_stages} | {stage_label} | progress={100.0 * completed_stages / total_stages:.1f}% "
                f"| elapsed={format_duration(total_elapsed)} | remaining~{remaining_text} | eta={eta_text}"
            )

        run_meta = {
            "analysis_title": "Global Analysis 6",
            "input_root": str(input_root),
            "output_root": str(output_root),
            "source_image_root": str(Path(args.source_image_root).resolve()),
            "source_dcm_root": str(Path(args.source_dcm_root).resolve()),
            "subset_roots": {
                "baseline_philips": str(baseline_input_root),
                "border_suppressed_philips": str(variant_input_root),
            },
            "arm_output_roots": {
                "baseline_philips": str(baseline_output_root),
                "border_suppressed_philips": str(variant_output_root),
            },
            "reports_root": str(reports_root),
            "anchoring_reports_root": str(anchoring_reports_root),
            "backbone_checkpoints": {
                "imagenet": str(args.imagenet_ckpt),
                "cag": str(args.cag_ckpt),
            },
            "probe_seeds": [int(v) for v in args.probe_seeds],
            "probe_lr_grid": [float(v) for v in args.probe_lr_grid],
            "probe_max_epoch": int(args.probe_max_epoch),
            "probe_patience": int(args.probe_patience),
            "device": str(args.device),
            "max_images_per_split": args.max_images_per_split,
            "suppression_parameters": {
                "center_x": int(args.center_x),
                "center_y": int(args.center_y),
                "radius": int(args.radius),
                "blur_sigma": float(args.blur_sigma),
                "feather_width": int(args.feather_width),
            },
            "expected_counts": EXPECTED_COUNTS,
            "run_log_path": str(log_path.resolve()),
        }
        (output_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        log("Global Analysis 6 completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
