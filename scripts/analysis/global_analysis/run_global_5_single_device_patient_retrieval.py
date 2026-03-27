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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

DEFAULT_IMAGE_ROOT = REPO_ROOT / "input/global_analysis_5_philips_integris_h_unique_view_subset/images"
DEFAULT_DCM_ROOT = REPO_ROOT / "input/global_analysis_5_philips_integris_h_unique_view_subset/dicoms"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/analysis2_rep_analysis/global_5_single_device_patient_retrieval_philips_integris_h_unique_view"
DEFAULT_RETRIEVAL_ROOT = DEFAULT_OUTPUT_ROOT / "retrieval_benchmark"
DEFAULT_IMAGENET_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
DEFAULT_CAG_CKPT = (
    "dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/"
    "3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth"
)
DEFAULT_ANALYSIS_TITLE = "Global Analysis 5"
DEFAULT_ANALYSIS_NAME = "global_5_single_device_patient_retrieval_philips_integris_h_unique_view"
DEFAULT_SUMMARY_PREFIX = "summary_global_5"
DEFAULT_FIGURE_PREFIX = "fig_global5"
DEFAULT_PATIENT_MARKDOWN_NAME = "analysis_global_5_patient_retrieval.md"
DEFAULT_ANCHORING_MARKDOWN_NAME = "analysis_global_5_cluster_anchoring_attribution.md"
DEFAULT_ANCHORING_LOG_PREFIX = "global_5_cluster_anchoring_attribution"


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


STAGE_SPECS = [
    ("retrieval", "run single-device retrieval benchmark"),
    ("export", "export GA5 patient retrieval summary"),
    ("anchoring", "run GA5 cluster anchoring attribution"),
]


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
        f"| total_elapsed={format_duration(total_elapsed)} "
        f"| remaining_est={remaining_text} | eta={eta_text}"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT))
    parser.add_argument("--dcm-root", default=str(DEFAULT_DCM_ROOT))
    parser.add_argument("--retrieval-root", default=str(DEFAULT_RETRIEVAL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--analysis-title", default=DEFAULT_ANALYSIS_TITLE)
    parser.add_argument("--analysis-name", default=DEFAULT_ANALYSIS_NAME)
    parser.add_argument("--summary-prefix", default=DEFAULT_SUMMARY_PREFIX)
    parser.add_argument("--figure-prefix", default=DEFAULT_FIGURE_PREFIX)
    parser.add_argument("--patient-markdown-name", default=DEFAULT_PATIENT_MARKDOWN_NAME)
    parser.add_argument("--anchoring-markdown-name", default=DEFAULT_ANCHORING_MARKDOWN_NAME)
    parser.add_argument("--anchoring-log-prefix", default=DEFAULT_ANCHORING_LOG_PREFIX)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = Path(args.image_root).resolve()
    dcm_root = Path(args.dcm_root).resolve()
    retrieval_root = Path(args.retrieval_root).resolve()
    output_root = Path(args.output_root).resolve()

    maybe_clear(output_root, overwrite=bool(args.overwrite))
    maybe_clear(retrieval_root, overwrite=bool(args.overwrite))
    ensure_dir(output_root)
    ensure_dir(retrieval_root)

    log_path, file_handle, original_stdout, original_stderr = setup_console_and_file_logging(
        output_root,
        args.log_file,
        default_prefix="run_global_5_single_device_patient_retrieval",
    )
    try:
        env = build_env()
        started_at = time.time()
        total_stages = len(STAGE_SPECS)
        completed_stages = 0
        log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")

        retrieval_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/global_analysis/global_2_study_patient_retrieval.py"),
            "--image-root",
            str(image_root),
            "--output-root",
            str(retrieval_root),
            "--log-file",
            "global_5_single_device_patient_retrieval.log",
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
            retrieval_cmd.extend(["--max-images-per-split", str(args.max_images_per_split)])
        if args.strict_deterministic:
            retrieval_cmd.append("--strict-deterministic")
        if args.cache_features:
            retrieval_cmd.append("--cache-features")

        export_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/global_analysis/export_global_4_1_patient_retrieval_summary.py"),
            "--source-root",
            str(retrieval_root),
            "--output-root",
            str(output_root),
            "--analysis-title",
            str(args.analysis_title),
            "--summary-prefix",
            str(args.summary_prefix),
            "--figure-prefix",
            str(args.figure_prefix),
            "--markdown-name",
            str(args.patient_markdown_name),
            "--probe-seeds",
            *[str(v) for v in args.probe_seeds],
        ]

        anchoring_cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts/analysis/global_analysis/global_4_1_cluster_anchoring_attribution.py"),
            "--global2-root",
            str(retrieval_root),
            "--image-root",
            str(image_root),
            "--dcm-root",
            str(dcm_root),
            "--output-root",
            str(output_root),
            "--analysis-title",
            str(args.analysis_title),
            "--summary-prefix",
            str(args.summary_prefix),
            "--figure-prefix",
            str(args.figure_prefix),
            "--markdown-name",
            str(args.anchoring_markdown_name),
            "--log-prefix",
            str(args.anchoring_log_prefix),
            "--probe-seeds",
            *[str(v) for v in args.probe_seeds],
            "--knn-k",
            "5",
            "10",
            "20",
            "--cluster-k",
            "8",
            "12",
            "16",
            "--probe-batch-size",
            str(args.probe_batch_size),
            "--device",
            str(args.device),
            "--seed",
            str(args.seed),
        ]
        if args.max_images_per_split is not None and int(args.max_images_per_split) > 0:
            anchoring_cmd.extend(["--max-images", str(args.max_images_per_split)])

        for stage_index, (stage_key, stage_label) in enumerate(STAGE_SPECS, start=1):
            stage_started_at = time.time()
            log(f"START {stage_index}/{total_stages} | {stage_label}")
            if stage_key == "retrieval":
                cmd = retrieval_cmd
            elif stage_key == "export":
                cmd = export_cmd
            elif stage_key == "anchoring":
                cmd = anchoring_cmd
            else:
                raise ValueError(stage_key)
            run_command(
                cmd,
                env,
                label=stage_label,
                started_at=started_at,
                completed_stages=completed_stages,
                total_stages=total_stages,
                current_stage_started_at=stage_started_at,
            )
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
            "analysis_title": str(args.analysis_title),
            "analysis_name": str(args.analysis_name),
            "image_root": str(image_root),
            "dcm_root": str(dcm_root),
            "retrieval_root": str(retrieval_root),
            "final_output_root": str(output_root),
            "backbone_checkpoints": {
                "imagenet": str(args.imagenet_ckpt),
                "cag": str(args.cag_ckpt),
            },
            "model_name": str(args.model_name),
            "repo_dir": str(args.repo_dir),
            "probe_seeds": [int(v) for v in args.probe_seeds],
            "probe_lr_grid": [float(v) for v in args.probe_lr_grid],
            "probe_max_epoch": int(args.probe_max_epoch),
            "probe_patience": int(args.probe_patience),
            "feature_batch_size": int(args.feature_batch_size),
            "probe_batch_size": int(args.probe_batch_size),
            "num_workers": int(args.num_workers),
            "device": str(args.device),
            "max_images_per_split": args.max_images_per_split,
            "strict_deterministic": bool(args.strict_deterministic),
            "cache_features": bool(args.cache_features),
            "run_log_path": str(log_path.resolve()),
            "child_outputs": {
                "retrieval_root": str(retrieval_root),
                "final_output_root": str(output_root),
                "export_markdown": str((output_root / str(args.patient_markdown_name)).resolve()),
                "anchoring_markdown": str((output_root / str(args.anchoring_markdown_name)).resolve()),
            },
        }
        (output_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        log(f"{args.analysis_title} completed successfully.")
    finally:
        restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
