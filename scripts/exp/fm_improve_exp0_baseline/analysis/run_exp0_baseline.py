#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent

VIEW_CLS_SCRIPT = REPO_ROOT / "scripts/exp/fm_improve_exp0_baseline/analysis/run_contrast_benchmark_view_classification.py"
PATIENT_RET_SCRIPT = REPO_ROOT / "scripts/exp/fm_improve_exp0_baseline/analysis/run_contrast_benchmark_patient_retrieval.py"
SEG_SCRIPT = REPO_ROOT / "scripts/analysis/local/local_2_segmentation_linear_probe.py"
SUMMARY_SCRIPT = REPO_ROOT / "scripts/exp/fm_improve_exp0_baseline/analysis/summarize_exp0_baseline.py"


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_clear(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)


def run_cmd(cmd: Sequence[str]) -> None:
    log("CMD: " + " ".join(subprocess.list2cmdline([part]) for part in cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(list(cmd), cwd=REPO_ROOT, check=True, env=env)


def format_duration(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def resolve_device(requested: str) -> str:
    lowered = str(requested).lower()
    if lowered.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            log(f"CUDA requested for Exp0 but unavailable in this runtime. Falling back to CPU.")
            return "cpu"
    return requested


def write_pending_status(output_root: Path, task_name: str, reason: str) -> None:
    ensure_dir(output_root)
    payload = {
        "task_name": task_name,
        "status": "pending",
        "reason": reason,
        "output_root": str(output_root.resolve()),
    }
    (output_root / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_root / "README.md").write_text(
        f"# {task_name}\n\n- status: `pending`\n- reason: {reason}\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FM Improvement Exp0 baseline benchmark package.")
    parser.add_argument("--benchmark-root", default="input/contrast_benchmark")
    parser.add_argument("--output-root", default="outputs/fm-imp-exp0_baseline")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imagenet-ckpt", default="dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth")
    parser.add_argument("--cag-ckpt", default="dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth")
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    parser.add_argument("--resize-size", type=int, default=480)
    parser.add_argument("--center-crop-size", type=int, default=448)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--probe-lr-grid", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-min-delta", type=float, default=0.0)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--input-policy", default="baseline_rgbtriplet")
    parser.add_argument("--input-stats-json", default=None)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--max-topk-log", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--skip-view-classification", action="store_true")
    parser.add_argument("--skip-patient-retrieval", action="store_true")
    parser.add_argument("--skip-coronary-segmentation", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_root = Path(args.benchmark_root).resolve()
    output_root = Path(args.output_root).resolve()
    if args.overwrite:
        maybe_clear(output_root, overwrite=True)
    ensure_dir(output_root)
    effective_device = resolve_device(args.device)

    view_dataset_root = benchmark_root / "1_global/1_view_classification"
    patient_dataset_root = benchmark_root / "1_global/2_patient_retrieval"
    coronary_root = benchmark_root / "2_dense/1_coronary_segmentation"

    steps: List[tuple[str, str]] = []
    if not args.skip_view_classification:
        steps.append(("view_classification", "Run contrast benchmark view classification"))
    if not args.skip_patient_retrieval:
        steps.append(("patient_retrieval", "Run contrast benchmark patient retrieval"))
    if not args.skip_coronary_segmentation:
        steps.append(("coronary_segmentation", "Run coronary segmentation linear probe"))
    steps.append(("pending_placeholders", "Write pending dense-task placeholders"))
    if not args.skip_summary:
        steps.append(("summary", "Summarize Exp0 baseline outputs"))

    log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
    start_time = time.time()
    completed = 0

    for step_key, step_name in steps:
        step_start = time.time()
        completed_display = completed + 1
        log(f"[ETA][EXP0] Step {completed_display}/{len(steps)} START | {step_name}")

        if step_key == "view_classification":
            run_cmd(
                [
                    sys.executable,
                    str(VIEW_CLS_SCRIPT),
                    "--dataset-root",
                    str(view_dataset_root),
                    "--output-root",
                    str(output_root / "1_global/1_view_classification"),
                    "--imagenet-ckpt",
                    args.imagenet_ckpt,
                    "--cag-ckpt",
                    args.cag_ckpt,
                    "--model-name",
                    args.model_name,
                    "--repo-dir",
                    args.repo_dir,
                    "--device",
                    effective_device,
                    "--resize-size",
                    str(args.resize_size),
                    "--center-crop-size",
                    str(args.center_crop_size),
                    "--feature-batch-size",
                    str(args.feature_batch_size),
                    "--input-policy",
                    args.input_policy,
                    "--probe-batch-size",
                    str(args.probe_batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--probe-seeds",
                    *[str(v) for v in args.probe_seeds],
                    "--probe-lr-grid",
                    *[str(v) for v in args.probe_lr_grid],
                    "--probe-max-epoch",
                    str(args.probe_max_epoch),
                    "--probe-patience",
                    str(args.probe_patience),
                    "--probe-min-delta",
                    str(args.probe_min_delta),
                    "--probe-weight-decay",
                    str(args.probe_weight_decay),
                    "--seed",
                    str(args.seed),
                ]
                + (["--input-stats-json", args.input_stats_json] if args.input_stats_json else [])
                + (["--max-images-per-split", str(args.max_images_per_split)] if args.max_images_per_split is not None else [])
                + (["--cache-features"] if args.cache_features else [])
                + (["--strict-deterministic"] if args.strict_deterministic else [])
            )
        elif step_key == "patient_retrieval":
            run_cmd(
                [
                    sys.executable,
                    str(PATIENT_RET_SCRIPT),
                    "--dataset-root",
                    str(patient_dataset_root),
                    "--output-root",
                    str(output_root / "1_global/2_patient_retrieval"),
                    "--imagenet-ckpt",
                    args.imagenet_ckpt,
                    "--cag-ckpt",
                    args.cag_ckpt,
                    "--model-name",
                    args.model_name,
                    "--repo-dir",
                    args.repo_dir,
                    "--device",
                    effective_device,
                    "--resize-size",
                    str(args.resize_size),
                    "--center-crop-size",
                    str(args.center_crop_size),
                    "--feature-batch-size",
                    str(args.feature_batch_size),
                    "--input-policy",
                    args.input_policy,
                    "--probe-batch-size",
                    str(args.probe_batch_size),
                    "--num-workers",
                    str(args.num_workers),
                    "--probe-seeds",
                    *[str(v) for v in args.probe_seeds],
                    "--probe-lr-grid",
                    *[str(v) for v in args.probe_lr_grid],
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
                    "--max-topk-log",
                    str(args.max_topk_log),
                    "--seed",
                    str(args.seed),
                ]
                + (["--input-stats-json", args.input_stats_json] if args.input_stats_json else [])
                + (["--max-images-per-split", str(args.max_images_per_split)] if args.max_images_per_split is not None else [])
                + (["--cache-features"] if args.cache_features else [])
                + (["--strict-deterministic"] if args.strict_deterministic else [])
            )
        elif step_key == "coronary_segmentation":
            run_cmd(
                [
                    sys.executable,
                    str(SEG_SCRIPT),
                    "--train-images",
                    str(coronary_root / "train_images"),
                    "--train-masks",
                    str(coronary_root / "train_labels"),
                    "--valid-images",
                    str(coronary_root / "valid_images"),
                    "--valid-masks",
                    str(coronary_root / "valid_labels"),
                    "--test-images",
                    str(coronary_root / "test_images"),
                    "--test-masks",
                    str(coronary_root / "test_labels"),
                    "--output-root",
                    str(output_root / "2_dense/1_coronary_segmentation"),
                    "--imagenet-ckpt",
                    args.imagenet_ckpt,
                    "--cag-ckpt",
                    args.cag_ckpt,
                    "--device",
                    effective_device,
                    "--feature-batch-size",
                    str(max(1, min(args.feature_batch_size, 32))),
                    "--probe-batch-size",
                    str(max(1, args.probe_batch_size)),
                    "--num-workers",
                    str(args.num_workers),
                    "--max-epoch",
                    str(args.probe_max_epoch),
                    "--early-stopping-patience",
                    str(args.probe_patience),
                    "--early-stopping-min-delta",
                    str(args.probe_min_delta),
                    "--lr-grid",
                    *[str(v) for v in args.probe_lr_grid],
                    "--seed",
                    str(args.seed),
                    "--model-name",
                    args.model_name,
                    "--repo-dir",
                    args.repo_dir,
                ]
                + (["--max-images-per-split", str(args.max_images_per_split)] if args.max_images_per_split is not None else [])
            )
        elif step_key == "pending_placeholders":
            write_pending_status(
                output_root / "2_dense/2_sub-segment_segmentation",
                task_name="2_sub-segment_segmentation",
                reason="Dataset slot exists in contrast_benchmark but Exp0 execution is deferred until the task-specific benchmark pipeline is implemented.",
            )
            write_pending_status(
                output_root / "2_dense/3_stenosis_detection",
                task_name="3_stenosis_detection",
                reason="Dataset slot exists in contrast_benchmark but Exp0 execution is deferred until the task-specific benchmark pipeline is implemented.",
            )
        elif step_key == "summary":
            run_cmd([sys.executable, str(SUMMARY_SCRIPT), "--exp-root", str(output_root)])
        else:
            raise ValueError(step_key)

        completed += 1
        elapsed = max(0.0, time.time() - start_time)
        avg = elapsed / completed
        remaining = avg * (len(steps) - completed)
        eta = datetime.now() + timedelta(seconds=remaining)
        log(
            f"[ETA][EXP0] Step {completed}/{len(steps)} DONE | {step_name} | "
            f"elapsed={format_duration(elapsed)} | remaining~{format_duration(remaining)} | eta={eta.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    run_meta = {
        "analysis": "fm_improve_exp0_baseline",
        "benchmark_root": str(benchmark_root),
        "patient_retrieval_root": str(patient_dataset_root),
        "output_root": str(output_root),
        "imagenet_ckpt": str(Path(args.imagenet_ckpt).resolve()),
        "cag_ckpt": str(Path(args.cag_ckpt).resolve()),
        "probe_seed_set": [int(v) for v in args.probe_seeds],
        "probe_lr_grid": [float(v) for v in args.probe_lr_grid],
        "device": args.device,
        "effective_device": effective_device,
        "executed_tasks": [step_key for step_key, _ in steps if step_key in {"view_classification", "patient_retrieval", "coronary_segmentation"}],
        "pending_tasks": ["2_sub-segment_segmentation", "3_stenosis_detection"],
        "reports_root": str((output_root / "reports").resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (output_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    log("FM Improvement Exp0 baseline completed successfully.")


if __name__ == "__main__":
    main()
