#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent

TASKS = {
    "global_4_1_patient_retrieval": {
        "script": REPO_ROOT / "scripts/analysis/global_analysis/global_2_study_patient_retrieval.py",
    },
    "global_4_2_same_dicom_retrieval": {
        "script": REPO_ROOT / "scripts/analysis/global_analysis/global_4_2_same_dicom_retrieval.py",
    },
    "global_4_3_view_classification": {
        "script": REPO_ROOT / "scripts/analysis/global_analysis/global_4_3_view_classification.py",
    },
}


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_clear(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(remaining_seconds: float) -> str:
    eta_dt = datetime.now() + timedelta(seconds=max(0.0, remaining_seconds))
    return eta_dt.strftime("%Y-%m-%d %H:%M:%S")


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


def materialize_same_dicom_smoke_subset(
    source_root: Path,
    source_manifest: Path,
    subset_root: Path,
    output_manifest: Path,
    max_images_per_split: int,
) -> tuple[Path, Path]:
    if max_images_per_split < 2:
        raise ValueError("max_images_per_split must be at least 2 for same-dicom smoke mode")
    df = pd.read_csv(source_manifest)
    limited_parts = []
    max_pairs = max(1, int(max_images_per_split) // 2)
    for _, split_df in df.groupby("split", sort=False):
        unique_dicom_ids = list(dict.fromkeys(split_df["dicom_id"].astype(str).tolist()))
        keep_ids = set(unique_dicom_ids[:max_pairs])
        subset = split_df[split_df["dicom_id"].astype(str).isin(keep_ids)].copy()
        limited_parts.append(subset)
    limited_df = pd.concat(limited_parts, axis=0, ignore_index=True)

    ensure_dir(subset_root)
    for row_idx, row in limited_df.iterrows():
        src = Path(str(row["img_path"])).resolve()
        rel_path = src.relative_to(source_root.resolve())
        dst = subset_root / rel_path
        ensure_dir(dst.parent)
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        limited_df.at[row_idx, "img_path"] = str(dst)

    output_manifest.write_text(limited_df.to_csv(index=False), encoding="utf-8")
    return subset_root, output_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run input_v1_cag_stats_normalization downstream-only benchmarks.")
    parser.add_argument("--unique-view-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--same-dicom-root", default="input/Stent-Contrast-same-dicom-unique-view")
    parser.add_argument("--dcm-root", default="input/stent_split_dcm_unique_view")
    parser.add_argument(
        "--output-root",
        default="outputs/fm_improve_exp1-input_policy/input_v1_cag_stats_normalization/downstream_only",
    )
    parser.add_argument(
        "--stats-json",
        default="outputs/fm_improve_exp1-input_policy/input_v1_cag_stats_normalization/stats/cag_stats_unique_view_train.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resize-size", type=int, default=480)
    parser.add_argument("--center-crop-size", type=int, default=448)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--probe-lr-grid", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-min-delta", type=float, default=0.0)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def execute_step(
    *,
    step_index: int,
    total_steps: int,
    step_label: str,
    cmd: Sequence[str],
    is_benchmark: bool,
    benchmark_runs_done: int,
    benchmark_runs_total: int,
    overall_start: float,
) -> int:
    log(
        f"[EXP][{step_index}/{total_steps}] START | current={step_label} | "
        f"benchmark_runs_done={benchmark_runs_done}/{benchmark_runs_total} | "
        f"benchmark_runs_remaining={benchmark_runs_total - benchmark_runs_done}"
    )
    step_start = perf_counter()
    run_cmd(cmd)
    step_elapsed = perf_counter() - step_start
    total_elapsed = perf_counter() - overall_start
    completed_steps = step_index
    remaining_steps = total_steps - completed_steps
    avg_step = total_elapsed / completed_steps if completed_steps > 0 else 0.0
    remaining_est = avg_step * remaining_steps
    benchmark_runs_done_after = benchmark_runs_done + (1 if is_benchmark else 0)
    log(
        f"[EXP][{step_index}/{total_steps}] DONE | current={step_label} | "
        f"step_elapsed={format_duration(step_elapsed)} | total_elapsed={format_duration(total_elapsed)} | "
        f"benchmark_runs_done={benchmark_runs_done_after}/{benchmark_runs_total} | "
        f"benchmark_runs_remaining={benchmark_runs_total - benchmark_runs_done_after} | "
        f"remaining_est~{format_duration(remaining_est)} | eta={format_eta(remaining_est)}"
    )
    return benchmark_runs_done_after


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    stats_json = Path(args.stats_json).resolve()
    ensure_dir(output_root)
    ensure_dir(stats_json.parent)
    log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")

    stats_builder = REPO_ROOT / "scripts/exp/fm_improve_exp1-input_policy/preprocess/build_input_v1_cag_stats.py"
    summary_script = REPO_ROOT / "scripts/exp/fm_improve_exp1-input_policy/analysis/summarize_input_v1_cag_stats_normalization.py"

    same_dicom_root = Path(args.same_dicom_root).resolve()
    same_dicom_manifest = same_dicom_root / "manifest_same_dicom_master.csv"
    if args.max_images_per_split is not None:
        tmp_root = output_root / "_tmp"
        ensure_dir(tmp_root)
        same_dicom_root, same_dicom_manifest = materialize_same_dicom_smoke_subset(
            source_root=same_dicom_root,
            source_manifest=same_dicom_manifest,
            subset_root=tmp_root / "same_dicom_smoke_subset_root",
            output_manifest=tmp_root / "manifest_same_dicom_smoke_subset.csv",
            max_images_per_split=args.max_images_per_split,
        )
        log(f"Using same-dicom smoke subset root: {same_dicom_root}")
        log(f"Using same-dicom subset manifest for smoke: {same_dicom_manifest}")

    common_args = [
        "--device", args.device,
        "--resize-size", str(args.resize_size),
        "--center-crop-size", str(args.center_crop_size),
        "--feature-batch-size", str(args.feature_batch_size),
        "--probe-batch-size", str(args.probe_batch_size),
        "--num-workers", str(args.num_workers),
        "--probe-seeds", *[str(v) for v in args.probe_seeds],
        "--probe-lr-grid", *[str(v) for v in args.probe_lr_grid],
        "--probe-max-epoch", str(args.probe_max_epoch),
        "--probe-patience", str(args.probe_patience),
        "--probe-min-delta", str(args.probe_min_delta),
        "--probe-weight-decay", str(args.probe_weight_decay),
        "--seed", str(args.seed),
    ]
    if args.strict_deterministic:
        common_args.append("--strict-deterministic")
    if args.cache_features:
        common_args.append("--cache-features")

    policy_layout = {
        "baseline_input": {
            "policy": "baseline_rgbtriplet",
            "extra": [],
        },
        "input_v1": {
            "policy": "input_v1_cag_stats_normalization",
            "extra": ["--input-stats-json", str(stats_json)],
        },
    }

    for policy_dir in policy_layout:
        base_root = output_root / policy_dir
        if args.overwrite:
            maybe_clear(base_root, overwrite=True)
        ensure_dir(base_root)

    steps: list[dict[str, object]] = []
    if not args.skip_stats and (args.overwrite or not stats_json.exists()):
        steps.append(
            {
                "label": "build input_v1 CAG stats",
                "kind": "setup",
                "cmd": [
                    sys.executable,
                    str(stats_builder),
                    "--image-root", args.unique_view_root,
                    "--split", "train",
                    "--output-json", str(stats_json),
                    "--resize-size", str(args.resize_size),
                    "--center-crop-size", str(args.center_crop_size),
                ],
            }
        )

    for policy_dir, policy_spec in policy_layout.items():
        base_root = output_root / policy_dir
        steps.append(
            {
                "label": f"{policy_dir} | global_4_1_patient_retrieval",
                "kind": "benchmark",
                "cmd": [
                    sys.executable,
                    str(TASKS["global_4_1_patient_retrieval"]["script"]),
                    "--image-root", args.unique_view_root,
                    "--output-root", str(base_root / "global_4_1_patient_retrieval"),
                    "--input-policy", str(policy_spec["policy"]),
                    *[str(v) for v in policy_spec["extra"]],
                    *common_args,
                    *([] if args.max_images_per_split is None else ["--max-images-per-split", str(args.max_images_per_split)]),
                    "--supcon-temperature", str(args.supcon_temperature),
                ],
            }
        )
        steps.append(
            {
                "label": f"{policy_dir} | global_4_2_same_dicom_retrieval",
                "kind": "benchmark",
                "cmd": [
                    sys.executable,
                    str(TASKS["global_4_2_same_dicom_retrieval"]["script"]),
                    "--image-root", str(same_dicom_root),
                    "--dataset-manifest", str(same_dicom_manifest),
                    "--output-root", str(base_root / "global_4_2_same_dicom_retrieval"),
                    "--input-policy", str(policy_spec["policy"]),
                    *[str(v) for v in policy_spec["extra"]],
                    *common_args,
                    "--supcon-temperature", str(args.supcon_temperature),
                ],
            }
        )
        steps.append(
            {
                "label": f"{policy_dir} | global_4_3_view_classification",
                "kind": "benchmark",
                "cmd": [
                    sys.executable,
                    str(TASKS["global_4_3_view_classification"]["script"]),
                    "--image-root", args.unique_view_root,
                    "--dcm-root", args.dcm_root,
                    "--output-root", str(base_root / "global_4_3_view_classification"),
                    "--input-policy", str(policy_spec["policy"]),
                    *[str(v) for v in policy_spec["extra"]],
                    *common_args,
                    *([] if args.max_images_per_split is None else ["--max-images-per-split", str(args.max_images_per_split)]),
                ],
            }
        )

    if not args.skip_summary:
        steps.append(
            {
                "label": "write experiment summary",
                "kind": "summary",
                "cmd": [
                    sys.executable,
                    str(summary_script),
                    "--exp-root", str(output_root),
                ],
            }
        )

    benchmark_runs_total = sum(1 for step in steps if step["kind"] == "benchmark")
    log(
        f"[EXP] Planned steps={len(steps)} | benchmark_runs_total={benchmark_runs_total} | "
        f"output_root={output_root}"
    )

    overall_start = perf_counter()
    benchmark_runs_done = 0
    for idx, step in enumerate(steps, start=1):
        benchmark_runs_done = execute_step(
            step_index=idx,
            total_steps=len(steps),
            step_label=str(step["label"]),
            cmd=[str(part) for part in step["cmd"]],
            is_benchmark=(step["kind"] == "benchmark"),
            benchmark_runs_done=benchmark_runs_done,
            benchmark_runs_total=benchmark_runs_total,
            overall_start=overall_start,
        )

    total_elapsed = perf_counter() - overall_start
    log(f"[EXP] All steps completed | total_elapsed={format_duration(total_elapsed)}")

    run_meta = {
        "experiment": "input_v1_cag_stats_normalization",
        "scope": "downstream_only_benchmark",
        "output_root": str(output_root),
        "stats_json": str(stats_json),
        "unique_view_root": str(Path(args.unique_view_root).resolve()),
        "same_dicom_root": str(same_dicom_root),
        "same_dicom_manifest": str(same_dicom_manifest),
        "dcm_root": str(Path(args.dcm_root).resolve()),
        "device": args.device,
        "probe_seeds": [int(v) for v in args.probe_seeds],
        "planned_steps_total": len(steps),
        "benchmark_runs_total": benchmark_runs_total,
        "total_elapsed_seconds": total_elapsed,
    }
    (output_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
