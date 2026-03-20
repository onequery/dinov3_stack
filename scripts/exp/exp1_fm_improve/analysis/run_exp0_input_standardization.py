#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent

ALLOWED_VARIANTS = ["norm_v1", "norm_v2", "norm_v3"]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp-0 PNG-space input-standardization control.")
    parser.add_argument("--variants", nargs="+", default=ALLOWED_VARIANTS)
    parser.add_argument("--unique-view-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--same-dicom-root", default="input/Stent-Contrast-same-dicom-unique-view")
    parser.add_argument("--dcm-root", default="input/stent_split_dcm_unique_view")
    parser.add_argument("--output-root", default="outputs/exp1_fm_improve/ablation/exp0_input_standardization")
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
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-anchoring", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    ensure_dir(out_root)
    log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")

    builder_script = REPO_ROOT / "scripts/exp/exp1_fm_improve/preprocess/build_exp0_input_standardization_cache.py"
    summary_script = REPO_ROOT / "scripts/exp/exp1_fm_improve/analysis/summarize_exp0_input_standardization.py"
    ga2_script = REPO_ROOT / "scripts/analysis/global_analysis/global_2_study_patient_retrieval.py"
    ga41_anchor_script = REPO_ROOT / "scripts/analysis/global_analysis/global_4_1_cluster_anchoring_attribution.py"
    ga42_script = REPO_ROOT / "scripts/analysis/global_analysis/global_4_2_same_dicom_retrieval.py"
    ga42_anchor_script = REPO_ROOT / "scripts/analysis/global_analysis/global_4_2_cluster_anchoring_attribution.py"
    ga43_script = REPO_ROOT / "scripts/analysis/global_analysis/global_4_3_view_classification.py"
    ga43_anchor_script = REPO_ROOT / "scripts/analysis/global_analysis/global_4_3_cluster_anchoring_attribution.py"

    common_benchmark_args = [
        "--device", args.device,
        "--resize-size", str(args.resize_size),
        "--center-crop-size", str(args.center_crop_size),
        "--feature-batch-size", str(args.feature_batch_size),
        "--probe-batch-size", str(args.probe_batch_size),
        "--num-workers", str(args.num_workers),
        "--probe-max-epoch", str(args.probe_max_epoch),
        "--probe-patience", str(args.probe_patience),
        "--probe-min-delta", str(args.probe_min_delta),
        "--probe-weight-decay", str(args.probe_weight_decay),
        "--seed", str(args.seed),
        "--probe-seeds", *[str(v) for v in args.probe_seeds],
        "--probe-lr-grid", *[str(v) for v in args.probe_lr_grid],
    ]
    if args.strict_deterministic:
        common_benchmark_args.append("--strict-deterministic")
    should_cache_features = args.cache_features or (not args.skip_benchmark and not args.skip_anchoring)
    if should_cache_features:
        common_benchmark_args.append("--cache-features")

    for variant in args.variants:
        variant_root = out_root / variant
        unique_cache_root = variant_root / "cache" / "unique_view"
        same_cache_root = variant_root / "cache" / "same_dicom"
        same_manifest = same_cache_root / "manifest_same_dicom_master.csv"

        if not args.skip_cache:
            cmd = [
                sys.executable,
                str(builder_script),
                "--variant", variant,
                "--unique-view-root", args.unique_view_root,
                "--same-dicom-root", args.same_dicom_root,
                "--output-root", str(out_root),
                "--seed", str(args.seed),
            ]
            if args.max_images_per_split is not None:
                cmd += ["--max-images-per-split", str(args.max_images_per_split)]
            if args.overwrite:
                cmd.append("--overwrite")
            run_cmd(cmd)

        if args.skip_benchmark:
            continue

        ga41_bench_root = variant_root / "global_4_1_patient_retrieval_control" / "benchmark"
        ga41_anchor_root = variant_root / "global_4_1_patient_retrieval_control" / "anchoring"
        ga42_root = variant_root / "global_4_2_same_dicom_control"
        ga43_root = variant_root / "global_4_3_view_classification_control"

        if args.overwrite:
            maybe_clear(ga41_bench_root, overwrite=True)
            maybe_clear(ga41_anchor_root, overwrite=True)
            maybe_clear(ga42_root, overwrite=True)
            maybe_clear(ga43_root, overwrite=True)

        run_cmd([
            sys.executable,
            str(ga2_script),
            "--image-root", str(unique_cache_root),
            "--output-root", str(ga41_bench_root),
            *common_benchmark_args,
            *([] if args.max_images_per_split is None else ["--max-images-per-split", str(args.max_images_per_split)]),
            "--supcon-temperature", str(args.supcon_temperature),
        ])

        run_cmd([
            sys.executable,
            str(ga42_script),
            "--image-root", str(same_cache_root),
            "--dataset-manifest", str(same_manifest),
            "--output-root", str(ga42_root),
            *common_benchmark_args,
            "--supcon-temperature", str(args.supcon_temperature),
        ])

        run_cmd([
            sys.executable,
            str(ga43_script),
            "--image-root", str(unique_cache_root),
            "--dcm-root", args.dcm_root,
            "--output-root", str(ga43_root),
            *common_benchmark_args,
            *([] if args.max_images_per_split is None else ["--max-images-per-split", str(args.max_images_per_split)]),
        ])

        if args.skip_anchoring:
            continue

        run_cmd([
            sys.executable,
            str(ga41_anchor_script),
            "--global2-root", str(ga41_bench_root),
            "--image-root", str(unique_cache_root),
            "--dcm-root", args.dcm_root,
            "--output-root", str(ga41_anchor_root),
            "--probe-seeds", *[str(v) for v in args.probe_seeds],
            "--probe-batch-size", str(max(256, args.probe_batch_size)),
            "--device", args.device,
            "--seed", str(args.seed),
        ])

        run_cmd([
            sys.executable,
            str(ga42_anchor_script),
            "--global2-root", str(ga42_root),
            "--image-root", str(same_cache_root),
            "--dcm-root", args.dcm_root,
            "--output-root", str(ga42_root),
            "--probe-seeds", *[str(v) for v in args.probe_seeds],
            "--probe-batch-size", str(max(256, args.probe_batch_size)),
            "--device", args.device,
            "--seed", str(args.seed),
        ])

        run_cmd([
            sys.executable,
            str(ga43_anchor_script),
            "--global3-root", str(ga43_root),
            "--image-root", str(unique_cache_root),
            "--dcm-root", args.dcm_root,
            "--output-root", str(ga43_root),
            "--probe-seeds", *[str(v) for v in args.probe_seeds],
            "--probe-batch-size", str(max(256, args.probe_batch_size)),
            "--device", args.device,
            "--seed", str(args.seed),
        ])

    if not args.skip_summary:
        run_cmd([
            sys.executable,
            str(summary_script),
            "--exp-root", str(out_root),
            "--variants", *list(args.variants),
        ])


if __name__ == "__main__":
    main()
