#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]


def _load_cache_builder():
    module_path = SCRIPT_DIR.parent / "preprocess/build_input_v2_percentile_cache.py"
    spec = importlib.util.spec_from_file_location("build_input_v2_percentile_cache", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load cache builder module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_cache_builder = _load_cache_builder()
build_input_v2_cache = _cache_builder.build_input_v2_cache


DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "outputs/fm_improve_exp1-input_policy/input_v2_percentile_canonicalization/downstream_only"
)
DEFAULT_UNIQUE_VIEW_ROOT = REPO_ROOT / "input/Stent-Contrast-unique-view"
DEFAULT_SAME_DICOM_ROOT = REPO_ROOT / "input/Stent-Contrast-same-dicom-unique-view"
DEFAULT_SAME_DICOM_MANIFEST = DEFAULT_SAME_DICOM_ROOT / "manifest_same_dicom_master.csv"


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_split_images(root: Path, split: str) -> list[Path]:
    split_root = root / split
    if not split_root.exists():
        return []
    return sorted(path for path in split_root.rglob("*.png") if path.is_file())


def build_unique_view_subset(source_root: Path, dest_root: Path, max_images_per_split: int, overwrite: bool) -> Path:
    ensure_clean_dir(dest_root, overwrite=overwrite)
    for split in ("train", "valid", "test"):
        selected = list_split_images(source_root, split)[:max_images_per_split]
        for src_path in selected:
            rel_path = src_path.relative_to(source_root)
            dst_path = dest_root / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
    return dest_root


def resolve_dataset_image_path(source_root: Path, raw_value: object) -> Path:
    candidate = Path(str(raw_value))
    if candidate.is_absolute():
        return candidate.resolve()
    options = [
        (source_root / candidate).resolve(),
        (REPO_ROOT / candidate).resolve(),
    ]
    for option in options:
        if option.exists():
            return option
    parts = candidate.parts
    for split in ("train", "valid", "test"):
        if split in parts:
            option = (source_root / Path(*parts[parts.index(split):])).resolve()
            if option.exists():
                return option
    return options[0]


def infer_relative_dataset_path(source_path: Path, dataset_root: Path) -> Path:
    try:
        return source_path.relative_to(dataset_root.resolve())
    except ValueError:
        parts = source_path.parts
        for split in ("train", "valid", "test"):
            if split in parts:
                return Path(*parts[parts.index(split):])
        raise ValueError(f"Unable to infer relative dataset path for: {source_path}")


def build_same_dicom_subset(
    source_root: Path,
    source_manifest: Path,
    dest_root: Path,
    dest_manifest: Path,
    max_images_per_split: int,
    overwrite: bool,
) -> tuple[Path, Path]:
    ensure_clean_dir(dest_root, overwrite=overwrite)
    manifest_df = pd.read_csv(source_manifest)
    if "split" not in manifest_df.columns or "dicom_id" not in manifest_df.columns:
        raise ValueError("same-DICOM manifest must contain 'split' and 'dicom_id' for smoke subset builds.")

    max_pairs = max(1, max_images_per_split // 2)
    selected_frames = []
    for split in ("train", "valid", "test"):
        split_df = manifest_df[manifest_df["split"] == split].copy()
        chosen_ids = sorted(split_df["dicom_id"].drop_duplicates().tolist())[:max_pairs]
        selected_frames.append(split_df[split_df["dicom_id"].isin(chosen_ids)].copy())
    subset_df = pd.concat(selected_frames, axis=0, ignore_index=True)

    new_paths: list[str] = []
    for value in subset_df["img_path"].tolist():
        src_path = resolve_dataset_image_path(source_root, value)
        rel_path = infer_relative_dataset_path(src_path, source_root)
        dst_path = dest_root / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        new_paths.append(str(dst_path.resolve()))
    subset_df["img_path"] = new_paths
    dest_manifest.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(dest_manifest, index=False)
    return dest_root, dest_manifest


def maybe_build_smoke_subsets(
    unique_view_root: Path,
    same_dicom_root: Path,
    same_dicom_manifest: Path,
    tmp_root: Path,
    max_images_per_split: int | None,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    if not max_images_per_split or max_images_per_split <= 0:
        return unique_view_root, same_dicom_root, same_dicom_manifest

    unique_subset_root = tmp_root / "unique_view_smoke_subset"
    same_subset_root = tmp_root / "same_dicom_smoke_subset"
    same_subset_manifest = tmp_root / "manifest_same_dicom_smoke_subset.csv"

    build_unique_view_subset(unique_view_root, unique_subset_root, max_images_per_split, overwrite=overwrite)
    build_same_dicom_subset(
        same_dicom_root,
        same_dicom_manifest,
        same_subset_root,
        same_subset_manifest,
        max_images_per_split=max_images_per_split,
        overwrite=overwrite,
    )
    return unique_subset_root, same_subset_root, same_subset_manifest


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    cache_root = REPO_ROOT / "outputs/fm_improve_exp1-input_policy/.runtime_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    env.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "estimating"
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def estimate_overall_remaining_seconds(
    *,
    started_at: float,
    completed_runs: int,
    total_runs: int,
    current_run_started_at: float | None,
) -> tuple[float, float | None]:
    now = time.time()
    total_elapsed = now - started_at
    if total_runs <= 0:
        return total_elapsed, 0.0

    if current_run_started_at is None:
        if completed_runs <= 0:
            return total_elapsed, None
        avg_completed = total_elapsed / completed_runs
        remaining = avg_completed * max(0, total_runs - completed_runs)
        return total_elapsed, remaining

    current_run_elapsed = max(0.0, now - current_run_started_at)
    if completed_runs <= 0:
        remaining = current_run_elapsed * max(0, total_runs - 1)
        return total_elapsed, remaining

    elapsed_before_current = max(0.0, current_run_started_at - started_at)
    avg_completed = elapsed_before_current / completed_runs
    estimated_current_total = max(avg_completed, current_run_elapsed)
    remaining_current = max(0.0, estimated_current_total - current_run_elapsed)
    remaining_future = avg_completed * max(0, total_runs - completed_runs - 1)
    return total_elapsed, remaining_current + remaining_future


def log_overall_eta(
    *,
    label: str,
    started_at: float,
    completed_runs: int,
    total_runs: int,
    current_run_started_at: float | None,
) -> None:
    total_elapsed, remaining_seconds = estimate_overall_remaining_seconds(
        started_at=started_at,
        completed_runs=completed_runs,
        total_runs=total_runs,
        current_run_started_at=current_run_started_at,
    )
    remaining_text = format_duration(remaining_seconds)
    eta_text = (
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() + remaining_seconds))
        if remaining_seconds is not None
        else "estimating"
    )
    log(
        f"[EXP-ETA] current={label} | benchmark_runs_done={completed_runs}/{total_runs} "
        f"| benchmark_runs_remaining={max(0, total_runs - completed_runs)} "
        f"| total_elapsed={format_duration(total_elapsed)} "
        f"| remaining_est={remaining_text} | eta={eta_text}"
    )


def run_command(
    cmd: list[str],
    env: dict[str, str],
    *,
    label: str | None = None,
    started_at: float | None = None,
    completed_runs: int | None = None,
    total_runs: int | None = None,
    current_run_started_at: float | None = None,
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
            if (
                label is not None
                and started_at is not None
                and completed_runs is not None
                and total_runs is not None
                and "[ETA]" in line
            ):
                log_overall_eta(
                    label=label,
                    started_at=started_at,
                    completed_runs=completed_runs,
                    total_runs=total_runs,
                    current_run_started_at=current_run_started_at,
                )
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def parse_int_list(raw: str) -> list[int]:
    return [int(token) for token in raw.split() if token.strip()]


def parse_float_list(raw: str) -> list[float]:
    return [float(token) for token in raw.split() if token.strip()]


def add_common_benchmark_args(
    cmd: list[str],
    args: argparse.Namespace,
    *,
    include_max_images_per_split: bool,
) -> list[str]:
    cmd.extend(["--device", str(args.device)])
    cmd.extend(["--feature-batch-size", str(args.feature_batch_size)])
    cmd.extend(["--num-workers", str(args.num_workers)])
    cmd.extend(["--probe-batch-size", str(args.probe_batch_size)])
    cmd.extend(["--probe-max-epoch", str(args.probe_max_epoch)])
    cmd.extend(["--probe-patience", str(args.probe_patience)])
    if include_max_images_per_split and int(args.max_images_per_split) > 0:
        cmd.extend(["--max-images-per-split", str(args.max_images_per_split)])
    probe_seeds = [str(seed) for seed in parse_int_list(args.probe_seeds_str)]
    if probe_seeds:
        cmd.extend(["--probe-seeds", *probe_seeds])
    probe_lr_grid = [str(value) for value in parse_float_list(args.probe_lr_grid_str)]
    if probe_lr_grid:
        cmd.extend(["--probe-lr-grid", *probe_lr_grid])
    if args.cache_features:
        cmd.append("--cache-features")
    if args.strict_deterministic:
        cmd.append("--strict-deterministic")
    if args.overwrite and args.pass_overwrite:
        cmd.append("--overwrite")
    return cmd


def build_benchmark_runs(
    baseline_unique_root: Path,
    baseline_same_root: Path,
    baseline_same_manifest: Path,
    variant_unique_root: Path,
    variant_same_root: Path,
    variant_same_manifest: Path,
    output_root: Path,
) -> list[tuple[str, list[str]]]:
    py = sys.executable
    runs: list[tuple[str, list[str]]] = []
    runs.append(
        (
            "baseline_input | global_4_1_patient_retrieval",
            [
                py,
                str(REPO_ROOT / "scripts/analysis/global_analysis/global_2_study_patient_retrieval.py"),
                "--image-root",
                str(baseline_unique_root),
                "--output-root",
                str(output_root / "baseline_input/global_4_1_patient_retrieval"),
            ],
        )
    )
    runs.append(
        (
            "input_v2 | global_4_1_patient_retrieval",
            [
                py,
                str(REPO_ROOT / "scripts/analysis/global_analysis/global_2_study_patient_retrieval.py"),
                "--image-root",
                str(variant_unique_root),
                "--output-root",
                str(output_root / "input_v2/global_4_1_patient_retrieval"),
            ],
        )
    )
    runs.append(
        (
            "baseline_input | global_4_2_same_dicom_retrieval",
            [
                py,
                str(REPO_ROOT / "scripts/analysis/global_analysis/global_4_2_same_dicom_retrieval.py"),
                "--image-root",
                str(baseline_same_root),
                "--dataset-manifest",
                str(baseline_same_manifest),
                "--output-root",
                str(output_root / "baseline_input/global_4_2_same_dicom_retrieval"),
            ],
        )
    )
    runs.append(
        (
            "input_v2 | global_4_2_same_dicom_retrieval",
            [
                py,
                str(REPO_ROOT / "scripts/analysis/global_analysis/global_4_2_same_dicom_retrieval.py"),
                "--image-root",
                str(variant_same_root),
                "--dataset-manifest",
                str(variant_same_manifest),
                "--output-root",
                str(output_root / "input_v2/global_4_2_same_dicom_retrieval"),
            ],
        )
    )
    runs.append(
        (
            "baseline_input | global_4_3_view_classification",
            [
                py,
                str(REPO_ROOT / "scripts/analysis/global_analysis/global_4_3_view_classification.py"),
                "--image-root",
                str(baseline_unique_root),
                "--output-root",
                str(output_root / "baseline_input/global_4_3_view_classification"),
            ],
        )
    )
    runs.append(
        (
            "input_v2 | global_4_3_view_classification",
            [
                py,
                str(REPO_ROOT / "scripts/analysis/global_analysis/global_4_3_view_classification.py"),
                "--image-root",
                str(variant_unique_root),
                "--output-root",
                str(output_root / "input_v2/global_4_3_view_classification"),
            ],
        )
    )
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--unique-view-root", default=str(DEFAULT_UNIQUE_VIEW_ROOT))
    parser.add_argument("--same-dicom-root", default=str(DEFAULT_SAME_DICOM_ROOT))
    parser.add_argument("--same-dicom-manifest", default=str(DEFAULT_SAME_DICOM_MANIFEST))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-seeds-str", default="11 22 33")
    parser.add_argument("--probe-lr-grid-str", default="0.001 0.003 0.01")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-images-per-split", type=int, default=0)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--skip-cache", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--pass-overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_clean_dir(output_root, overwrite=False)
    tmp_root = output_root / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    baseline_unique_root, baseline_same_root, baseline_same_manifest = maybe_build_smoke_subsets(
        unique_view_root=Path(args.unique_view_root).resolve(),
        same_dicom_root=Path(args.same_dicom_root).resolve(),
        same_dicom_manifest=Path(args.same_dicom_manifest).resolve(),
        tmp_root=tmp_root,
        max_images_per_split=int(args.max_images_per_split) if args.max_images_per_split else None,
        overwrite=args.overwrite,
    )

    cache_root = output_root / "input_v2" / "cache"
    if args.skip_cache:
        log(f"SKIP cache build | cache_root={cache_root}")
        variant_unique_root = cache_root / "unique_view"
        variant_same_root = cache_root / "same_dicom"
        variant_same_manifest = variant_same_root / "manifest_same_dicom_master.csv"
        cache_metadata = cache_root / "cache_metadata.json"
    else:
        cache_started_at = time.time()
        log(f"START cache build | cache_root={cache_root}")
        outputs = build_input_v2_cache(
            unique_view_root=baseline_unique_root,
            same_dicom_root=baseline_same_root,
            same_dicom_manifest=baseline_same_manifest,
            output_root=cache_root,
            overwrite=args.overwrite,
        )
        variant_unique_root = outputs.unique_view_root
        variant_same_root = outputs.same_dicom_root
        variant_same_manifest = outputs.same_dicom_manifest
        cache_metadata = outputs.metadata_json
        log(
            f"DONE cache build | elapsed={format_duration(time.time() - cache_started_at)} "
            f"| cache_metadata_json={cache_metadata}"
        )

    env = build_env()
    runs = build_benchmark_runs(
        baseline_unique_root=baseline_unique_root,
        baseline_same_root=baseline_same_root,
        baseline_same_manifest=baseline_same_manifest,
        variant_unique_root=variant_unique_root,
        variant_same_root=variant_same_root,
        variant_same_manifest=variant_same_manifest,
        output_root=output_root,
    )

    started_at = time.time()
    completed_runs = 0
    if not args.skip_benchmark:
        total_runs = len(runs)
        for label, cmd in runs:
            run_started_at = time.time()
            log(
                f"START {label} | benchmark_runs_done={completed_runs}/{total_runs} "
                f"| benchmark_runs_remaining={total_runs - completed_runs}"
            )
            cmd = add_common_benchmark_args(
                cmd,
                args,
                include_max_images_per_split="global_4_2_same_dicom_retrieval" not in label,
            )
            run_command(
                cmd,
                env=env,
                label=label,
                started_at=started_at,
                completed_runs=completed_runs,
                total_runs=total_runs,
                current_run_started_at=run_started_at,
            )
            completed_runs += 1
            elapsed = time.time() - started_at
            avg = elapsed / max(1, completed_runs)
            remaining = avg * (total_runs - completed_runs)
            log(
                f"DONE {label} | benchmark_runs_done={completed_runs}/{total_runs} "
                f"| benchmark_runs_remaining={total_runs - completed_runs} "
                f"| remaining_est_seconds~{remaining:.1f}"
            )

    reports_dir = output_root / "reports"
    if not args.skip_summary:
        summary_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "summarize_input_v2_percentile_canonicalization.py"),
            "--exp-root",
            str(output_root),
        ]
        run_command(summary_cmd, env=env)

    if not args.skip_plot:
        plot_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "plot_input_v2_task_comparison.py"),
            "--summary-csv",
            str(reports_dir / "summary_input_v2_benchmark_metrics.csv"),
            "--output-dir",
            str(reports_dir),
            "--split",
            "test",
        ]
        run_command(plot_cmd, env=env)

    run_meta = {
        "policy_name": "input_v2_percentile_canonicalization",
        "scope": "downstream_only_stage_a1",
        "output_root": str(output_root),
        "baseline_unique_view_root": str(baseline_unique_root),
        "baseline_same_dicom_root": str(baseline_same_root),
        "baseline_same_dicom_manifest": str(baseline_same_manifest),
        "variant_unique_view_root": str(variant_unique_root),
        "variant_same_dicom_root": str(variant_same_root),
        "variant_same_dicom_manifest": str(variant_same_manifest),
        "cache_metadata_json": str(cache_metadata),
        "max_images_per_split": int(args.max_images_per_split),
        "device": args.device,
        "feature_batch_size": int(args.feature_batch_size),
        "probe_batch_size": int(args.probe_batch_size),
        "probe_max_epoch": int(args.probe_max_epoch),
        "probe_patience": int(args.probe_patience),
        "probe_seeds": parse_int_list(args.probe_seeds_str),
        "probe_lr_grid": parse_float_list(args.probe_lr_grid_str),
        "num_workers": int(args.num_workers),
        "cache_features": bool(args.cache_features),
        "strict_deterministic": bool(args.strict_deterministic),
        "total_elapsed_seconds": time.time() - started_at,
    }
    (output_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    log(f"Finished input_v2 Stage A1. Output root: {output_root}")


if __name__ == "__main__":
    main()
