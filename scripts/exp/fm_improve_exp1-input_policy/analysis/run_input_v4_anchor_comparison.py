#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]

DEFAULT_EXP_ROOT = REPO_ROOT / "outputs/fm_improve_exp1-input_policy/input_v4_border_suppression/downstream_only"
DEFAULT_DCM_ROOT = REPO_ROOT / "input/stent_split_dcm_unique_view"

TASK_SPECS = {
    "ga4_1": {
        "label": "global_4_1_patient_retrieval",
        "script": REPO_ROOT / "scripts/analysis/global_analysis/global_4_1_cluster_anchoring_attribution.py",
        "baseline_global_root": ("baseline_input", "global_4_1_patient_retrieval"),
        "variant_global_root": ("input_v4", "global_4_1_patient_retrieval"),
        "baseline_output_root": ("baseline_input", "global_4_1_patient_retrieval", "anchoring"),
        "variant_output_root": ("input_v4", "global_4_1_patient_retrieval", "anchoring"),
        "baseline_image_root_kind": "unique_view",
        "variant_image_root_kind": "variant_unique_view",
        "global_root_arg": "--global2-root",
        "image_root_arg": "--image-root",
        "dcm_root_arg": "--dcm-root",
    },
    "ga4_2": {
        "label": "global_4_2_same_dicom_retrieval",
        "script": REPO_ROOT / "scripts/analysis/global_analysis/global_4_2_cluster_anchoring_attribution.py",
        "baseline_global_root": ("baseline_input", "global_4_2_same_dicom_retrieval"),
        "variant_global_root": ("input_v4", "global_4_2_same_dicom_retrieval"),
        "baseline_output_root": ("baseline_input", "global_4_2_same_dicom_retrieval", "anchoring"),
        "variant_output_root": ("input_v4", "global_4_2_same_dicom_retrieval", "anchoring"),
        "baseline_image_root_kind": "same_dicom",
        "variant_image_root_kind": "variant_same_dicom",
        "global_root_arg": "--global2-root",
        "image_root_arg": "--image-root",
        "dcm_root_arg": "--dcm-root",
    },
    "ga4_3": {
        "label": "global_4_3_view_classification",
        "script": REPO_ROOT / "scripts/analysis/global_analysis/global_4_3_cluster_anchoring_attribution.py",
        "baseline_global_root": ("baseline_input", "global_4_3_view_classification"),
        "variant_global_root": ("input_v4", "global_4_3_view_classification"),
        "baseline_output_root": ("baseline_input", "global_4_3_view_classification", "anchoring"),
        "variant_output_root": ("input_v4", "global_4_3_view_classification", "anchoring"),
        "baseline_image_root_kind": "unique_view",
        "variant_image_root_kind": "variant_unique_view",
        "global_root_arg": "--global3-root",
        "image_root_arg": "--image-root",
        "dcm_root_arg": "--dcm-root",
    },
}


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_clear(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)


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
        f"[EXP-ETA] current={label} | anchor_runs_done={completed_runs}/{total_runs} "
        f"| anchor_runs_remaining={max(0, total_runs - completed_runs)} "
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


def resolve_roots(exp_root: Path) -> dict[str, Path]:
    variant_cache = exp_root / "input_v4" / "cache"
    return {
        "unique_view": (REPO_ROOT / "input/Stent-Contrast-unique-view").resolve(),
        "same_dicom": (REPO_ROOT / "input/Stent-Contrast-same-dicom-unique-view").resolve(),
        "variant_unique_view": (variant_cache / "unique_view").resolve(),
        "variant_same_dicom": (variant_cache / "same_dicom").resolve(),
    }


def build_runs(exp_root: Path, args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    roots = resolve_roots(exp_root)
    runs: list[tuple[str, list[str]]] = []
    probe_seeds = [str(v) for v in parse_int_list(args.probe_seeds_str)]
    for task_name in args.tasks:
        spec = TASK_SPECS[task_name]
        for variant_name, global_key, output_key, image_kind in [
            ("baseline_input", "baseline_global_root", "baseline_output_root", spec["baseline_image_root_kind"]),
            ("input_v4", "variant_global_root", "variant_output_root", spec["variant_image_root_kind"]),
        ]:
            global_root = exp_root.joinpath(*spec[global_key])
            output_root = exp_root.joinpath(*spec[output_key])
            maybe_clear(output_root, overwrite=args.overwrite)
            ensure_dir(output_root)
            cmd = [
                sys.executable,
                str(spec["script"]),
                spec["global_root_arg"],
                str(global_root),
                spec["image_root_arg"],
                str(roots[image_kind]),
                spec["dcm_root_arg"],
                str(Path(args.dcm_root).resolve()),
                "--output-root",
                str(output_root),
                "--probe-seeds",
                *probe_seeds,
                "--probe-batch-size",
                str(args.probe_batch_size),
                "--device",
                str(args.device),
                "--seed",
                str(args.seed),
            ]
            if args.max_images is not None and int(args.max_images) > 0:
                cmd.extend(["--max-images", str(args.max_images)])
            runs.append((f"{variant_name} | {spec['label']} | anchoring", cmd))
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", default=str(DEFAULT_EXP_ROOT))
    parser.add_argument("--dcm-root", default=str(DEFAULT_DCM_ROOT))
    parser.add_argument("--tasks", nargs="+", default=["ga4_1"], choices=sorted(TASK_SPECS.keys()))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--probe-seeds-str", default="11 22 33")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    ensure_dir(exp_root)
    env = build_env()

    runs = build_runs(exp_root, args)
    started_at = time.time()
    completed_runs = 0
    total_runs = len(runs)

    for label, cmd in runs:
        run_started_at = time.time()
        log(
            f"START {label} | anchor_runs_done={completed_runs}/{total_runs} "
            f"| anchor_runs_remaining={total_runs - completed_runs}"
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
            f"DONE {label} | anchor_runs_done={completed_runs}/{total_runs} "
            f"| anchor_runs_remaining={total_runs - completed_runs} "
            f"| remaining_est_seconds~{remaining:.1f}"
        )

    if not args.skip_summary:
        summary_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "summarize_input_v4_anchor_comparison.py"),
            "--exp-root",
            str(exp_root),
            "--tasks",
            *list(args.tasks),
        ]
        run_command(summary_cmd, env=env)

    run_meta = {
        "exp_root": str(exp_root),
        "tasks": list(args.tasks),
        "device": str(args.device),
        "probe_batch_size": int(args.probe_batch_size),
        "probe_seeds": parse_int_list(args.probe_seeds_str),
        "max_images": args.max_images,
        "seed": int(args.seed),
        "total_elapsed_seconds": time.time() - started_at,
    }
    out_path = exp_root / "anchoring_reports" / "run_meta_input_v4_anchor_comparison.json"
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    log(f"Finished input_v4 anchor comparison. Exp root: {exp_root}")


if __name__ == "__main__":
    main()
