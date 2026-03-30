#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
BASE_RETRIEVAL_SCRIPT = REPO_ROOT / "scripts/analysis/global_analysis/global_2_study_patient_retrieval.py"
PLOT_COLORS = {"imagenet": "#4C78A8", "cag": "#F58518"}


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrast Benchmark — Patient Retrieval")
    parser.add_argument("--dataset-root", default="input/contrast_benchmark/1_global/2_patient_retrieval")
    parser.add_argument("--output-root", default="outputs/fm-imp-exp0_baseline/1_global/2_patient_retrieval")
    parser.add_argument("--imagenet-ckpt", default="dinov3/output/a6000/1_pretrain/dinov3_vits16/2_imagenet1k/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth")
    parser.add_argument("--cag-ckpt", default="dinov3/output/a6000/1_pretrain/dinov3_vits16/3_cagcontfm3m/3_stage3_high_res_adapt/eval/training_29999/teacher_checkpoint.pth")
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-size", type=int, default=480)
    parser.add_argument("--center-crop-size", type=int, default=448)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--input-policy", default="baseline_rgbtriplet")
    parser.add_argument("--input-stats-json", default=None)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--probe-lr-grid", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-min-delta", type=float, default=0.0)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--supcon-temperature", type=float, default=0.07)
    parser.add_argument("--max-topk-log", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sort_view_dirs(dataset_root: Path) -> List[Path]:
    def key(path: Path) -> tuple[int, str]:
        prefix, _, suffix = path.name.partition("_")
        try:
            return int(prefix), suffix
        except ValueError:
            return 9999, path.name

    return sorted([p for p in dataset_root.iterdir() if p.is_dir()], key=key)


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


def resolve_device(requested: str) -> str:
    lowered = str(requested).lower()
    if lowered.startswith("cuda"):
        if torch is None or not torch.cuda.is_available():
            log("CUDA requested for contrast benchmark patient retrieval but unavailable. Falling back to CPU.")
            return "cpu"
    return requested


def aggregate_macro(detail_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metric_cols = [
        "mAP_mean",
        "recall_at_1_mean",
        "recall_at_5_mean",
        "recall_at_10_mean",
        "median_first_positive_rank_mean",
        "queries_with_positive_mean",
        "queries_with_no_positive_mean",
    ]
    for (mode, backbone_name, split), group in detail_df.groupby(["mode", "backbone_name", "split"], sort=True):
        row: Dict[str, object] = {
            "mode": mode,
            "target": "patient",
            "backbone_name": backbone_name,
            "split": split,
            "num_views": int(group["view_label"].nunique()),
        }
        for metric in metric_cols:
            row[metric] = float(group[metric].mean())
            row[metric.replace("_mean", "_std_across_views")] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def make_grouped_bar_positions(group_count: int, bar_count: int, width: float = 0.18):
    centers = pd.Series(range(group_count), dtype="float64").to_numpy()
    offsets = []
    start = -width * (bar_count - 1) / 2.0
    for idx in range(bar_count):
        offsets.append(centers + start + idx * width)
    return centers, offsets


def save_macro_map_compare_figure(macro_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = macro_df.copy()
    splits = ["valid", "test"]
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    centers, xs = make_grouped_bar_positions(len(splits), len(conditions), width=0.18)
    plt.figure(figsize=(10, 6))
    for idx, (backbone_name, mode, label) in enumerate(conditions):
        means = []
        errs = []
        for split in splits:
            row = plot_df[(plot_df["split"] == split) & (plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)]
            if row.empty:
                means.append(float("nan"))
                errs.append(0.0)
            else:
                means.append(float(row.iloc[0]["mAP_mean"]))
                errs.append(float(row.iloc[0]["mAP_std_across_views"]))
        color = PLOT_COLORS[backbone_name]
        alpha = 0.55 if mode == "raw_frozen" else 0.95
        plt.bar(xs[idx], means, width=0.18, color=color, alpha=alpha, label=label, yerr=errs, capsize=4)
    plt.xticks(centers, ["Valid", "Test"])
    plt.ylabel("mAP")
    plt.title("Contrast Benchmark: Patient Retrieval mAP (Macro Across Views)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_macro_recall_compare_figure(macro_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = macro_df.copy()
    splits = ["valid", "test"]
    ks = [1, 5, 10]
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, split in zip(axes, splits):
        centers, xs = make_grouped_bar_positions(len(ks), len(conditions), width=0.18)
        for idx, (backbone_name, mode, label) in enumerate(conditions):
            means = []
            errs = []
            row = plot_df[(plot_df["split"] == split) & (plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)]
            for k in ks:
                if row.empty:
                    means.append(float("nan"))
                    errs.append(0.0)
                else:
                    means.append(float(row.iloc[0][f"recall_at_{k}_mean"]))
                    errs.append(float(row.iloc[0][f"recall_at_{k}_std_across_views"]))
            color = PLOT_COLORS[backbone_name]
            alpha = 0.55 if mode == "raw_frozen" else 0.95
            ax.bar(xs[idx], means, width=0.18, color=color, alpha=alpha, yerr=errs, capsize=4, label=label)
        ax.set_xticks(centers, ["R@1", "R@5", "R@10"])
        ax.set_title(split.title())
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Recall@K")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Contrast Benchmark: Patient Retrieval Recall@K (Macro Across Views)", y=1.12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(output_path: Path, macro_df: pd.DataFrame, detail_df: pd.DataFrame, run_meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Contrast Benchmark — Patient Retrieval")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- dataset root: `{run_meta['dataset_root']}`")
    lines.append(f"- output root: `{run_meta['output_root']}`")
    lines.append(f"- evaluated views: `{', '.join(run_meta['view_labels'])}`")
    lines.append(f"- probe seeds: `{', '.join(str(v) for v in run_meta['probe_seed_set'])}`")
    lines.append("")
    lines.append("## Macro Across Views")
    lines.append("")
    lines.append("| mode | split | backbone | mAP | R@1 | R@5 | R@10 | no-positive |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    display_df = macro_df.sort_values(["split", "mode", "backbone_name"]).reset_index(drop=True)
    for _, row in display_df.iterrows():
        lines.append(
            f"| {row['mode']} | {row['split']} | {row['backbone_name']} | "
            f"{float(row['mAP_mean']):.4f} | {float(row['recall_at_1_mean']):.4f} | {float(row['recall_at_5_mean']):.4f} | "
            f"{float(row['recall_at_10_mean']):.4f} | {float(row['queries_with_no_positive_mean']):.4f} |"
        )
    lines.append("")
    lines.append("## Probe Test by View")
    lines.append("")
    lines.append("| view | backbone | mAP | R@1 | R@5 | R@10 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    probe_test = detail_df[(detail_df["mode"] == "probe_linear") & (detail_df["split"] == "test")].copy()
    probe_test = probe_test.sort_values(["view_label", "backbone_name"]).reset_index(drop=True)
    for _, row in probe_test.iterrows():
        lines.append(
            f"| {row['view_label']} | {row['backbone_name']} | {float(row['mAP_mean']):.4f} | "
            f"{float(row['recall_at_1_mean']):.4f} | {float(row['recall_at_5_mean']):.4f} | {float(row['recall_at_10_mean']):.4f} |"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- macro mAP compare: `fig_patient_retrieval_map_compare.png`")
    lines.append("- macro Recall@K compare: `fig_patient_retrieval_recall_at_k_compare.png`")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    out_root = Path(args.output_root).resolve()
    ensure_dir(out_root)
    views_root = out_root / "views"
    ensure_dir(views_root)

    log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
    view_dirs = sort_view_dirs(dataset_root)
    if not view_dirs:
        raise FileNotFoundError(f"No view directories found under {dataset_root}")
    effective_device = resolve_device(args.device)

    started = time.time()
    patient_detail_rows: List[pd.DataFrame] = []
    for idx, view_dir in enumerate(view_dirs, start=1):
        view_output = views_root / view_dir.name
        ensure_dir(view_output)
        log(f"[{idx}/{len(view_dirs)}] START view={view_dir.name}")
        cmd = [
            sys.executable,
            str(BASE_RETRIEVAL_SCRIPT),
            "--image-root",
            str(view_dir),
            "--output-root",
            str(view_output),
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
        if args.input_stats_json:
            cmd.extend(["--input-stats-json", args.input_stats_json])
        if args.max_images_per_split is not None:
            cmd.extend(["--max-images-per-split", str(args.max_images_per_split)])
        if args.cache_features:
            cmd.append("--cache-features")
        if args.strict_deterministic:
            cmd.append("--strict-deterministic")
        run_cmd(cmd)

        summary_path = view_output / "summary_global_2_retrieval.csv"
        summary_df = pd.read_csv(summary_path)
        patient_df = summary_df.loc[summary_df["target"] == "patient"].copy()
        patient_df.insert(0, "view_label", view_dir.name)
        patient_df.insert(1, "view_root", str(view_dir))
        patient_detail_rows.append(patient_df)
        elapsed = max(0.0, time.time() - started)
        avg = elapsed / idx
        remaining = avg * (len(view_dirs) - idx)
        eta = datetime.now() + pd.to_timedelta(remaining, unit="s")
        log(f"[{idx}/{len(view_dirs)}] DONE view={view_dir.name} | elapsed={elapsed:.1f}s | remaining~{remaining:.1f}s | eta={eta.strftime('%Y-%m-%d %H:%M:%S')}")

    detail_df = pd.concat(patient_detail_rows, axis=0, ignore_index=True)
    detail_df.to_csv(out_root / "summary_patient_retrieval_by_view.csv", index=False)
    macro_df = aggregate_macro(detail_df)
    macro_df.to_csv(out_root / "summary_patient_retrieval.csv", index=False)
    save_macro_map_compare_figure(macro_df, out_root / "fig_patient_retrieval_map_compare.png")
    save_macro_recall_compare_figure(macro_df, out_root / "fig_patient_retrieval_recall_at_k_compare.png")

    run_meta = {
        "analysis": "contrast_benchmark_patient_retrieval",
        "dataset_root": str(dataset_root),
        "output_root": str(out_root),
        "view_labels": [path.name for path in view_dirs],
        "probe_seed_set": [int(v) for v in args.probe_seeds],
        "requested_device": args.device,
        "effective_device": effective_device,
        "aggregate_policy": "macro_across_views",
        "selected_targets": ["patient"],
        "detail_summary_path": str((out_root / "summary_patient_retrieval_by_view.csv").resolve()),
        "aggregate_summary_path": str((out_root / "summary_patient_retrieval.csv").resolve()),
        "macro_map_figure_path": str((out_root / "fig_patient_retrieval_map_compare.png").resolve()),
        "macro_recall_figure_path": str((out_root / "fig_patient_retrieval_recall_at_k_compare.png").resolve()),
    }
    write_markdown(out_root / "analysis_patient_retrieval.md", macro_df, detail_df, run_meta)
    (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    log("Contrast benchmark patient retrieval completed successfully.")


if __name__ == "__main__":
    main()
