#!/usr/bin/env python3
"""Export patient retrieval benchmark-style summary from GA2 outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}
TARGET_NAME = "patient"


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    return f"{value:.6f}"


def make_grouped_bar_positions(group_count: int, bar_count: int, width: float = 0.18):
    centers = np.arange(group_count, dtype=np.float64)
    offsets = []
    start = -width * (bar_count - 1) / 2.0
    for idx in range(bar_count):
        offsets.append(centers + start + idx * width)
    return centers, offsets


def save_map_compare_figure(summary_df: pd.DataFrame, output_path: Path, analysis_title: str) -> None:
    plot_df = summary_df[(summary_df["split"] == "test") & (summary_df["target"] == TARGET_NAME)].copy()
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    xs = np.arange(len(conditions), dtype=np.float64)
    means = []
    errs = []
    for backbone_name, mode, _ in conditions:
        row = plot_df[(plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)]
        if row.empty:
            means.append(np.nan)
            errs.append(0.0)
        else:
            means.append(float(row.iloc[0]["mAP_mean"]))
            errs.append(float(row.iloc[0]["mAP_std"]))
    plt.figure(figsize=(8, 5))
    bars = []
    for x, (backbone_name, mode, _), mean, err in zip(xs, conditions, means, errs):
        bar = plt.bar([x], [mean], yerr=[err], capsize=4, color=PLOT_COLORS[backbone_name], alpha=0.55 if mode == "raw_frozen" else 0.95)
        bars.append(bar[0])
    plt.xticks(xs, [label for _, _, label in conditions], rotation=15)
    for bar, value in zip(bars, means):
        if np.isfinite(value):
            plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.4f}", ha="center", va="bottom")
    plt.ylabel("mAP")
    plt.title(f"{analysis_title}: Patient Retrieval mAP")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_recall_compare_figure(summary_df: pd.DataFrame, output_path: Path, analysis_title: str) -> None:
    plot_df = summary_df[(summary_df["split"] == "test") & (summary_df["target"] == TARGET_NAME)].copy()
    ks = [1, 5, 10]
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    centers, xs = make_grouped_bar_positions(len(ks), len(conditions), width=0.18)
    plt.figure(figsize=(10, 5))
    for idx, (backbone_name, mode, label) in enumerate(conditions):
        row = plot_df[(plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)]
        means = []
        errs = []
        for k in ks:
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(row.iloc[0][f"recall_at_{k}_mean"]))
                errs.append(float(row.iloc[0][f"recall_at_{k}_std"]))
        plt.bar(xs[idx], means, width=0.18, color=PLOT_COLORS[backbone_name], alpha=0.55 if mode == "raw_frozen" else 0.95, yerr=errs, capsize=4, label=label)
    plt.xticks(centers, ["R@1", "R@5", "R@10"])
    plt.ylabel("Recall@K")
    plt.title(f"{analysis_title}: Patient Retrieval Recall@K")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def build_cdf(points: np.ndarray, max_rank: int) -> np.ndarray:
    if points.size == 0:
        return np.zeros(max_rank, dtype=np.float64)
    xs = np.arange(1, max_rank + 1)
    return np.asarray([(points <= x).mean() for x in xs], dtype=np.float64)


def save_rank_cdf_figure(
    raw_rank_map: Dict[str, np.ndarray],
    probe_rank_map: Dict[str, List[np.ndarray]],
    output_path: Path,
    analysis_title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    max_rank = 1
    for backbone_name in ["imagenet", "cag"]:
        raw_points = raw_rank_map.get(backbone_name, np.array([], dtype=np.int32))
        if raw_points.size:
            max_rank = max(max_rank, int(np.percentile(raw_points, 95)))
        for seed_points in probe_rank_map.get(backbone_name, []):
            if seed_points.size:
                max_rank = max(max_rank, int(np.percentile(seed_points, 95)))
    xs = np.arange(1, max_rank + 1)
    for backbone_name in ["imagenet", "cag"]:
        color = PLOT_COLORS[backbone_name]
        raw_points = raw_rank_map.get(backbone_name, np.array([], dtype=np.int32))
        raw_cdf = build_cdf(raw_points, max_rank)
        plt.plot(xs, raw_cdf, color=color, linestyle="--", linewidth=2, label=f"{backbone_name.title()} Raw")
        probe_curves = [build_cdf(points, max_rank) for points in probe_rank_map.get(backbone_name, [])]
        if probe_curves:
            probe_arr = np.stack(probe_curves, axis=0)
            mean = probe_arr.mean(axis=0)
            std = probe_arr.std(axis=0, ddof=1) if probe_arr.shape[0] > 1 else np.zeros_like(mean)
            plt.plot(xs, mean, color=color, linewidth=2, label=f"{backbone_name.title()} Probe")
            plt.fill_between(xs, np.clip(mean - std, 0.0, 1.0), np.clip(mean + std, 0.0, 1.0), color=color, alpha=0.18)
    plt.xlabel("First Positive Rank")
    plt.ylabel("Query Ratio (CDF)")
    plt.title(f"{analysis_title}: Patient First Positive Rank CDF")
    plt.grid(alpha=0.25)
    plt.legend(ncols=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_probe_minus_raw_delta_figure(summary_df: pd.DataFrame, output_path: Path, analysis_title: str) -> None:
    plot_df = summary_df[(summary_df["split"] == "test") & (summary_df["target"] == TARGET_NAME)].copy()
    labels = ["ImageNet", "CAG"]
    xs = np.arange(len(labels))
    means = []
    errs = []
    for backbone_name in ["imagenet", "cag"]:
        probe_row = plot_df[(plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == "probe_linear")]
        raw_row = plot_df[(plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == "raw_frozen")]
        if probe_row.empty or raw_row.empty:
            means.append(np.nan)
            errs.append(0.0)
        else:
            delta = float(probe_row.iloc[0]["mAP_mean"]) - float(raw_row.iloc[0]["mAP_mean"])
            means.append(delta)
            errs.append(float(probe_row.iloc[0]["mAP_std"]))
    plt.figure(figsize=(6, 5))
    bars = plt.bar(xs, means, color=[PLOT_COLORS["imagenet"], PLOT_COLORS["cag"]], yerr=errs, capsize=4)
    plt.axhline(0.0, color="black", linewidth=1)
    for bar, value in zip(bars, means):
        if np.isfinite(value):
            plt.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:+.4f}", ha="center", va="bottom")
    plt.xticks(xs, labels)
    plt.ylabel("Probe mAP - Raw mAP")
    plt.title(f"{analysis_title}: Probe Improvement over Raw Retrieval")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_markdown_summary(output_path: Path, summary_df: pd.DataFrame, source_root: Path, analysis_title: str) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    lines = []
    lines.append(f"# {analysis_title}: Patient Retrieval")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- source benchmark root: `{source_root}`")
    lines.append(f"- target: `{TARGET_NAME}`")
    lines.append("- modes: `raw_frozen`, `probe_linear`")
    lines.append("")
    lines.append("## Test Summary")
    lines.append("")
    lines.append("| Mode | Backbone | mAP | R@1 | R@5 | R@10 | Median First Positive Rank | Queries With No Positive |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, row in test_df.sort_values(["mode", "backbone_name"]).iterrows():
        lines.append(
            f"| {row['mode']} | {row['backbone_name']} | "
            f"{format_float(float(row['mAP_mean']))} +/- {format_float(float(row['mAP_std']))} | "
            f"{format_float(float(row['recall_at_1_mean']))} +/- {format_float(float(row['recall_at_1_std']))} | "
            f"{format_float(float(row['recall_at_5_mean']))} +/- {format_float(float(row['recall_at_5_std']))} | "
            f"{format_float(float(row['recall_at_10_mean']))} +/- {format_float(float(row['recall_at_10_std']))} | "
            f"{format_float(float(row['median_first_positive_rank_mean']))} | "
            f"{format_float(float(row['queries_with_no_positive_mean']))} |"
        )
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- `raw도 높음` -> patient identity is already linearly accessible in frozen CLS features.")
    lines.append("- `raw 약함 + probe 회복` -> patient signal exists but frozen global geometry is poorly organized for retrieval.")
    lines.append("- `CAG probe < ImageNet probe` -> later anchoring 결과와 연결해 global organization weakness를 해석할 수 있다.")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default="outputs/global_2_study_patient_retrieval_unique_view")
    parser.add_argument("--output-root", default="outputs/global_4_1_patient_retrieval_unique_view")
    parser.add_argument("--analysis-title", default="Global Analysis 4-1")
    parser.add_argument("--summary-prefix", default="summary_global_4_1")
    parser.add_argument("--figure-prefix", default="fig_global4_1")
    parser.add_argument("--markdown-name", default="analysis_global_4_1_patient_retrieval.md")
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(source_root / "summary_global_2_retrieval.csv")
    raw_df = pd.read_csv(source_root / "summary_global_2_retrieval_raw.csv")
    probe_raw_df = pd.read_csv(source_root / "summary_global_2_retrieval_probe_raw.csv")

    summary_patient = summary_df[summary_df["target"] == TARGET_NAME].copy().reset_index(drop=True)
    raw_patient = raw_df[raw_df["target"] == TARGET_NAME].copy().reset_index(drop=True)
    probe_patient = probe_raw_df[probe_raw_df["target"] == TARGET_NAME].copy().reset_index(drop=True)

    summary_patient.to_csv(output_root / f"{args.summary_prefix}_patient_retrieval.csv", index=False)
    raw_patient.to_csv(output_root / f"{args.summary_prefix}_patient_retrieval_raw.csv", index=False)
    probe_patient.to_csv(output_root / f"{args.summary_prefix}_patient_retrieval_probe_raw.csv", index=False)

    raw_rank_map: Dict[str, np.ndarray] = {}
    probe_rank_map: Dict[str, List[np.ndarray]] = {"imagenet": [], "cag": []}
    for backbone_name in ["imagenet", "cag"]:
        raw_per_query = pd.read_csv(source_root / f"per_query_patient_raw_{backbone_name}_test.csv")
        raw_rank_map[backbone_name] = pd.to_numeric(raw_per_query["first_positive_rank"], errors="coerce").dropna().to_numpy(dtype=np.int32)
        for seed in args.probe_seeds:
            probe_per_query = pd.read_csv(source_root / f"per_query_patient_probe_seed{seed}_{backbone_name}_test.csv")
            probe_rank_map[backbone_name].append(
                pd.to_numeric(probe_per_query["first_positive_rank"], errors="coerce").dropna().to_numpy(dtype=np.int32)
            )

    save_map_compare_figure(summary_patient, output_root / f"{args.figure_prefix}_map_compare.png", args.analysis_title)
    save_recall_compare_figure(summary_patient, output_root / f"{args.figure_prefix}_recall_at_k_compare.png", args.analysis_title)
    save_rank_cdf_figure(
        raw_rank_map,
        probe_rank_map,
        output_root / f"{args.figure_prefix}_rank_cdf_compare.png",
        args.analysis_title,
    )
    save_probe_minus_raw_delta_figure(
        summary_patient,
        output_root / f"{args.figure_prefix}_probe_minus_raw_delta.png",
        args.analysis_title,
    )
    write_markdown_summary(output_root / args.markdown_name, summary_patient, source_root, args.analysis_title)


if __name__ == "__main__":
    main()
