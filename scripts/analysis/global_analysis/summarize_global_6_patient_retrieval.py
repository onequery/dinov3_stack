#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_COLORS = {
    "imagenet": "#4C72B0",
    "cag": "#DD8452",
    "baseline_philips": "#7A7A7A",
    "border_suppressed_philips": "#2A9D8F",
}
TARGET = "patient"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "nan"
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-arm-name", default="baseline_philips")
    parser.add_argument("--variant-arm-name", default="border_suppressed_philips")
    parser.add_argument("--analysis-title", default="Global Analysis 6")
    return parser.parse_args()


def load_arm_summary(arm_root: Path, arm_name: str) -> tuple[pd.DataFrame, dict]:
    summary = pd.read_csv(arm_root / "summary_global_6_patient_retrieval.csv")
    summary["arm_name"] = arm_name
    run_meta = json.loads((arm_root / "run_meta.json").read_text(encoding="utf-8"))
    return summary, run_meta


def merge_delta_table(baseline_df: pd.DataFrame, variant_df: pd.DataFrame, baseline_name: str, variant_name: str) -> pd.DataFrame:
    key_cols = ["split", "mode", "backbone_name"]
    value_cols = [
        "mAP_mean",
        "mAP_std",
        "recall_at_1_mean",
        "recall_at_1_std",
        "recall_at_5_mean",
        "recall_at_5_std",
        "queries_with_no_positive_mean",
        "median_first_positive_rank_mean",
    ]
    baseline_core = baseline_df[key_cols + value_cols].rename(columns={col: f"{baseline_name}_{col}" for col in value_cols})
    variant_core = variant_df[key_cols + value_cols].rename(columns={col: f"{variant_name}_{col}" for col in value_cols})
    merged = baseline_core.merge(variant_core, on=key_cols, how="inner")
    for col in ["mAP_mean", "recall_at_1_mean", "recall_at_5_mean", "queries_with_no_positive_mean", "median_first_positive_rank_mean"]:
        merged[f"delta_{col}"] = merged[f"{variant_name}_{col}"] - merged[f"{baseline_name}_{col}"]
    return merged.sort_values(key_cols).reset_index(drop=True)


def build_gap_change(summary_df: pd.DataFrame, baseline_name: str, variant_name: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm_name in [baseline_name, variant_name]:
        arm = summary_df[summary_df["arm_name"] == arm_name].copy()
        for (split, mode), group in arm.groupby(["split", "mode"], sort=True):
            imagenet_row = group[group["backbone_name"] == "imagenet"].iloc[0]
            cag_row = group[group["backbone_name"] == "cag"].iloc[0]
            rows.append(
                {
                    "arm_name": arm_name,
                    "split": split,
                    "mode": mode,
                    "mAP_gap_cag_minus_imagenet": float(cag_row["mAP_mean"]) - float(imagenet_row["mAP_mean"]),
                    "recall_at_1_gap_cag_minus_imagenet": float(cag_row["recall_at_1_mean"]) - float(imagenet_row["recall_at_1_mean"]),
                    "recall_at_5_gap_cag_minus_imagenet": float(cag_row["recall_at_5_mean"]) - float(imagenet_row["recall_at_5_mean"]),
                }
            )
    arm_gap = pd.DataFrame(rows)
    baseline_gap = arm_gap[arm_gap["arm_name"] == baseline_name].rename(
        columns={
            "mAP_gap_cag_minus_imagenet": f"{baseline_name}_mAP_gap_cag_minus_imagenet",
            "recall_at_1_gap_cag_minus_imagenet": f"{baseline_name}_recall_at_1_gap_cag_minus_imagenet",
            "recall_at_5_gap_cag_minus_imagenet": f"{baseline_name}_recall_at_5_gap_cag_minus_imagenet",
        }
    )
    variant_gap = arm_gap[arm_gap["arm_name"] == variant_name].rename(
        columns={
            "mAP_gap_cag_minus_imagenet": f"{variant_name}_mAP_gap_cag_minus_imagenet",
            "recall_at_1_gap_cag_minus_imagenet": f"{variant_name}_recall_at_1_gap_cag_minus_imagenet",
            "recall_at_5_gap_cag_minus_imagenet": f"{variant_name}_recall_at_5_gap_cag_minus_imagenet",
        }
    )
    baseline_gap = baseline_gap.drop(columns=["arm_name"])
    variant_gap = variant_gap.drop(columns=["arm_name"])
    merged = baseline_gap.merge(variant_gap, on=["split", "mode"], how="inner")
    for metric in ["mAP", "recall_at_1", "recall_at_5"]:
        merged[f"gap_change_{metric}_cag_minus_imagenet"] = (
            merged[f"{variant_name}_{metric}_gap_cag_minus_imagenet"] - merged[f"{baseline_name}_{metric}_gap_cag_minus_imagenet"]
        )
    return merged.sort_values(["split", "mode"]).reset_index(drop=True)


def query_rel_series(df: pd.DataFrame, image_root: Path) -> pd.Series:
    rels = []
    resolved_root = image_root.resolve()
    for value in df["query_path"].astype(str).tolist():
        path = Path(value).resolve()
        rels.append(str(path.relative_to(resolved_root)))
    return pd.Series(rels, index=df.index, dtype=str)


def load_per_query_raw(arm_root: Path, run_meta: dict, backbone_name: str) -> pd.DataFrame:
    df = pd.read_csv(arm_root / "retrieval_benchmark" / f"per_query_patient_raw_{backbone_name}_test.csv")
    image_root = Path(run_meta["image_root"]).resolve()
    df["query_rel_path"] = query_rel_series(df, image_root)
    return df


def load_per_query_probe_mean(arm_root: Path, run_meta: dict, backbone_name: str) -> pd.DataFrame:
    image_root = Path(run_meta["image_root"]).resolve()
    frames = []
    for seed in run_meta["probe_seeds"]:
        df = pd.read_csv(arm_root / "retrieval_benchmark" / f"per_query_patient_probe_seed{seed}_{backbone_name}_test.csv")
        df["query_rel_path"] = query_rel_series(df, image_root)
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    agg = (
        merged.groupby("query_rel_path", sort=True)
        .agg(
            query_patient_id=("query_patient_id", "first"),
            query_study_id=("query_study_id", "first"),
            query_label=("query_label", "first"),
            num_positives=("num_positives", "first"),
            AP=("AP", "mean"),
            first_positive_rank=("first_positive_rank", "mean"),
            R_at_1=("R@1", "mean"),
            R_at_5=("R@5", "mean"),
            R_at_10=("R@10", "mean"),
            seed_count=("seed", "nunique"),
        )
        .reset_index()
    )
    return agg


def merge_per_query_delta(baseline_df: pd.DataFrame, variant_df: pd.DataFrame, baseline_name: str, variant_name: str, probe_mode: bool) -> pd.DataFrame:
    key_cols = ["query_rel_path"]
    if probe_mode:
        rename_map = {
            "query_patient_id": "query_patient_id",
            "query_study_id": "query_study_id",
            "query_label": "query_label",
            "num_positives": "num_positives",
            "AP": "AP",
            "first_positive_rank": "first_positive_rank",
            "R_at_1": "R_at_1",
            "R_at_5": "R_at_5",
            "R_at_10": "R_at_10",
            "seed_count": "seed_count",
        }
    else:
        rename_map = {
            "query_patient_id": "query_patient_id",
            "query_study_id": "query_study_id",
            "query_label": "query_label",
            "num_positives": "num_positives",
            "AP": "AP",
            "first_positive_rank": "first_positive_rank",
            "R@1": "R_at_1",
            "R@5": "R_at_5",
            "R@10": "R_at_10",
        }
    baseline_core = baseline_df[key_cols + list(rename_map.keys())].rename(columns={k: (rename_map[k] if rename_map[k].startswith("query_") or rename_map[k] == "num_positives" else f"{baseline_name}_{rename_map[k]}") for k in rename_map})
    variant_core = variant_df[key_cols + list(rename_map.keys())].rename(columns={k: (rename_map[k] if rename_map[k].startswith("query_") or rename_map[k] == "num_positives" else f"{variant_name}_{rename_map[k]}") for k in rename_map})
    merged = baseline_core.merge(variant_core, on=key_cols + ["query_patient_id", "query_study_id", "query_label", "num_positives"], how="inner")
    for metric in ["AP", "first_positive_rank", "R_at_1", "R_at_5", "R_at_10"]:
        merged[f"delta_{metric}"] = merged[f"{variant_name}_{metric}"] - merged[f"{baseline_name}_{metric}"]
    if probe_mode:
        merged = merged.rename(
            columns={
                f"{baseline_name}_seed_count": f"{baseline_name}_seed_count",
                f"{variant_name}_seed_count": f"{variant_name}_seed_count",
            }
        )
    return merged.sort_values("query_rel_path").reset_index(drop=True)


def save_map_compare(delta_df: pd.DataFrame, output_path: Path, analysis_title: str, baseline_name: str, variant_name: str) -> None:
    plot_df = delta_df[delta_df["split"] == "test"].copy()
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    xs = np.arange(len(conditions), dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    baseline_vals = []
    variant_vals = []
    for backbone_name, mode, _label in conditions:
        row = plot_df[(plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)].iloc[0]
        baseline_vals.append(float(row[f"{baseline_name}_mAP_mean"]))
        variant_vals.append(float(row[f"{variant_name}_mAP_mean"]))
    ax.bar(xs - width / 2, baseline_vals, width=width, color=PLOT_COLORS[baseline_name], label="Baseline")
    ax.bar(xs + width / 2, variant_vals, width=width, color=PLOT_COLORS[variant_name], label="Border Suppressed")
    ax.set_xticks(xs, [label for _, _, label in conditions], rotation=15)
    ax.set_ylabel("mAP")
    ax.set_title(f"{analysis_title}: Test Patient Retrieval mAP")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_recall_compare(delta_df: pd.DataFrame, output_path: Path, analysis_title: str, baseline_name: str, variant_name: str) -> None:
    plot_df = delta_df[delta_df["split"] == "test"].copy()
    conditions = [
        ("imagenet", "raw_frozen", "ImageNet Raw"),
        ("imagenet", "probe_linear", "ImageNet Probe"),
        ("cag", "raw_frozen", "CAG Raw"),
        ("cag", "probe_linear", "CAG Probe"),
    ]
    xs = np.arange(len(conditions), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, metric, ylabel in [
        (axes[0], "recall_at_1_mean", "R@1"),
        (axes[1], "recall_at_5_mean", "R@5"),
    ]:
        baseline_vals = []
        variant_vals = []
        for backbone_name, mode, _label in conditions:
            row = plot_df[(plot_df["backbone_name"] == backbone_name) & (plot_df["mode"] == mode)].iloc[0]
            baseline_vals.append(float(row[f"{baseline_name}_{metric}"]))
            variant_vals.append(float(row[f"{variant_name}_{metric}"]))
        ax.bar(xs - width / 2, baseline_vals, width=width, color=PLOT_COLORS[baseline_name], label="Baseline")
        ax.bar(xs + width / 2, variant_vals, width=width, color=PLOT_COLORS[variant_name], label="Border Suppressed")
        ax.set_xticks(xs, [label for _, _, label in conditions], rotation=15)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"{analysis_title}: Test Patient Retrieval Recall")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_probe_minus_raw(delta_df: pd.DataFrame, output_path: Path, analysis_title: str, baseline_name: str, variant_name: str) -> None:
    test_df = delta_df[delta_df["split"] == "test"].copy()
    rows = []
    for arm_name in [baseline_name, variant_name]:
        for backbone_name in ["imagenet", "cag"]:
            raw_row = test_df[(test_df["backbone_name"] == backbone_name) & (test_df["mode"] == "raw_frozen")].iloc[0]
            probe_row = test_df[(test_df["backbone_name"] == backbone_name) & (test_df["mode"] == "probe_linear")].iloc[0]
            rows.append(
                {
                    "arm_name": arm_name,
                    "backbone_name": backbone_name,
                    "probe_minus_raw_mAP": float(probe_row[f"{arm_name}_mAP_mean"]) - float(raw_row[f"{arm_name}_mAP_mean"]),
                }
            )
    plot_df = pd.DataFrame(rows)
    xs = np.arange(2, dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    for offset, arm_name in [(-width / 2, baseline_name), (width / 2, variant_name)]:
        vals = [float(plot_df[(plot_df["arm_name"] == arm_name) & (plot_df["backbone_name"] == b)].iloc[0]["probe_minus_raw_mAP"]) for b in ["imagenet", "cag"]]
        ax.bar(xs + offset, vals, width=width, color=PLOT_COLORS[arm_name], label="Baseline" if arm_name == baseline_name else "Border Suppressed")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(xs, ["ImageNet", "CAG"])
    ax.set_ylabel("Probe mAP - Raw mAP")
    ax.set_title(f"{analysis_title}: Probe Improvement over Raw")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_gap_change(gap_df: pd.DataFrame, output_path: Path, analysis_title: str, baseline_name: str, variant_name: str) -> None:
    plot_df = gap_df[gap_df["split"] == "test"].copy()
    xs = np.arange(2, dtype=float)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 5))
    baseline_vals = [float(plot_df[plot_df["mode"] == mode].iloc[0][f"{baseline_name}_mAP_gap_cag_minus_imagenet"]) for mode in ["raw_frozen", "probe_linear"]]
    variant_vals = [float(plot_df[plot_df["mode"] == mode].iloc[0][f"{variant_name}_mAP_gap_cag_minus_imagenet"]) for mode in ["raw_frozen", "probe_linear"]]
    ax.bar(xs - width / 2, baseline_vals, width=width, color=PLOT_COLORS[baseline_name], label="Baseline")
    ax.bar(xs + width / 2, variant_vals, width=width, color=PLOT_COLORS[variant_name], label="Border Suppressed")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(xs, ["Raw", "Probe"])
    ax.set_ylabel("CAG mAP - ImageNet mAP")
    ax.set_title(f"{analysis_title}: Backbone Gap Change")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(output_path: Path, delta_df: pd.DataFrame, gap_df: pd.DataFrame, analysis_title: str, baseline_name: str, variant_name: str) -> None:
    test_df = delta_df[delta_df["split"] == "test"].copy()
    lines = []
    lines.append(f"# {analysis_title}: Patient Retrieval")
    lines.append("")
    lines.append("## Test Summary")
    lines.append("")
    lines.append("| Mode | Backbone | Baseline mAP | Suppressed mAP | Delta | Baseline R@1 | Suppressed R@1 | Baseline R@5 | Suppressed R@5 | Queries With No Positive (Baseline/Suppressed) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, row in test_df.sort_values(["mode", "backbone_name"]).iterrows():
        lines.append(
            f"| {row['mode']} | {row['backbone_name']} | "
            f"{format_float(float(row[f'{baseline_name}_mAP_mean']))} | {format_float(float(row[f'{variant_name}_mAP_mean']))} | {format_float(float(row['delta_mAP_mean']))} | "
            f"{format_float(float(row[f'{baseline_name}_recall_at_1_mean']))} | {format_float(float(row[f'{variant_name}_recall_at_1_mean']))} | "
            f"{format_float(float(row[f'{baseline_name}_recall_at_5_mean']))} | {format_float(float(row[f'{variant_name}_recall_at_5_mean']))} | "
            f"{format_float(float(row[f'{baseline_name}_queries_with_no_positive_mean']))}/{format_float(float(row[f'{variant_name}_queries_with_no_positive_mean']))} |"
        )
    lines.append("")
    lines.append("## Backbone Gap")
    lines.append("")
    lines.append("| Mode | Baseline (CAG-ImageNet) | Suppressed (CAG-ImageNet) | Gap Change |")
    lines.append("| --- | ---: | ---: | ---: |")
    for _, row in gap_df[gap_df["split"] == "test"].sort_values("mode").iterrows():
        lines.append(
            f"| {row['mode']} | {format_float(float(row[f'{baseline_name}_mAP_gap_cag_minus_imagenet']))} | "
            f"{format_float(float(row[f'{variant_name}_mAP_gap_cag_minus_imagenet']))} | "
            f"{format_float(float(row['gap_change_mAP_cag_minus_imagenet']))} |"
        )
    lines.append("")
    lines.append("## Interpretation Guide")
    lines.append("")
    lines.append("- Positive `delta_mAP_mean` means border suppression improved patient retrieval for that backbone/mode.")
    lines.append("- Positive `gap_change_mAP_cag_minus_imagenet` means suppression moved the backbone gap toward `CAG > ImageNet`.")
    lines.append("- `queries_with_no_positive` should remain unchanged because the subset membership does not change.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    variant_root = Path(args.variant_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    baseline_summary, baseline_meta = load_arm_summary(baseline_root, args.baseline_arm_name)
    variant_summary, variant_meta = load_arm_summary(variant_root, args.variant_arm_name)
    comparison_df = pd.concat([baseline_summary, variant_summary], ignore_index=True).sort_values(["arm_name", "split", "mode", "backbone_name"]).reset_index(drop=True)
    delta_df = merge_delta_table(baseline_summary, variant_summary, args.baseline_arm_name, args.variant_arm_name)
    gap_df = build_gap_change(comparison_df, args.baseline_arm_name, args.variant_arm_name)

    comparison_df.to_csv(output_root / "summary_global_6_patient_retrieval_comparison.csv", index=False)
    delta_df.to_csv(output_root / "summary_global_6_patient_retrieval_deltas.csv", index=False)
    gap_df.to_csv(output_root / "summary_global_6_patient_retrieval_gap_change.csv", index=False)

    for backbone_name in ["imagenet", "cag"]:
        raw_baseline = load_per_query_raw(baseline_root, baseline_meta, backbone_name)
        raw_variant = load_per_query_raw(variant_root, variant_meta, backbone_name)
        raw_delta = merge_per_query_delta(raw_baseline, raw_variant, args.baseline_arm_name, args.variant_arm_name, probe_mode=False)
        raw_delta.to_csv(output_root / f"per_query_patient_test_delta_raw_{backbone_name}.csv", index=False)

        probe_baseline = load_per_query_probe_mean(baseline_root, baseline_meta, backbone_name)
        probe_variant = load_per_query_probe_mean(variant_root, variant_meta, backbone_name)
        probe_delta = merge_per_query_delta(probe_baseline, probe_variant, args.baseline_arm_name, args.variant_arm_name, probe_mode=True)
        probe_delta.to_csv(output_root / f"per_query_patient_test_delta_probe_{backbone_name}.csv", index=False)

    save_map_compare(delta_df, output_root / "fig_global6_patient_retrieval_map_compare.png", args.analysis_title, args.baseline_arm_name, args.variant_arm_name)
    save_recall_compare(delta_df, output_root / "fig_global6_patient_retrieval_recall_compare.png", args.analysis_title, args.baseline_arm_name, args.variant_arm_name)
    save_probe_minus_raw(delta_df, output_root / "fig_global6_patient_retrieval_probe_minus_raw_compare.png", args.analysis_title, args.baseline_arm_name, args.variant_arm_name)
    save_gap_change(gap_df, output_root / "fig_global6_patient_retrieval_gap_change.png", args.analysis_title, args.baseline_arm_name, args.variant_arm_name)
    write_markdown(output_root / "analysis_global_6_patient_retrieval.md", delta_df, gap_df, args.analysis_title, args.baseline_arm_name, args.variant_arm_name)


if __name__ == "__main__":
    main()
