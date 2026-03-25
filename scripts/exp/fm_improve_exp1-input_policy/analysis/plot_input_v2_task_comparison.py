#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUMMARY_CSV = (
    REPO_ROOT
    / "outputs/fm_improve_exp1-input_policy/input_v2_percentile_canonicalization/downstream_only/reports/summary_input_v2_benchmark_metrics.csv"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SUMMARY_CSV.parent

TASK_CONFIG = {
    "global_4_1_patient_retrieval": {
        "target": "patient",
        "metrics": [
            ("mAP_mean", "mAP"),
            ("recall_at_1_mean", "Recall@1"),
            ("recall_at_5_mean", "Recall@5"),
        ],
    },
    "global_4_2_same_dicom_retrieval": {
        "target": "same_dicom",
        "metrics": [
            ("mAP_mean", "mAP"),
            ("recall_at_1_mean", "Recall@1"),
            ("recall_at_5_mean", "Recall@5"),
        ],
    },
    "global_4_3_view_classification": {
        "target": "view_9way",
        "metrics": [
            ("accuracy_mean", "Accuracy"),
            ("macro_f1_mean", "Macro-F1"),
            ("balanced_accuracy_mean", "Balanced Acc"),
        ],
    },
}
ORDER = [
    ("imagenet", "raw", "ImageNet Raw"),
    ("imagenet", "probe", "ImageNet Probe"),
    ("cag", "raw", "CAG Raw"),
    ("cag", "probe", "CAG Probe"),
]
COLOR_MAP = {
    "baseline_input": "#4C78A8",
    "input_v2": "#F58518",
}


def plot_task(summary_df: pd.DataFrame, task: str, split: str, output_dir: Path) -> None:
    cfg = TASK_CONFIG[task]
    subset = summary_df[
        (summary_df["task"] == task)
        & (summary_df["split"] == split)
        & (summary_df["target"] == cfg["target"])
        & (summary_df["readout_mode"].isin(["raw", "probe"]))
        & (summary_df["backbone_name"].isin(["imagenet", "cag"]))
    ].copy()
    if subset.empty:
        return

    rows = []
    for input_policy_dir in ("baseline_input", "input_v2"):
        for backbone_name, readout_mode, label in ORDER:
            match = subset[
                (subset["input_policy_dir"] == input_policy_dir)
                & (subset["backbone_name"] == backbone_name)
                & (subset["readout_mode"] == readout_mode)
            ]
            if match.empty:
                continue
            row = match.iloc[0].to_dict()
            row["group_label"] = label
            rows.append(row)
    plot_df = pd.DataFrame.from_records(rows)
    if plot_df.empty:
        return

    out_csv = output_dir / f"summary_input_v2_{task}_combined_{split}.csv"
    plot_df.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, len(cfg["metrics"]), figsize=(16.5, 5.6))
    if len(cfg["metrics"]) == 1:
        axes = [axes]

    x = np.arange(len(ORDER))
    width = 0.36
    for ax, (metric_col, metric_label) in zip(axes, cfg["metrics"], strict=True):
        baseline_vals = []
        variant_vals = []
        tick_labels = []
        for backbone_name, readout_mode, label in ORDER:
            tick_labels.append(label)
            base_match = plot_df[
                (plot_df["input_policy_dir"] == "baseline_input")
                & (plot_df["backbone_name"] == backbone_name)
                & (plot_df["readout_mode"] == readout_mode)
            ]
            var_match = plot_df[
                (plot_df["input_policy_dir"] == "input_v2")
                & (plot_df["backbone_name"] == backbone_name)
                & (plot_df["readout_mode"] == readout_mode)
            ]
            baseline_vals.append(float(base_match.iloc[0][metric_col]) if not base_match.empty else np.nan)
            variant_vals.append(float(var_match.iloc[0][metric_col]) if not var_match.empty else np.nan)

        ax.bar(x - width / 2, baseline_vals, width=width, label="Baseline", color=COLOR_MAP["baseline_input"])
        ax.bar(x + width / 2, variant_vals, width=width, label="input_v2", color=COLOR_MAP["input_v2"])
        ax.set_title(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(f"{task} ({split})", y=1.06)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.78))
    out_fig = output_dir / f"fig_input_v2_{task}_combined_{split}.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_df = pd.read_csv(args.summary_csv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for task in TASK_CONFIG:
        plot_task(summary_df, task, args.split, output_dir)

    print(json.dumps({"output_dir": str(output_dir), "split": args.split}, indent=2))


if __name__ == "__main__":
    main()
