#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRACKED_FIELDS = [
    "ManufacturerModelName",
    "ShutterRightVerticalEdge",
    "WindowCenter",
    "WindowWidth",
    "intensity_p01",
]

TASK_SPECS = {
    "ga4_1_patient": {
        "variant_benchmark": ("global_4_1_patient_retrieval_control", "benchmark", "summary_global_2_retrieval.csv"),
        "variant_anchor": ("global_4_1_patient_retrieval_control", "anchoring", "summary_global_4_1_anchor_rank.csv"),
        "baseline_benchmark": Path("outputs/analysis2_rep_analysis/global_4_1_patient_retrieval_unique_view/summary_global_4_1_patient_retrieval.csv"),
        "baseline_anchor": Path("outputs/analysis2_rep_analysis/global_4_1_patient_retrieval_unique_view/summary_global_4_1_anchor_rank.csv"),
        "target": "patient",
    },
    "ga4_2_same_dicom": {
        "variant_benchmark": ("global_4_2_same_dicom_control", None, "summary_global_4_2_retrieval.csv"),
        "variant_anchor": ("global_4_2_same_dicom_control", None, "summary_global_4_2_anchor_rank.csv"),
        "baseline_benchmark": Path("outputs/analysis2_rep_analysis/global_4_2_same_dicom_retrieval_unique_view/summary_global_4_2_retrieval.csv"),
        "baseline_anchor": Path("outputs/analysis2_rep_analysis/global_4_2_same_dicom_retrieval_unique_view/summary_global_4_2_anchor_rank.csv"),
        "target": "same_dicom",
    },
    "ga4_3_view_9way": {
        "variant_benchmark": ("global_4_3_view_classification_control", None, "summary_global_4_3_view_classification.csv"),
        "variant_anchor": ("global_4_3_view_classification_control", None, "summary_global_4_3_anchor_rank.csv"),
        "baseline_benchmark": Path("outputs/analysis2_rep_analysis/global_4_3_view_classification_unique_view/summary_global_4_3_view_classification.csv"),
        "baseline_anchor": Path("outputs/analysis2_rep_analysis/global_4_3_view_classification_unique_view/summary_global_4_3_anchor_rank.csv"),
        "target": "view_9way",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def variant_file(variant_root: Path, segments: Sequence[str | None]) -> Path:
    parts = [variant_root]
    for item in segments:
        if item:
            parts.append(Path(item))
    return Path(*parts)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def collect_benchmark_rows(exp_root: Path, variants: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    delta_rows = []
    for variant in variants:
        variant_root = exp_root / variant
        for task_name, spec in TASK_SPECS.items():
            variant_path = variant_file(variant_root, spec["variant_benchmark"])
            baseline_path = spec["baseline_benchmark"]
            variant_df = load_csv(variant_path)
            baseline_df = load_csv(baseline_path)
            target = spec["target"]
            variant_df = variant_df[variant_df["target"] == target].copy()
            baseline_df = baseline_df[baseline_df["target"] == target].copy()
            variant_df["variant"] = variant
            variant_df["task"] = task_name
            metric_rows.append(variant_df)

            key_cols = [c for c in ["mode", "target", "backbone_name", "split"] if c in variant_df.columns and c in baseline_df.columns]
            baseline_core = baseline_df.copy()
            metric_cols = [c for c in variant_df.columns if c.endswith("_mean") and c in baseline_core.columns]
            merged = variant_df.merge(
                baseline_core[key_cols + metric_cols],
                on=key_cols,
                how="left",
                suffixes=("", "_baseline"),
            )
            for row in merged.itertuples(index=False):
                for metric in metric_cols:
                    variant_value = getattr(row, metric)
                    baseline_value = getattr(row, f"{metric}_baseline")
                    delta_rows.append(
                        {
                            "variant": variant,
                            "task": task_name,
                            "mode": getattr(row, "mode", None),
                            "target": target,
                            "backbone_name": getattr(row, "backbone_name", None),
                            "split": getattr(row, "split", None),
                            "metric_name": metric,
                            "baseline_value": baseline_value,
                            "variant_value": variant_value,
                            "delta_value": None if pd.isna(baseline_value) or pd.isna(variant_value) else float(variant_value) - float(baseline_value),
                        }
                    )
    metric_df = pd.concat(metric_rows, axis=0, ignore_index=True) if metric_rows else pd.DataFrame()
    delta_df = pd.DataFrame(delta_rows)
    return metric_df, delta_df


def collect_anchor_rows(exp_root: Path, variants: Sequence[str]) -> pd.DataFrame:
    rows = []
    for variant in variants:
        variant_root = exp_root / variant
        for task_name, spec in TASK_SPECS.items():
            variant_path = variant_file(variant_root, spec["variant_anchor"])
            baseline_path = spec["baseline_anchor"]
            variant_df = load_csv(variant_path)
            baseline_df = load_csv(baseline_path)
            target = spec["target"]
            variant_df = variant_df[variant_df["target"] == target].copy()
            baseline_df = baseline_df[baseline_df["target"] == target].copy()
            for backbone_name in ["imagenet", "cag"]:
                for field_name in TRACKED_FIELDS:
                    variant_row = variant_df[
                        (variant_df["backbone_name"] == backbone_name)
                        & (variant_df["field_name"] == field_name)
                    ]
                    baseline_row = baseline_df[
                        (baseline_df["backbone_name"] == backbone_name)
                        & (baseline_df["field_name"] == field_name)
                    ]
                    baseline_score = float(baseline_row.iloc[0]["combined_anchor_score_mean"]) if not baseline_row.empty else np.nan
                    variant_score = float(variant_row.iloc[0]["combined_anchor_score_mean"]) if not variant_row.empty else np.nan
                    baseline_rank = float(baseline_row.iloc[0]["rank_within_type"]) if not baseline_row.empty else np.nan
                    variant_rank = float(variant_row.iloc[0]["rank_within_type"]) if not variant_row.empty else np.nan
                    rows.append(
                        {
                            "variant": variant,
                            "task": task_name,
                            "backbone_name": backbone_name,
                            "target": target,
                            "field_name": field_name,
                            "baseline_score": baseline_score,
                            "variant_score": variant_score,
                            "delta_score": None if pd.isna(baseline_score) or pd.isna(variant_score) else float(variant_score) - float(baseline_score),
                            "baseline_rank": baseline_rank,
                            "variant_rank": variant_rank,
                            "delta_rank": None if pd.isna(baseline_rank) or pd.isna(variant_rank) else float(variant_rank) - float(baseline_rank),
                        }
                    )
    return pd.DataFrame(rows)


def build_markdown(metric_delta_df: pd.DataFrame, anchor_df: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Exp-0 Input Standardization")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    for variant in sorted(metric_delta_df["variant"].dropna().unique().tolist()):
        lines.append(f"### {variant}")
        lines.append("")
        for task_name in ["ga4_1_patient", "ga4_2_same_dicom", "ga4_3_view_9way"]:
            subset = metric_delta_df[
                (metric_delta_df["variant"] == variant)
                & (metric_delta_df["task"] == task_name)
                & (metric_delta_df["split"] == "test")
                & (metric_delta_df["backbone_name"] == "cag")
            ].copy()
            if subset.empty:
                continue
            if task_name == "ga4_3_view_9way":
                probe_metric = subset[(subset["mode"] == "probe") & (subset["metric_name"] == "accuracy_mean")]
                raw_metric = subset[(subset["mode"] == "raw_frozen") & (subset["metric_name"] == "accuracy_mean")]
            else:
                probe_metric = subset[(subset["mode"] == "probe") & (subset["metric_name"] == "mAP_mean")]
                raw_metric = subset[(subset["mode"] == "raw_frozen") & (subset["metric_name"] == "mAP_mean")]
            raw_delta = float(raw_metric.iloc[0]["delta_value"]) if not raw_metric.empty else float("nan")
            probe_delta = float(probe_metric.iloc[0]["delta_value"]) if not probe_metric.empty else float("nan")
            lines.append(f"- `{task_name}` raw delta: `{raw_delta:.4f}` | probe delta: `{probe_delta:.4f}`")
        lines.append("")
        anchor_subset = anchor_df[(anchor_df["variant"] == variant) & (anchor_df["backbone_name"] == "cag")]
        if not anchor_subset.empty:
            lines.append("Tracked anchor shifts (CAG):")
            for task_name in ["ga4_1_patient", "ga4_2_same_dicom", "ga4_3_view_9way"]:
                task_anchor = anchor_subset[anchor_subset["task"] == task_name]
                if task_anchor.empty:
                    continue
                pieces = []
                for field_name in TRACKED_FIELDS:
                    row_df = task_anchor[task_anchor["field_name"] == field_name]
                    if row_df.empty:
                        continue
                    pieces.append(f"`{field_name}` {float(row_df.iloc[0]['delta_score']):+.4f}")
                lines.append(f"- `{task_name}`: " + ", ".join(pieces))
            lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Exp-0 input-standardization results.")
    parser.add_argument("--exp-root", default="outputs/exp1_fm_improve/ablation/exp0_input_standardization")
    parser.add_argument("--variants", nargs="+", default=["norm_v1", "norm_v2", "norm_v3"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    reports_root = exp_root / "reports"
    ensure_dir(reports_root)

    metric_df, delta_df = collect_benchmark_rows(exp_root, args.variants)
    anchor_df = collect_anchor_rows(exp_root, args.variants)

    metric_df.to_csv(reports_root / "summary_exp0_benchmark_metrics.csv", index=False)
    delta_df.to_csv(reports_root / "summary_exp0_metric_deltas_vs_baseline.csv", index=False)
    anchor_df.to_csv(reports_root / "summary_exp0_anchor_shifts.csv", index=False)
    build_markdown(delta_df, anchor_df, reports_root / "analysis_exp0_input_standardization.md")


if __name__ == "__main__":
    main()
