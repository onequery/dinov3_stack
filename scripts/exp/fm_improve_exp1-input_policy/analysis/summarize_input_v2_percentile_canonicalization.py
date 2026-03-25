#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_EXP_ROOT = (
    REPO_ROOT
    / "outputs/fm_improve_exp1-input_policy/input_v2_percentile_canonicalization/downstream_only"
)

TASK_SPECS = {
    "global_4_1_patient_retrieval": {
        "task": "global_4_1_patient_retrieval",
        "summary_csv": "summary_global_2_retrieval.csv",
        "primary_target": "patient",
    },
    "global_4_2_same_dicom_retrieval": {
        "task": "global_4_2_same_dicom_retrieval",
        "summary_csv": "summary_global_4_2_retrieval.csv",
        "primary_target": "same_dicom",
    },
    "global_4_3_view_classification": {
        "task": "global_4_3_view_classification",
        "summary_csv": "summary_global_4_3_view_classification.csv",
        "primary_target": "view_9way",
    },
}
BASELINE_DIR = "baseline_input"
VARIANT_DIR = "input_v2"
VARIANT_POLICY = "input_v2_percentile_canonicalization"
BENCHMARK_POLICY = "baseline_rgbtriplet"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_summary(exp_root: Path, input_dir: str, task_dir: str) -> pd.DataFrame:
    spec = TASK_SPECS[task_dir]
    csv_path = exp_root / input_dir / task_dir / spec["summary_csv"]
    df = pd.read_csv(csv_path)
    df["task"] = spec["task"]
    df["input_policy_dir"] = input_dir
    df["input_policy"] = BENCHMARK_POLICY if input_dir == BASELINE_DIR else VARIANT_POLICY
    df["benchmark_input_policy"] = BENCHMARK_POLICY
    df["input_norm_mean"] = json.dumps(IMAGENET_MEAN)
    df["input_norm_std"] = json.dumps(IMAGENET_STD)
    if "readout_mode" not in df.columns:
        df["readout_mode"] = df["mode"].map({"raw_frozen": "raw", "probe_linear": "probe"}).fillna(df["mode"])
    return df


def build_benchmark_summary(exp_root: Path) -> pd.DataFrame:
    frames = []
    for input_dir in (BASELINE_DIR, VARIANT_DIR):
        for task_dir in TASK_SPECS:
            frames.append(read_summary(exp_root, input_dir, task_dir))
    return pd.concat(frames, ignore_index=True)


def iter_numeric_metrics(df: pd.DataFrame) -> Iterable[str]:
    excluded = {
        "num_seeds",
        "best_lr_vote",
        "input_norm_mean",
        "input_norm_std",
    }
    for column in df.columns:
        if not column.endswith(("_mean", "_std")):
            continue
        if column in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        yield column


def build_metric_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["task", "mode", "readout_mode", "target", "backbone_name", "split"]
    records: list[dict[str, object]] = []
    baseline_df = summary_df[summary_df["input_policy_dir"] == BASELINE_DIR].copy()
    variant_df = summary_df[summary_df["input_policy_dir"] == VARIANT_DIR].copy()

    for metric_name in iter_numeric_metrics(summary_df):
        merged = baseline_df[id_cols + ["input_policy", metric_name]].merge(
            variant_df[id_cols + ["input_policy", metric_name]],
            on=id_cols,
            suffixes=("_baseline", "_variant"),
            how="inner",
        )
        for row in merged.to_dict("records"):
            baseline_value = row.get(f"{metric_name}_baseline")
            variant_value = row.get(f"{metric_name}_variant")
            if pd.isna(baseline_value) or pd.isna(variant_value):
                continue
            records.append(
                {
                    **{key: row[key] for key in id_cols},
                    "metric_name": metric_name,
                    "baseline_input_policy": BENCHMARK_POLICY,
                    "variant_input_policy": VARIANT_POLICY,
                    "baseline_value": baseline_value,
                    "variant_value": variant_value,
                    "delta_value": float(variant_value) - float(baseline_value),
                }
            )
    return pd.DataFrame.from_records(records)


def build_analysis_markdown(summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# input_v2: Percentile Canonicalization")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Scope: downstream-only, benchmark-only, frozen backbone.")
    lines.append("- Variant: per-image percentile clip/rescale using p0.5 / p99.5.")
    lines.append("- Comparison: baseline_input vs input_v2 within the same experiment root.")
    lines.append("")
    lines.append("## Gate 1 Focus")
    lines.append("")
    primary_metrics = {
        "global_4_1_patient_retrieval": ("patient", "mAP_mean"),
        "global_4_2_same_dicom_retrieval": ("same_dicom", "mAP_mean"),
        "global_4_3_view_classification": ("view_9way", "accuracy_mean"),
    }
    for task, (target, metric_name) in primary_metrics.items():
        subset = delta_df[
            (delta_df["task"] == task)
            & (delta_df["target"] == target)
            & (delta_df["split"] == "test")
            & (delta_df["metric_name"] == metric_name)
        ].copy()
        if subset.empty:
            continue
        lines.append(f"### {task}")
        for row in subset.sort_values(["backbone_name", "readout_mode"]).to_dict("records"):
            lines.append(
                f"- {row['backbone_name']} / {row['readout_mode']}: "
                f"{row['baseline_value']:.6f} -> {row['variant_value']:.6f} "
                f"(delta {row['delta_value']:+.6f})"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", default=str(DEFAULT_EXP_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    reports_dir = exp_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    summary_df = build_benchmark_summary(exp_root)
    delta_df = build_metric_deltas(summary_df)

    summary_path = reports_dir / "summary_input_v2_benchmark_metrics.csv"
    delta_path = reports_dir / "summary_input_v2_metric_deltas.csv"
    analysis_path = reports_dir / "analysis_input_v2_percentile_canonicalization.md"

    summary_df.to_csv(summary_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    analysis_path.write_text(build_analysis_markdown(summary_df, delta_df), encoding="utf-8")

    print(json.dumps(
        {
            "summary_csv": str(summary_path),
            "delta_csv": str(delta_path),
            "analysis_md": str(analysis_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
