#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

TASK_SPECS = {
    "global_4_1_patient_retrieval": {
        "summary_file": "summary_global_2_retrieval.csv",
        "target": "patient",
        "primary_metric": "mAP_mean",
    },
    "global_4_2_same_dicom_retrieval": {
        "summary_file": "summary_global_4_2_retrieval.csv",
        "target": "same_dicom",
        "primary_metric": "mAP_mean",
    },
    "global_4_3_view_classification": {
        "summary_file": "summary_global_4_3_view_classification.csv",
        "target": "view_9way",
        "primary_metric": "accuracy_mean",
    },
}

INPUT_POLICY_DIRS = {
    "baseline_input": "baseline_rgbtriplet",
    "input_v1": "input_v1_cag_stats_normalization",
}

MODE_TO_READOUT = {
    "raw_frozen": "raw",
    "probe_linear": "probe",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_run_meta(task_root: Path) -> dict:
    meta_path = task_root / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_summary(task_root: Path, summary_file: str, target: str) -> pd.DataFrame:
    summary_path = task_root / summary_file
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    df = pd.read_csv(summary_path)
    if "target" in df.columns:
        df = df[df["target"] == target].copy()
    return df


def collect_metrics(exp_root: Path) -> pd.DataFrame:
    rows = []
    for policy_dir, _policy_name in INPUT_POLICY_DIRS.items():
        for task_name, spec in TASK_SPECS.items():
            task_root = exp_root / policy_dir / task_name
            run_meta = load_run_meta(task_root)
            df = load_summary(task_root, spec["summary_file"], spec["target"])
            df["task"] = task_name
            df["input_policy_dir"] = policy_dir
            df["input_policy"] = run_meta["input_policy"]
            df["input_norm_mean"] = json.dumps(run_meta["input_norm_mean"])
            df["input_norm_std"] = json.dumps(run_meta["input_norm_std"])
            df["readout_mode"] = df["mode"].map(MODE_TO_READOUT).fillna(df["mode"])
            rows.append(df)
    return pd.concat(rows, axis=0, ignore_index=True)


def collect_deltas(metric_df: pd.DataFrame) -> pd.DataFrame:
    delta_rows = []
    baseline_df = metric_df[metric_df["input_policy_dir"] == "baseline_input"].copy()
    variant_df = metric_df[metric_df["input_policy_dir"] == "input_v1"].copy()
    key_cols = ["task", "mode", "readout_mode", "target", "backbone_name", "split"]
    merged = variant_df.merge(
        baseline_df,
        on=key_cols,
        how="left",
        suffixes=("", "_baseline"),
    )
    metric_cols = []
    for col in sorted(c for c in metric_df.columns if c.endswith("_mean") and f"{c}_baseline" in merged.columns):
        variant_numeric = pd.to_numeric(merged[col], errors="coerce")
        baseline_numeric = pd.to_numeric(merged[f"{col}_baseline"], errors="coerce")
        if variant_numeric.notna().any() or baseline_numeric.notna().any():
            metric_cols.append(col)

    for row in merged.itertuples(index=False):
        for metric_name in metric_cols:
            variant_value = pd.to_numeric(pd.Series([getattr(row, metric_name)]), errors="coerce").iloc[0]
            baseline_value = pd.to_numeric(pd.Series([getattr(row, f"{metric_name}_baseline")]), errors="coerce").iloc[0]
            delta_rows.append(
                {
                    "task": getattr(row, "task"),
                    "mode": getattr(row, "mode"),
                    "readout_mode": getattr(row, "readout_mode"),
                    "target": getattr(row, "target"),
                    "backbone_name": getattr(row, "backbone_name"),
                    "split": getattr(row, "split"),
                    "metric_name": metric_name,
                    "baseline_input_policy": getattr(row, "input_policy_baseline"),
                    "variant_input_policy": getattr(row, "input_policy"),
                    "baseline_value": baseline_value,
                    "variant_value": variant_value,
                    "delta_value": None if pd.isna(baseline_value) or pd.isna(variant_value) else float(variant_value) - float(baseline_value),
                }
            )
    return pd.DataFrame(delta_rows)


def build_markdown(metric_df: pd.DataFrame, delta_df: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# input_v1_cag_stats_normalization")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    for task_name, spec in TASK_SPECS.items():
        lines.append(f"### {task_name}")
        lines.append("")
        for backbone_name in ["imagenet", "cag"]:
            task_rows = delta_df[
                (delta_df["task"] == task_name)
                & (delta_df["backbone_name"] == backbone_name)
                & (delta_df["split"] == "test")
                & (delta_df["metric_name"] == spec["primary_metric"])
            ].copy()
            if task_rows.empty:
                continue
            pieces = []
            for readout_mode in ["raw", "probe"]:
                mode_rows = task_rows[task_rows["readout_mode"] == readout_mode]
                if mode_rows.empty:
                    continue
                delta_value = mode_rows.iloc[0]["delta_value"]
                if pd.isna(delta_value):
                    pieces.append(f"{readout_mode} n/a")
                else:
                    pieces.append(f"{readout_mode} {float(delta_value):+.4f}")
            if pieces:
                lines.append(f"- `{backbone_name}`: " + " | ".join(pieces))
        lines.append("")

    lines.append("## Input Policy")
    lines.append("")
    policy_rows = metric_df[
        (metric_df["input_policy_dir"] == "input_v1")
        & (metric_df["split"] == "test")
    ]
    if not policy_rows.empty:
        sample_row = policy_rows.iloc[0]
        lines.append(f"- policy: `{sample_row['input_policy']}`")
        lines.append(f"- mean: `{sample_row['input_norm_mean']}`")
        lines.append(f"- std: `{sample_row['input_norm_std']}`")
    lines.append("")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize input_v1 downstream-only benchmark results.")
    parser.add_argument(
        "--exp-root",
        default="outputs/fm_improve_exp1-input_policy/input_v1_cag_stats_normalization/downstream_only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    reports_root = exp_root / "reports"
    ensure_dir(reports_root)

    metric_df = collect_metrics(exp_root)
    delta_df = collect_deltas(metric_df)
    metric_df.to_csv(reports_root / "summary_input_v1_benchmark_metrics.csv", index=False)
    delta_df.to_csv(reports_root / "summary_input_v1_metric_deltas.csv", index=False)
    build_markdown(metric_df, delta_df, reports_root / "analysis_input_v1_cag_stats_normalization.md")


if __name__ == "__main__":
    main()
