#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FM Improvement Exp0 baseline outputs.")
    parser.add_argument("--exp-root", default="outputs/fm-imp-exp0_baseline")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_unified_summary(exp_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    view_summary = pd.read_csv(exp_root / "1_global/1_view_classification/summary_view_classification.csv")
    view_probe_test = view_summary[(view_summary["mode"] == "probe_linear") & (view_summary["split"] == "test")].copy()
    for _, row in view_probe_test.iterrows():
        rows.append(
            {
                "task_group": "1_global",
                "task_name": "1_view_classification",
                "mode": row["mode"],
                "split": row["split"],
                "backbone_name": row["backbone_name"],
                "aggregation_scope": "global",
                "primary_metric_name": "accuracy",
                "primary_metric_value": float(row["accuracy_mean"]),
                "accuracy": float(row["accuracy_mean"]),
                "macro_f1": float(row["macro_f1_mean"]),
                "balanced_accuracy": float(row["balanced_accuracy_mean"]),
                "mAP": float("nan"),
                "recall_at_1": float("nan"),
                "recall_at_5": float("nan"),
                "recall_at_10": float("nan"),
                "miou": float("nan"),
                "dice": float("nan"),
                "pixel_acc": float("nan"),
                "source_summary_path": str((exp_root / "1_global/1_view_classification/summary_view_classification.csv").resolve()),
            }
        )

    patient_summary = pd.read_csv(exp_root / "1_global/2_patient_retrieval/summary_patient_retrieval.csv")
    patient_probe_test = patient_summary[(patient_summary["mode"] == "probe_linear") & (patient_summary["split"] == "test")].copy()
    for _, row in patient_probe_test.iterrows():
        rows.append(
            {
                "task_group": "1_global",
                "task_name": "2_patient_retrieval",
                "mode": row["mode"],
                "split": row["split"],
                "backbone_name": row["backbone_name"],
                "aggregation_scope": "macro_across_views",
                "primary_metric_name": "mAP",
                "primary_metric_value": float(row["mAP_mean"]),
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "balanced_accuracy": float("nan"),
                "mAP": float(row["mAP_mean"]),
                "recall_at_1": float(row["recall_at_1_mean"]),
                "recall_at_5": float(row["recall_at_5_mean"]),
                "recall_at_10": float(row["recall_at_10_mean"]),
                "miou": float("nan"),
                "dice": float("nan"),
                "pixel_acc": float("nan"),
                "source_summary_path": str((exp_root / "1_global/2_patient_retrieval/summary_patient_retrieval.csv").resolve()),
            }
        )

    seg_summary = pd.read_csv(exp_root / "2_dense/1_coronary_segmentation/summary_segmentation_linear_probe.csv")
    seg_test = seg_summary[seg_summary["split"] == "test"].copy()
    for _, row in seg_test.iterrows():
        rows.append(
            {
                "task_group": "2_dense",
                "task_name": "1_coronary_segmentation",
                "mode": "probe_linear",
                "split": row["split"],
                "backbone_name": row["backbone_name"],
                "aggregation_scope": "global",
                "primary_metric_name": "miou",
                "primary_metric_value": float(row["miou"]),
                "accuracy": float("nan"),
                "macro_f1": float("nan"),
                "balanced_accuracy": float("nan"),
                "mAP": float("nan"),
                "recall_at_1": float("nan"),
                "recall_at_5": float("nan"),
                "recall_at_10": float("nan"),
                "miou": float(row["miou"]),
                "dice": float(row["dice"]),
                "pixel_acc": float(row["pixel_acc"]),
                "source_summary_path": str((exp_root / "2_dense/1_coronary_segmentation/summary_segmentation_linear_probe.csv").resolve()),
            }
        )

    return pd.DataFrame(rows)


def build_pending_summary(exp_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for task_name in ["2_sub-segment_segmentation", "3_stenosis_detection"]:
        status_path = exp_root / "2_dense" / task_name / "status.json"
        payload = load_json(status_path)
        rows.append(payload)
    return pd.DataFrame(rows)


def write_markdown(output_path: Path, unified_df: pd.DataFrame, pending_df: pd.DataFrame, exp_root: Path) -> None:
    lines: List[str] = []
    lines.append("# FM Improvement Exp0 — Baseline Benchmark Evaluation")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- benchmark root: `input/contrast_benchmark`")
    lines.append("- patient retrieval root: `input/contrast_benchmark/1_global/2_patient_retrieval`")
    lines.append("- output root: `outputs/fm-imp-exp0_baseline`")
    lines.append("- official comparison mode: `probe_linear`")
    lines.append("")
    lines.append("## Scorecard")
    lines.append("")
    lines.append("| task | backbone | primary_metric | value | aux1 | aux2 | aux3 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    display_df = unified_df.sort_values(["task_group", "task_name", "backbone_name"]).reset_index(drop=True)
    for _, row in display_df.iterrows():
        task_name = f"{row['task_group']}/{row['task_name']}"
        if row["task_name"] == "1_view_classification":
            aux1 = f"macro_f1={row['macro_f1']:.4f}"
            aux2 = f"balanced_acc={row['balanced_accuracy']:.4f}"
            aux3 = "-"
        elif row["task_name"] == "2_patient_retrieval":
            aux1 = f"R@1={row['recall_at_1']:.4f}"
            aux2 = f"R@5={row['recall_at_5']:.4f}"
            aux3 = f"R@10={row['recall_at_10']:.4f}"
        else:
            aux1 = f"dice={row['dice']:.4f}"
            aux2 = f"pixel_acc={row['pixel_acc']:.4f}"
            aux3 = "-"
        lines.append(
            f"| {task_name} | {row['backbone_name']} | {row['primary_metric_name']} | {row['primary_metric_value']:.4f} | {aux1} | {aux2} | {aux3} |"
        )
    lines.append("")
    lines.append("## Pending Tasks")
    lines.append("")
    lines.append("| task | status | reason |")
    lines.append("| --- | --- | --- |")
    for _, row in pending_df.sort_values(["task_name"]).iterrows():
        lines.append(f"| {row['task_name']} | {row['status']} | {row['reason']} |")
    lines.append("")
    lines.append("## Task Output Roots")
    lines.append("")
    lines.append(f"- view classification: `{(exp_root / '1_global/1_view_classification').resolve()}`")
    lines.append(f"- patient retrieval: `{(exp_root / '1_global/2_patient_retrieval').resolve()}`")
    lines.append(f"- coronary segmentation: `{(exp_root / '2_dense/1_coronary_segmentation').resolve()}`")
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    reports_root = exp_root / "reports"
    ensure_dir(reports_root)

    unified_df = build_unified_summary(exp_root)
    unified_df.to_csv(reports_root / "summary_exp0_benchmark.csv", index=False)

    patient_detail_df = pd.read_csv(exp_root / "1_global/2_patient_retrieval/summary_patient_retrieval_by_view.csv")
    patient_detail_df.to_csv(reports_root / "summary_exp0_patient_retrieval_by_view.csv", index=False)

    pending_df = build_pending_summary(exp_root)
    pending_df.to_csv(reports_root / "summary_exp0_pending_tasks.csv", index=False)

    write_markdown(reports_root / "analysis_exp0_baseline.md", unified_df, pending_df, exp_root)


if __name__ == "__main__":
    main()
