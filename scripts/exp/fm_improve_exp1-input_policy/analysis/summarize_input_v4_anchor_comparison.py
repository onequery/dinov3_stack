#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd


TRACKED_FIELDS = [
    "ManufacturerModelName",
    "ShutterRightVerticalEdge",
    "WindowCenter",
    "WindowWidth",
    "intensity_p01",
]

TASK_SPECS = {
    "ga4_1": {
        "label": "global_4_1_patient_retrieval",
        "targets": ["patient", "study"],
        "baseline_anchor_csv": ("baseline_input", "global_4_1_patient_retrieval", "anchoring", "summary_global_4_1_anchor_rank.csv"),
        "variant_anchor_csv": ("input_v4", "global_4_1_patient_retrieval", "anchoring", "summary_global_4_1_anchor_rank.csv"),
    },
    "ga4_2": {
        "label": "global_4_2_same_dicom_retrieval",
        "targets": ["same_dicom"],
        "baseline_anchor_csv": ("baseline_input", "global_4_2_same_dicom_retrieval", "anchoring", "summary_global_4_2_anchor_rank.csv"),
        "variant_anchor_csv": ("input_v4", "global_4_2_same_dicom_retrieval", "anchoring", "summary_global_4_2_anchor_rank.csv"),
    },
    "ga4_3": {
        "label": "global_4_3_view_classification",
        "targets": ["view_9way"],
        "baseline_anchor_csv": ("baseline_input", "global_4_3_view_classification", "anchoring", "summary_global_4_3_anchor_rank.csv"),
        "variant_anchor_csv": ("input_v4", "global_4_3_view_classification", "anchoring", "summary_global_4_3_anchor_rank.csv"),
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def resolve_file(exp_root: Path, parts: Sequence[str]) -> Path:
    return exp_root.joinpath(*parts)


def collect_anchor_shifts(exp_root: Path, tasks: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task_name in tasks:
        spec = TASK_SPECS[task_name]
        baseline_df = load_csv(resolve_file(exp_root, spec["baseline_anchor_csv"]))
        variant_df = load_csv(resolve_file(exp_root, spec["variant_anchor_csv"]))
        baseline_df = baseline_df[baseline_df["target"].isin(spec["targets"])].copy()
        variant_df = variant_df[variant_df["target"].isin(spec["targets"])].copy()

        key_cols = ["backbone_name", "target", "field_name", "field_type", "field_group"]
        baseline_core = baseline_df[key_cols + ["combined_anchor_score_mean", "rank_within_type"]].rename(
            columns={
                "combined_anchor_score_mean": "baseline_score",
                "rank_within_type": "baseline_rank",
            }
        )
        variant_core = variant_df[key_cols + ["combined_anchor_score_mean", "rank_within_type"]].rename(
            columns={
                "combined_anchor_score_mean": "variant_score",
                "rank_within_type": "variant_rank",
            }
        )
        merged = baseline_core.merge(variant_core, on=key_cols, how="outer")
        merged["task"] = task_name
        merged["task_label"] = spec["label"]
        merged["delta_score"] = merged["variant_score"] - merged["baseline_score"]
        merged["delta_rank"] = merged["variant_rank"] - merged["baseline_rank"]
        rows.extend(merged.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_markdown(all_fields_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# input_v4 Anchor Comparison")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Comparison: `baseline_input` vs `input_v4_border_suppression` anchoring outputs within the same experiment root.")
    lines.append("- Negative `delta_score` on nuisance fields means the tracked anchor weakened after border suppression.")
    lines.append("- Positive `delta_rank` means the field moved lower in the ranking after border suppression.")
    lines.append("")

    for task_name in sorted(tracked_df["task"].dropna().unique().tolist()):
        task_df = tracked_df[tracked_df["task"] == task_name].copy()
        if task_df.empty:
            continue
        task_label = str(task_df.iloc[0]["task_label"])
        lines.append(f"## {task_label}")
        lines.append("")
        for backbone_name in ["imagenet", "cag"]:
            backbone_df = task_df[task_df["backbone_name"] == backbone_name].copy()
            if backbone_df.empty:
                continue
            lines.append(f"### {backbone_name}")
            lines.append("")
            for target in sorted(backbone_df["target"].dropna().unique().tolist()):
                target_df = backbone_df[backbone_df["target"] == target].copy()
                pieces = []
                for field_name in TRACKED_FIELDS:
                    row_df = target_df[target_df["field_name"] == field_name]
                    if row_df.empty:
                        continue
                    row = row_df.iloc[0]
                    delta_score = row["delta_score"]
                    delta_rank = row["delta_rank"]
                    score_text = "nan" if pd.isna(delta_score) else f"{float(delta_score):+.4f}"
                    rank_text = "nan" if pd.isna(delta_rank) else f"{float(delta_rank):+.1f}"
                    pieces.append(f"`{field_name}` score {score_text}, rank {rank_text}")
                if pieces:
                    lines.append(f"- `{target}`: " + "; ".join(pieces))
            lines.append("")

            top_before = (
                all_fields_df[
                    (all_fields_df["task"] == task_name)
                    & (all_fields_df["backbone_name"] == backbone_name)
                    & (all_fields_df["field_type"] == "categorical")
                    & (all_fields_df["field_group"] != "reference")
                ]
                .sort_values(["target", "baseline_rank", "field_name"])
                .groupby("target", sort=False)
                .head(3)
            )
            top_after = (
                all_fields_df[
                    (all_fields_df["task"] == task_name)
                    & (all_fields_df["backbone_name"] == backbone_name)
                    & (all_fields_df["field_type"] == "categorical")
                    & (all_fields_df["field_group"] != "reference")
                ]
                .sort_values(["target", "variant_rank", "field_name"])
                .groupby("target", sort=False)
                .head(3)
            )
            if not top_before.empty:
                lines.append("- top categorical anchors (baseline):")
                for target in sorted(top_before["target"].dropna().unique().tolist()):
                    subset = top_before[top_before["target"] == target]
                    text = ", ".join(
                        f"`{row.field_name}` ({float(row.baseline_score):.4f})" for row in subset.itertuples()
                    )
                    lines.append(f"  - `{target}`: {text}")
            if not top_after.empty:
                lines.append("- top categorical anchors (input_v4):")
                for target in sorted(top_after["target"].dropna().unique().tolist()):
                    subset = top_after[top_after["target"] == target]
                    text = ", ".join(
                        f"`{row.field_name}` ({float(row.variant_score):.4f})" for row in subset.itertuples()
                    )
                    lines.append(f"  - `{target}`: {text}")
            lines.append("")

    output_path.write_text(chr(10).join(lines).strip() + chr(10), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-root", default="outputs/fm_improve_exp1-input_policy/input_v4_border_suppression/downstream_only")
    parser.add_argument("--tasks", nargs="+", default=["ga4_1"], choices=sorted(TASK_SPECS.keys()))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp_root = Path(args.exp_root).resolve()
    reports_root = exp_root / "anchoring_reports"
    ensure_dir(reports_root)

    all_fields_df = collect_anchor_shifts(exp_root, args.tasks)
    tracked_df = all_fields_df[all_fields_df["field_name"].isin(TRACKED_FIELDS)].copy()

    all_fields_df.to_csv(reports_root / "summary_input_v4_anchor_shifts.csv", index=False)
    tracked_df.to_csv(reports_root / "summary_input_v4_anchor_tracked_fields.csv", index=False)
    build_markdown(all_fields_df, tracked_df, reports_root / "analysis_input_v4_anchor_shifts.md")


if __name__ == "__main__":
    main()
