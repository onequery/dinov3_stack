#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TRACKED_FIELDS = [
    "ShutterRightVerticalEdge",
    "WindowCenter",
    "WindowWidth",
    "intensity_p01",
    "intensity_p25",
    "gradient_mean",
    "gradient_std",
    "laplacian_variance",
    "laplacian_abs_mean",
    "edge_density_canny",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--variant-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--baseline-arm-name", default="baseline_philips")
    parser.add_argument("--variant-arm-name", default="border_suppressed_philips")
    parser.add_argument("--analysis-title", default="Global Analysis 6")
    return parser.parse_args()


def load_anchor_rank(arm_root: Path) -> pd.DataFrame:
    return pd.read_csv(arm_root / "summary_global_6_anchor_rank.csv")


def collect_anchor_shifts(
    baseline_df: pd.DataFrame,
    variant_df: pd.DataFrame,
    baseline_name: str,
    variant_name: str,
) -> pd.DataFrame:
    key_cols = ["backbone_name", "target", "field_name", "field_type", "field_group"]
    baseline_core = baseline_df[key_cols + ["combined_anchor_score_mean", "rank_within_type"]].rename(
        columns={
            "combined_anchor_score_mean": f"{baseline_name}_score",
            "rank_within_type": f"{baseline_name}_rank",
        }
    )
    variant_core = variant_df[key_cols + ["combined_anchor_score_mean", "rank_within_type"]].rename(
        columns={
            "combined_anchor_score_mean": f"{variant_name}_score",
            "rank_within_type": f"{variant_name}_rank",
        }
    )
    merged = baseline_core.merge(variant_core, on=key_cols, how="outer")
    merged["delta_score"] = merged[f"{variant_name}_score"] - merged[f"{baseline_name}_score"]
    merged["delta_rank"] = merged[f"{variant_name}_rank"] - merged[f"{baseline_name}_rank"]
    return merged.sort_values(["target", "backbone_name", "field_type", "field_group", "field_name"]).reset_index(drop=True)


def build_markdown(all_fields_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: Path, analysis_title: str) -> None:
    lines: list[str] = []
    lines.append(f"# {analysis_title}: Anchor Shifts")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- Comparison: `baseline_philips` vs `border_suppressed_philips` anchoring outputs within the same GA6 frame.")
    lines.append("- Negative `delta_score` means the tracked anchor weakened after border suppression.")
    lines.append("- Positive `delta_rank` means the field moved lower in the ranking after border suppression.")
    lines.append("")

    patient_df = tracked_df[tracked_df["target"] == "patient"].copy()
    for backbone_name in ["imagenet", "cag"]:
        backbone_df = patient_df[patient_df["backbone_name"] == backbone_name].copy()
        if backbone_df.empty:
            continue
        lines.append(f"## patient / {backbone_name}")
        lines.append("")
        for field_name in TRACKED_FIELDS:
            row_df = backbone_df[backbone_df["field_name"] == field_name]
            if row_df.empty:
                continue
            row = row_df.iloc[0]
            score_text = "nan" if pd.isna(row["delta_score"]) else f"{float(row['delta_score']):+.4f}"
            rank_text = "nan" if pd.isna(row["delta_rank"]) else f"{float(row['delta_rank']):+.1f}"
            lines.append(f"- `{field_name}`: score {score_text}, rank {rank_text}")
        lines.append("")

        top_before = (
            all_fields_df[
                (all_fields_df["target"] == "patient")
                & (all_fields_df["backbone_name"] == backbone_name)
                & (all_fields_df["field_group"] != "reference")
            ]
            .sort_values([f"baseline_philips_rank", "field_name"])
            .groupby("field_type", sort=False)
            .head(3)
        )
        top_after = (
            all_fields_df[
                (all_fields_df["target"] == "patient")
                & (all_fields_df["backbone_name"] == backbone_name)
                & (all_fields_df["field_group"] != "reference")
            ]
            .sort_values([f"border_suppressed_philips_rank", "field_name"])
            .groupby("field_type", sort=False)
            .head(3)
        )
        if not top_before.empty:
            lines.append("- top anchors (baseline):")
            for field_type in ["categorical", "continuous"]:
                subset = top_before[top_before["field_type"] == field_type]
                if subset.empty:
                    continue
                text = ", ".join(
                    f"`{row.field_name}` ({float(row.baseline_philips_score):.4f})" for row in subset.itertuples()
                )
                lines.append(f"  - `{field_type}`: {text}")
        if not top_after.empty:
            lines.append("- top anchors (border suppressed):")
            for field_type in ["categorical", "continuous"]:
                subset = top_after[top_after["field_type"] == field_type]
                if subset.empty:
                    continue
                text = ", ".join(
                    f"`{row.field_name}` ({float(row.border_suppressed_philips_score):.4f})" for row in subset.itertuples()
                )
                lines.append(f"  - `{field_type}`: {text}")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    baseline_root = Path(args.baseline_root).resolve()
    variant_root = Path(args.variant_root).resolve()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    baseline_df = load_anchor_rank(baseline_root)
    variant_df = load_anchor_rank(variant_root)
    all_fields_df = collect_anchor_shifts(baseline_df, variant_df, args.baseline_arm_name, args.variant_arm_name)
    tracked_df = all_fields_df[(all_fields_df["field_name"].isin(TRACKED_FIELDS)) & (all_fields_df["target"] == "patient")].copy()

    all_fields_df.to_csv(output_root / "summary_global_6_anchor_shifts.csv", index=False)
    tracked_df.to_csv(output_root / "summary_global_6_anchor_tracked_fields.csv", index=False)
    build_markdown(all_fields_df, tracked_df, output_root / "analysis_global_6_anchor_shifts.md", args.analysis_title)


if __name__ == "__main__":
    main()
