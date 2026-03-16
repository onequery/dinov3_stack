#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

DEFAULT_PRETRAINS = [
    ("1_lvd1689m", "LVD-1689M"),
    ("2_imagenet1k", "ImageNet-1K"),
    ("3_cagcontfm3m", "CAG-Contrast-FM-3M"),
]

TASK_CONFIGS = {
    "cls": {
        "task_dir": "1_cls",
        "primary_name": "f1_macro",
        "secondary_name": "accuracy",
        "plot_title": "CLS: F1 vs Unfrozen Blocks",
        "plot_ylabel": "F1 (macro)",
    },
    "ret": {
        "task_dir": "2_ret",
        "primary_name": "mAP",
        "secondary_name": "Recall@1",
        "plot_title": "RET: mAP vs Unfrozen Blocks",
        "plot_ylabel": "mAP",
    },
    "seg": {
        "task_dir": "3_seg",
        "primary_name": "mIoU",
        "secondary_name": "overall_acc",
        "plot_title": "SEG: mIoU vs Unfrozen Blocks",
        "plot_ylabel": "mIoU",
    },
}


def find_latest(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_cls_metric(eval_dir: str):
    pred_csv = find_latest(os.path.join(eval_dir, "per_image_predictions_*.csv"))
    if pred_csv is None:
        return None

    df = pd.read_csv(pred_csv)
    required_cols = {"gt_label", "pred_label"}
    if not required_cols.issubset(df.columns):
        return None

    y_true = df["gt_label"].astype(str).to_numpy()
    y_pred = df["pred_label"].astype(str).to_numpy()

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0.0))
    accuracy = float((y_true == y_pred).mean())

    return {
        "primary": f1_macro,
        "secondary": accuracy,
        "source": pred_csv,
    }


def parse_ret_metric(eval_dir: str):
    result_txt = find_latest(os.path.join(eval_dir, "retrieval_result_*.txt"))
    if result_txt is None:
        return None

    map_value = None
    r1_value = None

    with open(result_txt, "r") as f:
        for line in f:
            map_match = re.match(r"^mAP:\s*([-+0-9.eE]+)", line.strip())
            if map_match:
                map_value = float(map_match.group(1))
            r1_match = re.match(r"^Recall@1:\s*([-+0-9.eE]+)", line.strip())
            if r1_match:
                r1_value = float(r1_match.group(1))

    if map_value is None or r1_value is None:
        return None

    return {
        "primary": map_value,
        "secondary": r1_value,
        "source": result_txt,
    }


def parse_seg_metric(eval_dir: str):
    metrics_path = os.path.join(eval_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None

    with open(metrics_path, "r") as f:
        payload = json.load(f)

    if "mIoU" not in payload or "overall_acc" not in payload:
        return None

    return {
        "primary": float(payload["mIoU"]),
        "secondary": float(payload["overall_acc"]),
        "source": metrics_path,
    }


def collect_metrics(root: str, model_name: str, unfreeze_blocks: list[int]):
    parser_by_task = {
        "cls": parse_cls_metric,
        "ret": parse_ret_metric,
        "seg": parse_seg_metric,
    }

    rows = []
    for task, task_cfg in TASK_CONFIGS.items():
        task_dir_name = task_cfg["task_dir"]
        parser = parser_by_task[task]

        for pretrain_key, pretrain_label in DEFAULT_PRETRAINS:
            for ub in unfreeze_blocks:
                run_tag = f"u{ub:02d}"
                eval_dir = os.path.join(
                    root,
                    "eval",
                    task_dir_name,
                    model_name,
                    pretrain_key,
                    run_tag,
                )

                record = parser(eval_dir) if os.path.isdir(eval_dir) else None
                rows.append(
                    {
                        "task": task,
                        "pretrain_key": pretrain_key,
                        "pretrain_label": pretrain_label,
                        "unfreeze_blocks": ub,
                        "primary_metric": np.nan if record is None else record["primary"],
                        "secondary_metric": np.nan if record is None else record["secondary"],
                        "primary_metric_name": task_cfg["primary_name"],
                        "secondary_metric_name": task_cfg["secondary_name"],
                        "eval_dir": eval_dir,
                        "metric_source": "" if record is None else record["source"],
                        "status": "missing" if record is None else "ok",
                    }
                )
    return pd.DataFrame(rows)


def save_task_tables(df: pd.DataFrame, out_dir: str):
    for task, task_cfg in TASK_CONFIGS.items():
        task_df = df[df["task"] == task].copy()
        if task_df.empty:
            continue

        csv_path = os.path.join(out_dir, f"{task}_metrics.csv")
        task_df.to_csv(csv_path, index=False)

        pivot = task_df.pivot_table(
            index="unfreeze_blocks",
            columns="pretrain_label",
            values="primary_metric",
            aggfunc="first",
        )
        pivot = pivot.sort_index()
        pivot.to_csv(os.path.join(out_dir, f"{task}_primary_curve_table.csv"))

        if task == "ret":
            pivot_r1 = task_df.pivot_table(
                index="unfreeze_blocks",
                columns="pretrain_label",
                values="secondary_metric",
                aggfunc="first",
            )
            pivot_r1 = pivot_r1.sort_index()
            pivot_r1.to_csv(os.path.join(out_dir, "ret_r1_curve_table.csv"))


def plot_curve(df_task: pd.DataFrame, title: str, ylabel: str, out_path: str):
    plt.figure(figsize=(8, 5))
    for pretrain_label in [label for _, label in DEFAULT_PRETRAINS]:
        sub = df_task[df_task["pretrain_label"] == pretrain_label].copy()
        sub = sub.sort_values("unfreeze_blocks")
        if sub["primary_metric"].notna().sum() == 0:
            continue
        plt.plot(
            sub["unfreeze_blocks"],
            sub["primary_metric"],
            marker="o",
            linewidth=2,
            label=pretrain_label,
        )

    plt.xlabel("# Unfrozen Blocks")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(sorted(df_task["unfreeze_blocks"].unique()))
    plt.grid(alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_retrieval_r1(df_task: pd.DataFrame, out_path: str):
    plt.figure(figsize=(8, 5))
    for pretrain_label in [label for _, label in DEFAULT_PRETRAINS]:
        sub = df_task[df_task["pretrain_label"] == pretrain_label].copy()
        sub = sub.sort_values("unfreeze_blocks")
        if sub["secondary_metric"].notna().sum() == 0:
            continue
        plt.plot(
            sub["unfreeze_blocks"],
            sub["secondary_metric"],
            marker="o",
            linewidth=2,
            label=pretrain_label,
        )

    plt.xlabel("# Unfrozen Blocks")
    plt.ylabel("Recall@1")
    plt.title("RET: Recall@1 vs Unfrozen Blocks")
    plt.xticks(sorted(df_task["unfreeze_blocks"].unique()))
    plt.grid(alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def estimate_crossover_points(df_task: pd.DataFrame, task: str, metric_name: str):
    rows = []
    labels = [label for _, label in DEFAULT_PRETRAINS]

    for label_a, label_b in combinations(labels, 2):
        left = (
            df_task[df_task["pretrain_label"] == label_a][
                ["unfreeze_blocks", "primary_metric"]
            ]
            .rename(columns={"primary_metric": "metric_a"})
            .copy()
        )
        right = (
            df_task[df_task["pretrain_label"] == label_b][
                ["unfreeze_blocks", "primary_metric"]
            ]
            .rename(columns={"primary_metric": "metric_b"})
            .copy()
        )

        merged = pd.merge(left, right, on="unfreeze_blocks", how="inner")
        merged = merged.sort_values("unfreeze_blocks")
        merged = merged.dropna(subset=["metric_a", "metric_b"])

        if len(merged) < 2:
            rows.append(
                {
                    "task": task,
                    "metric": metric_name,
                    "pretrain_a": label_a,
                    "pretrain_b": label_b,
                    "status": "insufficient_points",
                    "first_crossover_unfreeze_blocks": np.nan,
                    "all_crossover_unfreeze_blocks": "",
                }
            )
            continue

        x = merged["unfreeze_blocks"].to_numpy(dtype=float)
        d = (merged["metric_a"] - merged["metric_b"]).to_numpy(dtype=float)

        crossings = []
        for i in range(len(x) - 1):
            x1, x2 = x[i], x[i + 1]
            d1, d2 = d[i], d[i + 1]

            if np.isnan(d1) or np.isnan(d2):
                continue

            if d1 == 0:
                crossings.append(float(x1))
                continue

            if d2 == 0:
                crossings.append(float(x2))
                continue

            if d1 * d2 < 0:
                ratio = abs(d1) / (abs(d1) + abs(d2))
                crossing_x = x1 + (x2 - x1) * ratio
                crossings.append(float(crossing_x))

        if crossings:
            rows.append(
                {
                    "task": task,
                    "metric": metric_name,
                    "pretrain_a": label_a,
                    "pretrain_b": label_b,
                    "status": "crossed",
                    "first_crossover_unfreeze_blocks": crossings[0],
                    "all_crossover_unfreeze_blocks": " | ".join(
                        f"{value:.3f}" for value in crossings
                    ),
                }
            )
        else:
            rows.append(
                {
                    "task": task,
                    "metric": metric_name,
                    "pretrain_a": label_a,
                    "pretrain_b": label_b,
                    "status": "no_cross",
                    "first_crossover_unfreeze_blocks": np.nan,
                    "all_crossover_unfreeze_blocks": "",
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="experiment root directory")
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--out-dir", required=True, help="summary output directory")
    parser.add_argument(
        "--unfreeze-blocks",
        nargs="+",
        type=int,
        default=[0, 1, 2, 4, 8, 12],
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = collect_metrics(args.root, args.model_name, args.unfreeze_blocks)
    metrics_df = metrics_df.sort_values(
        ["task", "pretrain_key", "unfreeze_blocks"]
    ).reset_index(drop=True)

    metrics_df.to_csv(out_dir / "ft_curve_metrics_long.csv", index=False)
    save_task_tables(metrics_df, str(out_dir))

    crossover_tables = []

    for task, cfg in TASK_CONFIGS.items():
        task_df = metrics_df[metrics_df["task"] == task].copy()

        plot_curve(
            task_df,
            title=cfg["plot_title"],
            ylabel=cfg["plot_ylabel"],
            out_path=str(out_dir / f"{task}_curve.png"),
        )

        crossover_tables.append(
            estimate_crossover_points(
                task_df,
                task=task,
                metric_name=cfg["primary_name"],
            )
        )

        if task == "ret":
            plot_retrieval_r1(task_df, str(out_dir / "ret_r1_curve.png"))

    crossover_df = pd.concat(crossover_tables, ignore_index=True)
    crossover_df.to_csv(out_dir / "crossover_points.csv", index=False)

    completed = int((metrics_df["status"] == "ok").sum())
    total = int(len(metrics_df))

    print("===== FT Curve Summary =====")
    print(f"Model: {args.model_name}")
    print(f"Root: {os.path.abspath(args.root)}")
    print(f"Out: {out_dir.resolve()}")
    print(f"Completed runs: {completed}/{total}")
    print(f"Saved: {out_dir / 'ft_curve_metrics_long.csv'}")
    print(f"Saved: {out_dir / 'crossover_points.csv'}")
    print(f"Saved: {out_dir / 'cls_curve.png'}")
    print(f"Saved: {out_dir / 'ret_curve.png'}")
    print(f"Saved: {out_dir / 'seg_curve.png'}")


if __name__ == "__main__":
    main()
