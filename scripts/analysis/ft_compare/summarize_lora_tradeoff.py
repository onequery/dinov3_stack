#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


TASK_META = {
    "cls": {"metric_col": "CLS_F1", "ylabel": "F1 (macro)", "title": "CLS Trade-off"},
    "ret": {"metric_col": "RET_mAP", "ylabel": "mAP", "title": "RET Trade-off"},
    "seg": {"metric_col": "SEG_mIoU", "ylabel": "mIoU", "title": "SEG Trade-off"},
}

CASE_LABELS = {
    "A": "Case A: General+Big Head",
    "B": "Case B: Domain+Big Head",
    "C": "Case C: General+LoRA+Small Head",
}

ABLATION_METHOD_LABELS = {
    "general_small_head": "General + Small Head",
    "general_big_head": "General + Big Head",
    "general_lora_small_head": "General + LoRA + Small Head",
}


def find_latest(pattern: str) -> str | None:
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_cls_metric(eval_dir: str) -> tuple[float | None, str | None]:
    pred_csv = find_latest(os.path.join(eval_dir, "per_image_predictions_*.csv"))
    if pred_csv is None:
        return None, None

    df = pd.read_csv(pred_csv)
    if not {"gt_label", "pred_label"}.issubset(df.columns):
        return None, pred_csv

    y_true = df["gt_label"].astype(str).to_numpy()
    y_pred = df["pred_label"].astype(str).to_numpy()
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0.0)), pred_csv


def parse_ret_metric(eval_dir: str) -> tuple[float | None, str | None]:
    txt_path = find_latest(os.path.join(eval_dir, "retrieval_result_*.txt"))
    if txt_path is None:
        return None, None

    value = None
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("mAP:"):
                try:
                    value = float(line.split(":", 1)[1].strip())
                except ValueError:
                    value = None
                break
    return value, txt_path


def parse_seg_metric(eval_dir: str) -> tuple[float | None, str | None]:
    metrics_path = os.path.join(eval_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        return None, None

    with open(metrics_path, "r") as f:
        payload = json.load(f)
    if "mIoU" not in payload:
        return None, metrics_path
    return float(payload["mIoU"]), metrics_path


def parse_metric(task: str, eval_dir: str) -> tuple[float | None, str | None]:
    if task == "cls":
        return parse_cls_metric(eval_dir)
    if task == "ret":
        return parse_ret_metric(eval_dir)
    if task == "seg":
        return parse_seg_metric(eval_dir)
    raise ValueError(f"Unknown task: {task}")


def load_param_stats(train_dir: str) -> tuple[float | None, float | None, str | None]:
    path = os.path.join(train_dir, "param_stats.json")
    if not os.path.isfile(path):
        return None, None, None
    with open(path, "r") as f:
        payload = json.load(f)
    return payload.get("trainable_params"), payload.get("total_params"), path


def annotate_unfreeze(ax, xs, ys, ns):
    for x, y, n in zip(xs, ys, ns):
        if pd.isna(x) or pd.isna(y):
            continue
        ax.annotate(
            f"N={int(n)}",
            (x, y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )


def build_report(tradeoff_long: pd.DataFrame, ablation_long: pd.DataFrame) -> str:
    lines = []
    lines.append("# LoRA Trade-off Analysis")
    lines.append("")

    lines.append("## 1) Low-parameter 영역 우위")
    for task, meta in TASK_META.items():
        df = tradeoff_long[(tradeoff_long["task"] == task) & tradeoff_long["metric"].notna() & tradeoff_long["trainable_params"].notna()]
        if df.empty:
            lines.append(f"- {task.upper()}: 데이터 부족")
            continue
        min_budget = df["trainable_params"].min()
        cand = df[df["trainable_params"] == min_budget]
        winner = cand.sort_values("metric", ascending=False).iloc[0]
        lines.append(
            f"- {task.upper()}: 최소 budget({int(min_budget):,})에서 {winner['case']} 우위 (metric={winner['metric']:.4f})"
        )

    lines.append("")
    lines.append("## 2) LoRA가 curve를 상향 이동시키는가")
    for task in TASK_META:
        a = tradeoff_long[(tradeoff_long["task"] == task) & (tradeoff_long["case"] == "A")][["unfreeze_blocks", "metric"]]
        c = tradeoff_long[(tradeoff_long["task"] == task) & (tradeoff_long["case"] == "C")][["unfreeze_blocks", "metric"]]
        m = pd.merge(a, c, on="unfreeze_blocks", how="inner", suffixes=("_a", "_c")).dropna()
        if m.empty:
            lines.append(f"- {task.upper()}: 비교 가능한 점 부족")
            continue
        uplift = (m["metric_c"] - m["metric_a"]).mean()
        lines.append(f"- {task.upper()}: Case C - Case A 평균 차이 = {uplift:+.4f}")

    lines.append("")
    lines.append("## 3) High-parameter ceiling 유지 여부 (N=12)")
    for task in TASK_META:
        a = tradeoff_long[(tradeoff_long["task"] == task) & (tradeoff_long["case"] == "A") & (tradeoff_long["unfreeze_blocks"] == 12)]
        c = tradeoff_long[(tradeoff_long["task"] == task) & (tradeoff_long["case"] == "C") & (tradeoff_long["unfreeze_blocks"] == 12)]
        if a.empty or c.empty or pd.isna(a.iloc[0]["metric"]) or pd.isna(c.iloc[0]["metric"]):
            lines.append(f"- {task.upper()}: N=12 비교 불가")
            continue
        delta = float(c.iloc[0]["metric"] - a.iloc[0]["metric"])
        lines.append(f"- {task.upper()}: N=12에서 Case C - Case A = {delta:+.4f}")

    lines.append("")
    lines.append("## 4) LoRA 효과와 head capacity 분리 (N=0)")
    for task in TASK_META:
        df = ablation_long[(ablation_long["task"] == task) & ablation_long["metric"].notna()].copy()
        if df.empty:
            lines.append(f"- {task.upper()}: ablation 결과 부족")
            continue
        m = {row["method"]: row["metric"] for _, row in df.iterrows()}
        s = m.get("general_small_head")
        b = m.get("general_big_head")
        l = m.get("general_lora_small_head")
        if s is None or b is None or l is None:
            lines.append(f"- {task.upper()}: 3-way 비교 불충분")
            continue
        lines.append(
            f"- {task.upper()}: Big-Small={b - s:+.4f}, LoRA+Small-Small={l - s:+.4f}, LoRA+Small-Big={l - b:+.4f}"
        )

    lines.append("")
    lines.append("## 참고")
    lines.append("- 본 리포트는 현재까지 완료된 run만 기반으로 자동 생성됨")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/5_lora_tradeoff/configs/run_manifest.csv")
    parser.add_argument("--summary-dir", default="outputs/5_lora_tradeoff/summary")
    parser.add_argument("--report-dir", default="outputs/5_lora_tradeoff/report")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    summary_dir = Path(args.summary_dir)
    report_dir = Path(args.report_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    rows = []
    for _, r in manifest.iterrows():
        task = str(r["task"])
        train_dir = str(r["train_dir"])
        eval_dir = str(r["eval_dir"])
        metric, metric_src = parse_metric(task, eval_dir)
        trainable_params, total_params, param_src = load_param_stats(train_dir)

        rows.append(
            {
                "kind": r["kind"],
                "task": task,
                "case": r["case"],
                "method": r["method"],
                "backbone_type": r["backbone_type"],
                "unfreeze_blocks": int(r["unfreeze_blocks"]),
                "lora_rank": int(r["lora_rank"]) if not pd.isna(r["lora_rank"]) else 0,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "metric": metric,
                "train_dir": train_dir,
                "eval_dir": eval_dir,
                "param_source": param_src,
                "metric_source": metric_src,
                "status": "ok" if metric is not None and trainable_params is not None else "missing",
            }
        )

    long_df = pd.DataFrame(rows)
    long_df.to_csv(summary_dir / "summary_long.csv", index=False)

    tradeoff_long = long_df[long_df["kind"] == "tradeoff"].copy()
    ablation_long = long_df[long_df["kind"] == "ablation"].copy()

    wide = tradeoff_long.pivot_table(
        index=["backbone_type", "case", "unfreeze_blocks", "lora_rank", "trainable_params", "total_params"],
        columns="task",
        values="metric",
        aggfunc="first",
    ).reset_index()
    wide = wide.rename(
        columns={
            "cls": "CLS_F1",
            "ret": "RET_mAP",
            "seg": "SEG_mIoU",
        }
    )
    for col in ["CLS_F1", "RET_mAP", "SEG_mIoU"]:
        if col not in wide.columns:
            wide[col] = np.nan
    wide = wide[[
        "backbone_type",
        "case",
        "unfreeze_blocks",
        "lora_rank",
        "trainable_params",
        "total_params",
        "CLS_F1",
        "RET_mAP",
        "SEG_mIoU",
    ]]
    wide = wide.sort_values(["case", "unfreeze_blocks"]).reset_index(drop=True)
    wide.to_csv(summary_dir / "summary_tradeoff.csv", index=False)

    # Trade-off plots
    for task, meta in TASK_META.items():
        task_df = tradeoff_long[(tradeoff_long["task"] == task) & tradeoff_long["metric"].notna() & tradeoff_long["trainable_params"].notna()].copy()

        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        for case in ["A", "B", "C"]:
            sub = task_df[task_df["case"] == case].sort_values("unfreeze_blocks")
            if sub.empty:
                continue
            xs = sub["trainable_params"].to_numpy(dtype=float)
            ys = sub["metric"].to_numpy(dtype=float)
            ns = sub["unfreeze_blocks"].to_numpy(dtype=float)
            ax.plot(xs, ys, marker="o", linewidth=2, label=CASE_LABELS.get(case, case))
            annotate_unfreeze(ax, xs, ys, ns)

        ax.set_xscale("log")
        ax.set_xlabel("Trainable Parameters (log scale)")
        ax.set_ylabel(meta["ylabel"])
        ax.set_title(meta["title"])
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        plt.tight_layout()
        plt.savefig(summary_dir / f"fig_tradeoff_{task}.png", dpi=300)
        plt.close()

    # Head ablation plots
    for task, meta in TASK_META.items():
        sub = ablation_long[(ablation_long["task"] == task) & ablation_long["metric"].notna()].copy()
        order = ["general_small_head", "general_big_head", "general_lora_small_head"]
        sub = sub[sub["method"].isin(order)]
        sub["method"] = pd.Categorical(sub["method"], categories=order, ordered=True)
        sub = sub.sort_values("method")

        plt.figure(figsize=(8, 5))
        ax = plt.gca()
        xs = np.arange(len(sub))
        ys = sub["metric"].to_numpy(dtype=float)
        labels = [ABLATION_METHOD_LABELS.get(m, m) for m in sub["method"].astype(str).tolist()]
        ax.bar(xs, ys)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(meta["ylabel"])
        ax.set_title(f"{task.upper()} Head Ablation (N=0)")
        ax.grid(alpha=0.3, axis="y")

        for i, (_, row) in enumerate(sub.iterrows()):
            params = row["trainable_params"]
            if pd.notna(params):
                ax.text(i, ys[i], f"{int(params):,}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(summary_dir / f"fig_head_ablation_{task}.png", dpi=300)
        plt.close()

    report_md = build_report(tradeoff_long=tradeoff_long, ablation_long=ablation_long)
    report_path = report_dir / "report_lora_tradeoff.md"
    with open(report_path, "w") as f:
        f.write(report_md)

    completed = int((long_df["status"] == "ok").sum())
    total = int(len(long_df))
    print("===== LoRA Trade-off Summary =====")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Summary: {summary_dir.resolve()}")
    print(f"Report: {report_path.resolve()}")
    print(f"Completed rows: {completed}/{total}")
    print(f"Saved: {summary_dir / 'summary_tradeoff.csv'}")


if __name__ == "__main__":
    main()
