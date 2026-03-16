#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, TextIO, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _seg_linear_probe_common as common

PLOT_COLORS = {"imagenet": "#4C72B0", "cag": "#DD8452"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Analysis 2-1 — Layer-wise Segmentation Linear Probe")
    parser.add_argument("--train-images", default="input/MPXA-Seg/train_images")
    parser.add_argument("--train-masks", default="input/MPXA-Seg/train_labels")
    parser.add_argument("--valid-images", default="input/MPXA-Seg/valid_images")
    parser.add_argument("--valid-masks", default="input/MPXA-Seg/valid_labels")
    parser.add_argument("--test-images", default="input/MPXA-Seg/test_images")
    parser.add_argument("--test-masks", default="input/MPXA-Seg/test_labels")
    parser.add_argument("--seg-config", default="configs_segmentation/mpxa-seg.yaml")
    parser.add_argument("--imagenet-ckpt", default=common.DEFAULT_IMAGENET_CKPT)
    parser.add_argument("--cag-ckpt", default=common.DEFAULT_CAG_CKPT)
    parser.add_argument("--output-root", default="outputs/local_2_1_layerwise_segmentation_linear_probe_multiseed")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--img-size", nargs=2, type=int, default=[640, 640], help="width height")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-grid", nargs="+", type=float, default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--layers", nargs="+", default=["all"], help="1-based layer ids or 'all'")
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--strict-deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--keep-feature-caches", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    return parser.parse_args()


def parse_layer_ids(layer_args: Sequence[str], total_layers: int = 12) -> List[int]:
    if len(layer_args) == 1 and str(layer_args[0]).lower() == "all":
        return list(range(1, total_layers + 1))
    values = sorted({int(item) for item in layer_args})
    if not values:
        raise ValueError("At least one layer must be provided.")
    if values[0] < 1 or values[-1] > total_layers:
        raise ValueError(f"Layer ids must be between 1 and {total_layers}: {values}")
    return values


def parse_seed_values(seed_args: Sequence[int]) -> List[int]:
    values = sorted({int(seed) for seed in seed_args})
    if not values:
        raise ValueError("At least one seed must be provided.")
    return values


def feature_cache_paths(out_root: Path, backbone_name: str, split_name: str, layer_id: int) -> Tuple[Path, Path, Path]:
    stem = f"features_{backbone_name}_{split_name}_layer{layer_id:02d}"
    return out_root / f"{stem}.pt", out_root / f"{stem}.meta.json", out_root / f"{stem}.mmap"


def layer_feature_meta(
    manifest: pd.DataFrame,
    ckpt_path: str,
    args: argparse.Namespace,
    layer_id: int,
) -> Dict[str, object]:
    grid_w = int(args.img_size[0]) // int(args.patch_size)
    grid_h = int(args.img_size[1]) // int(args.patch_size)
    return {
        "manifest_hash": common.hash_dataframe(manifest, ["image_id", "image_path", "mask_path"]),
        "checkpoint_path": str(Path(ckpt_path).resolve()),
        "model_name": args.model_name,
        "repo_dir": str(Path(args.repo_dir).resolve()),
        "img_size": [int(args.img_size[0]), int(args.img_size[1])],
        "patch_size": int(args.patch_size),
        "grid_size": [grid_h, grid_w],
        "layer_id": int(layer_id),
        "block_index": int(layer_id - 1),
        "n_images": int(len(manifest)),
    }


def load_cached_layer_feature(path: Path) -> torch.Tensor:
    cached = torch.load(path, map_location="cpu")
    if not isinstance(cached, torch.Tensor):
        raise ValueError(f"Invalid feature cache format: {path}")
    return cached


def prepare_layer_feature_caches(
    split_name: str,
    backbone_name: str,
    ckpt_path: str,
    manifest: pd.DataFrame,
    all_classes: List[str],
    label_colors_list: List[List[int]],
    args: argparse.Namespace,
    out_root: Path,
    layer_ids: Sequence[int],
) -> Dict[int, Path]:
    resolved_ckpt = str(Path(ckpt_path).expanduser().resolve())
    cache_paths: Dict[int, Path] = {}
    missing_layers: List[int] = []
    for layer_id in layer_ids:
        features_path, meta_path, _ = feature_cache_paths(out_root, backbone_name, split_name, layer_id)
        desired_meta = layer_feature_meta(manifest=manifest, ckpt_path=resolved_ckpt, args=args, layer_id=layer_id)
        cache_paths[layer_id] = features_path
        if args.cache_features and features_path.exists() and meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                saved_meta = json.load(f)
            if saved_meta == desired_meta:
                common.log(f"Reusing layer feature cache: {features_path}")
                continue
        missing_layers.append(layer_id)

    if not missing_layers:
        return cache_paths

    device = common.resolve_device(args.device)
    wrapper = common.Dinov3Backbone(
        weights=resolved_ckpt,
        model_name=args.model_name,
        repo_dir=args.repo_dir,
    )
    backbone = wrapper.backbone_model.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    dataset = common.build_seg_dataset(
        manifest=manifest,
        img_size=(int(args.img_size[0]), int(args.img_size[1])),
        all_classes=all_classes,
        label_colors_list=label_colors_list,
    )
    loader = common.build_feature_loader(
        dataset=dataset,
        batch_size=int(args.feature_batch_size),
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
    )
    embed_dim = backbone.norm.normalized_shape[0]
    grid_w = int(args.img_size[0]) // int(args.patch_size)
    grid_h = int(args.img_size[1]) // int(args.patch_size)
    block_indices = [int(layer_id - 1) for layer_id in missing_layers]

    memmaps: Dict[int, np.memmap] = {}
    mmap_paths: Dict[int, Path] = {}
    for layer_id in missing_layers:
        _, _, mmap_path = feature_cache_paths(out_root, backbone_name, split_name, layer_id)
        try:
            if mmap_path.exists():
                mmap_path.unlink()
        except OSError:
            pass
        mmap_paths[layer_id] = mmap_path
        memmaps[layer_id] = np.memmap(
            mmap_path,
            mode="w+",
            dtype=np.float16,
            shape=(len(dataset), embed_dim, grid_h, grid_w),
        )

    common.log(
        f"Extracting layerwise features ({backbone_name}, {split_name}) | "
        f"n_images={len(dataset):,} | layers={missing_layers} | grid={grid_h}x{grid_w} | device={device}"
    )
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extract-{backbone_name}-{split_name}-layers"):
            images = batch[0].to(device, non_blocking=(device.type == "cuda"))
            outputs = backbone.get_intermediate_layers(
                images,
                n=block_indices,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
            if len(outputs) != len(missing_layers):
                raise ValueError(
                    f"Layer output count mismatch: expected={len(missing_layers)} got={len(outputs)}"
                )
            batch_size_actual = int(images.shape[0])
            for layer_id, feat in zip(missing_layers, outputs):
                if feat.ndim != 4 or int(feat.shape[2]) != grid_h or int(feat.shape[3]) != grid_w:
                    raise ValueError(
                        f"Unexpected feature shape for layer {layer_id}: {tuple(feat.shape)}"
                    )
                memmaps[layer_id][offset : offset + batch_size_actual] = feat.detach().cpu().to(torch.float16).numpy()
            offset += batch_size_actual

    for layer_id in missing_layers:
        memmaps[layer_id].flush()
        tensor = torch.from_numpy(np.asarray(memmaps[layer_id]))
        features_path, meta_path, mmap_path = feature_cache_paths(out_root, backbone_name, split_name, layer_id)
        torch.save(tensor, features_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(layer_feature_meta(manifest=manifest, ckpt_path=resolved_ckpt, args=args, layer_id=layer_id), f, indent=2)
        del tensor
        del memmaps[layer_id]
        try:
            mmap_path.unlink()
        except OSError:
            pass
        common.log(f"Saved layer feature cache: {features_path}")

    del backbone
    del wrapper
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache_paths


def cleanup_backbone_feature_caches(out_root: Path, backbone_name: str, split_names: Sequence[str], layer_ids: Sequence[int]) -> None:
    for split_name in split_names:
        for layer_id in layer_ids:
            features_path, meta_path, mmap_path = feature_cache_paths(out_root, backbone_name, split_name, layer_id)
            common.remove_file_if_exists(features_path)
            common.remove_file_if_exists(meta_path)
            common.remove_file_if_exists(mmap_path)


def metrics_row_with_seed_and_layer(split_metrics: common.SplitMetrics, seed: int, layer_id: int) -> Dict[str, object]:
    row = split_metrics.to_row()
    row["seed"] = int(seed)
    row["layer_id"] = int(layer_id)
    row["block_index"] = int(layer_id - 1)
    preferred_order = [
        "seed",
        "backbone_name",
        "layer_id",
        "block_index",
        "split",
        "best_lr",
        "epochs_trained",
        "num_images",
        "pixel_acc",
        "miou",
        "dice",
        "per_class_iou",
        "per_class_dice",
        "probe_params",
    ]
    return {key: row[key] for key in preferred_order}


def _parse_json_list_column(series: pd.Series) -> List[List[float]]:
    return [list(map(float, json.loads(value))) for value in series.tolist()]


def build_aggregate_summary(raw_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (backbone_name, layer_id, block_index, split), group in raw_df.groupby(
        ["backbone_name", "layer_id", "block_index", "split"],
        sort=True,
    ):
        best_lr_counts = Counter(map(str, group["best_lr"].tolist()))
        best_lr_mode = float(group["best_lr"].mode().iloc[0])
        per_class_iou = np.asarray(_parse_json_list_column(group["per_class_iou"]), dtype=np.float64)
        per_class_dice = np.asarray(_parse_json_list_column(group["per_class_dice"]), dtype=np.float64)
        rows.append(
            {
                "backbone_name": backbone_name,
                "layer_id": int(layer_id),
                "block_index": int(block_index),
                "split": split,
                "num_seeds": int(len(group)),
                "miou_mean": float(group["miou"].mean()),
                "miou_std": float(group["miou"].std(ddof=0)),
                "dice_mean": float(group["dice"].mean()),
                "dice_std": float(group["dice"].std(ddof=0)),
                "pixel_acc_mean": float(group["pixel_acc"].mean()),
                "pixel_acc_std": float(group["pixel_acc"].std(ddof=0)),
                "best_lr_mode": best_lr_mode,
                "best_lr_counts": json.dumps(dict(sorted(best_lr_counts.items()))),
                "epochs_trained_mean": float(group["epochs_trained"].mean()),
                "epochs_trained_std": float(group["epochs_trained"].std(ddof=0)),
                "num_images": int(group["num_images"].iloc[0]),
                "per_class_iou_mean": json.dumps(np.mean(per_class_iou, axis=0).tolist()),
                "per_class_iou_std": json.dumps(np.std(per_class_iou, axis=0, ddof=0).tolist()),
                "per_class_dice_mean": json.dumps(np.mean(per_class_dice, axis=0).tolist()),
                "per_class_dice_std": json.dumps(np.std(per_class_dice, axis=0, ddof=0).tolist()),
                "probe_params": int(group["probe_params"].iloc[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(["layer_id", "backbone_name", "split"]).reset_index(drop=True)


def build_paired_delta_df(raw_df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    split_df = raw_df[raw_df["split"] == split].copy()
    pivot = split_df.pivot_table(
        index=["seed", "layer_id", "block_index"],
        columns="backbone_name",
        values=["miou", "dice"],
        aggfunc="first",
    ).sort_index()
    rows: List[Dict[str, object]] = []
    for (seed, layer_id, block_index), values in pivot.iterrows():
        if ("miou", "imagenet") not in pivot.columns or ("miou", "cag") not in pivot.columns:
            continue
        rows.append(
            {
                "seed": int(seed),
                "layer_id": int(layer_id),
                "block_index": int(block_index),
                "miou_delta": float(values[("miou", "cag")] - values[("miou", "imagenet")]),
                "dice_delta": float(values[("dice", "cag")] - values[("dice", "imagenet")]),
            }
        )
    return pd.DataFrame(rows).sort_values(["layer_id", "seed"]).reset_index(drop=True)


def _seed_offsets(num_points: int, width: float = 0.12) -> np.ndarray:
    if num_points <= 1:
        return np.asarray([0.0], dtype=np.float64)
    return np.linspace(-width, width, num_points, dtype=np.float64)


def save_layerwise_curve(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    metric_prefix: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    raw_test_df = raw_df[raw_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for backbone_name, label in [("imagenet", "ImageNet"), ("cag", "CAG")]:
        sub = test_df[test_df["backbone_name"] == backbone_name].sort_values("layer_id")
        mean_col = f"{metric_prefix}_mean"
        std_col = f"{metric_prefix}_std"
        ax.plot(
            sub["layer_id"],
            sub[mean_col],
            marker="o",
            linewidth=2.0,
            color=PLOT_COLORS[backbone_name],
            label=label,
        )
        ax.fill_between(
            sub["layer_id"],
            sub[mean_col] - sub[std_col],
            sub[mean_col] + sub[std_col],
            color=PLOT_COLORS[backbone_name],
            alpha=0.18,
            linewidth=0.0,
        )
        raw_sub = raw_test_df[raw_test_df["backbone_name"] == backbone_name].copy()
        if not raw_sub.empty:
            seed_order = sorted(raw_sub["seed"].unique().tolist())
            offsets = _seed_offsets(len(seed_order))
            seed_to_offset = {int(seed): float(offsets[idx]) for idx, seed in enumerate(seed_order)}
            xs = raw_sub["layer_id"].to_numpy(dtype=float) + raw_sub["seed"].map(seed_to_offset).to_numpy(dtype=float)
            ys = raw_sub[metric_prefix].to_numpy(dtype=float)
            ax.scatter(
                xs,
                ys,
                s=26,
                alpha=0.7,
                color=PLOT_COLORS[backbone_name],
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(sorted(test_df["layer_id"].unique().tolist()))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_delta_figure(delta_df: pd.DataFrame, output_path: Path) -> None:
    grouped = delta_df.groupby("layer_id", sort=True).agg(
        miou_mean=("miou_delta", "mean"),
        miou_std=("miou_delta", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
        dice_mean=("dice_delta", "mean"),
        dice_std=("dice_delta", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
    axes[0].plot(grouped["layer_id"], grouped["miou_mean"], marker="o", color="#C44E52")
    axes[1].plot(grouped["layer_id"], grouped["dice_mean"], marker="o", color="#55A868")
    axes[0].fill_between(
        grouped["layer_id"],
        grouped["miou_mean"] - grouped["miou_std"],
        grouped["miou_mean"] + grouped["miou_std"],
        color="#C44E52",
        alpha=0.18,
        linewidth=0.0,
    )
    axes[1].fill_between(
        grouped["layer_id"],
        grouped["dice_mean"] - grouped["dice_std"],
        grouped["dice_mean"] + grouped["dice_std"],
        color="#55A868",
        alpha=0.18,
        linewidth=0.0,
    )
    if not delta_df.empty:
        seed_order = sorted(delta_df["seed"].unique().tolist())
        offsets = _seed_offsets(len(seed_order))
        seed_to_offset = {int(seed): float(offsets[idx]) for idx, seed in enumerate(seed_order)}
        xs = delta_df["layer_id"].to_numpy(dtype=float) + delta_df["seed"].map(seed_to_offset).to_numpy(dtype=float)
        axes[0].scatter(
            xs,
            delta_df["miou_delta"].to_numpy(dtype=float),
            s=26,
            alpha=0.7,
            color="#C44E52",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        axes[1].scatter(
            xs,
            delta_df["dice_delta"].to_numpy(dtype=float),
            s=26,
            alpha=0.7,
            color="#55A868",
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
    for ax, ylabel in zip(axes, ["Paired Delta mIoU (CAG-ImageNet)", "Paired Delta Dice (CAG-ImageNet)"]):
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(grouped["layer_id"].tolist())
        ax.grid(alpha=0.25)
    fig.suptitle("Layer-wise Paired Delta: CAG minus ImageNet (mean ± std)", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_per_class_iou(
    summary_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy().copy()
    raw_test_df = raw_df[raw_df["split"] == "test"].copy()
    test_df["per_class_iou_mean_list"] = test_df["per_class_iou_mean"].apply(json.loads)
    test_df["per_class_iou_std_list"] = test_df["per_class_iou_std"].apply(json.loads)
    raw_test_df["per_class_iou_list"] = raw_test_df["per_class_iou"].apply(json.loads)
    fig, axes = plt.subplots(1, len(class_names), figsize=(5.4 * len(class_names), 4.8), sharex=True)
    if len(class_names) == 1:
        axes = [axes]
    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]
        for backbone_name, label in [("imagenet", "ImageNet"), ("cag", "CAG")]:
            sub = test_df[test_df["backbone_name"] == backbone_name].sort_values("layer_id")
            ys = [float(values[class_idx]) for values in sub["per_class_iou_mean_list"]]
            stds = [float(values[class_idx]) for values in sub["per_class_iou_std_list"]]
            xs = sub["layer_id"]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=PLOT_COLORS[backbone_name], label=label)
            ax.fill_between(
                xs,
                np.asarray(ys) - np.asarray(stds),
                np.asarray(ys) + np.asarray(stds),
                color=PLOT_COLORS[backbone_name],
                alpha=0.18,
                linewidth=0.0,
            )
            raw_sub = raw_test_df[raw_test_df["backbone_name"] == backbone_name].copy()
            if not raw_sub.empty:
                seed_order = sorted(raw_sub["seed"].unique().tolist())
                offsets = _seed_offsets(len(seed_order))
                seed_to_offset = {int(seed): float(offsets[idx]) for idx, seed in enumerate(seed_order)}
                raw_xs = raw_sub["layer_id"].to_numpy(dtype=float) + raw_sub["seed"].map(seed_to_offset).to_numpy(dtype=float)
                raw_ys = np.asarray(
                    [float(values[class_idx]) for values in raw_sub["per_class_iou_list"]],
                    dtype=np.float64,
                )
                ax.scatter(
                    raw_xs,
                    raw_ys,
                    s=26,
                    alpha=0.7,
                    color=PLOT_COLORS[backbone_name],
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )
        ax.set_title(f"IoU: {class_name}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("IoU")
        ax.set_xticks(sorted(test_df["layer_id"].unique().tolist()))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Layer-wise Per-class IoU", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def derive_interpretation(test_df: pd.DataFrame) -> str:
    pivot = test_df.pivot(index="layer_id", columns="backbone_name", values=["miou_mean", "dice_mean"]).sort_index()
    delta_miou = (pivot[("miou_mean", "cag")] - pivot[("miou_mean", "imagenet")]).to_numpy(dtype=float)
    delta_dice = (pivot[("dice_mean", "cag")] - pivot[("dice_mean", "imagenet")]).to_numpy(dtype=float)
    n_layers = len(delta_miou)
    if n_layers == 0:
        return "insufficient results"
    split_idx = max(1, int(math.ceil(2 * n_layers / 3)))
    early_delta_miou = float(np.mean(delta_miou[:split_idx]))
    early_delta_dice = float(np.mean(delta_dice[:split_idx]))
    late_delta_miou = float(np.mean(delta_miou[split_idx:])) if split_idx < n_layers else float("nan")
    late_delta_dice = float(np.mean(delta_dice[split_idx:])) if split_idx < n_layers else float("nan")
    if split_idx < n_layers and early_delta_miou >= 0.0 and early_delta_dice >= 0.0 and late_delta_miou < 0.0 and late_delta_dice < 0.0:
        return "supports the dense structure retention across depth hypothesis"
    if np.all(delta_miou < 0.0) and np.all(delta_dice < 0.0):
        return "suggests a dense transfer weakness beyond late-layer retention alone"
    if float(np.max(np.abs(np.concatenate([delta_miou, delta_dice])))) < 0.01:
        return "pattern is small and potentially variance-sensitive; confirm with additional seeds before method changes"
    return "shows a mixed depth-dependent pattern; inspect paired delta variance before drawing structural conclusions"


def write_markdown_summary(
    output_path: Path,
    summary_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    selected_layers: Sequence[int],
    seed_values: Sequence[int],
) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy().sort_values(["layer_id", "backbone_name"])
    interpretation = derive_interpretation(test_df)
    pivot = test_df.pivot(index="layer_id", columns="backbone_name", values=["miou_mean", "dice_mean"]).sort_index()
    delta_grouped = delta_df.groupby("layer_id", sort=True).agg(
        miou_delta_mean=("miou_delta", "mean"),
        miou_delta_std=("miou_delta", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
        dice_delta_mean=("dice_delta", "mean"),
        dice_delta_std=("dice_delta", lambda s: float(np.std(s.to_numpy(dtype=float), ddof=0))),
    ).reset_index()
    lines = [
        "# Local Analysis 2-1 — Layer-wise Segmentation Linear Probe",
        "",
        "## Setup",
        "- Strict probe: normalized patch token + 1x1 conv + bilinear upsampling",
        "- Backbone is fully frozen",
        "- Layers evaluated: " + ", ".join(map(str, selected_layers)),
        "- Seeds: " + ", ".join(map(str, seed_values)),
        "- Aggregate rule: mean ± std over fixed seed set",
        "- Delta rule: paired CAG(seed) - ImageNet(seed)",
        "- Input size: 640x640",
        "- Patch size: 16 (40x40 grid)",
        "- Probe parameters: 770 per layer",
        "",
        "## Test Metrics by Layer (test split)",
        "",
        "| Layer | ImageNet mIoU | CAG mIoU | ImageNet Dice | CAG Dice | Paired Delta mIoU | Paired Delta Dice |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    delta_lookup = delta_grouped.set_index("layer_id")
    for layer_id in pivot.index.tolist():
        im_miou = float(pivot.loc[layer_id, ("miou_mean", "imagenet")])
        cg_miou = float(pivot.loc[layer_id, ("miou_mean", "cag")])
        im_dice = float(pivot.loc[layer_id, ("dice_mean", "imagenet")])
        cg_dice = float(pivot.loc[layer_id, ("dice_mean", "cag")])
        miou_delta_mean = float(delta_lookup.loc[layer_id, "miou_delta_mean"])
        miou_delta_std = float(delta_lookup.loc[layer_id, "miou_delta_std"])
        dice_delta_mean = float(delta_lookup.loc[layer_id, "dice_delta_mean"])
        dice_delta_std = float(delta_lookup.loc[layer_id, "dice_delta_std"])
        lines.append(
            f"| {layer_id} | {im_miou:.6f} | {cg_miou:.6f} | {im_dice:.6f} | {cg_dice:.6f} | {miou_delta_mean:.6f} ± {miou_delta_std:.6f} | {dice_delta_mean:.6f} ± {dice_delta_std:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- {interpretation}",
            "- This analysis localizes the last-layer segmentation linear probe result across depth.",
            "- Positive earlier/mid-layer deltas with negative late-layer deltas across multiple seeds support a depth-wise dense retention failure hypothesis.",
            "- Small deltas with large std indicate a variance-sensitive probe result and should not be over-interpreted.",
            "",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def main() -> None:
    args = parse_args()
    seed_values = parse_seed_values(args.seeds)
    base_seed = int(seed_values[0])
    common.set_random_seed(base_seed, strict_deterministic=bool(args.strict_deterministic))

    if len(args.img_size) != 2:
        raise ValueError("--img-size must be two integers: width height")
    if args.img_size[0] % args.patch_size != 0 or args.img_size[1] % args.patch_size != 0:
        raise ValueError("--img-size must be divisible by --patch-size")

    selected_layers = parse_layer_ids(args.layers, total_layers=12)
    out_root = Path(args.output_root).resolve()
    common.ensure_dir(out_root)
    log_path: Path | None = None
    log_handle: TextIO | None = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        log_path, log_handle, original_stdout, original_stderr = common.setup_console_and_file_logging(
            output_root=out_root,
            log_file_arg=args.log_file,
            default_prefix="local_2_1_layerwise_segmentation_linear_probe",
        )
        common.log(f"Run args: {vars(args)}")
        common.log(f"Selected layers (1-based): {selected_layers}")
        common.log(f"Selected seeds: {seed_values}")
        config = common.load_seg_config(args.seg_config)
        all_classes = config["ALL_CLASSES"]
        label_colors_list = config["LABEL_COLORS_LIST"]
        common.log(f"Loaded seg config: {config}")
        tracker = common.ProgressTracker(total_steps=8)

        tracker.start_step("Build paired manifests")
        split_pairs = {}
        split_seed_offset = {"train": 0, "valid": 1, "test": 2}
        for split_name, image_root, mask_root in [
            ("train", args.train_images, args.train_masks),
            ("valid", args.valid_images, args.valid_masks),
            ("test", args.test_images, args.test_masks),
        ]:
            image_paths = common.collect_paths(image_root)
            mask_paths = common.collect_paths(mask_root)
            image_paths, mask_paths = common.pair_image_mask_paths(image_paths, mask_paths, image_root, mask_root)
            image_paths, mask_paths = common.maybe_subsample_pairs(
                image_paths=image_paths,
                mask_paths=mask_paths,
                max_images=args.max_images_per_split,
                seed=base_seed + split_seed_offset[split_name],
            )
            split_pairs[split_name] = (image_paths, mask_paths)
            common.log(f"{split_name}: paired_images={len(image_paths):,}")
        manifests = {
            split_name: common.build_or_load_manifest(
                split_name=split_name,
                image_paths=paths[0],
                mask_paths=paths[1],
                img_size=(int(args.img_size[0]), int(args.img_size[1])),
                patch_size=int(args.patch_size),
                out_root=out_root,
            )
            for split_name, paths in split_pairs.items()
        }
        tracker.finish_step("Build paired manifests")

        tracker.start_step("Build/load target caches")
        targets = {}
        for split_name in ["train", "valid", "test"]:
            targets[split_name] = common.extract_or_load_targets(
                split_name=split_name,
                manifest=manifests[split_name],
                img_size=(int(args.img_size[0]), int(args.img_size[1])),
                label_colors_list=label_colors_list,
                all_classes=all_classes,
                batch_size=int(args.feature_batch_size),
                num_workers=int(args.num_workers),
                out_root=out_root,
            )
            common.log(f"{split_name}: target_cache_shape={tuple(targets[split_name].shape)}")
        tracker.finish_step("Build/load target caches")

        device = common.resolve_device(args.device)
        raw_summary_rows: List[Dict[str, object]] = []
        lr_search_results: Dict[str, Dict[str, Dict[str, object]]] = {}
        total_training_runs = 2 * len(seed_values) * len(selected_layers)
        completed_probe_run_durations: List[float] = []

        for step_label, backbone_name, ckpt_path in [
            ("Run ImageNet layer-wise probes", "imagenet", str(Path(args.imagenet_ckpt).expanduser().resolve())),
            ("Run CAG layer-wise probes", "cag", str(Path(args.cag_ckpt).expanduser().resolve())),
        ]:
            tracker.start_step(step_label)
            lr_search_results[backbone_name] = {}
            split_cache_paths: Dict[str, Dict[int, Path]] = {}
            for split_name in ["train", "valid", "test"]:
                split_cache_paths[split_name] = prepare_layer_feature_caches(
                    split_name=split_name,
                    backbone_name=backbone_name,
                    ckpt_path=ckpt_path,
                    manifest=manifests[split_name],
                    all_classes=all_classes,
                    label_colors_list=label_colors_list,
                    args=args,
                    out_root=out_root,
                    layer_ids=selected_layers,
                )
            for seed_idx, seed in enumerate(seed_values, start=1):
                seed_key = f"seed_{int(seed)}"
                lr_search_results[backbone_name][seed_key] = {}
                common.log(
                    f"[{backbone_name}] Multi-seed run {seed_idx}/{len(seed_values)} START | seed={seed}"
                )
                for idx, layer_id in enumerate(selected_layers, start=1):
                    current_training_run_index = len(completed_probe_run_durations)
                    training_started_at = time.time()

                    def estimate_future_training(
                        substep_remaining: float,
                        *,
                        current_run_index: int = current_training_run_index,
                        started_at: float = training_started_at,
                    ) -> tuple[float, str]:
                        remaining_runs_after = max(0, total_training_runs - current_run_index - 1)
                        current_run_total_estimate = max(
                            float(substep_remaining),
                            max(0.0, time.time() - started_at) + float(substep_remaining),
                        )
                        if completed_probe_run_durations:
                            avg_training_run = sum(completed_probe_run_durations) / len(
                                completed_probe_run_durations
                            )
                        else:
                            avg_training_run = current_run_total_estimate
                        future_training_seconds = remaining_runs_after * avg_training_run
                        return (
                            future_training_seconds,
                            f"future_training_runs={remaining_runs_after} | "
                            f"avg_training_run~{common.format_duration(avg_training_run)}",
                        )

                    common.log(
                        f"[{backbone_name}] Seed {seed} | Layer-wise probe {idx}/{len(selected_layers)} START | "
                        f"layer_id={layer_id} block_index={layer_id - 1}"
                    )
                    train_features = load_cached_layer_feature(split_cache_paths["train"][layer_id])
                    valid_features = load_cached_layer_feature(split_cache_paths["valid"][layer_id])
                    test_features = load_cached_layer_feature(split_cache_paths["test"][layer_id])
                    training_key = f"seed{int(seed)}_{backbone_name}_layer{layer_id:02d}"
                    effective_seed = int(seed) * 1000 + int(layer_id) * 10 + (0 if backbone_name == "imagenet" else 1)
                    model, best_lr, epochs_trained, history, lr_search = common.train_probe_with_lr_search(
                        backbone_name=training_key,
                        backbone_ckpt_path=ckpt_path,
                        train_features=train_features,
                        train_targets=targets["train"],
                        valid_features=valid_features,
                        valid_targets=targets["valid"],
                        args=args,
                        out_root=out_root,
                        full_run_tracker=tracker,
                        future_training_estimator=estimate_future_training,
                        deterministic_seed=effective_seed,
                        strict_deterministic=bool(args.strict_deterministic),
                    )
                    valid_metrics, _ = common.evaluate_probe_split(
                        model=model,
                        backbone_name=backbone_name,
                        split_name="valid",
                        features=valid_features,
                        targets=targets["valid"],
                        manifest=manifests["valid"],
                        best_lr=best_lr,
                        epochs_trained=epochs_trained,
                        class_names=all_classes,
                        device=device,
                        batch_size=int(args.probe_batch_size),
                    )
                    test_metrics, _ = common.evaluate_probe_split(
                        model=model,
                        backbone_name=backbone_name,
                        split_name="test",
                        features=test_features,
                        targets=targets["test"],
                        manifest=manifests["test"],
                        best_lr=best_lr,
                        epochs_trained=epochs_trained,
                        class_names=all_classes,
                        device=device,
                        batch_size=int(args.probe_batch_size),
                    )
                    completed_probe_run_durations.append(max(0.0, time.time() - training_started_at))
                    raw_summary_rows.append(metrics_row_with_seed_and_layer(valid_metrics, seed=seed, layer_id=layer_id))
                    raw_summary_rows.append(metrics_row_with_seed_and_layer(test_metrics, seed=seed, layer_id=layer_id))
                    lr_search_results[backbone_name][seed_key][f"layer_{layer_id:02d}"] = {
                        **lr_search,
                        "seed": int(seed),
                        "effective_seed": int(effective_seed),
                    }
                    del train_features, valid_features, test_features, model, history
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if not args.keep_feature_caches:
                cleanup_backbone_feature_caches(
                    out_root=out_root,
                    backbone_name=backbone_name,
                    split_names=["train", "valid", "test"],
                    layer_ids=selected_layers,
                )
                common.log(f"[{backbone_name}] Removed temporary layer feature caches")
            tracker.finish_step(step_label)

        tracker.start_step("Save summaries")
        raw_summary_df = pd.DataFrame(raw_summary_rows).sort_values(["seed", "layer_id", "backbone_name", "split"]).reset_index(drop=True)
        raw_summary_path = out_root / "summary_layerwise_segmentation_linear_probe_raw.csv"
        raw_summary_df.to_csv(raw_summary_path, index=False)
        common.log(f"Saved raw summary: {raw_summary_path}")
        summary_df = build_aggregate_summary(raw_summary_df)
        summary_path = out_root / "summary_layerwise_segmentation_linear_probe.csv"
        summary_df.to_csv(summary_path, index=False)
        common.log(f"Saved aggregate summary: {summary_path}")
        lr_path = out_root / "layerwise_lr_search_results.json"
        with open(lr_path, "w", encoding="utf-8") as f:
            json.dump(lr_search_results, f, indent=2)
        common.log(f"Saved LR search results: {lr_path}")
        tracker.finish_step("Save summaries")

        tracker.start_step("Render figures")
        delta_df = build_paired_delta_df(raw_summary_df, split="test")
        save_layerwise_curve(summary_df, raw_summary_df, metric_prefix="miou", ylabel="mIoU", title="Layer-wise Test mIoU (mean ± std)", output_path=out_root / "fig_layerwise_miou_mean_std.png")
        save_layerwise_curve(summary_df, raw_summary_df, metric_prefix="dice", ylabel="Dice", title="Layer-wise Test Dice (mean ± std)", output_path=out_root / "fig_layerwise_dice_mean_std.png")
        save_delta_figure(delta_df, output_path=out_root / "fig_layerwise_delta_cag_minus_imagenet_mean_std.png")
        save_per_class_iou(summary_df, raw_summary_df, class_names=all_classes, output_path=out_root / "fig_layerwise_per_class_iou_mean_std.png")
        tracker.finish_step("Render figures")

        tracker.start_step("Write markdown")
        write_markdown_summary(
            output_path=out_root / "analysis_layerwise_segmentation_linear_probe.md",
            summary_df=summary_df,
            delta_df=delta_df,
            selected_layers=selected_layers,
            seed_values=seed_values,
        )
        tracker.finish_step("Write markdown")

        tracker.start_step("Save run metadata")
        run_meta = {
            "analysis": "local_2_1_layerwise_segmentation_linear_probe",
            "args": vars(args),
            "seed_set": list(map(int, seed_values)),
            "determinism_mode": "strict" if bool(args.strict_deterministic) else "practical_seed_reset",
            "aggregate_policy": "mean_std_with_paired_delta",
            "selected_layers": list(map(int, selected_layers)),
            "layer_to_block_index": {str(layer_id): int(layer_id - 1) for layer_id in selected_layers},
            "img_size": list(map(int, args.img_size)),
            "patch_grid": [
                int(args.img_size[1]) // int(args.patch_size),
                int(args.img_size[0]) // int(args.patch_size),
            ],
            "strict_linear_probe_params": 770,
            "manifests": {
                split_name: {
                    "hash": common.hash_dataframe(df, ["image_id", "image_path", "mask_path"]),
                    "num_images": int(len(df)),
                }
                for split_name, df in manifests.items()
            },
            "targets": {
                split_name: {
                    "shape": list(targets[split_name].shape),
                    "dtype": str(targets[split_name].dtype),
                }
                for split_name in ["train", "valid", "test"]
            },
            "keep_feature_caches": bool(args.keep_feature_caches),
            "raw_summary_path": str(raw_summary_path),
            "aggregate_summary_path": str(summary_path),
        }
        with open(out_root / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2)
        common.log(f"Saved run meta: {out_root / 'run_meta.json'}")
        tracker.finish_step("Save run metadata")

        common.log(f"Done. Log file: {log_path}")
    finally:
        if log_handle is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_handle.close()


if __name__ == "__main__":
    main()
