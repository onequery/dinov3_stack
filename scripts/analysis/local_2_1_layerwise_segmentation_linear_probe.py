#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
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
    parser.add_argument("--output-root", default="outputs/local_2_1_layerwise_segmentation_linear_probe")
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
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


def metrics_row_with_layer(split_metrics: common.SplitMetrics, layer_id: int) -> Dict[str, object]:
    row = split_metrics.to_row()
    row["layer_id"] = int(layer_id)
    row["block_index"] = int(layer_id - 1)
    preferred_order = [
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


def save_layerwise_curve(summary_df: pd.DataFrame, metric: str, ylabel: str, title: str, output_path: Path) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for backbone_name, label in [("imagenet", "ImageNet"), ("cag", "CAG")]:
        sub = test_df[test_df["backbone_name"] == backbone_name].sort_values("layer_id")
        ax.plot(
            sub["layer_id"],
            sub[metric],
            marker="o",
            linewidth=2.0,
            color=PLOT_COLORS[backbone_name],
            label=label,
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


def save_delta_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy()
    pivot = test_df.pivot(index="layer_id", columns="backbone_name", values=["miou", "dice"]).sort_index()
    delta_miou = pivot[("miou", "cag")] - pivot[("miou", "imagenet")]
    delta_dice = pivot[("dice", "cag")] - pivot[("dice", "imagenet")]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
    axes[0].plot(delta_miou.index, delta_miou.values, marker="o", color="#C44E52")
    axes[1].plot(delta_dice.index, delta_dice.values, marker="o", color="#55A868")
    for ax, ylabel in zip(axes, ["Delta mIoU (CAG-ImageNet)", "Delta Dice (CAG-ImageNet)"]):
        ax.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(delta_miou.index.tolist())
        ax.grid(alpha=0.25)
    fig.suptitle("Layer-wise Delta: CAG minus ImageNet", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_per_class_iou(summary_df: pd.DataFrame, class_names: Sequence[str], output_path: Path) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy().copy()
    test_df["per_class_iou_list"] = test_df["per_class_iou"].apply(json.loads)
    fig, axes = plt.subplots(1, len(class_names), figsize=(5.4 * len(class_names), 4.8), sharex=True)
    if len(class_names) == 1:
        axes = [axes]
    for class_idx, class_name in enumerate(class_names):
        ax = axes[class_idx]
        for backbone_name, label in [("imagenet", "ImageNet"), ("cag", "CAG")]:
            sub = test_df[test_df["backbone_name"] == backbone_name].sort_values("layer_id")
            ys = [float(values[class_idx]) for values in sub["per_class_iou_list"]]
            ax.plot(sub["layer_id"], ys, marker="o", linewidth=2.0, color=PLOT_COLORS[backbone_name], label=label)
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
    pivot = test_df.pivot(index="layer_id", columns="backbone_name", values=["miou", "dice"]).sort_index()
    delta_miou = (pivot[("miou", "cag")] - pivot[("miou", "imagenet")]).to_numpy(dtype=float)
    delta_dice = (pivot[("dice", "cag")] - pivot[("dice", "imagenet")]).to_numpy(dtype=float)
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
    return "shows a mixed depth-dependent pattern; retention and optimization effects remain entangled"


def write_markdown_summary(output_path: Path, summary_df: pd.DataFrame, selected_layers: Sequence[int]) -> None:
    test_df = summary_df[summary_df["split"] == "test"].copy().sort_values(["layer_id", "backbone_name"])
    interpretation = derive_interpretation(test_df)
    pivot = test_df.pivot(index="layer_id", columns="backbone_name", values=["miou", "dice"]).sort_index()
    lines = [
        "# Local Analysis 2-1 — Layer-wise Segmentation Linear Probe",
        "",
        "## Setup",
        "- Strict probe: normalized patch token + 1x1 conv + bilinear upsampling",
        "- Backbone is fully frozen",
        "- Layers evaluated: " + ", ".join(map(str, selected_layers)),
        "- Input size: 640x640",
        "- Patch size: 16 (40x40 grid)",
        "- Probe parameters: 770 per layer",
        "",
        "## Test Metrics by Layer",
        "",
        "| Layer | ImageNet mIoU | CAG mIoU | ImageNet Dice | CAG Dice | Delta mIoU | Delta Dice |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer_id in pivot.index.tolist():
        im_miou = float(pivot.loc[layer_id, ("miou", "imagenet")])
        cg_miou = float(pivot.loc[layer_id, ("miou", "cag")])
        im_dice = float(pivot.loc[layer_id, ("dice", "imagenet")])
        cg_dice = float(pivot.loc[layer_id, ("dice", "cag")])
        lines.append(
            f"| {layer_id} | {im_miou:.6f} | {cg_miou:.6f} | {im_dice:.6f} | {cg_dice:.6f} | {cg_miou - im_miou:.6f} | {cg_dice - im_dice:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            f"- {interpretation}",
            "- This analysis localizes the last-layer segmentation linear probe result across depth.",
            "- Positive earlier/mid-layer deltas with negative late-layer deltas support a depth-wise dense retention failure hypothesis.",
            "",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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
                seed=args.seed + split_seed_offset[split_name],
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
        summary_rows: List[Dict[str, object]] = []
        lr_search_results: Dict[str, Dict[str, object]] = {"imagenet": {}, "cag": {}}
        histories: Dict[str, Dict[int, List[Dict[str, object]]]] = {"imagenet": {}, "cag": {}}

        for step_label, backbone_name, ckpt_path in [
            ("Run ImageNet layer-wise probes", "imagenet", str(Path(args.imagenet_ckpt).expanduser().resolve())),
            ("Run CAG layer-wise probes", "cag", str(Path(args.cag_ckpt).expanduser().resolve())),
        ]:
            tracker.start_step(step_label)
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
            for idx, layer_id in enumerate(selected_layers, start=1):
                common.log(
                    f"[{backbone_name}] Layer-wise probe {idx}/{len(selected_layers)} START | "
                    f"layer_id={layer_id} block_index={layer_id - 1}"
                )
                train_features = load_cached_layer_feature(split_cache_paths["train"][layer_id])
                valid_features = load_cached_layer_feature(split_cache_paths["valid"][layer_id])
                test_features = load_cached_layer_feature(split_cache_paths["test"][layer_id])
                training_key = f"{backbone_name}_layer{layer_id:02d}"
                model, best_lr, epochs_trained, history, lr_search = common.train_probe_with_lr_search(
                    backbone_name=training_key,
                    backbone_ckpt_path=ckpt_path,
                    train_features=train_features,
                    train_targets=targets["train"],
                    valid_features=valid_features,
                    valid_targets=targets["valid"],
                    args=args,
                    out_root=out_root,
                    full_run_tracker=None,
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
                summary_rows.append(metrics_row_with_layer(valid_metrics, layer_id=layer_id))
                summary_rows.append(metrics_row_with_layer(test_metrics, layer_id=layer_id))
                histories[backbone_name][layer_id] = history
                lr_search_results[backbone_name][f"layer_{layer_id:02d}"] = lr_search
                del train_features, valid_features, test_features, model
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
        summary_df = pd.DataFrame(summary_rows).sort_values(["layer_id", "backbone_name", "split"]).reset_index(drop=True)
        summary_path = out_root / "summary_layerwise_segmentation_linear_probe.csv"
        summary_df.to_csv(summary_path, index=False)
        common.log(f"Saved summary: {summary_path}")
        lr_path = out_root / "layerwise_lr_search_results.json"
        with open(lr_path, "w", encoding="utf-8") as f:
            json.dump(lr_search_results, f, indent=2)
        common.log(f"Saved LR search results: {lr_path}")
        tracker.finish_step("Save summaries")

        tracker.start_step("Render figures")
        save_layerwise_curve(summary_df, metric="miou", ylabel="mIoU", title="Layer-wise Test mIoU", output_path=out_root / "fig_layerwise_miou.png")
        save_layerwise_curve(summary_df, metric="dice", ylabel="Dice", title="Layer-wise Test Dice", output_path=out_root / "fig_layerwise_dice.png")
        save_delta_figure(summary_df, output_path=out_root / "fig_layerwise_delta_cag_minus_imagenet.png")
        save_per_class_iou(summary_df, class_names=all_classes, output_path=out_root / "fig_layerwise_per_class_iou.png")
        tracker.finish_step("Render figures")

        tracker.start_step("Write markdown")
        write_markdown_summary(
            output_path=out_root / "analysis_layerwise_segmentation_linear_probe.md",
            summary_df=summary_df,
            selected_layers=selected_layers,
        )
        tracker.finish_step("Write markdown")

        tracker.start_step("Save run metadata")
        run_meta = {
            "analysis": "local_2_1_layerwise_segmentation_linear_probe",
            "args": vars(args),
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
