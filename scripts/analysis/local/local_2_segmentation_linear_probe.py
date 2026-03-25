#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import _seg_linear_probe_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Analysis 2 — Segmentation Linear Probe")
    parser.add_argument("--train-images", default="input/MPXA-Seg/train_images")
    parser.add_argument("--train-masks", default="input/MPXA-Seg/train_labels")
    parser.add_argument("--valid-images", default="input/MPXA-Seg/valid_images")
    parser.add_argument("--valid-masks", default="input/MPXA-Seg/valid_labels")
    parser.add_argument("--test-images", default="input/MPXA-Seg/test_images")
    parser.add_argument("--test-masks", default="input/MPXA-Seg/test_labels")
    parser.add_argument("--seg-config", default="configs_segmentation/mpxa-seg.yaml")
    parser.add_argument("--imagenet-ckpt", default=common.DEFAULT_IMAGENET_CKPT)
    parser.add_argument("--cag-ckpt", default=common.DEFAULT_CAG_CKPT)
    parser.add_argument("--output-root", default="outputs/local_2_segmentation_linear_probe")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--img-size", nargs=2, type=int, default=[640, 640], help="width height")
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--lr-grid", nargs="+", type=float, default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-example-images", type=int, default=8)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if len(args.img_size) != 2:
        raise ValueError("--img-size must be two integers: width height")
    if args.img_size[0] % args.patch_size != 0 or args.img_size[1] % args.patch_size != 0:
        raise ValueError("--img-size must be divisible by --patch-size")

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
            default_prefix="local_2_segmentation_linear_probe",
        )
        common.log(f"Run args: {vars(args)}")
        config = common.load_seg_config(args.seg_config)
        all_classes = config["ALL_CLASSES"]
        label_colors_list = config["LABEL_COLORS_LIST"]
        viz_map = config["VIS_LABEL_MAP"]
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

        rng = np.random.default_rng(args.seed)
        num_example_images = min(int(args.num_example_images), len(manifests["test"]))
        example_indices = np.sort(
            rng.choice(len(manifests["test"]), size=num_example_images, replace=False)
        ).tolist()
        common.log(f"Selected test example indices: {example_indices}")
        device = common.resolve_device(args.device)

        tracker.start_step("Run ImageNet linear probe")
        ckpt_imagenet = str(Path(args.imagenet_ckpt).expanduser().resolve())
        imagenet_train_features = common.extract_or_load_features(
            split_name="train",
            backbone_name="imagenet",
            ckpt_path=ckpt_imagenet,
            manifest=manifests["train"],
            all_classes=all_classes,
            label_colors_list=label_colors_list,
            args=args,
            out_root=out_root,
        )
        imagenet_valid_features = common.extract_or_load_features(
            split_name="valid",
            backbone_name="imagenet",
            ckpt_path=ckpt_imagenet,
            manifest=manifests["valid"],
            all_classes=all_classes,
            label_colors_list=label_colors_list,
            args=args,
            out_root=out_root,
        )
        imagenet_model, imagenet_best_lr, imagenet_epochs_trained, imagenet_history, imagenet_lr_search = common.train_probe_with_lr_search(
            backbone_name="imagenet",
            backbone_ckpt_path=ckpt_imagenet,
            train_features=imagenet_train_features,
            train_targets=targets["train"],
            valid_features=imagenet_valid_features,
            valid_targets=targets["valid"],
            args=args,
            out_root=out_root,
            full_run_tracker=tracker,
        )
        imagenet_valid_metrics, _ = common.evaluate_probe_split(
            model=imagenet_model,
            backbone_name="imagenet",
            split_name="valid",
            features=imagenet_valid_features,
            targets=targets["valid"],
            manifest=manifests["valid"],
            best_lr=imagenet_best_lr,
            epochs_trained=imagenet_epochs_trained,
            class_names=all_classes,
            device=device,
            batch_size=int(args.probe_batch_size),
        )
        del imagenet_train_features
        if device.type == "cuda":
            torch.cuda.empty_cache()
        imagenet_test_features = common.extract_or_load_features(
            split_name="test",
            backbone_name="imagenet",
            ckpt_path=ckpt_imagenet,
            manifest=manifests["test"],
            all_classes=all_classes,
            label_colors_list=label_colors_list,
            args=args,
            out_root=out_root,
        )
        imagenet_test_metrics, imagenet_test_df = common.evaluate_probe_split(
            model=imagenet_model,
            backbone_name="imagenet",
            split_name="test",
            features=imagenet_test_features,
            targets=targets["test"],
            manifest=manifests["test"],
            best_lr=imagenet_best_lr,
            epochs_trained=imagenet_epochs_trained,
            class_names=all_classes,
            device=device,
            batch_size=int(args.probe_batch_size),
        )
        imagenet_examples = common.collect_example_predictions(
            model=imagenet_model,
            test_features=imagenet_test_features,
            test_targets=targets["test"],
            test_manifest=manifests["test"],
            example_indices=example_indices,
            img_size=(int(args.img_size[0]), int(args.img_size[1])),
            viz_map=viz_map,
            device=device,
        )
        tracker.finish_step("Run ImageNet linear probe")

        tracker.start_step("Run CAG linear probe")
        ckpt_cag = str(Path(args.cag_ckpt).expanduser().resolve())
        cag_train_features = common.extract_or_load_features(
            split_name="train",
            backbone_name="cag",
            ckpt_path=ckpt_cag,
            manifest=manifests["train"],
            all_classes=all_classes,
            label_colors_list=label_colors_list,
            args=args,
            out_root=out_root,
        )
        cag_valid_features = common.extract_or_load_features(
            split_name="valid",
            backbone_name="cag",
            ckpt_path=ckpt_cag,
            manifest=manifests["valid"],
            all_classes=all_classes,
            label_colors_list=label_colors_list,
            args=args,
            out_root=out_root,
        )
        cag_model, cag_best_lr, cag_epochs_trained, cag_history, cag_lr_search = common.train_probe_with_lr_search(
            backbone_name="cag",
            backbone_ckpt_path=ckpt_cag,
            train_features=cag_train_features,
            train_targets=targets["train"],
            valid_features=cag_valid_features,
            valid_targets=targets["valid"],
            args=args,
            out_root=out_root,
            full_run_tracker=tracker,
        )
        cag_valid_metrics, _ = common.evaluate_probe_split(
            model=cag_model,
            backbone_name="cag",
            split_name="valid",
            features=cag_valid_features,
            targets=targets["valid"],
            manifest=manifests["valid"],
            best_lr=cag_best_lr,
            epochs_trained=cag_epochs_trained,
            class_names=all_classes,
            device=device,
            batch_size=int(args.probe_batch_size),
        )
        del cag_train_features
        if device.type == "cuda":
            torch.cuda.empty_cache()
        cag_test_features = common.extract_or_load_features(
            split_name="test",
            backbone_name="cag",
            ckpt_path=ckpt_cag,
            manifest=manifests["test"],
            all_classes=all_classes,
            label_colors_list=label_colors_list,
            args=args,
            out_root=out_root,
        )
        cag_test_metrics, cag_test_df = common.evaluate_probe_split(
            model=cag_model,
            backbone_name="cag",
            split_name="test",
            features=cag_test_features,
            targets=targets["test"],
            manifest=manifests["test"],
            best_lr=cag_best_lr,
            epochs_trained=cag_epochs_trained,
            class_names=all_classes,
            device=device,
            batch_size=int(args.probe_batch_size),
        )
        cag_examples = common.collect_example_predictions(
            model=cag_model,
            test_features=cag_test_features,
            test_targets=targets["test"],
            test_manifest=manifests["test"],
            example_indices=example_indices,
            img_size=(int(args.img_size[0]), int(args.img_size[1])),
            viz_map=viz_map,
            device=device,
        )
        tracker.finish_step("Run CAG linear probe")

        tracker.start_step("Save summaries")
        summary_df = pd.DataFrame(
            [
                imagenet_valid_metrics.to_row(),
                imagenet_test_metrics.to_row(),
                cag_valid_metrics.to_row(),
                cag_test_metrics.to_row(),
            ]
        )
        summary_path = out_root / "summary_segmentation_linear_probe.csv"
        summary_df.to_csv(summary_path, index=False)
        common.log(f"Saved summary: {summary_path}")

        imagenet_per_image_path = out_root / "per_image_metrics_imagenet_test.csv"
        cag_per_image_path = out_root / "per_image_metrics_cag_test.csv"
        imagenet_test_df.to_csv(imagenet_per_image_path, index=False)
        cag_test_df.to_csv(cag_per_image_path, index=False)
        common.log(f"Saved per-image metrics: {imagenet_per_image_path}")
        common.log(f"Saved per-image metrics: {cag_per_image_path}")

        lr_search_path = out_root / "lr_search_results.json"
        with open(lr_search_path, "w", encoding="utf-8") as f:
            json.dump({"imagenet": imagenet_lr_search, "cag": cag_lr_search}, f, indent=2)
        common.log(f"Saved LR search results: {lr_search_path}")
        tracker.finish_step("Save summaries")

        tracker.start_step("Render figures")
        common.save_bar_compare(summary_df, out_root / "fig_seg_linear_probe_bar_compare.png")
        common.save_learning_curves(
            histories={"imagenet": imagenet_history, "cag": cag_history},
            output_path=out_root / "fig_seg_linear_probe_learning_curves.png",
        )
        common.save_example_figure(
            imagenet_examples=imagenet_examples,
            cag_examples=cag_examples,
            output_path=out_root / "fig_seg_linear_probe_examples.png",
        )
        tracker.finish_step("Render figures")

        tracker.start_step("Write markdown")
        common.write_markdown_summary(
            output_path=out_root / "analysis_segmentation_linear_probe.md",
            summary_df=summary_df,
            probe_params=int(imagenet_test_metrics.probe_params),
        )
        tracker.finish_step("Write markdown")

        tracker.start_step("Save run metadata")
        run_meta = {
            "analysis": "local_2_segmentation_linear_probe",
            "args": vars(args),
            "img_size": list(map(int, args.img_size)),
            "patch_grid": [
                int(args.img_size[1]) // int(args.patch_size),
                int(args.img_size[0]) // int(args.patch_size),
            ],
            "strict_linear_probe_params": int(imagenet_test_metrics.probe_params),
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
            "examples": [manifests["test"].iloc[idx]["image_path"] for idx in example_indices],
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
