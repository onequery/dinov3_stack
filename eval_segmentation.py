#!/usr/bin/env python3
"""
Segmentation evaluation script (DINOv3).

Outputs:
  - Console summary (overall accuracy, mIoU)
  - metrics.json
  - per_class_metrics.csv
"""

import argparse
import glob
import json
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from src.img_seg.datasets import SegmentationDataset, valid_transforms, collate_fn
from src.img_seg.metrics import IOUEval
from src.img_seg.model import Dinov3Segmentation
from src.utils.common import get_dinov3_paths


def collect_paths(root_dir: str) -> List[str]:
    paths = glob.glob(os.path.join(root_dir, "*"))
    paths.sort()
    return paths


def build_dataloader(
    image_dir: str,
    mask_dir: str,
    img_size: List[int],
    label_colors_list: List[List[int]],
    classes: List[str],
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    image_paths = collect_paths(image_dir)
    mask_paths = collect_paths(mask_dir)

    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {image_dir}")
    if len(mask_paths) == 0:
        raise ValueError(f"No masks found in: {mask_dir}")
    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Image/Mask count mismatch: {len(image_paths)} vs {len(mask_paths)}"
        )

    tfms = valid_transforms(img_size)
    dataset = SegmentationDataset(
        image_paths=image_paths,
        mask_paths=mask_paths,
        tfms=tfms,
        label_colors_list=label_colors_list,
        classes_to_train=classes,
        all_classes=classes,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return dataloader


def load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, List[float], List[float], float]:
    iou_eval = IOUEval(num_classes)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values, target = batch[0].to(device), batch[1].to(device)
            outputs = model(pixel_values)

            upsampled_logits = F.interpolate(
                outputs,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preds = upsampled_logits.max(1)[1]
            iou_eval.addBatch(preds, target)

    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval.getMetric()
    return float(overall_acc), per_class_acc.tolist(), per_class_iu.tolist(), float(mIOU)


def save_metrics(
    out_dir: str,
    class_names: List[str],
    overall_acc: float,
    per_class_acc: List[float],
    per_class_iou: List[float],
    mIOU: float,
    num_images: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "overall_acc": overall_acc,
        "mIoU": mIOU,
        "num_images": num_images,
        "per_class_acc": {
            class_names[i]: per_class_acc[i] for i in range(len(class_names))
        },
        "per_class_iou": {
            class_names[i]: per_class_iou[i] for i in range(len(class_names))
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    csv_path = os.path.join(out_dir, "per_class_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("class_id,class_name,acc,iou\n")
        for i, name in enumerate(class_names):
            f.write(f"{i},{name},{per_class_acc[i]:.6f},{per_class_iou[i]:.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-images", required=True, help="path to eval images")
    parser.add_argument("--eval-masks", required=True, help="path to eval masks")
    parser.add_argument("--config", required=True, help="dataset config YAML")
    parser.add_argument("--weights", required=True, help="trained model weights path")
    parser.add_argument("--out-dir", default="outputs/eval_segmentation")
    parser.add_argument("--imgsz", default=[640, 640], type=int, nargs="+")
    parser.add_argument("--batch", default=4, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument(
        "--feature-extractor",
        dest="feature_extractor",
        default="multi",
        choices=["last", "multi"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--repo-dir", default=None)
    parser.add_argument(
        "--backbone-weights",
        default=None,
        help="optional backbone weights (relative to DINOv3_WEIGHTS if not absolute)",
    )
    args = parser.parse_args()

    if len(args.imgsz) != 2:
        raise ValueError("--imgsz must be two integers: width height")

    if args.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    all_classes = config["ALL_CLASSES"]
    label_colors_list = config["LABEL_COLORS_LIST"]

    # Resolve DINOv3 repo / weights root
    dinov3_repo, dinov3_weights = get_dinov3_paths()
    repo_dir = args.repo_dir if args.repo_dir is not None else dinov3_repo

    backbone_weights = None
    if args.backbone_weights:
        if os.path.isabs(args.backbone_weights):
            backbone_weights = args.backbone_weights
        else:
            backbone_weights = os.path.join(dinov3_weights, args.backbone_weights)

    # Build dataloader
    dataloader = build_dataloader(
        image_dir=args.eval_images,
        mask_dir=args.eval_masks,
        img_size=args.imgsz,
        label_colors_list=label_colors_list,
        classes=all_classes,
        batch_size=args.batch,
        num_workers=args.num_workers,
    )

    # Build model
    model = Dinov3Segmentation(
        fine_tune=False,
        num_classes=len(all_classes),
        weights=backbone_weights,
        model_name=args.model_name,
        repo_dir=repo_dir,
        feature_extractor=args.feature_extractor,
    ).to(device)

    state_dict = load_checkpoint(args.weights, device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        raise RuntimeError(
            "Failed to load weights. Pass a full model checkpoint "
            "(e.g., best_model_iou.pth), not decode_head-only weights."
        ) from e

    # Evaluate
    overall_acc, per_class_acc, per_class_iou, mIOU = evaluate(
        model, dataloader, device, num_classes=len(all_classes)
    )

    # Save + print
    save_metrics(
        out_dir=args.out_dir,
        class_names=all_classes,
        overall_acc=overall_acc,
        per_class_acc=per_class_acc,
        per_class_iou=per_class_iou,
        mIOU=mIOU,
        num_images=len(dataloader.dataset),
    )

    print("========== Segmentation Evaluation ==========")
    print(f"Num images: {len(dataloader.dataset)}")
    print(f"Overall Acc: {overall_acc:.4f}")
    print(f"mIoU: {mIOU:.4f}")
    print(f"Saved metrics to: {args.out_dir}")


if __name__ == "__main__":
    main()
