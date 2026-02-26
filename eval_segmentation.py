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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def collect_paths(root_dir: str, extensions: set[str]) -> List[str]:
    paths = glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
    paths = [
        path
        for path in paths
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions
    ]
    paths.sort()
    return paths


def pair_image_mask_paths(
    image_paths: List[str],
    mask_paths: List[str],
    image_root: str,
    mask_root: str,
) -> Tuple[List[str], List[str]]:
    image_map = {os.path.relpath(path, image_root): path for path in image_paths}
    mask_map = {os.path.relpath(path, mask_root): path for path in mask_paths}

    shared_paths = sorted(set(image_map.keys()) & set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))

    if missing_images or missing_masks:
        print(
            "[WARN] eval: paired using intersection only "
            f"(missing_images={len(missing_images)}, missing_masks={len(missing_masks)})."
        )

    if not shared_paths:
        raise ValueError(
            "No paired image/mask files found. "
            f"images_root={image_root}, masks_root={mask_root}"
        )

    return [image_map[path] for path in shared_paths], [mask_map[path] for path in shared_paths]


def build_dataloader(
    image_dir: str,
    mask_dir: str,
    img_size: List[int],
    label_colors_list: List[List[int]],
    classes: List[str],
    batch_size: int,
    num_workers: int,
) -> torch.utils.data.DataLoader:
    image_paths = collect_paths(image_dir, IMAGE_EXTENSIONS)
    mask_paths = collect_paths(mask_dir, IMAGE_EXTENSIONS)

    image_paths, mask_paths = pair_image_mask_paths(
        image_paths, mask_paths, image_dir, mask_dir
    )

    if len(image_paths) == 0:
        raise ValueError(f"No images found in: {image_dir}")
    if len(mask_paths) == 0:
        raise ValueError(f"No masks found in: {mask_dir}")

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
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
    )
    return dataloader


def load_checkpoint(path: str, device: torch.device) -> dict:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def normalize_state_dict_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint state_dict must be a dictionary.")
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {
            key[len("module.") :]: value
            for key, value in state_dict.items()
            if key.startswith("module.")
        }
    return state_dict


def infer_decoder_hidden_channels(state_dict: dict) -> int:
    key = "decode_head.decode.0.weight"
    if key not in state_dict:
        raise ValueError(f"Cannot infer decoder hidden channels: missing key `{key}`")
    weight = state_dict[key]
    if weight.ndim != 4:
        raise ValueError(f"Unexpected tensor shape for `{key}`: {tuple(weight.shape)}")
    return int(weight.shape[0])


def infer_lora_rank(state_dict: dict) -> int | None:
    ranks = set()
    for key, value in state_dict.items():
        if key.startswith("backbone_model.") and key.endswith(".lora_A"):
            ranks.add(int(value.shape[0]))
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"Multiple LoRA ranks detected in checkpoint: {sorted(ranks)}")
    return list(ranks)[0]


def resolve_repo_path(repo_dir_arg: str | None, env_repo_dir: str | None) -> str:
    if repo_dir_arg:
        repo_path = os.path.abspath(os.path.expanduser(repo_dir_arg))
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"DINOv3 repository not found at: {repo_path}")
        return repo_path

    if not env_repo_dir:
        raise ValueError(
            "DINOv3 repository path is missing. "
            "Set DINOV3_REPO in .env or pass --repo-dir."
        )

    return env_repo_dir


def resolve_checkpoint_path(weights_arg: str) -> str:
    candidate = os.path.abspath(os.path.expanduser(weights_arg))
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"Checkpoint not found at: {candidate}")
    return candidate


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

    dinov3_repo, _ = get_dinov3_paths(
        require_repo=not bool(args.repo_dir),
        require_weights=False,
    )
    repo_dir = resolve_repo_path(args.repo_dir, dinov3_repo)
    checkpoint_path = resolve_checkpoint_path(args.weights)

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

    state_dict = normalize_state_dict_keys(load_checkpoint(checkpoint_path, device))
    decoder_hidden_channels = infer_decoder_hidden_channels(state_dict)
    lora_rank = infer_lora_rank(state_dict)
    print(
        "Inferred segmentation config | "
        f"decoder_hidden_channels={decoder_hidden_channels} "
        f"| lora_rank={lora_rank if lora_rank is not None else 'none'}"
    )

    # Build model
    model = Dinov3Segmentation(
        fine_tune=False,
        num_classes=len(all_classes),
        decoder_hidden_channels=decoder_hidden_channels,
        enable_lora=(lora_rank is not None),
        lora_rank=lora_rank,
        lora_alpha=lora_rank,
        lora_target="attn_qkv_proj",
        weights=checkpoint_path,
        model_name=args.model_name,
        repo_dir=repo_dir,
        feature_extractor=args.feature_extractor,
    ).to(device)

    if not any(key.startswith("backbone_model.") for key in state_dict.keys()):
        raise RuntimeError(
            "The checkpoint does not include full segmentation model weights "
            "(missing `backbone_model.*` keys). "
            "Please pass a full checkpoint such as `best_model_iou.pth`."
        )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        detail = str(e).splitlines()[0]
        raise RuntimeError(
            "Failed to load segmentation checkpoint. "
            f"checkpoint={checkpoint_path}. "
            f"details={detail}"
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
