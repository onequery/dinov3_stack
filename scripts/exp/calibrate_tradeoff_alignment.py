#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import yaml


@dataclass
class ModelSpec:
    embed_dim: int
    num_blocks: int


MODEL_SPECS = {
    "dinov3_vits16": ModelSpec(embed_dim=384, num_blocks=12),
    "dinov3_vits16plus": ModelSpec(embed_dim=384, num_blocks=12),
    "dinov3_vitb16": ModelSpec(embed_dim=768, num_blocks=12),
    "dinov3_vitl16": ModelSpec(embed_dim=1024, num_blocks=24),
}


def cls_head_params(embed_dim: int, num_classes: int, head_size: str, hidden_dim: int | None = None) -> int:
    if head_size == "small":
        return embed_dim * num_classes + num_classes
    if hidden_dim is None or hidden_dim <= 0:
        raise ValueError("big CLS head requires positive hidden_dim")
    return (embed_dim * hidden_dim + hidden_dim) + (hidden_dim * num_classes + num_classes)


def ret_head_params(embed_dim: int, proj_dim: int, head_size: str, hidden_dim: int | None = None) -> int:
    if head_size == "small":
        h = embed_dim
    else:
        if hidden_dim is None or hidden_dim <= 0:
            raise ValueError("big RET head requires positive hidden_dim")
        h = hidden_dim

    # Linear(embed->h) + BN(h) + Linear(h->proj)
    linear1 = embed_dim * h + h
    bn = 2 * h
    linear2 = h * proj_dim + proj_dim
    return linear1 + bn + linear2


def seg_head_params(in_channels: int, num_classes: int, hidden_channels: int) -> int:
    if hidden_channels <= 0:
        raise ValueError("hidden_channels must be > 0")
    # Conv3x3(in->h) + Conv1x1(h->num_classes)
    conv1 = in_channels * hidden_channels * 3 * 3 + hidden_channels
    conv2 = hidden_channels * num_classes + num_classes
    return conv1 + conv2


def lora_params_per_block(embed_dim: int, rank: int) -> int:
    # qkv: d -> 3d => rank*(d+3d)
    # proj: d -> d  => rank*(d+d)
    return rank * ((embed_dim + 3 * embed_dim) + (embed_dim + embed_dim))


def total_lora_params(spec: ModelSpec, rank: int) -> int:
    return spec.num_blocks * lora_params_per_block(spec.embed_dim, rank)


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml dict at {path}")
    return data


def select_best_pair(
    small_head_params: int,
    big_candidates: list[int],
    lora_ranks: list[int],
    spec: ModelSpec,
):
    best = None
    for big_size in big_candidates:
        for rank in lora_ranks:
            big_params = big_size
            lora_params = total_lora_params(spec, rank)
            target = small_head_params + lora_params
            delta = abs(big_params - target)
            tie_budget = max(big_params, target)
            key = (delta, tie_budget)
            payload = {
                "big_params": int(big_params),
                "lora_rank": int(rank),
                "lora_params": int(lora_params),
                "target_params": int(target),
                "delta_params": int(delta),
                "delta_ratio": float(delta / max(big_params, 1)),
            }
            if best is None or key < best[0]:
                best = (key, payload)
    if best is None:
        raise RuntimeError("Failed to select alignment pair")
    return best[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--cls-config", default="configs_classification/stent.yaml")
    parser.add_argument("--seg-config", default="configs_segmentation/mpxa-seg.yaml")
    parser.add_argument("--ret-proj-dim", type=int, default=128)
    parser.add_argument("--feature-extractor", choices=["last", "multi"], default="multi")
    parser.add_argument("--lora-ranks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--cls-big-hidden-candidates", nargs="+", type=int, default=[128, 192, 256, 384, 512, 768, 1024])
    parser.add_argument("--ret-big-hidden-candidates", nargs="+", type=int, default=[512, 768, 1024, 1536])
    parser.add_argument("--seg-big-hidden-candidates", nargs="+", type=int, default=[384, 512, 640, 768, 1024])
    parser.add_argument("--out-config", default="outputs/5_lora_tradeoff/configs/alignment_selected.yaml")
    args = parser.parse_args()

    if args.model_name not in MODEL_SPECS:
        raise ValueError(f"Unsupported model-name for calibration: {args.model_name}")
    spec = MODEL_SPECS[args.model_name]

    cls_cfg = load_yaml(args.cls_config)
    seg_cfg = load_yaml(args.seg_config)

    cls_num_classes = len(cls_cfg.get("CLASS_NAMES", []))
    seg_num_classes = len(seg_cfg.get("ALL_CLASSES", []))
    if cls_num_classes <= 0:
        raise ValueError("CLASS_NAMES not found in cls config")
    if seg_num_classes <= 0:
        raise ValueError("ALL_CLASSES not found in seg config")

    embed_dim = spec.embed_dim
    seg_in_channels = embed_dim if args.feature_extractor == "last" else embed_dim * 4

    cls_small = cls_head_params(embed_dim, cls_num_classes, "small")
    cls_big_param_candidates = [
        cls_head_params(embed_dim, cls_num_classes, "big", hidden_dim=h)
        for h in args.cls_big_hidden_candidates
    ]
    cls_selected = select_best_pair(
        small_head_params=cls_small,
        big_candidates=cls_big_param_candidates,
        lora_ranks=args.lora_ranks,
        spec=spec,
    )
    cls_hidden_lookup = {
        cls_head_params(embed_dim, cls_num_classes, "big", hidden_dim=h): h
        for h in args.cls_big_hidden_candidates
    }

    ret_small = ret_head_params(embed_dim, args.ret_proj_dim, "small")
    ret_big_param_candidates = [
        ret_head_params(embed_dim, args.ret_proj_dim, "big", hidden_dim=h)
        for h in args.ret_big_hidden_candidates
    ]
    ret_selected = select_best_pair(
        small_head_params=ret_small,
        big_candidates=ret_big_param_candidates,
        lora_ranks=args.lora_ranks,
        spec=spec,
    )
    ret_hidden_lookup = {
        ret_head_params(embed_dim, args.ret_proj_dim, "big", hidden_dim=h): h
        for h in args.ret_big_hidden_candidates
    }

    seg_small_hidden = 256
    seg_small = seg_head_params(seg_in_channels, seg_num_classes, seg_small_hidden)
    seg_big_param_candidates = [
        seg_head_params(seg_in_channels, seg_num_classes, hidden_channels=h)
        for h in args.seg_big_hidden_candidates
    ]
    seg_selected = select_best_pair(
        small_head_params=seg_small,
        big_candidates=seg_big_param_candidates,
        lora_ranks=args.lora_ranks,
        spec=spec,
    )
    seg_hidden_lookup = {
        seg_head_params(seg_in_channels, seg_num_classes, hidden_channels=h): h
        for h in args.seg_big_hidden_candidates
    }

    payload = {
        "model_name": args.model_name,
        "calibration_rule": "min_abs_error_then_min_budget",
        "lora_target": "attn_qkv_proj",
        "small_head_definition": {
            "cls": "current linear head",
            "ret": "current projector hidden=embed_dim",
            "seg": "current decoder hidden_channels=256",
        },
        "tasks": {
            "cls": {
                "small_head_params": int(cls_small),
                "selected_big_head_hidden_dim": int(cls_hidden_lookup[cls_selected["big_params"]]),
                "selected_lora_rank": int(cls_selected["lora_rank"]),
                "big_head_params": int(cls_selected["big_params"]),
                "small_plus_lora_params": int(cls_selected["target_params"]),
                "delta_params": int(cls_selected["delta_params"]),
                "delta_ratio": float(cls_selected["delta_ratio"]),
            },
            "ret": {
                "small_head_params": int(ret_small),
                "selected_big_head_hidden_dim": int(ret_hidden_lookup[ret_selected["big_params"]]),
                "selected_lora_rank": int(ret_selected["lora_rank"]),
                "big_head_params": int(ret_selected["big_params"]),
                "small_plus_lora_params": int(ret_selected["target_params"]),
                "delta_params": int(ret_selected["delta_params"]),
                "delta_ratio": float(ret_selected["delta_ratio"]),
            },
            "seg": {
                "small_head_hidden_channels": int(seg_small_hidden),
                "small_head_params": int(seg_small),
                "selected_big_decoder_hidden_channels": int(seg_hidden_lookup[seg_selected["big_params"]]),
                "selected_lora_rank": int(seg_selected["lora_rank"]),
                "big_head_params": int(seg_selected["big_params"]),
                "small_plus_lora_params": int(seg_selected["target_params"]),
                "delta_params": int(seg_selected["delta_params"]),
                "delta_ratio": float(seg_selected["delta_ratio"]),
            },
        },
        "spec": {
            "embed_dim": int(embed_dim),
            "num_blocks": int(spec.num_blocks),
            "ret_proj_dim": int(args.ret_proj_dim),
            "seg_feature_extractor": args.feature_extractor,
            "seg_in_channels": int(seg_in_channels),
        },
    }

    os.makedirs(os.path.dirname(args.out_config), exist_ok=True)
    with open(args.out_config, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    json_path = os.path.splitext(args.out_config)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("=== Alignment Calibration ===")
    print(f"model={args.model_name}")
    print(f"out={os.path.abspath(args.out_config)}")
    for task in ("cls", "ret", "seg"):
        row = payload["tasks"][task]
        if task == "seg":
            head_size = row["selected_big_decoder_hidden_channels"]
        else:
            head_size = row["selected_big_head_hidden_dim"]
        print(
            f"{task}: big_head_size={head_size} | lora_rank={row['selected_lora_rank']} "
            f"| delta={row['delta_params']} ({row['delta_ratio']:.4f})"
        )


if __name__ == "__main__":
    main()
