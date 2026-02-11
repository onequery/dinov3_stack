"""
Build classification model on top of DINOv3 backbone.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import yaml


def _unwrap_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return {}

    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
        "teacher",
        "student",
        "module",
    ):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            nested = _unwrap_state_dict(value)
            if nested:
                return nested

    for value in checkpoint.values():
        if isinstance(value, dict):
            nested = _unwrap_state_dict(value)
            if nested:
                return nested

    return {}


def _extract_backbone_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], str]:
    prefix_candidates = (
        "module.backbone_model.",
        "backbone_model.",
        "module.backbone.",
        "backbone.",
        "model.backbone_model.",
        "teacher.backbone.",
        "student.backbone.",
        "model.backbone.",
    )
    for prefix in prefix_candidates:
        extracted = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted, f"prefix:{prefix}"

    if any(key.startswith("module.") for key in state_dict.keys()):
        stripped = {
            key[len("module.") :]: value
            for key, value in state_dict.items()
            if key.startswith("module.")
        }
        if stripped:
            return stripped, "prefix:module."

    return state_dict, "raw"


def _find_pretrain_config(weights_path: str) -> str | None:
    path = Path(weights_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    for parent in [path.parent, *path.parents]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return str(candidate)
    return None


def _load_pretrain_config(weights_path: str) -> Tuple[str | None, Dict[str, Any]]:
    config_path = _find_pretrain_config(weights_path)
    if not config_path:
        return None, {}
    with open(config_path, "r") as file:
        config = yaml.safe_load(file) or {}
    return config_path, config if isinstance(config, dict) else {}


def _build_model_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    student = cfg.get("student", {}) if isinstance(cfg, dict) else {}
    if not isinstance(student, dict):
        return {}

    keys = (
        "img_size",
        "patch_size",
        "drop_path_rate",
        "ffn_layer",
        "ffn_ratio",
        "qkv_bias",
        "proj_bias",
        "ffn_bias",
        "norm_layer",
        "layerscale_init",
        "n_storage_tokens",
        "mask_k_bias",
        "untie_cls_and_patch_norms",
        "untie_global_and_local_cls_norm",
        "pos_embed_rope_base",
        "pos_embed_rope_min_period",
        "pos_embed_rope_max_period",
        "pos_embed_rope_normalize_coords",
        "pos_embed_rope_shift_coords",
        "pos_embed_rope_jitter_coords",
        "pos_embed_rope_rescale_coords",
        "pos_embed_rope_dtype",
        "in_chans",
    )

    model_kwargs: Dict[str, Any] = {}
    crops = cfg.get("crops", {}) if isinstance(cfg, dict) else {}
    global_crops = crops.get("global_crops_size") if isinstance(crops, dict) else None
    if isinstance(global_crops, list) and global_crops:
        model_kwargs["img_size"] = max(global_crops)
    elif isinstance(global_crops, int):
        model_kwargs["img_size"] = global_crops

    if "patch_size" in student and student["patch_size"] is not None:
        model_kwargs["patch_size"] = student["patch_size"]
    if "drop_path_rate" in student and student["drop_path_rate"] is not None:
        model_kwargs["drop_path_rate"] = student["drop_path_rate"]
    if "layerscale" in student and student["layerscale"] is not None:
        model_kwargs["layerscale_init"] = student["layerscale"]

    for key in keys:
        if key in student and student[key] is not None:
            model_kwargs[key] = student[key]
    return model_kwargs


def _build_backbone(
    repo_dir: str,
    model_name: str,
    arch: str | None,
    model_kwargs: Dict[str, Any],
):
    if arch and isinstance(arch, str):
        repo_path = Path(repo_dir).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from dinov3.models import convnext as convnext_models
        from dinov3.models.vision_transformer import DinoVisionTransformer

        vit_arch = {
            "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "ffn_ratio": 4.0},
            "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "ffn_ratio": 4.0},
            "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "ffn_ratio": 4.0},
            "vit_huge2": {"embed_dim": 1280, "depth": 32, "num_heads": 20, "ffn_ratio": 4.0},
            "vit_giant2": {"embed_dim": 1536, "depth": 40, "num_heads": 24, "ffn_ratio": 4.0},
            "vit_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "ffn_ratio": 3.0},
        }

        if arch in vit_arch:
            arch_cfg = vit_arch[arch]
            img_size = model_kwargs.pop("img_size", 224)
            patch_size = model_kwargs.pop("patch_size", 16)
            ffn_ratio = model_kwargs.pop("ffn_ratio", arch_cfg["ffn_ratio"])
            return DinoVisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=arch_cfg["embed_dim"],
                depth=arch_cfg["depth"],
                num_heads=arch_cfg["num_heads"],
                ffn_ratio=ffn_ratio,
                **model_kwargs,
            )

        if "convnext" in arch:
            convnext_cls = convnext_models.get_convnext_arch(arch)
            return convnext_cls(**model_kwargs)

    return torch.hub.load(repo_dir, model_name, source="local", pretrained=False)


def _load_backbone_from_checkpoint(
    repo_dir: str,
    model_name: str,
    weights_path: str,
):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Local weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    raw_state_dict = _unwrap_state_dict(checkpoint)
    if not raw_state_dict:
        raise ValueError(
            f"Could not parse a valid state_dict from checkpoint: {weights_path}"
        )

    candidate_state_dict, state_source = _extract_backbone_state_dict(raw_state_dict)

    config_path, cfg = _load_pretrain_config(weights_path)
    model_kwargs = _build_model_kwargs_from_cfg(cfg) if cfg else {}
    student_cfg = cfg.get("student", {}) if isinstance(cfg, dict) else {}
    arch = student_cfg.get("arch") if isinstance(student_cfg, dict) else None
    if not arch:
        model_name_to_arch = {
            "dinov3_vits16": "vit_small",
            "dinov3_vits16plus": "vit_small",
            "dinov3_vitb16": "vit_base",
            "dinov3_vitl16": "vit_large",
            "dinov3_vit7b16": "vit_7b",
        }
        arch = model_name_to_arch.get(model_name)

    embed_dim_from_arch = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
        "vit_huge2": 1280,
        "vit_giant2": 1536,
        "vit_7b": 4096,
    }
    embed_dim = embed_dim_from_arch.get(arch, 384)

    if not cfg and isinstance(arch, str) and arch.startswith("vit_"):
        model_kwargs.setdefault("norm_layer", "layernormbf16")
        model_kwargs.setdefault("pos_embed_rope_base", 100)
        model_kwargs.setdefault("pos_embed_rope_normalize_coords", "separate")
        model_kwargs.setdefault("pos_embed_rope_rescale_coords", 2)
        model_kwargs.setdefault("pos_embed_rope_dtype", "fp32")

    model_kwargs["qkv_bias"] = any(
        key.endswith("attn.qkv.bias") for key in candidate_state_dict.keys()
    )

    if "storage_tokens" in candidate_state_dict and torch.is_tensor(
        candidate_state_dict["storage_tokens"]
    ):
        storage_tokens = candidate_state_dict["storage_tokens"]
        if storage_tokens.ndim == 3:
            model_kwargs["n_storage_tokens"] = int(storage_tokens.shape[1])

    if any(key.endswith("attn.qkv.bias_mask") for key in candidate_state_dict.keys()):
        model_kwargs["mask_k_bias"] = True

    if any(
        key.endswith(".ls1.gamma") or key.endswith(".ls2.gamma")
        for key in candidate_state_dict.keys()
    ):
        model_kwargs["layerscale_init"] = 1.0e-5

    if any(key.endswith("local_cls_norm.weight") for key in candidate_state_dict.keys()):
        model_kwargs["untie_global_and_local_cls_norm"] = True

    if any(
        key.endswith("cls_norm.weight") and "local_cls_norm" not in key
        for key in candidate_state_dict.keys()
    ):
        model_kwargs["untie_cls_and_patch_norms"] = True

    has_swiglu = any(
        key.endswith("mlp.w1.weight") or key.endswith("mlp.w12.weight")
        for key in candidate_state_dict.keys()
    )
    if has_swiglu:
        model_kwargs["ffn_layer"] = "swiglu64"
        for key, value in candidate_state_dict.items():
            if not torch.is_tensor(value):
                continue
            if key.endswith("mlp.w1.weight"):
                swiglu_hidden = value.shape[0]
                hidden_features = int(swiglu_hidden * 3 / 2)
                model_kwargs["ffn_ratio"] = hidden_features / float(embed_dim)
                break
            if key.endswith("mlp.w12.weight"):
                hidden_features = int(value.shape[0] // 2)
                model_kwargs["ffn_ratio"] = hidden_features / float(embed_dim)
                break
    elif any(key.endswith("mlp.fc1.weight") for key in candidate_state_dict.keys()):
        model_kwargs["ffn_layer"] = "mlp"
        for key, value in candidate_state_dict.items():
            if key.endswith("mlp.fc1.weight") and torch.is_tensor(value):
                model_kwargs["ffn_ratio"] = value.shape[0] / float(embed_dim)
                break

    backbone = _build_backbone(
        repo_dir=repo_dir,
        model_name=model_name,
        arch=arch,
        model_kwargs=model_kwargs,
    )
    model_state = backbone.state_dict()

    loadable_state_dict = {}
    skipped_not_found = 0
    skipped_shape = 0
    skipped_not_found_keys = []
    skipped_shape_keys = []
    for key, value in candidate_state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key not in model_state:
            skipped_not_found += 1
            if len(skipped_not_found_keys) < 10:
                skipped_not_found_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape += 1
            if len(skipped_shape_keys) < 10:
                skipped_shape_keys.append(
                    f"{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}"
                )
            continue
        loadable_state_dict[key] = value

    if not loadable_state_dict:
        raise ValueError(
            "No compatible backbone weights were found in the checkpoint. "
            f"model={model_name}, file={weights_path}, source={state_source}"
        )

    if skipped_not_found > 0 or skipped_shape > 0:
        raise ValueError(
            "Backbone checkpoint does not fully match the constructed architecture. "
            f"model={model_name}, file={weights_path}, source={state_source}, "
            f"skipped_key={skipped_not_found}, skipped_shape={skipped_shape}, "
            f"sample_missing={skipped_not_found_keys}, "
            f"sample_shape_mismatch={skipped_shape_keys}"
        )

    strict_load = set(model_state.keys()).issubset(set(loadable_state_dict.keys()))
    incompatible = backbone.load_state_dict(loadable_state_dict, strict=strict_load)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Backbone load resulted in incompatible keys. "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    print(
        "Backbone load | "
        f"source={state_source} config={config_path or 'none'} "
        f"builder={arch or model_name} strict={strict_load} "
        f"loaded={len(loadable_state_dict)}/{len(model_state)} "
        "missing=0 unexpected=0"
    )
    return backbone


def load_model(weights: str | None = None, model_name: str | None = None, repo_dir: str | None = None):
    if weights is None:
        print("No pretrained weights path given. Loading with random weights.")
        return torch.hub.load(repo_dir, model_name, source="local", pretrained=False)

    weights_path = Path(weights).expanduser()
    if weights_path.exists():
        return _load_backbone_from_checkpoint(
            repo_dir=repo_dir,
            model_name=model_name,
            weights_path=str(weights_path),
        )

    print("Loading DINOv3 hub weights spec: ", weights)
    return torch.hub.load(repo_dir, model_name, source="local", weights=weights)


class Dinov3Classification(nn.Module):
    def __init__(
        self,
        fine_tune: bool = False,
        num_classes: int = 2,
        weights: str | None = None,
        model_name: str | None = None,
        repo_dir: str | None = None,
    ):
        super(Dinov3Classification, self).__init__()

        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )

        self.head = nn.Linear(
            in_features=self.backbone_model.norm.normalized_shape[0],
            out_features=num_classes,
            bias=True,
        )

        if not fine_tune:
            for params in self.backbone_model.parameters():
                params.requires_grad = False

    def forward(self, x):
        features = self.backbone_model(x)
        classifier_out = self.head(features)
        return classifier_out


class Dinov3Backbone(nn.Module):
    def __init__(
        self,
        weights: str | None = None,
        model_name: str | None = None,
        repo_dir: str | None = None,
    ):
        super(Dinov3Backbone, self).__init__()
        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )

    def forward(self, x):
        return self.backbone_model(x)
