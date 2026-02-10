import os
import sys
import torch
import torch.nn as nn
import yaml

from pathlib import Path
from typing import Any, Dict, Tuple

from torchinfo import summary

model_feature_layers = {
    "dinov3_vits16": [3, 5, 7, 11],
    "dinov3_vits16plus": [3, 5, 7, 11],
    "dinov3_vitb16": [3, 5, 7, 11],
    "dinov3_vitl16": [7, 11, 15, 23],
    "dinov3_vith16plus": [9, 13, 18, 26],
    "dinov3_vit7b16": [11, 16, 21, 31],
}


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
        "module.backbone.",
        "backbone.",
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
            "vit_large": {
                "embed_dim": 1024,
                "depth": 24,
                "num_heads": 16,
                "ffn_ratio": 4.0,
            },
            "vit_huge2": {
                "embed_dim": 1280,
                "depth": 32,
                "num_heads": 20,
                "ffn_ratio": 4.0,
            },
            "vit_giant2": {
                "embed_dim": 1536,
                "depth": 40,
                "num_heads": 24,
                "ffn_ratio": 4.0,
            },
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


def _load_backbone_from_checkpoint(repo_dir: str, model_name: str, weights_path: str):
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

    if any(key.endswith("mlp.w1.weight") for key in candidate_state_dict.keys()):
        model_kwargs["ffn_layer"] = "swiglu64"
        for key, value in candidate_state_dict.items():
            if key.endswith("mlp.w1.weight") and torch.is_tensor(value):
                swiglu_hidden = value.shape[0]
                hidden_features = int(swiglu_hidden * 3 / 2)
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
        "Loaded pretrained backbone weights from: "
        f"{weights_path} (source={state_source}, config={config_path or 'none'})"
    )
    return backbone


def load_model(weights: str = None, model_name: str = None, repo_dir: str = None):
    if weights is not None:
        model = _load_backbone_from_checkpoint(
            repo_dir=repo_dir,
            model_name=model_name,
            weights_path=weights,
        )
    else:
        print("No pretrained weights path given. Loading with random weights.")
        model = torch.hub.load(repo_dir, model_name, source="local", pretrained=False)

    return model


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, nc=1):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, nc, kernel_size=1),
        )

    def forward(self, x):
        return self.decode(x)


class Dinov3Segmentation(nn.Module):
    def __init__(
        self,
        fine_tune: bool = False,
        num_classes: int = 2,
        weights: str = None,
        model_name: str = None,
        repo_dir: str = None,
        feature_extractor: str = "last",  # OR 'multi'
    ):
        super(Dinov3Segmentation, self).__init__()

        self.model_name = model_name

        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )
        self.num_classes = num_classes

        if fine_tune:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.backbone_model.named_parameters():
                param.requires_grad = False

        self.feature_extractor_layers = (
            1 if feature_extractor == "last" else model_feature_layers[self.model_name]
        )
        decode_head_in_channels = (
            self.backbone_model.norm.normalized_shape[0]
            if feature_extractor == "last"
            else self.backbone_model.norm.normalized_shape[0] * 4
        )

        self.decode_head = SimpleDecoder(
            in_channels=decode_head_in_channels, nc=self.num_classes
        )

    def forward(self, x):
        # Backbone forward pass
        features = self.backbone_model.get_intermediate_layers(
            x,
            n=self.feature_extractor_layers,
            reshape=True,
            return_class_token=False,
            norm=True,
        )

        # for i, feat in enumerate(features):
        #     print(f"Feature {i}: {feat.shape}")

        concatednated_features = torch.cat(features, dim=1)

        # print('Final feature shape: ', concatednated_features.shape)
        # exit(0)

        # Decoder forward pass
        classifier_out = self.decode_head(concatednated_features)
        return classifier_out


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms
    from src.utils.common import get_dinov3_paths

    import numpy as np
    import os

    DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

    input_size = 640

    transform = transforms.Compose(
        [
            transforms.Resize(
                input_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model_names = {
        "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        # 'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth',
        # 'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        # 'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        # 'dinov3_vith16plus': 'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth',
        # 'dinov3_vit7b16': [11, 16, 21, 31]
    }

    for model_name in model_names:
        print("Testing: ", model_name)
        model = Dinov3Segmentation(
            repo_dir=DINOV3_REPO,
            weights=os.path.join(DINOV3_WEIGHTS, model_names[model_name]),
            model_name=model_name,
            feature_extractor="multi",  # OR 'last'
        )
        model.eval()
        print(model)

        random_image = Image.fromarray(
            np.ones((input_size, input_size, 3), dtype=np.uint8)
        )
        x = transform(random_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(x)

        print(outputs.shape)

        summary(
            model,
            input_data=x,
            col_names=("input_size", "output_size", "num_params"),
            row_settings=["var_names"],
        )
        print("#" * 50, "\n\n")
