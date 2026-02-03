"""
Building a linear classifier on top of DINOv3 backbone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _extract_backbone_state_dict(checkpoint: Any) -> dict[str, Any]:
    """
    Normalize various checkpoint formats into a DINOv3 backbone state_dict.

    Supported formats:
    - Plain backbone state_dict (keys like "blocks.0.*", "cls_token", ...)
    - DINOv3 eval checkpoint (train.py) saved as {"teacher": <ModuleDict state_dict>}
      where the teacher ModuleDict uses "backbone.*" prefix for the backbone weights.
    """
    state_dict = checkpoint
    if isinstance(state_dict, dict):
        for key in ("teacher", "state_dict", "model_state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict)}")

    # Remove DistributedDataParallel prefix.
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {
            (k.removeprefix("module.") if isinstance(k, str) else k): v
            for k, v in state_dict.items()
        }

    # DINOv3 SSLMetaArch teacher/student checkpoints typically store backbone under "backbone.*".
    backbone_items = {k: v for k, v in state_dict.items() if isinstance(k, str) and k.startswith("backbone.")}
    if backbone_items:
        return {k.removeprefix("backbone."): v for k, v in backbone_items.items()}

    # Already a backbone state_dict.
    return state_dict


def load_model(weights: str | None = None, model_name: str | None = None, repo_dir: str | None = None):
    if weights is None:
        print("No pretrained weights path given. Loading with random weights.")
        return torch.hub.load(repo_dir, model_name, source="local", pretrained=False)

    weights_path = Path(weights).expanduser()
    if weights_path.exists():
        print("Loading pretrained backbone weights from: ", str(weights_path))
        model = torch.hub.load(repo_dir, model_name, source="local", pretrained=False)
        checkpoint = torch.load(str(weights_path), map_location="cpu")
        backbone_state_dict = _extract_backbone_state_dict(checkpoint)

        # Filter to model keys (protects against extra keys like dino_head/ibot_head).
        model_keys = set(model.state_dict().keys())
        backbone_state_dict = {k: v for k, v in backbone_state_dict.items() if k in model_keys}
        msg = model.load_state_dict(backbone_state_dict, strict=False)
        print(f"Backbone weights loaded (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}).")
        if msg.missing_keys:
            print("Missing keys (first 20):", msg.missing_keys[:20])
        if msg.unexpected_keys:
            print("Unexpected keys (first 20):", msg.unexpected_keys[:20])
        return model

    # Not a file on disk: assume it's a DINOv3 hub weights identifier or URL.
    print("Loading DINOv3 hub weights spec: ", weights)
    return torch.hub.load(repo_dir, model_name, source="local", weights=weights)


# def build_model(
#     num_classes: int=10,
#     fine_tune: bool=False,
#     weights: str=None,
#     model_name: str=None,
#     repo_dir: str=None
# ):
#     backbone_model = load_model(
#         weights=weights, model_name=model_name, repo_dir=repo_dir
#     )

#     model = torch.nn.Sequential(OrderedDict([
#         ('backbone', backbone_model),
#         ('head', torch.nn.Linear(
#             in_features=backbone_model.norm.normalized_shape[0],
#             out_features=num_classes,
#             bias=True
#         ))
#     ]))

#     if not fine_tune:
#         for params in model.backbone.parameters():
#             params.requires_grad = False

#     return model


class Dinov3Classification(nn.Module):
    def __init__(
        self,
        fine_tune: bool = False,
        num_classes: int = 2,
        weights: str = None,
        model_name: str = None,
        repo_dir: str = None,
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
        # Through classifier head.
        classifier_out = self.head(features)
        return classifier_out


class Dinov3Backbone(nn.Module):
    def __init__(
        self,
        weights: str = None,
        model_name: str = None,
        repo_dir: str = None,
    ):
        super(Dinov3Backbone, self).__init__()

        self.backbone_model = load_model(
            weights=weights, model_name=model_name, repo_dir=repo_dir
        )

    def forward(self, x):
        features = self.backbone_model(x)
        return features


if __name__ == "__main__":
    import os

    import numpy as np
    from PIL import Image
    from torchinfo import summary
    from torchvision import transforms

    from src.utils.common import get_dinov3_paths

    DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()

    weight_file = "dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth"
    model_name = "dinov3_convnext_tiny"

    sample_size = 224

    # Define image transformation
    transform = transforms.Compose(
        [
            transforms.Resize(
                sample_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(sample_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Loading the pretrained model without classification head.
    model = load_model(
        repo_dir=DINOV3_REPO,
        weights=os.path.join(DINOV3_WEIGHTS, weight_file),
        model_name=model_name,
    )

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.")

    # Testing forward pass.
    pil_image = Image.fromarray(np.ones((sample_size, sample_size, 3), dtype=np.uint8))
    model_input = transform(pil_image).unsqueeze(0)

    # summary(
    #     model,
    #     input_data=model_input,
    #     col_names=('input_size', 'output_size', 'num_params'),
    #     row_settings=['var_names']
    # )

    # Manual torch forward pass.
    with torch.no_grad():
        features = model.forward_features(model_input)
        patch_features = features["x_norm_patchtokens"]

    print(features.keys())
    print(f"Patch features shape: {patch_features.shape}")

    # Check the forward passes through the complete model.
    # To check what gets fed to the classification layer.
    print(model)

    model_cls = Dinov3Classification(
        repo_dir=DINOV3_REPO,
        weights=os.path.join(DINOV3_WEIGHTS, weight_file),
        model_name=model_name,
    )

    summary(
        model_cls,
        input_data=model_input,
        col_names=("input_size", "output_size", "num_params"),
        row_settings=["var_names"],
    )

    features = model_cls.backbone_model(model_input)
    print(f"Shape of features getting fed to classification layer: {features.shape}")
