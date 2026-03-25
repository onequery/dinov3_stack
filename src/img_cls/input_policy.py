from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import torchvision.transforms as transforms

IMAGENET_RGB_MEAN = (0.485, 0.456, 0.406)
IMAGENET_RGB_STD = (0.229, 0.224, 0.225)
INPUT_POLICY_BASELINE = "baseline_rgbtriplet"
INPUT_POLICY_CAG_STATS = "input_v1_cag_stats_normalization"
INPUT_POLICY_CHOICES = [INPUT_POLICY_BASELINE, INPUT_POLICY_CAG_STATS]


@dataclass(frozen=True)
class InputPolicySpec:
    name: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    stats_json: str | None
    stats_source: str

    def to_meta(self) -> Dict[str, Any]:
        return {
            "input_policy": self.name,
            "input_stats_json": self.stats_json,
            "input_norm_mean": [float(v) for v in self.mean],
            "input_norm_std": [float(v) for v in self.std],
            "input_stats_source": self.stats_source,
        }


def _validate_triplet(values: Sequence[float], field_name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{field_name} must contain exactly 3 values, got {len(values)}")
    triplet = tuple(float(v) for v in values)
    if any(not (v == v) for v in triplet):
        raise ValueError(f"{field_name} contains NaN: {triplet}")
    return triplet


def eval_geometry_signature(resize_size: int, center_crop_size: int) -> Dict[str, int]:
    return {
        "resize_size": int(resize_size),
        "center_crop_size": int(center_crop_size),
    }


def build_pre_normalize_eval_transform(resize_size: int, center_crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop((center_crop_size, center_crop_size)),
            transforms.ToTensor(),
        ]
    )


def build_eval_transform(
    resize_size: int,
    center_crop_size: int,
    policy_spec: InputPolicySpec,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop((center_crop_size, center_crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(policy_spec.mean),
                std=list(policy_spec.std),
            ),
        ]
    )


def resolve_input_policy(policy_name: str, stats_json: str | None = None) -> InputPolicySpec:
    if policy_name == INPUT_POLICY_BASELINE:
        return InputPolicySpec(
            name=INPUT_POLICY_BASELINE,
            mean=IMAGENET_RGB_MEAN,
            std=IMAGENET_RGB_STD,
            stats_json=None,
            stats_source="imagenet_builtin",
        )

    if policy_name != INPUT_POLICY_CAG_STATS:
        raise ValueError(f"Unsupported input policy: {policy_name}")

    if not stats_json:
        raise ValueError("--input-stats-json is required for input_v1_cag_stats_normalization")
    stats_path = Path(stats_json).expanduser().resolve()
    if not stats_path.exists():
        raise FileNotFoundError(f"Input stats JSON not found: {stats_path}")
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    mean = _validate_triplet(payload.get("mean_rgb", []), "mean_rgb")
    std = _validate_triplet(payload.get("std_rgb", []), "std_rgb")
    if any(v <= 0 for v in std):
        raise ValueError(f"std_rgb must be positive, got {std}")
    return InputPolicySpec(
        name=INPUT_POLICY_CAG_STATS,
        mean=mean,
        std=std,
        stats_json=str(stats_path),
        stats_source="json",
    )
