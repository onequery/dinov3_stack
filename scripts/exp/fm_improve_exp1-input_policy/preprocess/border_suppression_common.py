from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

DEFAULT_CENTER_X = 256
DEFAULT_CENTER_Y = 256
DEFAULT_RADIUS = 332
DEFAULT_BLUR_SIGMA = 25.0
DEFAULT_FEATHER_WIDTH = 18
TARGET_MODEL_NAME_NORMALIZED = "P H I L I P S INTEGRIS H"


def normalize_manufacturer_model_name(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).upper().split())


def is_target_model_name(value: object) -> bool:
    return normalize_manufacturer_model_name(value) == TARGET_MODEL_NAME_NORMALIZED


def build_valid_roi_mask(
    height: int,
    width: int,
    center_x: int = DEFAULT_CENTER_X,
    center_y: int = DEFAULT_CENTER_Y,
    radius: int = DEFAULT_RADIUS,
) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2


def estimate_background_field(image: np.ndarray, mask: np.ndarray, blur_sigma: float = DEFAULT_BLUR_SIGMA) -> np.ndarray:
    image_f = image.astype(np.float32)
    mask_f = mask.astype(np.float32)
    weighted = image_f * mask_f
    blurred_weighted = cv2.GaussianBlur(weighted, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    blurred_mask = cv2.GaussianBlur(mask_f, ksize=(0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    background = blurred_weighted / np.maximum(blurred_mask, 1e-6)
    return np.clip(background, 0, 255)


def build_feather_alpha(mask: np.ndarray, feather_width: int = DEFAULT_FEATHER_WIDTH) -> np.ndarray:
    dist_inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(dist_inside / max(float(feather_width), 1.0), 0.0, 1.0)
    return alpha.astype(np.float32)


def apply_border_suppression(
    image: np.ndarray,
    *,
    center_x: int = DEFAULT_CENTER_X,
    center_y: int = DEFAULT_CENTER_Y,
    radius: int = DEFAULT_RADIUS,
    blur_sigma: float = DEFAULT_BLUR_SIGMA,
    feather_width: int = DEFAULT_FEATHER_WIDTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = build_valid_roi_mask(
        height=image.shape[0],
        width=image.shape[1],
        center_x=center_x,
        center_y=center_y,
        radius=radius,
    )
    background = estimate_background_field(image=image, mask=mask, blur_sigma=blur_sigma)
    alpha = build_feather_alpha(mask=mask, feather_width=feather_width)
    image_f = image.astype(np.float32)
    blended = alpha * image_f + (1.0 - alpha) * background
    suppressed = np.clip(blended, 0, 255).astype(np.uint8)
    return suppressed, mask, background.astype(np.uint8), (alpha * 255.0).astype(np.uint8)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def overlay_mask_boundary(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = _to_bgr(image)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    outside = ~mask
    rgb[outside] = (rgb[outside] * 0.35).astype(np.uint8)
    return rgb


def make_three_panel(
    left: np.ndarray,
    center: np.ndarray,
    right: np.ndarray,
    labels: Sequence[str] = ("Original", "Suppressed", "Mask Overlay"),
) -> np.ndarray:
    left_rgb = _to_bgr(left)
    center_rgb = _to_bgr(center)
    right_rgb = _to_bgr(right)
    label_h = 36
    gap = 8
    width = left_rgb.shape[1]
    panel_h = left_rgb.shape[0] + label_h
    panel = np.full((panel_h, width * 3 + gap * 2, 3), 255, dtype=np.uint8)
    panel[label_h:, :width] = left_rgb
    panel[label_h:, width + gap : width * 2 + gap] = center_rgb
    panel[label_h:, width * 2 + gap * 2 :] = right_rgb
    for idx, label in enumerate(labels[:3]):
        x = idx * (width + gap) + 16
        cv2.putText(panel, str(label), (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def safe_name_from_rel_path(parts: Iterable[str]) -> str:
    return "__".join(str(part).replace("/", "_") for part in parts)
