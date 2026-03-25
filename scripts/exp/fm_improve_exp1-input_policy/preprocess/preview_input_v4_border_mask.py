from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# Prototype fixed valid-ROI definition for the observed circular FOV.
# These parameters are easy to adjust after visual inspection.
DEFAULT_CENTER_X = 256
DEFAULT_CENTER_Y = 256
DEFAULT_RADIUS = 332
DEFAULT_FILL_VALUE = 0


def build_valid_roi_mask(height: int, width: int, center_x: int, center_y: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2


def apply_border_suppression(image: np.ndarray, mask: np.ndarray, fill_value: int) -> np.ndarray:
    out = image.copy()
    out[~mask] = np.uint8(fill_value)
    return out


def overlay_mask_boundary(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, (0, 255, 0), 2)
    outside = ~mask
    rgb[outside] = (rgb[outside] * 0.35).astype(np.uint8)
    return rgb


def make_panel(original: np.ndarray, suppressed: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    left = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    mid = cv2.cvtColor(suppressed, cv2.COLOR_GRAY2BGR)
    label_h = 36
    gap = 8
    width = original.shape[1]
    panel_h = original.shape[0] + label_h
    panel = np.full((panel_h, width * 3 + gap * 2, 3), 255, dtype=np.uint8)
    panel[label_h:, :width] = left
    panel[label_h:, width + gap: width * 2 + gap] = mid
    panel[label_h:, width * 2 + gap * 2:] = overlay
    cv2.putText(panel, 'Original', (16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Suppressed', (width + gap + 16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(panel, 'Mask Overlay', (width * 2 + gap * 2 + 16, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return panel


def relative_name(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def process_images(
    image_paths: Iterable[Path],
    output_dir: Path,
    center_x: int,
    center_y: int,
    radius: int,
    fill_value: int,
    repo_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        h, w = image.shape
        mask = build_valid_roi_mask(h, w, center_x=center_x, center_y=center_y, radius=radius)
        suppressed = apply_border_suppression(image, mask, fill_value=fill_value)
        overlay = overlay_mask_boundary(image, mask)
        panel = make_panel(image, suppressed, overlay)

        stem = image_path.stem
        out_suppressed = output_dir / f'{stem}_suppressed.png'
        out_overlay = output_dir / f'{stem}_overlay.png'
        out_panel = output_dir / f'{stem}_panel.png'
        out_mask = output_dir / f'{stem}_mask.png'

        cv2.imwrite(str(out_suppressed), suppressed)
        cv2.imwrite(str(out_overlay), overlay)
        cv2.imwrite(str(out_panel), panel)
        cv2.imwrite(str(out_mask), mask.astype(np.uint8) * 255)

        rows.append(
            ','.join(
                [
                    relative_name(image_path, repo_root),
                    str(h),
                    str(w),
                    str(center_x),
                    str(center_y),
                    str(radius),
                    str(int(mask.sum())),
                    str(int((~mask).sum())),
                    relative_name(out_panel, repo_root),
                ]
            )
        )

    summary_path = output_dir / 'preview_summary.csv'
    summary_path.write_text(
        'image_path,height,width,center_x,center_y,radius,valid_pixels,invalid_pixels,panel_path\n' + '\n'.join(rows) + '\n',
        encoding='utf-8',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Preview fixed circular border suppression on example PNGs.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--center-x', type=int, default=DEFAULT_CENTER_X)
    parser.add_argument('--center-y', type=int, default=DEFAULT_CENTER_Y)
    parser.add_argument('--radius', type=int, default=DEFAULT_RADIUS)
    parser.add_argument('--fill-value', type=int, default=DEFAULT_FILL_VALUE)
    parser.add_argument('images', nargs='+', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    process_images(
        image_paths=args.images,
        output_dir=args.output_dir,
        center_x=args.center_x,
        center_y=args.center_y,
        radius=args.radius,
        fill_value=args.fill_value,
        repo_root=repo_root,
    )
    print(f'Wrote preview outputs to {args.output_dir}')


if __name__ == '__main__':
    main()
