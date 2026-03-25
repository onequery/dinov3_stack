from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2

from border_suppression_common import (
    DEFAULT_BLUR_SIGMA,
    DEFAULT_CENTER_X,
    DEFAULT_CENTER_Y,
    DEFAULT_FEATHER_WIDTH,
    DEFAULT_RADIUS,
    apply_border_suppression,
    make_three_panel,
    overlay_mask_boundary,
)


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
    blur_sigma: float,
    feather_width: int,
    repo_root: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        suppressed, mask, background, alpha = apply_border_suppression(
            image=image,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            blur_sigma=blur_sigma,
            feather_width=feather_width,
        )
        overlay = overlay_mask_boundary(image, mask)
        panel = make_three_panel(image, suppressed, overlay)

        stem = image_path.stem
        out_suppressed = output_dir / f"{stem}_suppressed.png"
        out_overlay = output_dir / f"{stem}_overlay.png"
        out_panel = output_dir / f"{stem}_panel.png"
        out_mask = output_dir / f"{stem}_mask.png"
        out_background = output_dir / f"{stem}_background.png"
        out_alpha = output_dir / f"{stem}_alpha.png"

        cv2.imwrite(str(out_suppressed), suppressed)
        cv2.imwrite(str(out_overlay), overlay)
        cv2.imwrite(str(out_panel), panel)
        cv2.imwrite(str(out_mask), mask.astype("uint8") * 255)
        cv2.imwrite(str(out_background), background)
        cv2.imwrite(str(out_alpha), alpha)

        rows.append(
            ",".join(
                [
                    relative_name(image_path, repo_root),
                    str(image.shape[0]),
                    str(image.shape[1]),
                    str(center_x),
                    str(center_y),
                    str(radius),
                    f"{blur_sigma:.2f}",
                    str(feather_width),
                    str(int(mask.sum())),
                    str(int((~mask).sum())),
                    relative_name(out_panel, repo_root),
                ]
            )
        )

    summary_path = output_dir / "preview_summary.csv"
    summary_path.write_text(
        "image_path,height,width,center_x,center_y,radius,blur_sigma,feather_width,valid_pixels,invalid_pixels,panel_path\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview fixed circular border suppression on example PNGs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-x", type=int, default=DEFAULT_CENTER_X)
    parser.add_argument("--center-y", type=int, default=DEFAULT_CENTER_Y)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    parser.add_argument("--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA)
    parser.add_argument("--feather-width", type=int, default=DEFAULT_FEATHER_WIDTH)
    parser.add_argument("images", nargs="+", type=Path)
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
        blur_sigma=args.blur_sigma,
        feather_width=args.feather_width,
        repo_root=repo_root,
    )
    print(f"Wrote preview outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
