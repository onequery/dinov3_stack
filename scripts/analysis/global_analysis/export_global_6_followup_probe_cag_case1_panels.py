#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CASE_CSV = (
    REPO_ROOT
    / "outputs/analysis2_rep_analysis/global_6_patient_retrieval_border_suppressed_philips/reports"
    / "followup_probe_cag_r1_improved_ap_worsened.csv"
)
DEFAULT_BASELINE_ROOT = (
    REPO_ROOT
    / "input/global_analysis_6_patient_retrieval_border_suppressed_philips"
    / "baseline_philips_unique_view_subset/images"
)
DEFAULT_VARIANT_ROOT = (
    REPO_ROOT
    / "input/global_analysis_6_patient_retrieval_border_suppressed_philips"
    / "border_suppressed_philips_unique_view_subset/images"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/analysis2_rep_analysis/global_6_patient_retrieval_border_suppressed_philips/reports"
    / "followup_probe_cag_r1_improved_ap_worsened_panels"
)

ROW_IMAGE_WIDTH = 320
ROW_IMAGE_HEIGHT = 320
LEFT_MARGIN = 24
RIGHT_MARGIN = 24
TOP_MARGIN = 28
BOTTOM_MARGIN = 24
ROW_GAP = 18
COLUMN_GAP = 18
BG_COLOR = (250, 250, 250)
TEXT_COLOR = (20, 20, 20)
SUBTEXT_COLOR = (90, 90, 90)
ACCENT_COLOR = (40, 120, 180)
ROW_TOP_PADDING = 14
ROW_BOTTOM_PADDING = 16
ROW_TEXT_GAP = 8
PATH_LINE_GAP = 18
BODY_LINE_GAP = 18
IMAGE_LABEL_GAP = 10


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-csv", type=Path, default=DEFAULT_CASE_CSV)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--variant-root", type=Path, default=DEFAULT_VARIANT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_image(path: Path) -> Image.Image:
    image = Image.open(path).convert("L")
    image = image.resize((ROW_IMAGE_WIDTH, ROW_IMAGE_HEIGHT), Image.Resampling.BILINEAR)
    return image.convert("RGB")


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
    draw.text(xy, text, font=font, fill=fill)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    return float(draw.textlength(text, font=font))


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    lines: list[str] = []
    current = ""
    last_break = -1
    break_chars = set("/_| -")
    for ch in text:
        trial = current + ch
        if text_width(draw, trial, font) <= max_width:
            current = trial
            if ch in break_chars:
                last_break = len(current) - 1
            continue
        if current:
            if last_break >= 0:
                line = current[: last_break + 1].rstrip()
                remainder = current[last_break + 1 :].lstrip() + ch
                lines.append(line)
                current = remainder
            else:
                lines.append(current.rstrip())
                current = ch
        else:
            lines.append(ch)
            current = ""
        last_break = -1
        for idx, existing in enumerate(current):
            if existing in break_chars:
                last_break = idx
    if current:
        lines.append(current.rstrip())
    return lines


def build_row(row: pd.Series, baseline_root: Path, variant_root: Path, font_main: ImageFont.ImageFont, font_sub: ImageFont.ImageFont) -> Image.Image:
    baseline_img = read_image(baseline_root / row["query_rel_path"])
    variant_img = read_image(variant_root / row["query_rel_path"])

    row_width = LEFT_MARGIN + ROW_IMAGE_WIDTH + COLUMN_GAP + ROW_IMAGE_WIDTH + RIGHT_MARGIN
    measure = Image.new("RGB", (row_width, 10), BG_COLOR)
    measure_draw = ImageDraw.Draw(measure)
    content_width = row_width - LEFT_MARGIN - RIGHT_MARGIN

    file_name = str(row["query_rel_path"])
    metric_line = (
        f"num_pos={int(row['num_positives'])} | "
        f"delta_AP={float(row['delta_AP']):+.4f} | "
        f"delta_R@1={float(row['delta_R_at_1']):+.3f} | "
        f"delta_rank={float(row['delta_first_positive_rank']):+.3f}"
    )
    baseline_line = (
        f"baseline: AP={float(row['baseline_philips_AP']):.4f}, "
        f"R@1={float(row['baseline_philips_R_at_1']):.3f}, "
        f"first_pos_rank={float(row['baseline_philips_first_positive_rank']):.3f}"
    )
    variant_line = (
        f"suppressed: AP={float(row['border_suppressed_philips_AP']):.4f}, "
        f"R@1={float(row['border_suppressed_philips_R_at_1']):.3f}, "
        f"first_pos_rank={float(row['border_suppressed_philips_first_positive_rank']):.3f}"
    )
    path_lines = wrap_text(measure_draw, file_name, font_sub, content_width)
    metric_lines = wrap_text(measure_draw, metric_line, font_sub, content_width)
    baseline_lines = wrap_text(measure_draw, baseline_line, font_sub, content_width)
    variant_lines = wrap_text(measure_draw, variant_line, font_sub, content_width)

    top_text_height = len(path_lines) * PATH_LINE_GAP
    labels_height = BODY_LINE_GAP
    bottom_text_height = (
        len(metric_lines) * BODY_LINE_GAP
        + len(baseline_lines) * BODY_LINE_GAP
        + len(variant_lines) * BODY_LINE_GAP
        + 2 * ROW_TEXT_GAP
    )
    row_height = (
        ROW_TOP_PADDING
        + top_text_height
        + ROW_TEXT_GAP
        + labels_height
        + IMAGE_LABEL_GAP
        + ROW_IMAGE_HEIGHT
        + ROW_TEXT_GAP
        + bottom_text_height
        + ROW_BOTTOM_PADDING
    )
    canvas = Image.new("RGB", (row_width, row_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    y = ROW_TOP_PADDING
    for line in path_lines:
        draw_text(draw, (LEFT_MARGIN, y), line, font=font_sub, fill=TEXT_COLOR)
        y += PATH_LINE_GAP
    y += ROW_TEXT_GAP

    x0 = LEFT_MARGIN
    x1 = LEFT_MARGIN + ROW_IMAGE_WIDTH + COLUMN_GAP
    draw_text(draw, (x0, y), "Baseline PHILIPS", font=font_sub, fill=SUBTEXT_COLOR)
    draw_text(draw, (x1, y), "Border-suppressed PHILIPS", font=font_sub, fill=SUBTEXT_COLOR)
    y += labels_height + IMAGE_LABEL_GAP
    y0 = y
    canvas.paste(baseline_img, (x0, y0))
    canvas.paste(variant_img, (x1, y0))

    y = y0 + ROW_IMAGE_HEIGHT + ROW_TEXT_GAP
    for line in metric_lines:
        draw_text(draw, (LEFT_MARGIN, y), line, font=font_sub, fill=ACCENT_COLOR)
        y += BODY_LINE_GAP
    y += ROW_TEXT_GAP
    for line in baseline_lines:
        draw_text(draw, (LEFT_MARGIN, y), line, font=font_sub, fill=SUBTEXT_COLOR)
        y += BODY_LINE_GAP
    for line in variant_lines:
        draw_text(draw, (LEFT_MARGIN, y), line, font=font_sub, fill=SUBTEXT_COLOR)
        y += BODY_LINE_GAP
    return canvas


def build_patient_panel(group: pd.DataFrame, baseline_root: Path, variant_root: Path) -> Image.Image:
    font_title = load_font(24)
    font_main = load_font(18)
    font_sub = load_font(15)

    patient_id = str(group["query_patient_id"].iloc[0])
    rows = [build_row(row, baseline_root, variant_root, font_main, font_sub) for _, row in group.iterrows()]
    body_width = max(row.width for row in rows)
    body_height = sum(row.height for row in rows) + ROW_GAP * (len(rows) - 1)

    panel_height = TOP_MARGIN + 72 + body_height + BOTTOM_MARGIN
    panel = Image.new("RGB", (body_width, panel_height), BG_COLOR)
    draw = ImageDraw.Draw(panel)
    draw_text(draw, (LEFT_MARGIN, TOP_MARGIN), f"GA6 follow-up | probe CAG | patient {patient_id}", font=font_title, fill=TEXT_COLOR)
    draw_text(draw, (LEFT_MARGIN, TOP_MARGIN + 30), f"queries in this panel: {len(group)}", font=font_main, fill=SUBTEXT_COLOR)
    draw_text(draw, (LEFT_MARGIN, TOP_MARGIN + 52), "Condition: delta_R@1 > 0 and delta_AP < 0", font=font_main, fill=ACCENT_COLOR)

    y = TOP_MARGIN + 72
    for idx, row_panel in enumerate(rows):
        panel.paste(row_panel, (0, y))
        y += row_panel.height
        if idx < len(rows) - 1:
            y += ROW_GAP
    return panel


def main() -> None:
    args = parse_args()
    case_csv = args.case_csv.resolve()
    baseline_root = args.baseline_root.resolve()
    variant_root = args.variant_root.resolve()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    df = pd.read_csv(case_csv)
    df = df.sort_values(["query_patient_id", "query_rel_path"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows found in {case_csv}")

    inventory_rows = []
    for patient_id, group in df.groupby("query_patient_id", sort=True):
        group = group.sort_values("query_rel_path").reset_index(drop=True)
        panel = build_patient_panel(group, baseline_root, variant_root)
        out_path = output_root / f"patient_{patient_id}.png"
        panel.save(out_path)
        inventory_rows.append(
            {
                "query_patient_id": int(patient_id),
                "query_count": int(len(group)),
                "panel_path": str(out_path),
                "query_rel_paths": " | ".join(group["query_rel_path"].astype(str).tolist()),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows).sort_values("query_patient_id").reset_index(drop=True)
    inventory_df.to_csv(output_root / "panel_inventory.csv", index=False)

    lines = [
        "# Global Analysis 6 Follow-up Panels",
        "",
        "- condition: `delta_R@1 > 0` and `delta_AP < 0` on `per_query_patient_test_delta_probe_cag.csv`",
        f"- source case csv: `{case_csv}`",
        f"- baseline image root: `{baseline_root}`",
        f"- suppressed image root: `{variant_root}`",
        "",
        "## Patient Panels",
        "",
    ]
    for row in inventory_df.itertuples(index=False):
        lines.append(f"- patient `{row.query_patient_id}`: `{row.panel_path}`")
    lines.append("")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
