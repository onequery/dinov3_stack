#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORT_ROOT = (
    REPO_ROOT
    / "outputs/analysis2_rep_analysis/global_6_patient_retrieval_border_suppressed_philips/reports"
)
DEFAULT_CASE_CSV = DEFAULT_REPORT_ROOT / "followup_probe_cag_r1_improved_ap_worsened.csv"
DEFAULT_BASELINE_PER_QUERY_CSV = (
    REPO_ROOT
    / "outputs/analysis2_rep_analysis/global_6_patient_retrieval_border_suppressed_philips"
    / "baseline_philips/retrieval_benchmark/per_query_patient_probe_seed11_cag_test.csv"
)
DEFAULT_VARIANT_PER_QUERY_CSV = (
    REPO_ROOT
    / "outputs/analysis2_rep_analysis/global_6_patient_retrieval_border_suppressed_philips"
    / "border_suppressed_philips/retrieval_benchmark/per_query_patient_probe_seed11_cag_test.csv"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_REPORT_ROOT / "followup_probe_cag_r1_improved_ap_worsened_ranking_panels"

BG_COLOR = (248, 248, 248)
TEXT_COLOR = (24, 24, 24)
SUBTEXT_COLOR = (88, 88, 88)
ACCENT_COLOR = (30, 90, 160)
POS_COLOR = (46, 125, 50)
NEG_COLOR = (198, 40, 40)
FIRST_POS_COLOR = (33, 150, 243)
CARD_BG = (255, 255, 255)
QUERY_LABEL_BG = (244, 247, 252)
SECTION_BG = (252, 252, 252)

PANEL_WIDTH = 920
MARGIN_X = 24
TOP_MARGIN = 24
BOTTOM_MARGIN = 24
SECTION_GAP = 22
TEXT_LINE_GAP = 18
QUERY_SIZE = 240
QUERY_GAP = 24
THUMB_SIZE = 150
GRID_COLS = 5
GRID_GAP_X = 12
GRID_GAP_Y = 14
CAPTION_GAP = 4
TILE_BORDER = 5
QUERY_BORDER = 4
INFO_BOX_HEIGHT = 52


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-csv", type=Path, default=DEFAULT_CASE_CSV)
    parser.add_argument("--baseline-per-query-csv", type=Path, default=DEFAULT_BASELINE_PER_QUERY_CSV)
    parser.add_argument("--variant-per-query-csv", type=Path, default=DEFAULT_VARIANT_PER_QUERY_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    draw.text(xy, text, font=font, fill=fill)


def text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    return float(draw.textlength(text, font=font))


def font_line_height(font: ImageFont.ImageFont, *, padding: int = 4) -> int:
    bbox = font.getbbox("Ag")
    return int(bbox[3] - bbox[1] + padding)


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


def read_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("L")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return image.convert("RGB")


def to_rel_path(path_str: str) -> str:
    return path_str.split("/images/", 1)[-1]


def parse_pipe_list(value: str) -> list[str]:
    text = str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(" | ")]


def format_score(value: str) -> str:
    try:
        return f"{float(value):.3f}"
    except ValueError:
        return value


def build_query_card(
    title: str,
    image_path: Path,
    info_lines: list[str],
    *,
    card_width: int,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    image = read_image(image_path, QUERY_SIZE)
    measure = Image.new("RGB", (card_width, 10), BG_COLOR)
    measure_draw = ImageDraw.Draw(measure)
    wrapped_lines: list[str] = []
    for line in info_lines:
        wrapped_lines.extend(wrap_text(measure_draw, line, small_font, card_width - 24))
    info_text_height = max(INFO_BOX_HEIGHT, 14 + len(wrapped_lines) * TEXT_LINE_GAP)
    height = 40 + QUERY_SIZE + 10 + info_text_height + 16

    canvas = Image.new("RGB", (card_width, height), CARD_BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, card_width - 1, height - 1), radius=10, fill=CARD_BG, outline=(220, 220, 220), width=1)
    draw_text(draw, (12, 10), title, font=title_font, fill=TEXT_COLOR)
    x = (card_width - QUERY_SIZE) // 2
    y = 40
    draw.rectangle((x - QUERY_BORDER, y - QUERY_BORDER, x + QUERY_SIZE + QUERY_BORDER, y + QUERY_SIZE + QUERY_BORDER), fill=ACCENT_COLOR)
    canvas.paste(image, (x, y))
    info_y = y + QUERY_SIZE + 12
    draw.rounded_rectangle((12, info_y, card_width - 12, info_y + info_text_height), radius=8, fill=QUERY_LABEL_BG, outline=(230, 235, 242), width=1)
    ty = info_y + 8
    for line in wrapped_lines:
        draw_text(draw, (20, ty), line, font=small_font, fill=SUBTEXT_COLOR)
        ty += TEXT_LINE_GAP
    return canvas


def short_tail(rel_path: str) -> str:
    parts = rel_path.split("/")
    if len(parts) >= 5:
        return f"{parts[-5]}/{parts[-2]}/{parts[-1]}"
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return rel_path


def build_gallery_tile(
    rank_idx: int,
    image_path: Path,
    *,
    score: str,
    patient_id: str,
    study_id: str,
    is_positive: bool,
    is_first_positive: bool,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    image = read_image(image_path, THUMB_SIZE)
    border_color = FIRST_POS_COLOR if is_first_positive else (POS_COLOR if is_positive else NEG_COLOR)
    rel_path = to_rel_path(str(image_path))
    caption_lines = [
        f"#{rank_idx} {'POS' if is_positive else 'NEG'} | s={format_score(score)}",
        f"pid {patient_id} | study {study_id}",
        short_tail(rel_path),
    ]
    tile_width = THUMB_SIZE + 18
    tile_height = THUMB_SIZE + 68
    canvas = Image.new("RGB", (tile_width, tile_height), CARD_BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, tile_width - 1, tile_height - 1), radius=8, fill=CARD_BG, outline=(225, 225, 225), width=1)
    x = 9
    y = 9
    draw.rectangle((x - TILE_BORDER, y - TILE_BORDER, x + THUMB_SIZE + TILE_BORDER, y + THUMB_SIZE + TILE_BORDER), fill=border_color)
    canvas.paste(image, (x, y))
    cy = y + THUMB_SIZE + 8
    for idx, line in enumerate(caption_lines):
        font = body_font if idx == 0 else small_font
        fill = border_color if idx == 0 else SUBTEXT_COLOR
        draw_text(draw, (9, cy), line, font=font, fill=fill)
        cy += 16
    return canvas


def build_gallery_section(
    title: str,
    row: pd.Series,
    *,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> Image.Image:
    top_paths = [Path(path) for path in parse_pipe_list(row["top_paths"])]
    top_scores = parse_pipe_list(row["top_scores"])
    top_patient_ids = parse_pipe_list(row["top_patient_ids"])
    top_study_ids = parse_pipe_list(row["top_study_ids"])
    top_is_positive = [item == "1" for item in parse_pipe_list(row["top_is_positive"])]
    first_positive_rank = float(row["first_positive_rank"])

    tiles: list[Image.Image] = []
    for idx, img_path in enumerate(top_paths[:10], start=1):
        is_first_positive = top_is_positive[idx - 1] and abs(first_positive_rank - idx) < 1e-6
        tiles.append(
            build_gallery_tile(
                idx,
                img_path,
                score=top_scores[idx - 1] if idx - 1 < len(top_scores) else "",
                patient_id=top_patient_ids[idx - 1] if idx - 1 < len(top_patient_ids) else "",
                study_id=top_study_ids[idx - 1] if idx - 1 < len(top_study_ids) else "",
                is_positive=top_is_positive[idx - 1] if idx - 1 < len(top_is_positive) else False,
                is_first_positive=is_first_positive,
                body_font=body_font,
                small_font=small_font,
            )
        )

    section_width = PANEL_WIDTH - 2 * MARGIN_X
    grid_width = GRID_COLS * tiles[0].width + (GRID_COLS - 1) * GRID_GAP_X if tiles else section_width
    grid_x = MARGIN_X + max(0, (section_width - grid_width) // 2)
    measure = Image.new("RGB", (PANEL_WIDTH, 10), BG_COLOR)
    measure_draw = ImageDraw.Draw(measure)
    meta_line = (
        f"AP={float(row['AP']):.4f} | first_pos_rank={float(row['first_positive_rank']):.1f} | "
        f"R@1={int(row['R@1'])} | R@5={int(row['R@5'])} | R@10={int(row['R@10'])} | "
        f"num_pos={int(row['num_positives'])}"
    )
    meta_lines = wrap_text(measure_draw, meta_line, small_font, section_width)

    top_banner = 24 + len(meta_lines) * TEXT_LINE_GAP + 16
    rows = 2
    grid_height = rows * tiles[0].height + (rows - 1) * GRID_GAP_Y if tiles else 0
    section_height = top_banner + grid_height + 16
    canvas = Image.new("RGB", (PANEL_WIDTH, section_height), SECTION_BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((MARGIN_X, 0, PANEL_WIDTH - MARGIN_X, section_height - 1), radius=10, fill=SECTION_BG, outline=(228, 228, 228), width=1)
    draw_text(draw, (MARGIN_X + 14, 12), title, font=title_font, fill=TEXT_COLOR)
    ty = 36
    for line in meta_lines:
        draw_text(draw, (MARGIN_X + 14, ty), line, font=small_font, fill=SUBTEXT_COLOR)
        ty += TEXT_LINE_GAP

    for idx, tile in enumerate(tiles):
        row_idx = idx // GRID_COLS
        col_idx = idx % GRID_COLS
        x = grid_x + col_idx * (tile.width + GRID_GAP_X)
        y = top_banner + row_idx * (tile.height + GRID_GAP_Y)
        canvas.paste(tile, (x, y))
    return canvas


def build_panel(case_row: pd.Series, base_row: pd.Series, var_row: pd.Series) -> Image.Image:
    title_font = load_font(24)
    body_font = load_font(15)
    small_font = load_font(13)

    measure = Image.new("RGB", (PANEL_WIDTH, 10), BG_COLOR)
    measure_draw = ImageDraw.Draw(measure)

    header_lines = [
        "Global Analysis 6 follow-up | probe CAG | seed11 ranking view",
        str(case_row["query_rel_path"]),
        (
            f"query_patient={case_row['query_patient_id']} | query_study={case_row['query_study_id']} | "
            f"num_positives={int(case_row['num_positives'])}"
        ),
        (
            f"averaged selection condition: delta_AP={float(case_row['delta_AP']):+.4f}, "
            f"delta_R@1={float(case_row['delta_R_at_1']):+.3f}, "
            f"delta_first_positive_rank={float(case_row['delta_first_positive_rank']):+.3f}"
        ),
        "Interpretation note: R@1 can improve even when AP drops if one positive moves up but the remaining positives move down.",
    ]
    wrapped_header: list[tuple[str, tuple[int, int, int], ImageFont.ImageFont, int]] = []
    for idx, line in enumerate(header_lines):
        font = title_font if idx == 0 else (body_font if idx == 1 else small_font)
        color = TEXT_COLOR if idx == 0 else (TEXT_COLOR if idx == 1 else (ACCENT_COLOR if idx == 3 else SUBTEXT_COLOR))
        line_height = font_line_height(font, padding=8 if idx == 0 else 5)
        for wrapped in wrap_text(measure_draw, line, font, PANEL_WIDTH - 2 * MARGIN_X):
            wrapped_header.append((wrapped, color, font, line_height))

    header_height = TOP_MARGIN + sum(line_height for _, _, _, line_height in wrapped_header) + 12
    card_width = (PANEL_WIDTH - 2 * MARGIN_X - QUERY_GAP) // 2
    query_cards = [
        build_query_card(
            "Baseline query",
            Path(str(base_row["query_path"])),
            [
                f"AP={float(base_row['AP']):.4f} | first_pos_rank={float(base_row['first_positive_rank']):.1f}",
                f"R@1={int(base_row['R@1'])} | R@5={int(base_row['R@5'])} | R@10={int(base_row['R@10'])}",
            ],
            card_width=card_width,
            title_font=body_font,
            body_font=body_font,
            small_font=small_font,
        ),
        build_query_card(
            "Border-suppressed query",
            Path(str(var_row["query_path"])),
            [
                f"AP={float(var_row['AP']):.4f} | first_pos_rank={float(var_row['first_positive_rank']):.1f}",
                f"R@1={int(var_row['R@1'])} | R@5={int(var_row['R@5'])} | R@10={int(var_row['R@10'])}",
            ],
            card_width=card_width,
            title_font=body_font,
            body_font=body_font,
            small_font=small_font,
        ),
    ]
    query_row_height = max(card.height for card in query_cards)
    baseline_section = build_gallery_section("Baseline PHILIPS top-10 gallery", base_row, title_font=body_font, body_font=body_font, small_font=small_font)
    variant_section = build_gallery_section("Border-suppressed PHILIPS top-10 gallery", var_row, title_font=body_font, body_font=body_font, small_font=small_font)

    panel_height = (
        header_height
        + query_row_height
        + SECTION_GAP
        + baseline_section.height
        + SECTION_GAP
        + variant_section.height
        + BOTTOM_MARGIN
    )
    panel = Image.new("RGB", (PANEL_WIDTH, panel_height), BG_COLOR)
    draw = ImageDraw.Draw(panel)

    y = TOP_MARGIN
    for line, color, font, line_height in wrapped_header:
        draw_text(draw, (MARGIN_X, y), line, font=font, fill=color)
        y += line_height
    y += 10

    x0 = MARGIN_X
    x1 = MARGIN_X + card_width + QUERY_GAP
    panel.paste(query_cards[0], (x0, y))
    panel.paste(query_cards[1], (x1, y))
    y += query_row_height + SECTION_GAP

    panel.paste(baseline_section, (0, y))
    y += baseline_section.height + SECTION_GAP
    panel.paste(variant_section, (0, y))
    return panel


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    cases = pd.read_csv(args.case_csv.resolve()).sort_values(["query_patient_id", "query_rel_path"]).reset_index(drop=True)
    base_per_query = pd.read_csv(args.baseline_per_query_csv.resolve())
    var_per_query = pd.read_csv(args.variant_per_query_csv.resolve())
    base_per_query["query_rel_path"] = base_per_query["query_path"].map(to_rel_path)
    var_per_query["query_rel_path"] = var_per_query["query_path"].map(to_rel_path)

    base_lookup = base_per_query.set_index("query_rel_path", drop=False)
    var_lookup = var_per_query.set_index("query_rel_path", drop=False)

    inventory_rows: list[dict[str, object]] = []
    for row in cases.itertuples(index=False):
        rel_path = str(row.query_rel_path)
        if rel_path not in base_lookup.index or rel_path not in var_lookup.index:
            raise KeyError(f"Missing query row for {rel_path}")
        panel = build_panel(pd.Series(row._asdict()), base_lookup.loc[rel_path], var_lookup.loc[rel_path])
        patient_id = str(row.query_patient_id)
        stem = rel_path.replace("/", "__")
        out_name = f"patient_{patient_id}__{stem[:-4]}.png" if stem.endswith(".png") else f"patient_{patient_id}__{stem}.png"
        out_path = output_root / out_name
        panel.save(out_path)
        inventory_rows.append(
            {
                "query_rel_path": rel_path,
                "query_patient_id": int(row.query_patient_id),
                "query_study_id": int(row.query_study_id),
                "num_positives": int(row.num_positives),
                "delta_AP": float(row.delta_AP),
                "delta_R_at_1": float(row.delta_R_at_1),
                "delta_first_positive_rank": float(row.delta_first_positive_rank),
                "panel_path": str(out_path),
            }
        )

    inventory_df = pd.DataFrame(inventory_rows).sort_values(["query_patient_id", "query_rel_path"]).reset_index(drop=True)
    inventory_df.to_csv(output_root / "panel_inventory.csv", index=False)

    lines = [
        "# Global Analysis 6 Follow-up Ranking Panels",
        "",
        "- condition source: `followup_probe_cag_r1_improved_ap_worsened.csv`",
        "- query selection: averaged over 3 seeds (`delta_R@1 > 0` and `delta_AP < 0`)",
        "- ranking view: concrete `seed11` top-10 gallery from `per_query_patient_probe_seed11_cag_test.csv`",
        "- border colors: `green=positive`, `red=negative`, `blue=first positive`",
        "",
        "## Query Panels",
        "",
    ]
    for row in inventory_df.itertuples(index=False):
        lines.append(f"- `{row.query_rel_path}`: `{row.panel_path}`")
    lines.append("")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
