#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans

from scripts.analysis.global_analysis.global_4_1_cluster_anchoring_attribution import (
    _build_umap_coords,
    apply_probe_checkpoint,
    ensure_dir,
    hash_dataframe,
    load_test_features,
    make_feature_hash,
    resolve_device,
)


def load_manifest_and_hash(analysis_root: Path, global2_root: Path) -> tuple[pd.DataFrame, str]:
    manifest = pd.read_csv(analysis_root / "test_manifest_with_anchor_features.csv").sort_values("image_id").reset_index(drop=True)
    base_manifest = pd.read_csv(global2_root / "image_manifest_test.csv")
    manifest_hash = hash_dataframe(base_manifest, ["image_id", "img_path", "patient_id", "study_id"])
    return manifest, manifest_hash


def reconstruct_probe_embedding(
    global2_root: Path,
    manifest: pd.DataFrame,
    manifest_hash: str,
    backbone_name: str,
    target: str,
    seed: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    features, _meta = load_test_features(global2_root, backbone_name)
    feature_hash = make_feature_hash(features)
    full_probe_embeddings, _summary = apply_probe_checkpoint(
        global2_root=global2_root,
        backbone_name=backbone_name,
        target=target,
        seed=seed,
        features=features,
        manifest_hash=manifest_hash,
        feature_hash=feature_hash,
        device=resolve_device(device),
        batch_size=batch_size,
    )
    active_indices = manifest["image_id"].astype(int).to_numpy(dtype=np.int64)
    active_probe_embeddings = full_probe_embeddings[active_indices].contiguous()
    return active_probe_embeddings.detach().cpu().numpy().astype(np.float32, copy=False)


def cluster_umap(coords: np.ndarray, n_clusters: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_
    return labels.astype(np.int32), centers.astype(np.float32)


def choose_representative_rows(cluster_df: pd.DataFrame, center_xy: np.ndarray, max_samples: int) -> pd.DataFrame:
    work = cluster_df.copy()
    work["dist_to_center"] = np.sqrt((work["umap_x"] - float(center_xy[0])) ** 2 + (work["umap_y"] - float(center_xy[1])) ** 2)
    work = work.sort_values(["dist_to_center", "patient_id", "study_id", "img_path"]).reset_index(drop=True)

    selected_indices: List[int] = []
    used_pairs: set[str] = set()
    used_patients: set[str] = set()

    for idx, row in work.iterrows():
        pair_key = f"{row['patient_id']}::{row['study_id']}"
        patient_key = str(row["patient_id"])
        if pair_key in used_pairs:
            continue
        if patient_key not in used_patients or len(used_patients) >= max_samples // 2:
            selected_indices.append(idx)
            used_pairs.add(pair_key)
            used_patients.add(patient_key)
        if len(selected_indices) >= max_samples:
            break

    if len(selected_indices) < max_samples:
        for idx, row in work.iterrows():
            if idx in selected_indices:
                continue
            pair_key = f"{row['patient_id']}::{row['study_id']}"
            if pair_key in used_pairs:
                continue
            selected_indices.append(idx)
            used_pairs.add(pair_key)
            if len(selected_indices) >= max_samples:
                break

    return work.iloc[selected_indices].reset_index(drop=True)


def _get_font(size: int) -> ImageFont.ImageFont:
    for name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _wrap_text_by_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int | None = None,
) -> list[str]:
    raw = str(text).strip()
    if not raw:
        return [""]
    words = raw.split()
    if not words:
        return [raw]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        candidate_w, _ = _measure_text(draw, candidate, font)
        if candidate_w <= max_width:
            current = candidate
            continue
        lines.append(current)
        current = word
    lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        kept = lines[: max_lines - 1]
        overflow = " ".join(lines[max_lines - 1 :])
        trimmed = overflow
        while trimmed:
            candidate = f"{trimmed}..."
            candidate_w, _ = _measure_text(draw, candidate, font)
            if candidate_w <= max_width:
                break
            trimmed = trimmed[:-1]
        kept.append(f"{trimmed}..." if trimmed else "...")
        return kept
    return lines


def render_contact_sheet(
    sample_df: pd.DataFrame,
    output_path: Path,
    title: str,
    cols: int = 4,
    thumb_w: int = 220,
    thumb_h: int = 220,
) -> None:
    ensure_dir(output_path.parent)
    rows = max(1, math.ceil(len(sample_df) / cols))
    pad = 12
    inner_pad = 6
    title_font = _get_font(20)
    caption_font = _get_font(14)
    scratch = Image.new("RGB", (max(1, cols * thumb_w), 200), "white")
    scratch_draw = ImageDraw.Draw(scratch)
    title_max_width = cols * thumb_w + (cols - 1) * pad
    title_lines = _wrap_text_by_width(scratch_draw, title, title_font, title_max_width)
    _, title_line_h = _measure_text(scratch_draw, "Ag", title_font)
    title_line_gap = 4
    header_h = pad + len(title_lines) * title_line_h + max(0, len(title_lines) - 1) * title_line_gap + pad

    caption_w = thumb_w - 2 * inner_pad
    caption_line_gap = 3
    caption_template = [
        ("p:{patient_id} s:{study_id}", caption_font, None),
        ("{field_value}", caption_font, 3),
        ("({umap_x:.2f}, {umap_y:.2f})", caption_font, None),
    ]
    row_captions: list[list[list[tuple[str, ImageFont.ImageFont]]]] = []
    row_caption_h: list[int] = []
    _, caption_line_h = _measure_text(scratch_draw, "Ag", caption_font)

    for r in range(rows):
        row_start = r * cols
        row_end = min(len(sample_df), row_start + cols)
        row_items = sample_df.iloc[row_start:row_end]
        wrapped_cells: list[list[tuple[str, ImageFont.ImageFont]]] = []
        max_h = 0
        for _, row in row_items.iterrows():
            cell_lines: list[tuple[str, ImageFont.ImageFont]] = []
            for template, font, max_lines in caption_template:
                text = template.format(
                    patient_id=row["patient_id"],
                    study_id=row["study_id"],
                    field_value=row["field_value"],
                    umap_x=row["umap_x"],
                    umap_y=row["umap_y"],
                )
                wrapped = _wrap_text_by_width(scratch_draw, text, font, caption_w, max_lines=max_lines)
                cell_lines.extend((line, font) for line in wrapped)
            wrapped_cells.append(cell_lines)
            cell_h = inner_pad * 2 + len(cell_lines) * caption_line_h + max(0, len(cell_lines) - 1) * caption_line_gap
            max_h = max(max_h, cell_h)
        row_captions.append(wrapped_cells)
        row_caption_h.append(max_h)

    canvas_w = cols * thumb_w + (cols + 1) * pad
    canvas_h = header_h + pad
    for caption_h in row_caption_h:
        canvas_h += thumb_h + caption_h + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    title_y = pad
    for line in title_lines:
        draw.text((pad, title_y), line, fill="black", font=title_font)
        title_y += title_line_h + title_line_gap

    row_top = header_h + pad
    for idx, row in sample_df.iterrows():
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (thumb_w + pad)
        y0 = row_top + sum(thumb_h + row_caption_h[i] + pad for i in range(r))
        img = Image.open(row["img_path"]).convert("L").resize((thumb_w, thumb_h))
        rgb = Image.merge("RGB", (img, img, img))
        canvas.paste(rgb, (x0, y0))
        draw.rectangle([x0, y0, x0 + thumb_w, y0 + thumb_h], outline="#444444", width=1)
        caption_box_top = y0 + thumb_h
        cell_lines = row_captions[r][c]
        cy = caption_box_top + inner_pad
        for line, font in cell_lines:
            draw.text((x0 + inner_pad, cy), line, fill="black", font=font)
            cy += caption_line_h + caption_line_gap
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-root", default="outputs/global_4_1_cluster_anchoring_attribution_unique_view")
    parser.add_argument("--global2-root", default="outputs/global_2_study_patient_retrieval_unique_view")
    parser.add_argument("--field-name", required=True)
    parser.add_argument("--backbone-name", required=True, choices=["imagenet", "cag"])
    parser.add_argument("--target", required=True, choices=["patient", "study"])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--umap-seed", type=int, default=42)
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples-per-cluster", type=int, default=12)
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    analysis_root = Path(args.analysis_root).resolve()
    global2_root = Path(args.global2_root).resolve()
    output_root = (
        Path(args.output_root).resolve()
        if args.output_root
        else analysis_root
        / "cluster_examples"
        / args.target
        / args.backbone_name
        / args.field_name
    )
    ensure_dir(output_root)

    manifest, manifest_hash = load_manifest_and_hash(analysis_root, global2_root)
    if args.field_name not in manifest.columns:
        raise KeyError(f"Field not found in manifest: {args.field_name}")

    embedding = reconstruct_probe_embedding(
        global2_root=global2_root,
        manifest=manifest,
        manifest_hash=manifest_hash,
        backbone_name=args.backbone_name,
        target=args.target,
        seed=int(args.seed),
        device=args.device,
        batch_size=int(args.batch_size),
    )
    coords = _build_umap_coords(embedding, random_state=int(args.umap_seed))

    field_series = manifest[args.field_name].astype(str).fillna("<missing>")
    unique_values = sorted(field_series.unique().tolist())
    n_clusters = int(args.num_clusters) if args.num_clusters is not None else len(unique_values)
    labels, centers = cluster_umap(coords, n_clusters=n_clusters, seed=int(args.umap_seed))

    cluster_df = manifest[["image_id", "img_path", "patient_id", "study_id", "class_name"]].copy()
    cluster_df["field_name"] = args.field_name
    cluster_df["field_value"] = field_series
    cluster_df["umap_x"] = coords[:, 0]
    cluster_df["umap_y"] = coords[:, 1]
    cluster_df["cluster_id"] = labels

    summary_rows: list[dict[str, object]] = []
    for cluster_id in sorted(cluster_df["cluster_id"].unique().tolist()):
        sub = cluster_df[cluster_df["cluster_id"] == cluster_id].copy().reset_index(drop=True)
        dominant = sub["field_value"].value_counts()
        dominant_value = str(dominant.index[0])
        dominant_share = float(dominant.iloc[0] / len(sub))
        center_xy = centers[int(cluster_id)]
        sub["dist_to_center"] = np.sqrt((sub["umap_x"] - float(center_xy[0])) ** 2 + (sub["umap_y"] - float(center_xy[1])) ** 2)
        sub = sub.sort_values(["dist_to_center", "patient_id", "study_id", "img_path"]).reset_index(drop=True)
        sub.to_csv(output_root / f"cluster_{cluster_id}_members.csv", index=False)

        selected = choose_representative_rows(sub, center_xy=center_xy, max_samples=int(args.samples_per_cluster))
        selected.to_csv(output_root / f"cluster_{cluster_id}_selected_samples.csv", index=False)
        render_contact_sheet(
            selected,
            output_root / f"cluster_{cluster_id}_contact_sheet.png",
            title=(
                f"cluster {cluster_id} | dominant {args.field_name}={dominant_value} "
                f"| purity={dominant_share:.3f} | patients={sub['patient_id'].nunique()} | studies={sub['study_id'].nunique()}"
            ),
        )

        summary_rows.append(
            {
                "cluster_id": int(cluster_id),
                "num_images": int(len(sub)),
                "num_unique_patients": int(sub["patient_id"].nunique()),
                "num_unique_studies": int(sub["study_id"].nunique()),
                "num_unique_field_values": int(sub["field_value"].nunique()),
                "dominant_field_value": dominant_value,
                "dominant_field_value_share": dominant_share,
                "center_x": float(center_xy[0]),
                "center_y": float(center_xy[1]),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("cluster_id").reset_index(drop=True)
    summary_df.to_csv(output_root / "cluster_summary.csv", index=False)
    cluster_df.to_csv(output_root / "cluster_assignments.csv", index=False)
    print(str((output_root / "cluster_summary.csv").resolve()))


if __name__ == "__main__":
    main()
