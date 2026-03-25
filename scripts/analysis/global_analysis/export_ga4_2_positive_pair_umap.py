#!/usr/bin/env python3
"""Export GA4-2 positive-pair UMAP figures for raw/probe embeddings."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import torch


def _load_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ga42_anchor_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--analysis-script",
        default="scripts/analysis/global_4_2_cluster_anchoring_attribution.py",
    )
    parser.add_argument(
        "--global2-root",
        default="outputs/global_4_2_same_dicom_retrieval_unique_view",
    )
    parser.add_argument("--probe-seed", type=int, default=11)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cosine_similarity_pairs(embeddings: np.ndarray, index_pairs: np.ndarray) -> np.ndarray:
    left = embeddings[index_pairs[:, 0]]
    right = embeddings[index_pairs[:, 1]]
    return np.sum(left * right, axis=1)


def pairwise_umap_distance(coords: np.ndarray, index_pairs: np.ndarray) -> np.ndarray:
    deltas = coords[index_pairs[:, 0]] - coords[index_pairs[:, 1]]
    return np.linalg.norm(deltas, axis=1)


def build_pair_index(manifest: pd.DataFrame) -> np.ndarray:
    grouped = manifest.groupby("dicom_id")["image_id"].apply(list)
    bad = grouped[grouped.apply(len) != 2]
    if not bad.empty:
        raise ValueError(f"Each dicom_id must map to exactly 2 images. Violations: {bad.head().to_dict()}")
    return np.asarray([[int(ids[0]), int(ids[1])] for ids in grouped.tolist()], dtype=np.int64)


def plot_pair_umap(
    coords_by_backbone: dict[str, np.ndarray],
    manifest: pd.DataFrame,
    index_pairs: np.ndarray,
    cosine_by_backbone: dict[str, np.ndarray],
    output_path: Path,
    title_prefix: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), squeeze=False)
    offset_palette = {-3: "#4C78A8", 3: "#F58518"}

    for col_idx, backbone_name in enumerate(["imagenet", "cag"]):
        ax = axes[0, col_idx]
        coords = coords_by_backbone[backbone_name]
        pair_cosine = cosine_by_backbone[backbone_name]

        # Background points.
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=10,
            c="#D6D6D6",
            alpha=0.45,
            linewidths=0,
        )

        # Pair segments.
        segments = np.stack(
            [coords[index_pairs[:, 0]], coords[index_pairs[:, 1]]],
            axis=1,
        )
        norm = plt.Normalize(vmin=float(np.percentile(pair_cosine, 5)), vmax=float(np.percentile(pair_cosine, 95)))
        lc = LineCollection(
            segments,
            cmap="viridis",
            norm=norm,
            linewidths=0.8,
            alpha=0.28,
        )
        lc.set_array(pair_cosine)
        ax.add_collection(lc)

        # Frame-offset points.
        for offset_value, color in offset_palette.items():
            mask = manifest["frame_offset"].to_numpy(dtype=np.int64) == int(offset_value)
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=12,
                c=color,
                alpha=0.7,
                linewidths=0,
                label=f"offset {offset_value:+d}",
            )

        pair_dist = pairwise_umap_distance(coords, index_pairs)
        ax.set_title(
            f"{backbone_name} | median cos={np.median(pair_cosine):.3f} | "
            f"p10 cos={np.percentile(pair_cosine, 10):.3f} | median 2D={np.median(pair_dist):.3f}"
        )
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle(title_prefix, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    analysis_script = repo_root / args.analysis_script
    out_root = repo_root / args.global2_root
    mod = _load_module(analysis_script)

    manifest = pd.read_csv(out_root / "image_manifest_test.csv")
    manifest = manifest.sort_values("image_id").reset_index(drop=True)
    pair_index = build_pair_index(manifest)

    test_manifest_hash = mod.hash_dataframe(
        manifest,
        ["image_id", "img_path", "patient_id", "study_id", "dicom_id", "frame_index", "frame_offset"],
    )
    device = torch.device(args.device)

    summary_rows: list[dict[str, object]] = []

    for mode in ["raw", "probe"]:
        mode_embeddings: dict[str, np.ndarray] = {}
        coords_by_backbone: dict[str, np.ndarray] = {}
        pair_cos_by_backbone: dict[str, np.ndarray] = {}

        for backbone_name in mod.BACKBONES:
            features, _ = mod.load_test_features(out_root, backbone_name)
            if mode == "raw":
                embedding = mod.l2_normalize_cpu(features).numpy()
            else:
                feature_hash = mod.make_feature_hash(features)
                probe_embedding, _ = mod.apply_probe_checkpoint(
                    global2_root=out_root,
                    backbone_name=backbone_name,
                    target="same_dicom",
                    seed=int(args.probe_seed),
                    features=features,
                    manifest_hash=test_manifest_hash,
                    feature_hash=feature_hash,
                    device=device,
                    batch_size=int(args.probe_batch_size),
                )
                embedding = probe_embedding.numpy()

            mode_embeddings[backbone_name] = embedding
            coords = mod._build_umap_coords(embedding, random_state=int(args.seed))
            coords_by_backbone[backbone_name] = coords
            pair_cos_by_backbone[backbone_name] = cosine_similarity_pairs(embedding, pair_index)

            pair_dist = pairwise_umap_distance(coords, pair_index)
            summary_rows.append(
                {
                    "mode": mode,
                    "backbone_name": backbone_name,
                    "num_pairs": int(len(pair_index)),
                    "median_pair_cosine": float(np.median(pair_cos_by_backbone[backbone_name])),
                    "p10_pair_cosine": float(np.percentile(pair_cos_by_backbone[backbone_name], 10)),
                    "median_pair_umap_distance": float(np.median(pair_dist)),
                    "p90_pair_umap_distance": float(np.percentile(pair_dist, 90)),
                }
            )

        plot_pair_umap(
            coords_by_backbone=coords_by_backbone,
            manifest=manifest,
            index_pairs=pair_index,
            cosine_by_backbone=pair_cos_by_backbone,
            output_path=out_root / f"fig_global4_2_umap_positive_pairs_{mode}.png",
            title_prefix=f"Global Analysis 4-2: Positive Pair UMAP ({mode})",
        )

    pd.DataFrame(summary_rows).to_csv(out_root / "summary_global_4_2_positive_pair_umap.csv", index=False)


if __name__ == "__main__":
    main()
