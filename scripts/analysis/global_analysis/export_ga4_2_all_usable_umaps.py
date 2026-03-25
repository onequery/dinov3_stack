#!/usr/bin/env python3
"""Fill GA4-2 per-field UMAP overlays for all ranked fields.

This reuses existing GA4-2 retrieval outputs and anchor ranking CSVs without
rerunning the full attribution pipeline. It is intended for backfilling
`umap_overlays_score_ge_0/` so that every ranked field in
`summary_global_4_2_anchor_rank.csv` gets a per-field UMAP PNG.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def _load_ga42_module(script_path: Path) -> Any:
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
    parser.add_argument(
        "--output-root",
        default="outputs/global_4_2_same_dicom_retrieval_unique_view/umap_overlays_score_ge_0",
    )
    parser.add_argument("--target", default="same_dicom")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--probe-seed", type=int, default=11)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()
    analysis_script = repo_root / args.analysis_script
    global2_root = repo_root / args.global2_root
    output_root = repo_root / args.output_root

    mod = _load_ga42_module(analysis_script)

    anchor_df = pd.read_csv(global2_root / "summary_global_4_2_anchor_rank.csv")
    manifest = pd.read_csv(global2_root / "test_manifest_with_anchor_features.csv")
    test_manifest = pd.read_csv(global2_root / "image_manifest_test.csv")

    expected_hash = mod.hash_dataframe(
        test_manifest,
        ["image_id", "img_path", "patient_id", "study_id", "dicom_id", "frame_index", "frame_offset"],
    )
    if "image_id" in manifest.columns:
        if len(manifest) != len(test_manifest) or not np.array_equal(
            manifest["image_id"].to_numpy(),
            test_manifest["image_id"].to_numpy(),
        ):
            raise RuntimeError("anchor feature manifest order mismatch with GA4-2 test manifest")

    device = torch.device(args.device)
    embedding_map: dict[tuple[str, str], np.ndarray] = {}
    for backbone_name in mod.BACKBONES:
        features, _ = mod.load_test_features(global2_root, backbone_name)
        feature_hash = mod.make_feature_hash(features)
        embedding, _ = mod.apply_probe_checkpoint(
            global2_root=global2_root,
            backbone_name=backbone_name,
            target=args.target,
            seed=int(args.probe_seed),
            features=features,
            manifest_hash=expected_hash,
            feature_hash=feature_hash,
            device=device,
            batch_size=int(args.probe_batch_size),
        )
        embedding_map[(backbone_name, args.target)] = embedding.numpy()

    rows: list[dict[str, object]] = []
    created = 0
    for row in anchor_df.itertuples(index=False):
        field_type = str(row.field_type)
        target = str(row.target)
        backbone_name = str(row.backbone_name)
        field_name = str(row.field_name)
        field_group = str(row.field_group)
        rank = float(row.rank_within_type) if pd.notna(row.rank_within_type) else np.nan

        export_dir = output_root / field_type / target / backbone_name
        mod.ensure_dir(export_dir)
        if pd.notna(rank):
            filename = f"{int(rank):03d}_{mod._safe_field_filename(field_name)}.png"
        else:
            filename = f"{mod._safe_field_filename(field_name)}.png"
        output_path = export_dir / filename

        if not output_path.exists():
            if field_type == "categorical":
                mod.save_umap_overlay_field_categorical(
                    embedding_map[(backbone_name, target)],
                    manifest,
                    field_name,
                    output_path,
                    seed=int(args.seed),
                )
            else:
                mod.save_umap_overlay_field_continuous(
                    embedding_map[(backbone_name, target)],
                    manifest,
                    field_name,
                    output_path,
                    seed=int(args.seed),
                )
            created += 1

        rows.append(
            {
                "field_type": field_type,
                "target": target,
                "backbone_name": backbone_name,
                "field_name": field_name,
                "field_group": field_group,
                "rank_within_type": rank,
                "combined_anchor_score_mean": float(row.combined_anchor_score_mean),
                "combined_anchor_score_std": float(row.combined_anchor_score_std),
                "output_path": str(output_path.resolve()),
            }
        )

    index_df = pd.DataFrame(rows)
    index_df.to_csv(output_root / "index_all_usable.csv", index=False)
    index_df.to_csv(output_root / "index_score_ge_0.csv", index=False)
    print(f"created={created} indexed={len(index_df)} output_root={output_root}")


if __name__ == "__main__":
    main()
