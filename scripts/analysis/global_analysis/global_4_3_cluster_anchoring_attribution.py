#!/usr/bin/env python3
"""
Global Analysis 4-3 — Cluster Anchoring Attribution Analysis.

This analysis reuses Global Analysis 4-3 view-classification probe checkpoints and test
features to find which nuisance modalities anchor the view-classification probe embedding space.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.global_analysis import global_4_2_cluster_anchoring_attribution as ga42  # noqa: E402
from scripts.analysis.global_analysis.global_4_3_view_classification import (  # noqa: E402
    StrictLinearClassificationProbe,
    TARGET_NAME as VIEW_TARGET_NAME,
    VIEW_COLORS,
    VIEW_LABELS_9WAY,
)


BACKBONES = ["imagenet", "cag"]
TARGETS = [VIEW_TARGET_NAME]
REFERENCE_FIELDS = ["view_label_9way", "patient_id", "study_id"]

# Reuse GA4-2 generic plotting/scoring helpers with GA4-3 target/reference semantics.
ga42.TARGETS = TARGETS
ga42.REFERENCE_FIELDS = REFERENCE_FIELDS


@dataclass(frozen=True)
class JobDef:
    kind: str
    backbone_name: str | None = None
    target: str | None = None
    seed: int | None = None
    name: str | None = None

    @property
    def display_name(self) -> str:
        if self.kind == "postprocess":
            return self.name or "postprocess"
        return f"probe/{self.backbone_name}/{self.target}/seed{self.seed}"


def load_test_features(global3_root: Path, backbone_name: str) -> Tuple[torch.Tensor, Dict[str, object]]:
    return ga42.load_test_features(global3_root, backbone_name)


def apply_probe_checkpoint(
    global3_root: Path,
    backbone_name: str,
    target: str,
    seed: int,
    features: torch.Tensor,
    manifest_hash: str,
    feature_hash: str,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    ckpt_path = global3_root / f"probe_checkpoint_seed{seed}_{backbone_name}_{target}.pt"
    payload = torch.load(ckpt_path, map_location="cpu")
    signature = payload.get("signature")
    if not isinstance(signature, dict):
        raise ValueError(f"Missing signature in checkpoint: {ckpt_path}")
    if signature.get("seed") != int(seed) or signature.get("backbone_name") != backbone_name or signature.get("target") != target:
        raise ValueError(f"Checkpoint signature mismatch: {ckpt_path}")
    if signature.get("test_manifest_hash") != manifest_hash:
        raise ValueError(f"Checkpoint test manifest hash mismatch: {ckpt_path}")
    feature_hashes = signature.get("feature_hashes", {})
    if feature_hashes.get("test") != feature_hash:
        raise ValueError(f"Checkpoint test feature hash mismatch: {ckpt_path}")
    model = StrictLinearClassificationProbe(int(features.shape[1]), len(VIEW_LABELS_9WAY))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, int(features.shape[0]), int(batch_size)):
            x = features[start : start + int(batch_size)].to(device, non_blocking=True)
            z = model.encode(x)
            outputs.append(z.cpu())
    embeddings = torch.cat(outputs, dim=0).contiguous().float()
    return embeddings, payload.get("summary", {})


def build_anchor_manifest(
    global3_root: Path,
    image_root: Path,
    dcm_root: Path,
    output_root: Path,
    max_images: int | None,
    seed: int,
) -> Tuple[pd.DataFrame, str, pd.DataFrame]:
    base_manifest = ga42.load_csv(global3_root / "image_manifest_test.csv")
    full_manifest_hash = ga42.hash_dataframe(
        base_manifest,
        [
            "image_id",
            "img_path",
            "patient_id",
            "study_id",
            "dicom_rel_path",
            "view_label_9way",
            "view_label_index",
        ],
    )
    manifest = ga42.maybe_subsample_manifest(base_manifest, max_images, seed)
    rows: List[Dict[str, object]] = []
    field_specs: Dict[str, ga42.FieldSpec] = {}
    for _, row in manifest.iterrows():
        img_path = Path(str(row["img_path"])).resolve()
        dcm_path = dcm_root.resolve() / str(row["dicom_rel_path"])
        if not dcm_path.exists():
            raise FileNotFoundError(f"Missing DICOM for image: {img_path}")
        meta, dicom_specs = ga42.extract_dicom_scalar_fields(dcm_path)
        for spec in dicom_specs.values():
            ga42.register_field(field_specs, spec.field_name, spec.field_type, spec.field_group, spec.source)
        image_stats = ga42.compute_image_attributes(img_path)
        for image_field in image_stats:
            ga42.register_field(field_specs, image_field, "continuous", "image", "image_stats")
        primary = ga42.clean_numeric(meta.get("PositionerPrimaryAngle"))
        secondary = ga42.clean_numeric(meta.get("PositionerSecondaryAngle"))
        derived_values = {
            "primary_angle_bin_10deg": ga42.quantize_angle(primary, 10.0),
            "secondary_angle_bin_10deg": ga42.quantize_angle(secondary, 10.0),
            "view_bin_2d_10deg": None
            if primary is None or secondary is None
            else f"({ga42.quantize_angle(primary, 10.0)},{ga42.quantize_angle(secondary, 10.0)})",
            "primary_angle_bin_20deg": ga42.quantize_angle(primary, 20.0),
            "secondary_angle_bin_20deg": ga42.quantize_angle(secondary, 20.0),
            "view_bin_2d_20deg": None
            if primary is None or secondary is None
            else f"({ga42.quantize_angle(primary, 20.0)},{ga42.quantize_angle(secondary, 20.0)})",
            "primary_angle_abs": None if primary is None else abs(float(primary)),
            "secondary_angle_abs": None if secondary is None else abs(float(secondary)),
        }
        for name in [
            "primary_angle_bin_10deg",
            "secondary_angle_bin_10deg",
            "view_bin_2d_10deg",
            "primary_angle_bin_20deg",
            "secondary_angle_bin_20deg",
            "view_bin_2d_20deg",
        ]:
            ga42.register_field(field_specs, name, "categorical", "derived", "angle_derived")
        for name in ["primary_angle_abs", "secondary_angle_abs"]:
            ga42.register_field(field_specs, name, "continuous", "derived", "angle_derived")
        combined = {
            "image_id": int(row["image_id"]),
            "img_path": str(img_path),
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "class_name": str(row.get("class_name", "")),
            "dicom_rel_path": str(row.get("dicom_rel_path", "")),
            "view_horizontal_10deg": str(row.get("view_horizontal_10deg", "")),
            "view_vertical_10deg": str(row.get("view_vertical_10deg", "")),
            "view_label_9way": str(row.get("view_label_9way", "")),
            "view_label_index": int(row.get("view_label_index", -1)),
            **meta,
            **derived_values,
            **image_stats,
        }
        rows.append(combined)
    out_df = pd.DataFrame(rows)
    if out_df["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image_id detected in anchor manifest.")
    out_df = out_df.sort_values("image_id").reset_index(drop=True)
    out_df.to_csv(output_root / "test_manifest_with_anchor_features.csv", index=False)
    field_specs_df = pd.DataFrame([spec.__dict__ for spec in field_specs.values()]).sort_values(
        ["field_group", "field_type", "field_name"]
    ).reset_index(drop=True)
    field_specs_df.to_csv(output_root / "summary_global_4_3_field_catalog.csv", index=False)
    return out_df, full_manifest_hash, field_specs_df


def save_view_category_overlay_probe(
    embedding_map: Dict[Tuple[str, str], np.ndarray],
    manifest: pd.DataFrame,
    output_path: Path,
    seed: int,
) -> None:
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(15.5, 6.3), squeeze=False)
    for col_idx, backbone_name in enumerate(BACKBONES):
        ax = axes[0, col_idx]
        coords = ga42._build_umap_coords(embedding_map[(backbone_name, VIEW_TARGET_NAME)], random_state=seed)
        labels = manifest["view_label_9way"].astype(str).to_numpy()
        for label in VIEW_LABELS_9WAY:
            mask = labels == label
            if not np.any(mask):
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=18,
                alpha=0.78,
                linewidths=0,
                color=VIEW_COLORS[label],
                label=label,
            )
        ax.set_title(f"{backbone_name} | {VIEW_TARGET_NAME}")
        ax.grid(alpha=0.18)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    output_path: Path,
    audit_df: pd.DataFrame,
    field_specs_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    field_audit_path: Path,
    field_catalog_path: Path,
) -> None:
    lines: List[str] = []
    usable = audit_df[audit_df["usable"] == 1]
    dropped = audit_df[audit_df["usable"] == 0]
    lines.append("# Global Analysis 4-3: Cluster Anchoring Attribution Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- discovered categorical fields: {int((field_specs_df['field_type'] == 'categorical').sum())}")
    lines.append(f"- discovered continuous fields: {int((field_specs_df['field_type'] == 'continuous').sum())}")
    lines.append(f"- usable categorical fields: {len(usable[usable['field_type'] == 'categorical'])}")
    lines.append(f"- usable continuous fields: {len(usable[usable['field_type'] == 'continuous'])}")
    lines.append(f"- field audit CSV: `{field_audit_path}`")
    lines.append(f"- field catalog CSV: `{field_catalog_path}`")
    lines.append("")
    lines.append("## Top Anchors")
    lines.append("")
    for backbone_name in BACKBONES:
        subset = anchor_df[(anchor_df["target"] == VIEW_TARGET_NAME) & (anchor_df["backbone_name"] == backbone_name)]
        top_cat = subset[(subset["field_type"] == "categorical") & (subset["field_group"] != "reference")].sort_values("combined_anchor_score_mean", ascending=False).head(5)
        top_cont = subset[subset["field_type"] == "continuous"].sort_values("combined_anchor_score_mean", ascending=False).head(5)
        lines.append(f"### {backbone_name}")
        lines.append("")
        lines.append("- top categorical anchors: " + ", ".join(f"`{row.field_name}` ({row.combined_anchor_score_mean:.6f})" for row in top_cat.itertuples()))
        lines.append("- top continuous anchors: " + ", ".join(f"`{row.field_name}` ({row.combined_anchor_score_mean:.6f})" for row in top_cont.itertuples()))
        lines.append("")
    lines.append("## Dropped Fields")
    lines.append("")
    lines.append("| Field | Type | Reason |")
    lines.append("| --- | --- | --- |")
    for row in dropped.sort_values(["field_type", "field_name"]).itertuples():
        lines.append(f"| {row.field_name} | {row.field_type} | {row.drop_reason} |")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- If `view_label_9way` remains weaker than device / protocol fields, even the easier view task is not the dominant organizer.")
    lines.append("- If angle-derived fields dominate but device fields remain high, the representation is jointly view- and device-anchored.")
    lines.append("- Compare nuisance anchors against `view_label_9way`, `patient_id`, `study_id` before concluding that the task signal dominates the geometry.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_jobs(probe_seeds: Sequence[int]) -> List[JobDef]:
    jobs: List[JobDef] = []
    for seed in probe_seeds:
        for backbone_name in BACKBONES:
            jobs.append(JobDef(kind="analysis", backbone_name=backbone_name, target=VIEW_TARGET_NAME, seed=int(seed)))
    jobs.append(JobDef(kind="postprocess", name="aggregate_and_render"))
    jobs.append(JobDef(kind="postprocess", name="write_markdown_and_meta"))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global3-root", default="outputs/global_4_3_view_classification_unique_view")
    parser.add_argument("--image-root", default="input/Stent-Contrast-unique-view")
    parser.add_argument("--dcm-root", default="input/stent_split_dcm_unique_view")
    parser.add_argument("--output-root", default="outputs/global_4_3_view_classification_unique_view")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--knn-k", type=int, nargs="+", default=ga42.KNN_K_DEFAULT)
    parser.add_argument("--cluster-k", type=int, nargs="+", default=ga42.KMEANS_K_DEFAULT)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.output_root)
    ga42.ensure_dir(out_root)
    log_path, file_handle, original_stdout, original_stderr = ga42.setup_console_and_file_logging(
        out_root,
        args.log_file,
        default_prefix="global_4_3_cluster_anchoring_attribution",
    )

    try:
        ga42.set_global_seed(int(args.seed))
        ga42.log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        global3_root = Path(args.global3_root).resolve()
        image_root = Path(args.image_root).resolve()
        dcm_root = Path(args.dcm_root).resolve()
        device = ga42.resolve_device(args.device)
        knn_k_values = [int(v) for v in args.knn_k]
        cluster_k_values = [int(v) for v in args.cluster_k]

        manifest, full_manifest_hash, field_specs_df = build_anchor_manifest(
            global3_root=global3_root,
            image_root=image_root,
            dcm_root=dcm_root,
            output_root=out_root,
            max_images=args.max_images,
            seed=int(args.seed),
        )
        active_manifest_hash = ga42.hash_dataframe(
            manifest,
            ["image_id", "img_path", "patient_id", "study_id", "dicom_rel_path", "view_label_9way", "view_label_index"],
        )
        audit_df, usable_categorical, usable_continuous = ga42.audit_fields(manifest, field_specs_df)
        audit_df.to_csv(out_root / "summary_global_4_3_field_audit.csv", index=False)
        ga42.save_field_audit_figure(audit_df, out_root / "fig_global4_3_field_audit.png")
        ga42.log(
            f"Discovered fields | categorical={int((field_specs_df['field_type'] == 'categorical').sum())} "
            f"continuous={int((field_specs_df['field_type'] == 'continuous').sum())}"
        )
        ga42.log(f"Usable categorical fields: {usable_categorical}")
        ga42.log(f"Usable continuous fields: {usable_continuous}")

        field_group_map = dict(zip(field_specs_df["field_name"].tolist(), field_specs_df["field_group"].tolist()))
        for ref_name in REFERENCE_FIELDS:
            field_group_map[ref_name] = "reference"
        categorical_fields_for_scoring = list(dict.fromkeys(REFERENCE_FIELDS + usable_categorical))

        full_features: Dict[str, torch.Tensor] = {}
        feature_hashes: Dict[str, str] = {}
        global3_meta: Dict[str, Dict[str, object]] = {}
        for backbone_name in BACKBONES:
            features, meta = load_test_features(global3_root, backbone_name)
            full_features[backbone_name] = features
            feature_hashes[backbone_name] = ga42.make_feature_hash(features)
            global3_meta[backbone_name] = meta
            ga42.log(
                f"Loaded test features | backbone={backbone_name} | shape={tuple(features.shape)} | "
                f"feature_hash={feature_hashes[backbone_name][:12]}"
            )
        active_indices = manifest["image_id"].astype(int).to_numpy(dtype=np.int64)

        jobs = build_jobs(args.probe_seeds)
        tracker = ga42.FullRunTracker(jobs)

        neighborhood_categorical_raw_tables: List[pd.DataFrame] = []
        neighborhood_continuous_raw_tables: List[pd.DataFrame] = []
        cluster_categorical_raw_tables: List[pd.DataFrame] = []
        cluster_continuous_raw_tables: List[pd.DataFrame] = []
        anchor_rank_raw_tables: List[pd.DataFrame] = []
        seed11_embedding_map: Dict[Tuple[str, str], np.ndarray] = {}

        job_index = 0
        for seed in args.probe_seeds:
            for backbone_name in BACKBONES:
                tracker.start_job(job_index)
                tracker.update_phase("load_probe_embedding", 1, 6)
                full_probe_embeddings, _summary = apply_probe_checkpoint(
                    global3_root=global3_root,
                    backbone_name=backbone_name,
                    target=VIEW_TARGET_NAME,
                    seed=int(seed),
                    features=full_features[backbone_name],
                    manifest_hash=full_manifest_hash,
                    feature_hash=feature_hashes[backbone_name],
                    device=device,
                    batch_size=int(args.probe_batch_size),
                )
                active_probe_embeddings = full_probe_embeddings[active_indices].contiguous()
                active_np = active_probe_embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
                if int(seed) == 11:
                    seed11_embedding_map[(backbone_name, VIEW_TARGET_NAME)] = active_np

                tracker.update_phase("build_neighbors", 2, 6)
                sims = ga42.compute_similarity_matrix(active_probe_embeddings)
                topk_neighbors = ga42.build_topk_neighbors(sims, max(knn_k_values))

                tracker.update_phase("neighborhood_scores", 3, 6)
                neighborhood_categorical_raw_tables.append(
                    ga42.compute_categorical_neighborhood_rows(
                        topk_neighbors=topk_neighbors,
                        manifest=manifest,
                        field_names=categorical_fields_for_scoring,
                        backbone_name=backbone_name,
                        target=VIEW_TARGET_NAME,
                        seed=int(seed),
                        k_values=knn_k_values,
                        field_group_map=field_group_map,
                    )
                )
                neighborhood_continuous_raw_tables.append(
                    ga42.compute_continuous_neighborhood_rows(
                        topk_neighbors=topk_neighbors,
                        manifest=manifest,
                        field_names=usable_continuous,
                        backbone_name=backbone_name,
                        target=VIEW_TARGET_NAME,
                        seed=int(seed),
                        k_values=knn_k_values,
                        field_group_map=field_group_map,
                    )
                )

                tracker.update_phase("cluster_scores", 4, 6)
                cluster_categorical_raw_tables.append(
                    ga42.compute_categorical_cluster_rows(
                        embeddings=active_np,
                        manifest=manifest,
                        field_names=categorical_fields_for_scoring,
                        backbone_name=backbone_name,
                        target=VIEW_TARGET_NAME,
                        seed=int(seed),
                        cluster_k_values=cluster_k_values,
                        field_group_map=field_group_map,
                    )
                )
                cluster_continuous_raw_tables.append(
                    ga42.compute_continuous_cluster_rows(
                        embeddings=active_np,
                        manifest=manifest,
                        field_names=usable_continuous,
                        backbone_name=backbone_name,
                        target=VIEW_TARGET_NAME,
                        seed=int(seed),
                        cluster_k_values=cluster_k_values,
                        field_group_map=field_group_map,
                    )
                )

                tracker.update_phase("rank_attributes", 5, 6)
                neighborhood_cat_job = neighborhood_categorical_raw_tables[-1]
                neighborhood_cont_job = neighborhood_continuous_raw_tables[-1]
                cluster_cat_job = cluster_categorical_raw_tables[-1]
                cluster_cont_job = cluster_continuous_raw_tables[-1]
                anchor_rank_raw_tables.append(
                    ga42.compute_combined_anchor_rows(
                        neighborhood_df=neighborhood_cat_job,
                        cluster_df=cluster_cat_job,
                        field_type="categorical",
                        backbone_name=backbone_name,
                        target=VIEW_TARGET_NAME,
                        seed=int(seed),
                    )
                )
                anchor_rank_raw_tables.append(
                    ga42.compute_combined_anchor_rows(
                        neighborhood_df=neighborhood_cont_job,
                        cluster_df=cluster_cont_job,
                        field_type="continuous",
                        backbone_name=backbone_name,
                        target=VIEW_TARGET_NAME,
                        seed=int(seed),
                    )
                )
                tracker.update_phase("write_raw_rows", 6, 6)
                ga42.log(
                    f"Finished analysis job | backbone={backbone_name} target={VIEW_TARGET_NAME} seed={seed} | "
                    f"usable_cat={len(categorical_fields_for_scoring)} usable_cont={len(usable_continuous)}"
                )
                tracker.finish_job()
                job_index += 1

        tracker.start_job(job_index)
        tracker.update_phase("aggregate", 1, 2)
        neighborhood_categorical_raw_df = pd.concat(neighborhood_categorical_raw_tables, axis=0, ignore_index=True)
        neighborhood_continuous_raw_df = pd.concat(neighborhood_continuous_raw_tables, axis=0, ignore_index=True)
        cluster_categorical_raw_df = pd.concat(cluster_categorical_raw_tables, axis=0, ignore_index=True)
        cluster_continuous_raw_df = pd.concat(cluster_continuous_raw_tables, axis=0, ignore_index=True)
        anchor_rank_raw_df = pd.concat(anchor_rank_raw_tables, axis=0, ignore_index=True)

        neighborhood_categorical_df = ga42.aggregate_rows(
            neighborhood_categorical_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "k"],
            metric_cols=["num_valid_queries", "base_match_rate", "neighbor_match_rate", "purity_uplift"],
        )
        neighborhood_continuous_df = ga42.aggregate_rows(
            neighborhood_continuous_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "k"],
            metric_cols=["num_valid_queries", "global_abs_diff_mean", "neighbor_abs_diff_mean", "neighbor_consistency"],
        )
        cluster_categorical_df = ga42.aggregate_rows(
            cluster_categorical_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "kmeans_k"],
            metric_cols=["num_valid_samples", "nmi"],
        )
        cluster_continuous_df = ga42.aggregate_rows(
            cluster_continuous_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group", "kmeans_k"],
            metric_cols=["num_valid_samples", "eta_squared"],
        )
        anchor_rank_df = ga42.aggregate_rows(
            anchor_rank_raw_df,
            group_cols=["backbone_name", "target", "field_name", "field_type", "field_group"],
            metric_cols=["local_metric", "global_metric", "local_z", "global_z", "combined_anchor_score"],
        )
        anchor_rank_df = ga42.assign_anchor_ranks(anchor_rank_df)

        neighborhood_categorical_raw_df.to_csv(out_root / "summary_global_4_3_neighborhood_categorical_raw.csv", index=False)
        neighborhood_continuous_raw_df.to_csv(out_root / "summary_global_4_3_neighborhood_continuous_raw.csv", index=False)
        cluster_categorical_raw_df.to_csv(out_root / "summary_global_4_3_cluster_categorical_raw.csv", index=False)
        cluster_continuous_raw_df.to_csv(out_root / "summary_global_4_3_cluster_continuous_raw.csv", index=False)
        anchor_rank_raw_df.to_csv(out_root / "summary_global_4_3_anchor_rank_raw.csv", index=False)

        neighborhood_categorical_df.to_csv(out_root / "summary_global_4_3_neighborhood_categorical.csv", index=False)
        neighborhood_continuous_df.to_csv(out_root / "summary_global_4_3_neighborhood_continuous.csv", index=False)
        cluster_categorical_df.to_csv(out_root / "summary_global_4_3_cluster_categorical.csv", index=False)
        cluster_continuous_df.to_csv(out_root / "summary_global_4_3_cluster_continuous.csv", index=False)
        anchor_rank_df.to_csv(out_root / "summary_global_4_3_anchor_rank.csv", index=False)

        ga42.save_anchor_rank_figure(anchor_rank_df, "categorical", out_root / "fig_global4_3_anchor_rank_categorical.png", top_n=None)
        ga42.save_anchor_rank_target_figure(anchor_rank_df, "categorical", VIEW_TARGET_NAME, out_root / "fig_global4_3_anchor_rank_categorical_view_9way.png", top_n=None)
        ga42.save_anchor_rank_figure(anchor_rank_df, "continuous", out_root / "fig_global4_3_anchor_rank_continuous.png", top_n=None)
        ga42.save_anchor_rank_target_figure(anchor_rank_df, "continuous", VIEW_TARGET_NAME, out_root / "fig_global4_3_anchor_rank_continuous_view_9way.png", top_n=None)
        ga42.save_umap_overlay_top_categorical(
            aggregate_anchor_df=anchor_rank_df,
            embedding_map=seed11_embedding_map,
            manifest=manifest,
            output_path=out_root / "fig_global4_3_umap_overlay_top_categorical.png",
            seed=int(args.seed),
        )
        ga42.save_umap_overlay_top_continuous(
            aggregate_anchor_df=anchor_rank_df,
            embedding_map=seed11_embedding_map,
            manifest=manifest,
            output_path=out_root / "fig_global4_3_umap_overlay_top_continuous.png",
            seed=int(args.seed),
        )
        save_view_category_overlay_probe(
            embedding_map=seed11_embedding_map,
            manifest=manifest,
            output_path=out_root / "fig_global4_3_umap_overlay_view_category_probe.png",
            seed=int(args.seed),
        )
        ga42.export_umap_overlays_for_all_usable_fields(
            aggregate_anchor_df=anchor_rank_df,
            embedding_map=seed11_embedding_map,
            manifest=manifest,
            output_root=out_root / "umap_overlays_all_usable",
            seed=int(args.seed),
        )
        ga42.save_reference_vs_nuisance_compare(anchor_rank_df, out_root / "fig_global4_3_reference_vs_nuisance_compare.png")
        tracker.finish_job()
        job_index += 1

        tracker.start_job(job_index)
        tracker.update_phase("write_markdown", 1, 2)
        write_markdown(
            output_path=out_root / "analysis_global_4_3_cluster_anchoring_attribution.md",
            audit_df=audit_df,
            field_specs_df=field_specs_df,
            anchor_df=anchor_rank_df,
            field_audit_path=out_root / "summary_global_4_3_field_audit.csv",
            field_catalog_path=out_root / "summary_global_4_3_field_catalog.csv",
        )
        run_meta = {
            "global3_root": str(global3_root),
            "image_root": str(image_root),
            "dcm_root": str(dcm_root),
            "output_root": str(out_root.resolve()),
            "probe_seeds": [int(v) for v in args.probe_seeds],
            "targets": TARGETS,
            "knn_k": knn_k_values,
            "cluster_k": cluster_k_values,
            "max_images": args.max_images,
            "seed": int(args.seed),
            "device": str(device),
            "full_test_manifest_hash": full_manifest_hash,
            "active_manifest_hash": active_manifest_hash,
            "usable_categorical_fields": usable_categorical,
            "usable_continuous_fields": usable_continuous,
            "reference_fields": REFERENCE_FIELDS,
            "field_catalog_path": str((out_root / "summary_global_4_3_field_catalog.csv").resolve()),
            "num_discovered_fields": int(len(field_specs_df)),
            "feature_hashes": feature_hashes,
            "log_path": str(log_path.resolve()),
        }
        (out_root / "run_meta_global_4_3_cluster_anchoring.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        tracker.finish_job()

        ga42.log("Global Analysis 4-3 cluster anchoring attribution completed successfully.")
    finally:
        ga42.restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
