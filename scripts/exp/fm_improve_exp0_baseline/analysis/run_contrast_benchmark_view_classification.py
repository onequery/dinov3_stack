#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.global_analysis import global_4_3_view_classification as base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contrast Benchmark — View Classification")
    parser.add_argument("--dataset-root", default="input/contrast_benchmark/1_global/1_view_classification")
    parser.add_argument("--output-root", default="outputs/fm-imp-exp0_baseline/1_global/1_view_classification")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--imagenet-ckpt", default=base.DEFAULT_IMAGENET_CKPT)
    parser.add_argument("--cag-ckpt", default=base.DEFAULT_CAG_CKPT)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-size", type=int, default=480)
    parser.add_argument("--center-crop-size", type=int, default=448)
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--input-policy", choices=base.INPUT_POLICY_CHOICES, default=base.INPUT_POLICY_BASELINE)
    parser.add_argument("--input-stats-json", default=None)
    parser.add_argument("--probe-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-features", action="store_true")
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--strict-deterministic", action="store_true")
    parser.add_argument("--probe-lr-grid", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--probe-max-epoch", type=int, default=200)
    parser.add_argument("--probe-patience", type=int, default=20)
    parser.add_argument("--probe-min-delta", type=float, default=0.0)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-images-per-split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sort_view_labels(labels: Sequence[str]) -> List[str]:
    def key(label: str) -> tuple[int, str]:
        prefix, _, suffix = str(label).partition("_")
        try:
            return int(prefix), suffix
        except ValueError:
            return 9999, str(label)

    return sorted({str(v) for v in labels}, key=key)


def set_view_globals(view_labels: Sequence[str]) -> None:
    view_labels = list(view_labels)
    base.VIEW_LABELS_9WAY = view_labels
    base.VIEW_INDEX = {label: idx for idx, label in enumerate(view_labels)}
    cmap = [
        "#2166AC",
        "#4393C3",
        "#92C5DE",
        "#D1E5F0",
        "#F7F7F7",
        "#FDDBC7",
        "#F4A582",
        "#D6604D",
        "#B2182B",
    ]
    base.VIEW_COLORS = {label: cmap[idx % len(cmap)] for idx, label in enumerate(view_labels)}


def build_manifest(dataset_root: Path, max_images_per_split: int | None, seed: int) -> pd.DataFrame:
    image_paths = sorted([p for p in dataset_root.rglob("*") if p.suffix.lower() in base.IMAGE_EXTS])
    if not image_paths:
        raise FileNotFoundError(f"No images found under {dataset_root}")

    rows: List[Dict[str, object]] = []
    for image_path in image_paths:
        rel = image_path.resolve().relative_to(dataset_root.resolve())
        if len(rel.parts) < 6:
            raise ValueError(f"Unexpected dataset path: {image_path}")
        split, view_label, patient_id, study_id = rel.parts[:4]
        if split not in base.ALLOWED_SPLITS:
            raise ValueError(f"Unexpected split in path: {image_path}")
        rows.append(
            {
                "img_path": str(image_path.resolve()),
                "image_rel_path": str(rel),
                "split": split,
                "class_name": view_label,
                "patient_id": patient_id,
                "study_id": study_id,
                "dicom_rel_path": "",
                "PositionerPrimaryAngle": float("nan"),
                "PositionerSecondaryAngle": float("nan"),
                "view_horizontal_10deg": "",
                "view_vertical_10deg": "",
                "view_label_9way": view_label,
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["split", "class_name", "patient_id", "study_id", "img_path"]).reset_index(drop=True)
    view_labels = sort_view_labels(manifest["view_label_9way"].astype(str).tolist())
    set_view_globals(view_labels)
    manifest["view_label_index"] = manifest["view_label_9way"].map(base.VIEW_INDEX).astype(np.int64)

    sampled_parts: List[pd.DataFrame] = []
    for split_name in base.ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"No rows found for split={split_name}")
        if max_images_per_split is not None:
            split_df = base._balanced_downsample_split(split_df, int(max_images_per_split), seed + hash(split_name) % 1000)
        sampled_parts.append(split_df)
    manifest = pd.concat(sampled_parts, axis=0, ignore_index=True)
    manifest = manifest.sort_values(["split", "class_name", "patient_id", "study_id", "img_path"]).reset_index(drop=True)
    manifest.insert(0, "image_id", np.arange(len(manifest), dtype=np.int64))
    return manifest


def validate_manifest(manifest: pd.DataFrame, allow_missing_classes: bool) -> None:
    required = [
        "image_id",
        "img_path",
        "split",
        "class_name",
        "patient_id",
        "study_id",
        "view_label_9way",
        "view_label_index",
    ]
    missing = [column for column in required if column not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")
    for split_name in base.ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split_name]
        if split_df.empty:
            raise ValueError(f"No rows found for split={split_name}")
        present = set(split_df["view_label_9way"].astype(str).tolist())
        missing_classes = [label for label in base.VIEW_LABELS_9WAY if label not in present]
        if missing_classes and not allow_missing_classes:
            raise ValueError(f"Split {split_name} is missing classes: {missing_classes}")


def split_manifest_map(manifest: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    result: Dict[str, pd.DataFrame] = {}
    for split_name in base.ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"No rows found for split={split_name}")
        result[split_name] = split_df
    return result


def save_distribution_summary(manifest: pd.DataFrame, output_root: Path) -> None:
    rows: List[Dict[str, object]] = []
    for split_name in base.ALLOWED_SPLITS:
        split_df = manifest.loc[manifest["split"] == split_name]
        counts = split_df["view_label_9way"].value_counts().to_dict()
        for label in base.VIEW_LABELS_9WAY:
            rows.append(
                {
                    "split": split_name,
                    "view_label": label,
                    "count": int(counts.get(label, 0)),
                }
            )
    pd.DataFrame(rows).to_csv(output_root / "summary_view_class_distribution.csv", index=False)


def write_markdown_summary(output_path: Path, summary_df: pd.DataFrame, run_meta: Dict[str, object]) -> None:
    lines: List[str] = []
    lines.append("# Contrast Benchmark — View Classification")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- dataset root: `{run_meta['dataset_root']}`")
    lines.append(f"- output root: `{run_meta['output_root']}`")
    lines.append(f"- probe seeds: `{', '.join(str(v) for v in run_meta['probe_seed_set'])}`")
    lines.append(f"- input policy: `{run_meta['input_policy']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| mode | split | backbone | accuracy | macro_f1 | balanced_accuracy |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    display_df = summary_df.sort_values(["split", "mode", "backbone_name"]).reset_index(drop=True)
    for _, row in display_df.iterrows():
        lines.append(
            f"| {row['mode']} | {row['split']} | {row['backbone_name']} | "
            f"{float(row['accuracy_mean']):.4f} | {float(row['macro_f1_mean']):.4f} | {float(row['balanced_accuracy_mean']):.4f} |"
        )
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_steps(args: argparse.Namespace) -> List[base.StepDef]:
    steps: List[base.StepDef] = [base.StepDef("Build benchmark manifests", "short")]
    for backbone_name in ["imagenet", "cag"]:
        for split_name in base.ALLOWED_SPLITS:
            steps.append(base.StepDef(f"Extract features | {backbone_name} | {split_name}", "short"))
    for backbone_name in ["imagenet", "cag"]:
        steps.append(base.StepDef(f"Run raw evaluation | {backbone_name}", "short"))
    for seed in args.probe_seeds:
        for backbone_name in ["imagenet", "cag"]:
            steps.append(base.StepDef(f"Run probe evaluation | {backbone_name} | seed={seed}", "probe"))
    steps.extend(
        [
            base.StepDef("Aggregate summaries", "short"),
            base.StepDef("Render figures", "short"),
            base.StepDef("Write markdown", "short"),
            base.StepDef("Write run meta", "short"),
        ]
    )
    return steps


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    base.ensure_dir(out_root)
    log_path, file_handle, original_stdout, original_stderr = base.setup_console_and_file_logging(
        out_root,
        args.log_file,
        default_prefix="contrast_benchmark_view_classification",
    )

    try:
        base.set_global_seed(args.seed, strict_deterministic=args.strict_deterministic)
        policy_spec = base.resolve_input_policy(args.input_policy, args.input_stats_json)
        base.log(f"Arguments: {json.dumps(vars(args), indent=2, sort_keys=True)}")
        dataset_root = Path(args.dataset_root).resolve()
        steps = build_steps(args)
        run_tracker = base.AnalysisRunTracker(steps)
        step_idx = 0

        run_tracker.start_step(step_idx)
        manifest = build_manifest(dataset_root, args.max_images_per_split, args.seed)
        validate_manifest(manifest, allow_missing_classes=args.max_images_per_split is not None)
        split_manifests = split_manifest_map(manifest)
        for split_name, split_df in split_manifests.items():
            split_df.to_csv(out_root / f"image_manifest_{split_name}.csv", index=False)
        save_distribution_summary(manifest, out_root)
        manifest_hashes = {
            split_name: base.hash_dataframe(
                split_df,
                ["image_id", "img_path", "patient_id", "study_id", "view_label_9way", "view_label_index"],
            )
            for split_name, split_df in split_manifests.items()
        }
        (out_root / "view_labels.json").write_text(json.dumps(base.VIEW_LABELS_9WAY, indent=2), encoding="utf-8")
        run_tracker.finish_step()
        step_idx += 1

        backbone_ckpts = {"imagenet": args.imagenet_ckpt, "cag": args.cag_ckpt}
        feature_store: Dict[str, Dict[str, torch.Tensor]] = {"imagenet": {}, "cag": {}}
        feature_hashes: Dict[str, Dict[str, str]] = {"imagenet": {}, "cag": {}}
        for backbone_name in ["imagenet", "cag"]:
            for split_name in base.ALLOWED_SPLITS:
                run_tracker.start_step(step_idx)
                features = base.extract_or_load_features(
                    manifest=split_manifests[split_name],
                    split_name=split_name,
                    backbone_name=backbone_name,
                    ckpt_path=backbone_ckpts[backbone_name],
                    args=args,
                    out_root=out_root,
                )
                feature_store[backbone_name][split_name] = features
                feature_hashes[backbone_name][split_name] = base.make_feature_hash(features)
                run_tracker.finish_step()
                step_idx += 1

        raw_metric_rows: List[Dict[str, object]] = []
        train_labels = split_manifests["train"]["view_label_index"].to_numpy(dtype=np.int64)
        for backbone_name in ["imagenet", "cag"]:
            run_tracker.start_step(step_idx)
            centroids, centroid_labels = base.build_nearest_centroids(feature_store[backbone_name]["train"], train_labels)
            for split_name in ["valid", "test"]:
                split_df = split_manifests[split_name]
                y_true = split_df["view_label_index"].to_numpy(dtype=np.int64)
                y_pred, best_scores = base.predict_nearest_centroid(feature_store[backbone_name][split_name], centroids, centroid_labels)
                metrics = base.compute_classification_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    mode="raw_frozen",
                    target="view_label_9way",
                    backbone_name=backbone_name,
                    split_name=split_name,
                    seed=None,
                    best_lr=None,
                    epochs_trained=None,
                )
                raw_metric_rows.append(metrics.to_row())
                pred_df = base.build_prediction_df(
                    split_df,
                    y_true,
                    y_pred,
                    best_scores,
                    mode="raw_frozen",
                    backbone_name=backbone_name,
                    split_name=split_name,
                    seed=None,
                )
                pred_df.to_csv(out_root / f"per_image_view_raw_{backbone_name}_{split_name}.csv", index=False)
                if split_name == "test":
                    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(base.VIEW_LABELS_9WAY))))
                    base.save_confusion_figure(cm, out_root / f"fig_view_confusion_raw_{backbone_name}_{split_name}.png", f"raw | {backbone_name} | {split_name}")
            run_tracker.finish_step()
            step_idx += 1

        probe_metric_rows: List[Dict[str, object]] = []
        probe_lr_search_results: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
        for seed in args.probe_seeds:
            for backbone_name in ["imagenet", "cag"]:
                probe_lr_search_results.setdefault(str(seed), {})
                run_tracker.start_step(step_idx)
                tracker = base.ProbeEpochTracker(
                    run_tracker=run_tracker,
                    backbone_name=backbone_name,
                    target="view_label_9way",
                    seed=seed,
                    lr_values=args.probe_lr_grid,
                    max_epoch=args.probe_max_epoch,
                )
                model, best_summary, lr_rows = base.train_probe_with_lr_search(
                    backbone_name=backbone_name,
                    target="view_label_9way",
                    seed=seed,
                    train_features=feature_store[backbone_name]["train"],
                    valid_features=feature_store[backbone_name]["valid"],
                    train_labels=split_manifests["train"]["view_label_index"].to_numpy(dtype=np.int64),
                    valid_labels=split_manifests["valid"]["view_label_index"].to_numpy(dtype=np.int64),
                    out_root=out_root,
                    args=args,
                    tracker=tracker,
                    feature_hashes=feature_hashes[backbone_name],
                    manifest_hashes=manifest_hashes,
                )
                probe_lr_search_results[str(seed)][backbone_name] = lr_rows
                device = torch.device(args.device if args.device and torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
                model = model.to(device)
                for split_name in ["valid", "test"]:
                    split_df = split_manifests[split_name]
                    y_true = split_df["view_label_index"].to_numpy(dtype=np.int64)
                    probe_embeddings, probe_logits = base.evaluate_probe_embeddings_and_logits(
                        model,
                        feature_store[backbone_name][split_name],
                        int(args.probe_batch_size),
                        device,
                    )
                    y_pred = probe_logits.argmax(dim=1).cpu().numpy().astype(np.int64)
                    scores = probe_logits.max(dim=1).values.cpu().numpy().astype(np.float64)
                    metrics = base.compute_classification_metrics(
                        y_true=y_true,
                        y_pred=y_pred,
                        mode="probe_linear",
                        target="view_label_9way",
                        backbone_name=backbone_name,
                        split_name=split_name,
                        seed=seed,
                        best_lr=float(best_summary["lr"]),
                        epochs_trained=int(best_summary["epochs_trained"]),
                    )
                    probe_metric_rows.append(metrics.to_row())
                    pred_df = base.build_prediction_df(
                        split_df,
                        y_true,
                        y_pred,
                        scores,
                        mode="probe_linear",
                        backbone_name=backbone_name,
                        split_name=split_name,
                        seed=seed,
                    )
                    pred_df.to_csv(out_root / f"per_image_view_probe_seed{seed}_{backbone_name}_{split_name}.csv", index=False)
                    if split_name == "test":
                        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(base.VIEW_LABELS_9WAY))))
                        base.save_confusion_figure(cm, out_root / f"fig_view_confusion_probe_seed{seed}_{backbone_name}_{split_name}.png", f"probe seed{seed} | {backbone_name} | {split_name}")
                run_tracker.finish_step()
                step_idx += 1

        run_tracker.start_step(step_idx)
        raw_summary_df = pd.DataFrame(raw_metric_rows)
        probe_raw_df = pd.DataFrame(probe_metric_rows)
        raw_aggregate_df = base.summarize_raw_metrics(raw_summary_df)
        probe_aggregate_df = base.summarize_probe_metrics(probe_raw_df)
        summary_df = pd.concat([raw_aggregate_df, probe_aggregate_df], axis=0, ignore_index=True)
        raw_summary_df.to_csv(out_root / "summary_view_classification_raw.csv", index=False)
        probe_raw_df.to_csv(out_root / "summary_view_classification_probe_raw.csv", index=False)
        summary_df.to_csv(out_root / "summary_view_classification.csv", index=False)
        (out_root / "probe_lr_search_results.json").write_text(json.dumps(probe_lr_search_results, indent=2), encoding="utf-8")
        run_tracker.finish_step()
        step_idx += 1

        run_tracker.start_step(step_idx)
        base.save_metric_compare_figure(summary_df, "accuracy", "Accuracy", out_root / "fig_view_classification_accuracy_compare.png")
        base.save_metric_compare_figure(summary_df, "macro_f1", "Macro-F1", out_root / "fig_view_classification_macro_f1_compare.png")
        base.save_metric_compare_figure(summary_df, "balanced_accuracy", "Balanced Accuracy", out_root / "fig_view_classification_balanced_accuracy_compare.png")
        run_tracker.finish_step()
        step_idx += 1

        run_meta = {
            "analysis": "contrast_benchmark_view_classification",
            "dataset_root": str(dataset_root),
            "output_root": str(out_root),
            "log_path": str(log_path.resolve()),
            "probe_seed_set": [int(v) for v in args.probe_seeds],
            "feature_source": "x_norm_clstoken",
            "view_labels": list(base.VIEW_LABELS_9WAY),
            "aggregate_policy": "probe=mean+-std, raw=single-run",
            "selected_targets": ["view_label_9way"],
            "manifest_hashes": manifest_hashes,
            "feature_hashes": feature_hashes,
            "aggregate_summary_path": str((out_root / "summary_view_classification.csv").resolve()),
            **policy_spec.to_meta(),
            "step_count": len(steps),
            "steps": [{"name": step.name, "kind": step.kind} for step in steps],
        }

        run_tracker.start_step(step_idx)
        write_markdown_summary(out_root / "analysis_view_classification.md", summary_df, run_meta)
        run_tracker.finish_step()
        step_idx += 1

        run_tracker.start_step(step_idx)
        (out_root / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        run_tracker.finish_step()
        base.log("Contrast benchmark view classification completed successfully.")
    finally:
        base.restore_console_logging(file_handle, original_stdout, original_stderr)


if __name__ == "__main__":
    main()
