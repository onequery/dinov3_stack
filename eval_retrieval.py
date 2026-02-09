#!/usr/bin/env python3
"""
Patient-ID Image–Image Retrieval Evaluation (DINOv3)
"""

import argparse
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from train_retrieval import load_backbone_from_local_checkpoint


# -----------------------------
# Utils
# -----------------------------
def collect_image_paths(
    root_dir: str, exts: Tuple[str, ...] = ("png", "jpg", "jpeg")
) -> List[str]:
    paths: List[str] = []
    for ext in exts:
        paths.extend(
            glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        )
    return sorted(paths)


_PATIENT_RE = re.compile(r"(^|[\\/])(\d{8})(?=([\\/]|$))")


def extract_patient_id(path: str) -> str:
    m = _PATIENT_RE.search(path)
    if not m:
        raise ValueError(
            f"[ERROR] Could not find 8-digit patient_id folder in path:\n{path}"
        )
    return m.group(2)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


class Dinov3Retrieval(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        backbone_embed_dim: int,
        retrieval_embedding_dim: int = 128,
    ):
        super().__init__()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(backbone_embed_dim, backbone_embed_dim),
            nn.BatchNorm1d(backbone_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_embed_dim, retrieval_embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        z = self.projector(feat)
        return l2_normalize(z)


class BackboneRetrieval(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.backbone(x))


def load_retrieval_model(
    repo_dir: str,
    model_name: str,
    weights_path: Optional[str],
    proj_dim: int,
) -> Tuple[nn.Module, str, int]:
    if weights_path is None:
        backbone = torch.hub.load(repo_dir, model_name, source="local")
        backbone_embed_dim = backbone.norm.normalized_shape[0]
        return BackboneRetrieval(backbone), "backbone_only_default", backbone_embed_dim

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Invalid checkpoint format. Expected a state_dict dictionary.")

    has_backbone_prefix = any(k.startswith("backbone.") for k in state_dict.keys())
    has_projector_prefix = any(k.startswith("projector.") for k in state_dict.keys())

    if has_backbone_prefix and has_projector_prefix:
        backbone, load_report = load_backbone_from_local_checkpoint(
            repo_dir=repo_dir,
            model_name=model_name,
            weights_path=weights_path,
            device="cpu",
        )
        backbone_embed_dim = backbone.norm.normalized_shape[0]
        model = Dinov3Retrieval(
            backbone,
            backbone_embed_dim=backbone_embed_dim,
            retrieval_embedding_dim=proj_dim,
        )
        model.load_state_dict(state_dict, strict=True)
        return (
            model,
            f"retrieval_with_projector:{load_report['builder']}",
            backbone_embed_dim,
        )

    backbone, load_report = load_backbone_from_local_checkpoint(
        repo_dir=repo_dir,
        model_name=model_name,
        weights_path=weights_path,
        device="cpu",
    )
    backbone_embed_dim = backbone.norm.normalized_shape[0]
    return (
        BackboneRetrieval(backbone),
        f"backbone_only:{load_report['builder']}",
        backbone_embed_dim,
    )


def build_transform(resize_size: int, center_crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop((center_crop_size, center_crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--repo-dir", default="dinov3")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--out-dir", default="outputs/eval_retrieval/patient_retrieval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--max-topk-log", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    resize_size = int(cfg["RESIZE_SIZE"])
    center_crop_size = int(cfg["CENTER_CROP_SIZE"])
    transform = build_transform(resize_size, center_crop_size)

    # Collect images
    all_paths = collect_image_paths(args.input)
    if len(all_paths) == 0:
        raise ValueError(f"No images found under: {args.input}")
    patient_ids = np.array([extract_patient_id(p) for p in all_paths])

    patient_to_indices: Dict[str, np.ndarray] = {
        pid: np.where(patient_ids == pid)[0] for pid in np.unique(patient_ids)
    }

    # Load retrieval model
    model, embedding_source, backbone_embed_dim = load_retrieval_model(
        args.repo_dir, args.model_name, args.weights, args.proj_dim
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Feature extraction
    feats_list: List[torch.Tensor] = []
    bs = max(1, args.batch_size)

    with torch.no_grad():
        for start in tqdm(range(0, len(all_paths), bs), desc="Extracting"):
            batch_imgs = []
            for p in all_paths[start : start + bs]:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                batch_imgs.append(transform(img))
            x = torch.stack(batch_imgs).to(device)
            emb = model(x).cpu()
            feats_list.append(emb)

    retrieval_features = torch.cat(feats_list, dim=0)
    n, retrieval_embedding_dim = retrieval_features.shape

    # Retrieval evaluation
    K_LIST = sorted(set(int(k) for k in args.k))
    recall_hits = {k: [] for k in K_LIST}
    ap_list: List[float] = []
    ap_valid_mask: List[bool] = []

    per_query_rows = []

    first_positive_ranks: List[int] = []
    positive_sims: List[float] = []
    negative_sims: List[float] = []

    for i in tqdm(range(n), desc="Queries"):
        pid = patient_ids[i]
        pos_idx = patient_to_indices[pid]
        pos_idx = pos_idx[pos_idx != i]

        sims = torch.mv(retrieval_features, retrieval_features[i])
        sims[i] = -1e9
        sorted_idx = torch.argsort(sims, descending=True)

        for k in K_LIST:
            hit = int(
                len(pos_idx) > 0 and np.isin(sorted_idx[:k].numpy(), pos_idx).any()
            )
            recall_hits[k].append(hit)

        if len(pos_idx) == 0:
            ap = np.nan
            ap_valid = False
        else:
            y_true = np.zeros(n, dtype=np.int32)
            y_true[pos_idx] = 1
            ap = float(average_precision_score(y_true, sims.numpy()))
            ap_valid = True
            ap_list.append(ap)

            for rank, idx in enumerate(sorted_idx.tolist(), start=1):
                if idx in pos_idx:
                    first_positive_ranks.append(rank)
                    break

            positive_sims.extend(sims[pos_idx].tolist())
            neg_mask = patient_ids != pid
            negative_sims.extend(sims[neg_mask].tolist())

        ap_valid_mask.append(ap_valid)

        nn_idx = sorted_idx[: args.max_topk_log].numpy()
        nn_paths = [all_paths[j] for j in nn_idx]
        nn_pids = [patient_ids[j] for j in nn_idx]
        nn_sims = [float(sims[j].item()) for j in nn_idx]
        nn_is_pos = [int(patient_ids[j] == pid) for j in nn_idx]

        per_query_rows.append(
            {
                "query_index": i,
                "query_path": all_paths[i],
                "patient_id": pid,
                "num_positives": int(len(patient_to_indices[pid]) - 1),
                "AP": ap,
                **{f"R@{k}": recall_hits[k][-1] for k in K_LIST},
                "top1_path": nn_paths[0] if len(nn_paths) > 0 else "",
                "top1_patient_id": nn_pids[0] if len(nn_pids) > 0 else "",
                "top1_sim": nn_sims[0] if len(nn_sims) > 0 else np.nan,
                "top1_is_positive": nn_is_pos[0] if len(nn_is_pos) > 0 else 0,
                "nn_paths": " | ".join(nn_paths),
                "nn_patient_ids": " | ".join(nn_pids),
                "nn_sims": " | ".join([f"{v:.6f}" for v in nn_sims]),
                "nn_is_positive": " | ".join([str(v) for v in nn_is_pos]),
            }
        )

    results = {f"Recall@{k}": float(np.mean(recall_hits[k])) for k in K_LIST}
    results["mAP"] = float(np.mean(ap_list)) if len(ap_list) > 0 else float("nan")
    num_no_pos = int(np.sum(np.array(ap_valid_mask) == 0))

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = os.path.join(args.out_dir, f"retrieval_result_{timestamp}.txt")
    per_query_csv = os.path.join(args.out_dir, f"per_query_{timestamp}.csv")

    with open(summary_path, "w") as f:
        f.write("========== Patient Retrieval Evaluation ==========\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Repo: {args.repo_dir}\n")
        f.write(f"Weights: {args.weights if args.weights else '(torch.hub default)'}\n")
        f.write(f"Embedding source: {embedding_source}\n")
        f.write(f"Backbone embed dim: {backbone_embed_dim}\n")
        f.write(f"Retrieval embedding dim: {retrieval_embedding_dim}\n")
        f.write(f"Projection dim arg: {args.proj_dim}\n")
        f.write(f"Dataset: {args.input}\n")
        f.write(f"N images: {n}\n")
        f.write(f"Unique patients: {len(np.unique(patient_ids))}\n")
        f.write(f"Queries with no positives: {num_no_pos}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    pd.DataFrame(per_query_rows).to_csv(per_query_csv, index=False)

    # ===== Per-query Retrieval Rank Plot (CDF) =====
    plt.figure(figsize=(8, 6))
    if len(first_positive_ranks) > 0:
        ranks = np.array(first_positive_ranks)
        max_rank = max(1, int(np.percentile(ranks, 95)))
        xs = np.arange(1, max_rank + 1)
        ys = [(ranks <= x).mean() for x in xs]
        plt.plot(xs, ys, linewidth=2)
        plt.xlabel("First Positive Rank")
        plt.ylabel("Query Ratio (CDF)")
        plt.title("Per-query Retrieval Rank (CDF)")
    else:
        plt.text(0.5, 0.5, "No positive pairs found", ha="center", va="center")
        plt.axis("off")
        plt.title("Per-query Retrieval Rank (CDF)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    cdf_path = os.path.join(
        args.out_dir, f"per_query_retrieval_rank_cdf_{timestamp}.png"
    )
    plt.savefig(cdf_path, dpi=300)
    plt.close()

    # ===== Positive vs Negative Similarity Distribution =====
    plt.figure(figsize=(8, 6))
    plt.hist(positive_sims, bins=100, density=True, alpha=0.6, label="Positive")
    plt.hist(negative_sims, bins=100, density=True, alpha=0.6, label="Negative")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Positive vs Negative Similarity Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    simdist_path = os.path.join(
        args.out_dir, f"positive_vs_negative_similarity_{timestamp}.png"
    )
    plt.savefig(simdist_path, dpi=300)
    plt.close()

    # ===== Query-wise AP Distribution =====
    plt.figure(figsize=(8, 6))
    plt.hist(ap_list, bins=30, density=True, alpha=0.75)
    plt.xlabel("Average Precision (AP)")
    plt.ylabel("Density")
    plt.title("Query-wise AP Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    apdist_path = os.path.join(
        args.out_dir, f"query_wise_ap_distribution_{timestamp}.png"
    )
    plt.savefig(apdist_path, dpi=300)
    plt.close()

    print("\n========== Retrieval Results ==========")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("\n[Saved]")
    print(f"- {summary_path}")
    print(f"- {per_query_csv}")
    print(f"- {cdf_path}")
    print(f"- {simdist_path}")
    print(f"- {apdist_path}")


if __name__ == "__main__":
    main()
