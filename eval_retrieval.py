#!/usr/bin/env python3
"""
Patient-ID Image–Image Retrieval Evaluation
(using trained DINOv3 + projection head)
"""

import argparse
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import average_precision_score
from tqdm import tqdm

# -----------------------------
# Utils
# -----------------------------
_PATIENT_RE = re.compile(r"(^|[\\/])(\d{8})(?=([\\/]|$))")


def extract_patient_id(path: str) -> str:
    m = _PATIENT_RE.search(path)
    if not m:
        raise ValueError(f"Cannot extract patient id from: {path}")
    return m.group(2)


def collect_image_paths(
    root_dir: str, exts: Tuple[str, ...] = ("png", "jpg", "jpeg")
) -> List[str]:
    paths = []
    for ext in exts:
        paths.extend(
            glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        )
    return sorted(paths)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


def build_transform(resize_size: int, center_crop_size: int):
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
# Model
# -----------------------------
class Dinov3Retrieval(nn.Module):
    def __init__(self, model_name, feat_dim=384, proj_dim=128):
        super().__init__()
        repo_dir = "dinov3"
        self.backbone = torch.hub.load(repo_dir, model_name, source="local")

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        z = self.projector(feat)
        return l2_normalize(z)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--weights", required=True, help="best_model.pth")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load config
    # -----------------------------
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    transform = build_transform(cfg["RESIZE_SIZE"], cfg["CENTER_CROP_SIZE"])

    # -----------------------------
    # Load images
    # -----------------------------
    all_paths = collect_image_paths(args.input)
    patient_ids = np.array([extract_patient_id(p) for p in all_paths])

    patient_to_indices: Dict[str, np.ndarray] = {
        pid: np.where(patient_ids == pid)[0] for pid in np.unique(patient_ids)
    }

    # -----------------------------
    # Load trained model
    # -----------------------------
    ckpt = torch.load(args.weights, map_location=device)
    model = Dinov3Retrieval(model_name=args.model_name).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    # -----------------------------
    # Feature extraction
    # -----------------------------
    feats_list = []
    bs = max(1, args.batch_size)

    with torch.no_grad():
        for start in tqdm(range(0, len(all_paths), bs), desc="Extracting"):
            imgs = []
            for p in all_paths[start : start + bs]:
                img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                imgs.append(transform(img))
            x = torch.stack(imgs).to(device)
            emb = model(x).cpu()
            feats_list.append(emb)

    features = torch.cat(feats_list, dim=0)
    n, d = features.shape

    # -----------------------------
    # Retrieval evaluation
    # -----------------------------
    K_LIST = sorted(set(args.k))
    recall_hits = {k: [] for k in K_LIST}
    ap_list = []
    ap_valid_mask = []

    first_positive_ranks = []
    positive_sims, negative_sims = [], []

    per_query_rows = []
    topk_rows = []

    for i in tqdm(range(n), desc="Queries"):
        pid = patient_ids[i]
        pos_idx = patient_to_indices[pid]
        pos_idx = pos_idx[pos_idx != i]

        sims = torch.mv(features, features[i])
        sims[i] = -1e9
        sorted_idx = torch.argsort(sims, descending=True)

        query_recalls = {}

        for k in K_LIST:
            hit = int(
                len(pos_idx) > 0 and np.isin(sorted_idx[:k].numpy(), pos_idx).any()
            )
            recall_hits[k].append(hit)
            query_recalls[f"R@{k}"] = hit

        if len(pos_idx) == 0:
            ap_valid_mask.append(False)
            per_query_rows.append(
                dict(
                    query_index=i,
                    query_path=all_paths[i],
                    patient_id=pid,
                    num_positives=0,
                    first_positive_rank=None,
                    AP=None,
                    **query_recalls,
                    top1_path=None,
                    top1_pid=None,
                    top1_sim=None,
                    top1_is_positive=None,
                )
            )
            continue

        ap_valid_mask.append(True)

        y_true = np.zeros(n, dtype=np.int32)
        y_true[pos_idx] = 1
        ap = float(average_precision_score(y_true, sims.numpy()))
        ap_list.append(ap)

        first_rank = None
        for rank, idx in enumerate(sorted_idx.tolist(), start=1):
            if idx in pos_idx:
                first_positive_ranks.append(rank)
                first_rank = rank
                break

        for r, j in enumerate(sorted_idx[: max(K_LIST)].tolist(), start=1):
            topk_rows.append(
                dict(
                    query_index=i,
                    query_path=all_paths[i],
                    query_patient_id=pid,
                    rank=r,
                    retrieved_path=all_paths[j],
                    retrieved_patient_id=patient_ids[j],
                    similarity=float(sims[j]),
                    is_positive=int(j in pos_idx),
                )
            )

        top1 = sorted_idx[0].item()
        per_query_rows.append(
            dict(
                query_index=i,
                query_path=all_paths[i],
                patient_id=pid,
                num_positives=len(pos_idx),
                first_positive_rank=first_rank,
                AP=ap,
                **query_recalls,
                top1_path=all_paths[top1],
                top1_pid=patient_ids[top1],
                top1_sim=float(sims[top1]),
                top1_is_positive=int(top1 in pos_idx),
            )
        )

        positive_sims.extend(sims[pos_idx].tolist())
        negative_sims.extend(sims[patient_ids != pid].tolist())

    # -----------------------------
    # Save results & plots
    # -----------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    pd.DataFrame(per_query_rows).to_csv(
        os.path.join(args.out_dir, f"per_query_{timestamp}.csv"), index=False
    )
    pd.DataFrame(topk_rows).to_csv(
        os.path.join(args.out_dir, f"topk_retrieval_{timestamp}.csv"), index=False
    )

    results = {f"Recall@{k}": float(np.mean(recall_hits[k])) for k in K_LIST}
    results["mAP"] = float(np.mean(ap_list)) if len(ap_list) > 0 else float("nan")

    num_no_pos = int(np.sum(np.array(ap_valid_mask) == 0))

    # ----- Summary -----
    with open(os.path.join(args.out_dir, f"summary_{timestamp}.txt"), "w") as f:
        f.write("========== Patient Retrieval Evaluation ==========\n")
        f.write(f"Model: {args.model_name}\n")
        f.write("Repo: dinov3\n")
        f.write(f"Weights: {args.weights}\n")
        f.write(f"Dataset: {args.input}\n")
        f.write(f"N images: {n}\n")
        f.write(f"Embedding dim: {d}\n")
        f.write(f"Unique patients: {len(np.unique(patient_ids))}\n")
        f.write(f"Queries with no positives: {num_no_pos}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    # ----- Per-query Retrieval Rank CDF -----
    ranks = np.array(first_positive_ranks)
    max_rank = int(np.percentile(ranks, 95))
    xs = np.arange(1, max_rank + 1)
    ys = [(ranks <= x).mean() for x in xs]

    plt.figure(figsize=(8, 6))
    plt.plot(xs, ys, linewidth=2)
    plt.xlabel("First Positive Rank")
    plt.ylabel("Query Ratio (CDF)")
    plt.title("Per-query Retrieval Rank (CDF)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"rank_cdf_{timestamp}.png"), dpi=300)
    plt.close()

    # ----- Positive vs Negative Similarity -----
    plt.figure(figsize=(8, 6))
    plt.hist(positive_sims, bins=100, density=True, alpha=0.6, label="Positive")
    plt.hist(negative_sims, bins=100, density=True, alpha=0.6, label="Negative")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    plt.title("Positive vs Negative Similarity Distribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"sim_dist_{timestamp}.png"), dpi=300)
    plt.close()

    # ----- Query-wise AP Distribution -----
    plt.figure(figsize=(8, 6))
    plt.hist(ap_list, bins=30, density=True, alpha=0.75)
    plt.xlabel("Average Precision (AP)")
    plt.ylabel("Density")
    plt.title("Query-wise AP Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"ap_dist_{timestamp}.png"), dpi=300)
    plt.close()

    print("\n========== Retrieval Results ==========")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"[Saved] summary + CDF + sim dist + AP dist → {args.out_dir}")


if __name__ == "__main__":
    main()
