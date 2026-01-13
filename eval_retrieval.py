#!/usr/bin/env python3
"""
Patient-ID Image–Image Retrieval Evaluation (DINOv3)

Task:
- Query: each image
- Gallery: all other images
- Positive: images with the same patient_id (8-digit folder name), excluding itself
- Ignore: stent/no_stent, date, modality folders

Protocol:
- Frozen DINOv3 backbone
- Global embedding = model(x) output (e.g., CLS pooled embedding)
- L2 normalization
- Cosine similarity via dot product

Metrics:
- Recall@K
- mAP (queries with >=1 positive only)

USAGE:
python eval_retrieval.py \
  --input input/stent_split_img/test \
  --config configs_retrieval/patients.yaml \
  --model-name dinov3_vits16 \
  --repo-dir dinov3 \
  --backbone-weights /path/to/dinov3_vits16_pretrain_*.pth \
  --out-dir outputs/eval/pat_ret

Notes:
- If --backbone-weights is omitted, torch.hub may try to download weights depending on your hub entry;
  in restricted environments this can fail (e.g., HTTP 403). Prefer providing local weights.
"""

import argparse
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import average_precision_score
from tqdm import tqdm


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
    """
    Find an 8-digit directory name anywhere in the path.
    Example:
      input/stent_split_img/test/no_stent/10003493/20060405/XA/004.png -> 10003493
    """
    m = _PATIENT_RE.search(path)
    if not m:
        raise ValueError(
            f"[ERROR] Could not find 8-digit patient_id folder in path:\n  {path}\n"
            "Expected a directory named like '10003493' somewhere in the path."
        )
    return m.group(2)


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=1)


def load_dinov3_backbone(
    repo_dir: str, model_name: str, weights_path: Optional[str]
) -> torch.nn.Module:
    """
    Loads DINOv3 backbone via torch.hub.load(source='local').
    If weights_path is provided, it should point to a local .pth file.
    """
    if weights_path is not None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"[ERROR] --backbone-weights not found: {weights_path}"
            )
        print(f"[INFO] Loading backbone with local weights: {weights_path}")
        model = torch.hub.load(
            repo_dir, model_name, source="local", weights=weights_path
        )
    else:
        print(
            "[WARN] No --backbone-weights provided. torch.hub may attempt to download weights."
        )
        model = torch.hub.load(repo_dir, model_name, source="local")
    return model


def build_transform(resize_size: int, center_crop_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop((center_crop_size, center_crop_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="dataset root (contains stent/no_stent subfolders)",
    )
    parser.add_argument(
        "--config", required=True, help="yaml with RESIZE_SIZE and CENTER_CROP_SIZE"
    )
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument(
        "--repo-dir",
        default="dinov3",
        help="path to cloned dinov3 repo for torch.hub.load",
    )
    parser.add_argument(
        "--backbone-weights",
        default=None,
        help="local .pth for backbone weights (recommended)",
    )
    parser.add_argument("--out-dir", default="outputs/eval_retrieval/patient_retrieval")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 5, 10], help="Recall@K list"
    )
    parser.add_argument(
        "--max-topk-log",
        type=int,
        default=10,
        help="how many top neighbors to log per query",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if "RESIZE_SIZE" not in cfg or "CENTER_CROP_SIZE" not in cfg:
        raise ValueError(
            "[ERROR] config yaml must contain RESIZE_SIZE and CENTER_CROP_SIZE"
        )

    resize_size = int(cfg["RESIZE_SIZE"])
    center_crop_size = int(cfg["CENTER_CROP_SIZE"])
    transform = build_transform(resize_size, center_crop_size)

    # Collect images and labels (patient_id)
    all_paths = collect_image_paths(args.input, exts=("png", "jpg", "jpeg"))
    if len(all_paths) == 0:
        raise RuntimeError(f"[FATAL] No images found under: {args.input}")

    patient_ids = []
    for p in all_paths:
        patient_ids.append(extract_patient_id(p))
    patient_ids = np.array(patient_ids)

    # Build patient -> indices for positives
    patient_to_indices: Dict[str, np.ndarray] = {}
    for pid in np.unique(patient_ids):
        patient_to_indices[pid] = np.where(patient_ids == pid)[0]

    # Load backbone
    backbone = load_dinov3_backbone(
        args.repo_dir, args.model_name, args.backbone_weights
    ).to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # Feature extraction (batched)
    print("\n[INFO] Extracting features...\n")
    feats_list: List[torch.Tensor] = []

    bs = max(1, int(args.batch_size))
    with torch.no_grad():
        for start in tqdm(range(0, len(all_paths), bs), desc="Batches"):
            batch_paths = all_paths[start : start + bs]
            batch_imgs = []
            for p in batch_paths:
                img = cv2.imread(p)
                if img is None:
                    raise RuntimeError(f"[ERROR] Failed to read image: {p}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_imgs.append(transform(img))
            x = torch.stack(batch_imgs, dim=0).to(device)  # (B,3,H,W)

            # DINOv3 hub backbones typically return a global embedding when called as model(x)
            emb = backbone(x)  # (B, D)
            if emb.ndim != 2:
                raise RuntimeError(
                    f"[ERROR] Unexpected backbone output shape: {tuple(emb.shape)}. "
                    "Expected (B, D). You may need to adjust how embeddings are extracted."
                )
            emb = l2_normalize(emb).detach().cpu()  # normalize on GPU then move to CPU
            feats_list.append(emb)

    features = torch.cat(feats_list, dim=0)  # (N, D) on CPU
    n, d = features.shape
    print(f"[INFO] Features shape: {features.shape} (N={n}, D={d})")
    print(f"[INFO] Unique patients: {len(np.unique(patient_ids))}")

    # Retrieval evaluation (no full NxN matrix; per-query dot)
    K_LIST = sorted(set(int(k) for k in args.k))
    max_k = max(K_LIST)
    topk_log = max(1, int(args.max_topk_log))

    recall_hits = {k: [] for k in K_LIST}
    ap_list: List[float] = []
    ap_valid_mask: List[bool] = []

    per_query_rows = []

    print("\n[INFO] Running retrieval evaluation...\n")

    features_t = features.t().contiguous()  # (D, N) for faster matmul

    for i in tqdm(range(n), desc="Queries"):
        pid = patient_ids[i]
        pos_idx = patient_to_indices[pid]
        # exclude self
        pos_idx = pos_idx[pos_idx != i]

        # similarity to all (cosine since L2 normalized)
        # sims: (N,)
        sims = torch.mv(features, features[i])  # dot(features[j], features[i])
        sims[i] = -1e9  # remove self match

        # rank
        sorted_idx = torch.argsort(sims, descending=True)
        topk_idx = sorted_idx[:max_k].numpy()

        # Recall@K
        for k in K_LIST:
            hit = int(np.isin(topk_idx[:k], pos_idx).any()) if len(pos_idx) > 0 else 0
            recall_hits[k].append(hit)

        # AP (skip queries with no positives)
        if len(pos_idx) == 0:
            ap = np.nan
            ap_valid = False
        else:
            y_true = np.zeros(n, dtype=np.int32)
            y_true[pos_idx] = 1
            # sklearn expects scores aligned with y_true (same length)
            ap = float(average_precision_score(y_true, sims.numpy()))
            ap_valid = True
            ap_list.append(ap)

        ap_valid_mask.append(ap_valid)

        # Log top neighbors
        nn_idx = sorted_idx[:topk_log].numpy()
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

    # Aggregate metrics
    results = {f"Recall@{k}": float(np.mean(recall_hits[k])) for k in K_LIST}
    if len(ap_list) > 0:
        results["mAP"] = float(np.mean(ap_list))
    else:
        results["mAP"] = float("nan")

    num_no_pos = int(np.sum(np.array(ap_valid_mask) == 0))

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = os.path.join(args.out_dir, f"retrieval_result_{timestamp}.txt")
    per_query_csv = os.path.join(args.out_dir, f"per_query_{timestamp}.csv")

    with open(summary_path, "w") as f:
        f.write("========== Patient Retrieval Evaluation ==========\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Repo: {args.repo_dir}\n")
        f.write(
            f"Weights: {args.backbone_weights if args.backbone_weights else '(torch.hub default)'}\n"
        )
        f.write(f"Dataset: {args.input}\n")
        f.write(f"Resize/CenterCrop: {resize_size}/{center_crop_size}\n")
        f.write(f"N images: {n}\n")
        f.write(f"Embedding dim: {d}\n")
        f.write(f"Unique patients: {len(np.unique(patient_ids))}\n")
        f.write(f"Queries with no positives (patient has 1 image): {num_no_pos}\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    pd.DataFrame(per_query_rows).to_csv(per_query_csv, index=False)

    # Console
    print("\n========== Retrieval Results ==========")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print(f"\n[Saved]\n- {summary_path}\n- {per_query_csv}")


if __name__ == "__main__":
    main()
