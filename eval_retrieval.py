"""
Image–Image Retrieval Evaluation (DINOv3)

Protocol:
- Frozen DINOv3 backbone
- CLS token as global representation
- L2 normalization
- Cosine similarity
- Folder name = GT label

Metrics:
- Recall@K
- mAP

USAGE:
python eval_retrieval.py \
    --input input/archive/test \
    --config classification_configs/cards.yaml \
    --model-name dinov3_vits16 \
    --out-dir outputs/eval_retrieval/cards
"""

import argparse
import glob
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from src.img_cls.model import Dinov3Classification
from src.utils.common import get_dinov3_paths


# --------------------------------------------------
# Utils
# --------------------------------------------------
def collect_image_paths(root_dir, exts=("png", "jpg", "jpeg")):
    paths = []
    for ext in exts:
        paths.extend(
            glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        )
    return sorted(paths)


def l2_normalize(x):
    return F.normalize(x, dim=1)


# --------------------------------------------------
# Argument parser
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="dataset root directory")
parser.add_argument("--config", required=True, help="yaml with CLASS_NAMES")
parser.add_argument("--model-name", default="dinov3_vits16")
parser.add_argument("--out-dir", default="outputs/eval_retrieval")
args = parser.parse_args()


# --------------------------------------------------
# Environment
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()


# --------------------------------------------------
# Load config
# --------------------------------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

CLASS_NAMES = config["CLASS_NAMES"]
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

RESIZE_SIZE = config["RESIZE_SIZE"]
CENTER_CROP_SIZE = config["CENTER_CROP_SIZE"]


# --------------------------------------------------
# Transform
# --------------------------------------------------
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((RESIZE_SIZE, RESIZE_SIZE)),
        transforms.CenterCrop((CENTER_CROP_SIZE, CENTER_CROP_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# --------------------------------------------------
# Load model (backbone only)
# --------------------------------------------------
model = Dinov3Classification(
    num_classes=1,  # dummy
    model_name=args.model_name,
    repo_dir=DINOV3_REPO,
).to(DEVICE)

model.eval()


# --------------------------------------------------
# Feature extraction
# --------------------------------------------------
features = []
labels = []
paths = []

print("\nExtracting features...\n")

for class_name in tqdm(sorted(os.listdir(args.input)), desc="Classes"):
    class_dir = os.path.join(args.input, class_name)
    if not os.path.isdir(class_dir):
        continue

    if class_name not in class_to_idx:
        raise ValueError(f"Unknown class: {class_name}")

    label = class_to_idx[class_name]
    image_paths = collect_image_paths(class_dir)

    for p in image_paths:
        img = cv2.imread(p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            feat = model.forward_features(img)  # CLS token
            feat = l2_normalize(feat)

        features.append(feat.cpu())
        labels.append(label)
        paths.append(p)

features = torch.cat(features, dim=0)
labels = np.array(labels)


# --------------------------------------------------
# Retrieval evaluation
# --------------------------------------------------
print("\nRunning retrieval evaluation...\n")

sim_matrix = torch.matmul(features, features.T)  # cosine similarity
sim_matrix.fill_diagonal_(-1)  # remove self-match

K_LIST = [1, 5, 10]
recall_at_k = {k: [] for k in K_LIST}
ap_list = []

for i in tqdm(range(len(features)), desc="Queries"):
    sims = sim_matrix[i]
    sorted_idx = torch.argsort(sims, descending=True)

    gt = labels[i]
    gt_mask = (labels == gt).astype(np.int32)

    ranked_gt = gt_mask[sorted_idx.numpy()]

    # Recall@K
    for k in K_LIST:
        recall_at_k[k].append(int(ranked_gt[:k].sum() > 0))

    # AP
    ap = average_precision_score(ranked_gt, sims[sorted_idx].numpy())
    ap_list.append(ap)


# --------------------------------------------------
# Metrics summary
# --------------------------------------------------
results = {f"Recall@{k}": np.mean(recall_at_k[k]) for k in K_LIST}
results["mAP"] = np.mean(ap_list)


# --------------------------------------------------
# Save results
# --------------------------------------------------
os.makedirs(args.out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

summary_path = os.path.join(args.out_dir, f"retrieval_result_{timestamp}.txt")

with open(summary_path, "w") as f:
    f.write("========== Image Retrieval Evaluation ==========\n")
    f.write(f"Model: {args.model_name}\n")
    f.write(f"Dataset: {args.input}\n\n")
    for k, v in results.items():
        f.write(f"{k}: {v:.4f}\n")

# per-query AP log
pd.DataFrame(
    {
        "image_path": paths,
        "label": labels,
        "AP": ap_list,
    }
).to_csv(
    os.path.join(args.out_dir, f"per_query_ap_{timestamp}.csv"),
    index=False,
)


# --------------------------------------------------
# Console output
# --------------------------------------------------
print("\n========== Retrieval Results ==========")
for k, v in results.items():
    print(f"{k}: {v:.4f}")

print(f"\nSaved results to:\n{summary_path}")
