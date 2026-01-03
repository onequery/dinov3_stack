"""
Inference + Evaluation script for image classification (DINOv3).

- Folder-based GT labels (recursive)
- Supports png / jpg / jpeg
- Outputs:
  - Accuracy / Precision / Recall / F1-score
  - Confusion Matrix (CSV / TXT / PNG)
  - Per-image prediction log (CSV / TXT)

USAGE:
python eval_classifier.py \
    --weights outputs/train/stent_cls/best_model.pth \
    --input input/stent_split_img/test \
    --config classification_configs/stent.yaml \
    --model-name dinov3_convnext_tiny \
    --out-dir outputs/eval/stent_cls
"""

import argparse
import glob
import os
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as transforms
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from src.img_cls.model import Dinov3Classification
from src.utils.common import get_dinov3_paths


# --------------------------------------------------
# Utility: recursive image loader
# --------------------------------------------------
def collect_image_paths(root_dir, exts=("png", "jpg", "jpeg")):
    image_paths = []
    for ext in exts:
        image_paths.extend(
            glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        )
    return sorted(image_paths)


# --------------------------------------------------
# Argument parser
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True, help="path to model weights (.pth)")
parser.add_argument("--input", required=True, help="test dataset root directory")
parser.add_argument("--config", required=True, help="yaml file with CLASS_NAMES")
parser.add_argument("--model-name", default="dinov3_vits16")
parser.add_argument("--out-dir", default="outputs/eval_results")
args = parser.parse_args()


# --------------------------------------------------
# Environment
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINOV3_REPO, _ = get_dinov3_paths()


# --------------------------------------------------
# Load config
# --------------------------------------------------
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

CLASS_NAMES = config["CLASS_NAMES"]
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

RESIZE_SIZE = config.get("RESIZE_SIZE")
if RESIZE_SIZE is None:
    raise ValueError("[ERROR] 'RESIZE_SIZE' not found in the config YAML.")

CENTER_CROP_SIZE = config.get("CENTER_CROP_SIZE")
if CENTER_CROP_SIZE is None:
    raise ValueError("[ERROR] 'CENTER_CROP_SIZE' not found in the config YAML.")


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
# Load model
# --------------------------------------------------
checkpoint = torch.load(args.weights, map_location="cpu")

model = Dinov3Classification(
    num_classes=len(CLASS_NAMES),
    model_name=args.model_name,
    repo_dir=DINOV3_REPO,
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
y_true = []
y_pred = []
records = []

print("\nRunning inference + evaluation...\n")

for class_name in tqdm(sorted(os.listdir(args.input)), desc="Classes"):
    class_dir = os.path.join(args.input, class_name)
    if not os.path.isdir(class_dir):
        continue

    if class_name not in class_to_idx:
        raise ValueError(
            f"[ERROR] Folder name '{class_name}' not found in CLASS_NAMES {CLASS_NAMES}"
        )

    gt_label = class_to_idx[class_name]
    image_paths = collect_image_paths(class_dir)

    if len(image_paths) == 0:
        raise RuntimeError(
            f"[ERROR] No image files found under:\n  {class_dir}\n"
            "Check folder structure or file extensions."
        )

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"[ERROR] Failed to read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()

        y_true.append(gt_label)
        y_pred.append(pred_label)

        records.append(
            {
                "image_path": img_path,
                "gt_label": CLASS_NAMES[gt_label],
                "pred_label": CLASS_NAMES[pred_label],
                "confidence": confidence,
                "correct": gt_label == pred_label,
            }
        )

if len(y_true) == 0:
    raise RuntimeError(
        "[FATAL] No samples processed. Check dataset path and structure."
    )


# --------------------------------------------------
# Metrics
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
cls_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
conf_mat = confusion_matrix(y_true, y_pred)


# --------------------------------------------------
# Save results
# --------------------------------------------------
os.makedirs(args.out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1) Summary TXT
eval_txt_path = os.path.join(args.out_dir, f"eval_result_{timestamp}.txt")
with open(eval_txt_path, "w") as f:
    f.write("========== Classification Evaluation ==========\n")
    f.write(f"Weights: {args.weights}\n")
    f.write(f"Model: {args.model_name}\n")
    f.write(f"Dataset: {args.input}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(cls_report)
    f.write("\n")

# 2) Confusion Matrix
cm_df = pd.DataFrame(conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)

cm_df.to_csv(os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.csv"))

with open(os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.txt"), "w") as f:
    f.write("Confusion Matrix (rows = GT, cols = Prediction)\n\n")
    f.write(cm_df.to_string())

# Heatmap
cm_normalized = conf_mat.astype(np.float32) / conf_mat.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.xlabel("Predicted")
plt.ylabel("Ground Truth")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.png"), dpi=200)
plt.close()

# 3) Per-image prediction log
pred_df = pd.DataFrame(records)
pred_csv_path = os.path.join(args.out_dir, f"per_image_predictions_{timestamp}.csv")
pred_df.to_csv(pred_csv_path, index=False)

pred_txt_path = os.path.join(args.out_dir, f"per_image_predictions_{timestamp}.txt")
with open(pred_txt_path, "w") as f:
    for r in records:
        f.write(
            f"[{'OK' if r['correct'] else 'WRONG'}] "
            f"GT={r['gt_label']} | "
            f"PRED={r['pred_label']} "
            f"(conf={r['confidence']:.3f}) | "
            f"{r['image_path']}\n"
        )


# --------------------------------------------------
# Console output
# --------------------------------------------------
print("\n========== Evaluation Results ==========")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(cls_report)

print("\nSaved files:")
print(f" - Evaluation summary: {eval_txt_path}")
print(f" - Confusion matrix (CSV/TXT/PNG)")
print(f" - Per-image predictions (CSV): {pred_csv_path}")
print(f" - Per-image predictions (TXT): {pred_txt_path}")
