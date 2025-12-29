"""
Inference + Evaluation script for image classification (DINOv3).

- Folder-based GT labels
- Outputs Accuracy / Precision / Recall / F1-score
- Saves evaluation results
  - TXT (summary)
  - CSV (confusion matrix, numeric)
  - TXT (confusion matrix, readable)
  - PNG (confusion matrix heatmap, colored)

USAGE:
python infer_and_eval_classifier.py \
    --weights outputs/best_model.pth \
    --input data/test \
    --config configs/classification.yaml \
    --out-dir outputs/eval_results
"""

import os
import glob
import cv2
import yaml
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

from src.img_cls.model import Dinov3Classification
from src.utils.common import get_dinov3_paths

# --------------------------------------------------
# Argument parser
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--weights', required=True, help='path to model weights (.pth)')
parser.add_argument('--input', required=True, help='test dataset root directory')
parser.add_argument('--config', required=True, help='yaml file with CLASS_NAMES')
parser.add_argument('--model-name', default='dinov3_vits16')
parser.add_argument('--out-dir', default='outputs/eval_results')
args = parser.parse_args()

# --------------------------------------------------
# Environment
# --------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_SIZE = 224

DINOV3_REPO, _ = get_dinov3_paths()

# --------------------------------------------------
# Load class names
# --------------------------------------------------
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

CLASS_NAMES = config['CLASS_NAMES']
class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

# --------------------------------------------------
# Transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Load model
# --------------------------------------------------
checkpoint = torch.load(args.weights, map_location='cpu')

model = Dinov3Classification(
    num_classes=len(CLASS_NAMES),
    model_name=args.model_name,
    repo_dir=DINOV3_REPO
).to(DEVICE)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
y_true = []
y_pred = []

print("\nRunning inference + evaluation...\n")

for class_name in tqdm(sorted(os.listdir(args.input)), desc="Classes"):
    class_dir = os.path.join(args.input, class_name)
    if not os.path.isdir(class_dir):
        continue

    gt_label = class_to_idx[class_name]
    image_paths = glob.glob(os.path.join(class_dir, "*.jpg"))

    for img_path in image_paths:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image)
            pred_label = torch.argmax(logits, dim=1).item()

        y_true.append(gt_label)
        y_pred.append(pred_label)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
cls_report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)
conf_mat = confusion_matrix(y_true, y_pred)

# --------------------------------------------------
# Save results
# --------------------------------------------------
os.makedirs(args.out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1) Evaluation summary TXT
eval_txt_path = os.path.join(
    args.out_dir,
    f"eval_result_{timestamp}.txt"
)

with open(eval_txt_path, "w") as f:
    f.write("========== Classification Evaluation ==========\n")
    f.write(f"Weights: {args.weights}\n")
    f.write(f"Model: {args.model_name}\n")
    f.write(f"Dataset: {args.input}\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(cls_report)
    f.write("\n")

# 2) Confusion Matrix (DataFrame)
cm_df = pd.DataFrame(
    conf_mat,
    index=CLASS_NAMES,   # GT
    columns=CLASS_NAMES  # Prediction
)

# CSV
cm_csv_path = os.path.join(
    args.out_dir,
    f"confusion_matrix_{timestamp}.csv"
)
cm_df.to_csv(cm_csv_path)

# TXT
cm_txt_path = os.path.join(
    args.out_dir,
    f"confusion_matrix_{timestamp}.txt"
)
with open(cm_txt_path, "w") as f:
    f.write("Confusion Matrix (rows = GT, cols = Prediction)\n\n")
    f.write(cm_df.to_string())

# --------------------------------------------------
# Confusion Matrix Heatmap (PNG)
# --------------------------------------------------

plt.figure(figsize=(1.2 * len(CLASS_NAMES), 1.0 * len(CLASS_NAMES)))

# 정규화 (row-wise)
cm_normalized = conf_mat.astype(np.float32) / conf_mat.sum(axis=1, keepdims=True)

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cbar=True
)

plt.xlabel("Predicted Label")
plt.ylabel("Ground Truth Label")
plt.title("Normalized Confusion Matrix")

plt.tight_layout()

cm_png_path = os.path.join(
    args.out_dir,
    f"confusion_matrix_{timestamp}.png"
)
plt.savefig(cm_png_path, dpi=200)
plt.close()

# --------------------------------------------------
# Console output
# --------------------------------------------------
print("\n========== Evaluation Results ==========")
print(f"Accuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(cls_report)

print("\nSaved files:")
print(f" - Evaluation summary: {eval_txt_path}")
print(f" - Confusion matrix (CSV): {cm_csv_path}")
print(f" - Confusion matrix (TXT): {cm_txt_path}")
print(f" - Confusion matrix (Heatmap PNG): {cm_png_path}")
