"""
Inference + Evaluation script for image classification (DINOv3).

Outputs:
  - Accuracy / Precision / Recall / F1-score
  - Confusion Matrix (CSV / PNG)
  - Per-image prediction log (CSV)
  - Confidence histogram (correct vs wrong)
  - Confidence x Correctness scatter
  - Softmax-UMAP (prediction space visualization)
"""

import argparse
import glob
import os
from datetime import datetime
from matplotlib.lines import Line2D
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
import math

from src.img_cls.model import Dinov3Classification
from src.utils.common import get_dinov3_paths


# --------------------------------------------------
# Utility
# --------------------------------------------------
def collect_image_paths(root_dir, exts=("png", "jpg", "jpeg")):
    image_paths = []
    for ext in exts:
        image_paths.extend(
            glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        )
    return sorted(image_paths)


def plot_simplex_2class(df, softmax_arr, class_names, out_path):
    """
    2-class simplex: probability strip
    """
    p0 = softmax_arr[:, 0]
    jitter = np.random.normal(0, 0.02, size=len(p0))

    plt.figure(figsize=(10, 3))

    color_map = {
        class_names[0]: "#1f77b4",  # blue
        class_names[1]: "#ff7f0e",  # orange
    }

    point_colors = df["gt_label"].map(color_map)

    plt.scatter(
        p0,
        jitter,
        c=point_colors,
        s=20 + 80 * df["confidence"],
        alpha=0.65,
    )

    wrong = df.correct == 0
    plt.scatter(
        p0[wrong],
        jitter[wrong],
        facecolors="none",
        edgecolors="red",
        s=80,
        linewidths=1.2,
        label="Wrong",
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=class_names[0],
            markerfacecolor=color_map[class_names[0]],
            markersize=8,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=class_names[1],
            markerfacecolor=color_map[class_names[1]],
            markersize=8,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            label="Wrong",
            markerfacecolor="none",
            markersize=8,
            linewidth=1.2,
        ),
    ]

    plt.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=False,
    )

    plt.axvline(0.5, linestyle="--", color="gray", alpha=0.5)
    plt.yticks([])
    plt.xlabel(f"p({class_names[0]})")
    plt.title("2-class Simplex (Probability Strip)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_simplex_3class(df, softmax_arr, class_names, out_path):
    """
    True 3-class simplex (equilateral triangle)
    """
    # Triangle vertices
    V = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, math.sqrt(3) / 2],
        ]
    )

    coords = softmax_arr @ V

    plt.figure(figsize=(8, 7))

    color_map = {
        class_names[0]: "#1f77b4",  # blue
        class_names[1]: "#ff7f0e",  # orange
        class_names[2]: "#2ca02c",  # green
    }

    point_colors = df["gt_label"].map(color_map)

    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=point_colors,
        s=20 + 80 * df["confidence"],
        alpha=0.65,
    )

    wrong = df.correct == 0
    plt.scatter(
        coords[wrong, 0],
        coords[wrong, 1],
        facecolors="none",
        edgecolors="red",
        s=100,
        linewidths=1.3,
        label="Wrong",
    )

    # Draw simplex edges
    for i in range(3):
        p1 = V[i]
        p2 = V[(i + 1) % 3]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k-", lw=1)

    # Vertex labels
    for i, name in enumerate(class_names):
        plt.text(V[i, 0], V[i, 1] + 0.03, name, ha="center", fontsize=10)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=class_names[0],
            markerfacecolor=color_map[class_names[0]],
            markersize=8,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=class_names[1],
            markerfacecolor=color_map[class_names[1]],
            markersize=8,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=class_names[2],
            markerfacecolor=color_map[class_names[2]],
            markersize=8,
            alpha=0.8,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="red",
            label="Wrong",
            markerfacecolor="none",
            markersize=8,
            linewidth=1.2,
        ),
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
    )

    plt.axis("equal")
    plt.axis("off")
    plt.title("3-class Simplex (Prediction Geometry)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# Args
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--model-name", default="dinov3_vits16")
parser.add_argument("--out-dir", default="outputs/eval_results")
args = parser.parse_args()


# --------------------------------------------------
# Env
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINOV3_REPO, _ = get_dinov3_paths()


# --------------------------------------------------
# Config
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
# Model
# --------------------------------------------------
checkpoint = torch.load(args.weights, map_location="cpu")
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model_state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict):
    model_state_dict = checkpoint
else:
    raise ValueError(
        f"Invalid checkpoint format for classification eval: {args.weights}"
    )

if not isinstance(model_state_dict, dict):
    raise ValueError(
        "Invalid `model_state_dict` in checkpoint. Expected a dictionary of tensors."
    )

model = Dinov3Classification(
    num_classes=len(CLASS_NAMES),
    weights=args.weights,
    model_name=args.model_name,
    repo_dir=DINOV3_REPO,
).to(DEVICE)

model.load_state_dict(model_state_dict, strict=True)
model.eval()

checkpoint_abs = os.path.abspath(args.weights)
checkpoint_epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
if checkpoint_epoch is not None:
    print(
        f"Loaded classification checkpoint (backbone+head, strict=True): "
        f"{checkpoint_abs} (epoch={checkpoint_epoch})"
    )
else:
    print(
        f"Loaded classification checkpoint (backbone+head, strict=True): "
        f"{checkpoint_abs}"
    )


# --------------------------------------------------
# Inference
# --------------------------------------------------
records = []
y_true, y_pred = [], []
softmax_vecs = []  # ⭐ NEW

print("\nRunning inference...\n")

for class_name in tqdm(sorted(os.listdir(args.input)), desc="Classes"):
    class_dir = os.path.join(args.input, class_name)
    if not os.path.isdir(class_dir):
        continue

    gt_label = class_to_idx[class_name]

    for img_path in collect_image_paths(class_dir):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(image)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label])

        y_true.append(gt_label)
        y_pred.append(pred_label)
        softmax_vecs.append(probs)

        records.append(
            {
                "image_path": img_path,
                "gt_label": CLASS_NAMES[gt_label],
                "pred_label": CLASS_NAMES[pred_label],
                "confidence": confidence,
                "correct": int(gt_label == pred_label),
            }
        )


# --------------------------------------------------
# Metrics
# --------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)
cls_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
conf_mat = confusion_matrix(y_true, y_pred)


# --------------------------------------------------
# Save outputs
# --------------------------------------------------
os.makedirs(args.out_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

df = pd.DataFrame(records)
df.to_csv(
    os.path.join(args.out_dir, f"per_image_predictions_{timestamp}.csv"),
    index=False,
)

with open(os.path.join(args.out_dir, f"eval_result_{timestamp}.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(cls_report)


# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
cm_df = pd.DataFrame(conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
cm_df.to_csv(os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.csv"))

cm_norm = conf_mat.astype(np.float32)
cm_norm /= cm_norm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12, 12))
sns.heatmap(
    cm_norm,
    cmap="Blues",
    square=True,
    annot=False,
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, f"confusion_matrix_{timestamp}.png"), dpi=300)
plt.close()


# --------------------------------------------------
# Confidence Histogram
# --------------------------------------------------
plt.figure(figsize=(10, 6))
plt.hist(df[df.correct == 1].confidence, bins=30, alpha=0.6, label="Correct")
plt.hist(df[df.correct == 0].confidence, bins=30, alpha=0.6, label="Wrong")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.title("Confidence Distribution (Correct vs Wrong)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(args.out_dir, f"confidence_histogram_{timestamp}.png"), dpi=300
)
plt.close()


# --------------------------------------------------
# Confidence x Correctness Scatter
# --------------------------------------------------
jitter = np.random.normal(0, 0.03, size=len(df))
plt.figure(figsize=(10, 4))
plt.scatter(
    df["confidence"],
    df["correct"] + jitter,
    c=df["correct"],
    cmap="coolwarm",
    alpha=0.5,
    s=20,
)
plt.yticks([0, 1], ["Wrong", "Correct"])
plt.xlabel("Confidence")
plt.title("Confidence x Correctness Scatter")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(args.out_dir, f"confidence_correctness_scatter_{timestamp}.png"),
    dpi=300,
)
plt.close()


# --------------------------------------------------
# ⭐ Simplex Visualization (2-class / 3-class)
# --------------------------------------------------
print("Running Simplex visualization...")

softmax_arr = np.array(softmax_vecs)
num_classes = softmax_arr.shape[1]

simplex_path = os.path.join(args.out_dir, f"simplex_{num_classes}class_{timestamp}.png")

if num_classes == 2:
    plot_simplex_2class(df, softmax_arr, CLASS_NAMES, simplex_path)

elif num_classes == 3:
    plot_simplex_3class(df, softmax_arr, CLASS_NAMES, simplex_path)

else:
    print(
        f"[WARNING] Simplex visualization skipped: "
        f"{num_classes}-class not supported (only 2 or 3)."
    )


print("\nEvaluation complete.")
