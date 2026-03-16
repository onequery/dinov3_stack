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
import json
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm
import math

from src.img_cls.model import Dinov3Classification
from src.utils.common import get_dinov3_paths


# --------------------------------------------------
# Utility
# --------------------------------------------------
def resolve_repo_path(repo_dir_arg, env_repo_dir):
    if repo_dir_arg:
        repo_path = os.path.abspath(os.path.expanduser(repo_dir_arg))
        if not os.path.exists(repo_path):
            raise FileNotFoundError(f"DINOv3 repository not found at: {repo_path}")
        return repo_path

    if not env_repo_dir:
        raise ValueError(
            "DINOv3 repository path is missing. "
            "Set DINOV3_REPO in .env or pass --repo-dir."
        )

    return env_repo_dir


def collect_image_paths(root_dir, exts=("png", "jpg", "jpeg")):
    image_paths = []
    for ext in exts:
        image_paths.extend(
            glob.glob(os.path.join(root_dir, "**", f"*.{ext}"), recursive=True)
        )
    return sorted(image_paths)


def plot_simplex_2class(df, softmax_arr, class_names, out_path, decision_boundary_p0=None):
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

    boundary_x = 0.5 if decision_boundary_p0 is None else float(decision_boundary_p0)
    plt.axvline(boundary_x, linestyle="--", color="gray", alpha=0.7)
    plt.yticks([])
    plt.xlabel(f"p({class_names[0]})")
    if decision_boundary_p0 is None:
        plt.title("2-class Simplex (Probability Strip)")
    else:
        plt.title(f"2-class Simplex (Decision boundary on p({class_names[0]})={boundary_x:.3f})")
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


def infer_head_config_from_state_dict(state_dict):
    if "head.weight" in state_dict and "head.bias" in state_dict:
        return {"head_size": "small", "head_hidden_dim": None}

    if (
        "head.0.weight" in state_dict
        and "head.0.bias" in state_dict
        and "head.2.weight" in state_dict
        and "head.2.bias" in state_dict
    ):
        hidden_dim = int(state_dict["head.0.weight"].shape[0])
        return {"head_size": "big", "head_hidden_dim": hidden_dim}

    raise ValueError(
        "Cannot infer classification head type from checkpoint keys. "
        "Expected either small head (`head.weight`) or big head (`head.0.weight`, `head.2.weight`)."
    )


def infer_lora_rank_from_state_dict(state_dict):
    ranks = set()
    for key, value in state_dict.items():
        if key.startswith("backbone_model.") and key.endswith(".lora_A"):
            ranks.add(int(value.shape[0]))
    if not ranks:
        return None
    if len(ranks) != 1:
        raise ValueError(f"Multiple LoRA ranks detected in checkpoint: {sorted(ranks)}")
    return list(ranks)[0]


def normalize_state_dict_keys(state_dict):
    if any(key.startswith("module.") for key in state_dict.keys()):
        return {
            key[len("module.") :]: value
            for key, value in state_dict.items()
            if key.startswith("module.")
        }
    return state_dict


def resolve_positive_class_idx(class_names, class_to_idx, positive_class_name):
    if positive_class_name is None:
        return 1
    if positive_class_name not in class_to_idx:
        raise ValueError(
            f"Unknown positive class '{positive_class_name}'. "
            f"Available classes: {class_names}"
        )
    return class_to_idx[positive_class_name]


def collect_samples(input_dir, class_to_idx, transform, model, device):
    samples = []
    for class_name in tqdm(sorted(os.listdir(input_dir)), desc="Classes"):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        gt_label = class_to_idx[class_name]
        for img_path in collect_image_paths(class_dir):
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            samples.append(
                {
                    "image_path": img_path,
                    "gt_idx": int(gt_label),
                    "probs": probs,
                }
            )
    return samples


def select_best_threshold_macro_f1(samples, positive_idx, steps=1001):
    y_true_pos = np.array(
        [1 if s["gt_idx"] == positive_idx else 0 for s in samples], dtype=np.int64
    )
    pos_probs = np.array([float(s["probs"][positive_idx]) for s in samples], dtype=np.float64)
    thresholds = np.linspace(0.0, 1.0, steps)

    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        y_pred_pos = (pos_probs >= threshold).astype(np.int64)
        metric = f1_score(y_true_pos, y_pred_pos, average="macro", zero_division=0.0)
        if metric > best_f1 + 1e-12:
            best_f1 = float(metric)
            best_threshold = float(threshold)
        elif abs(metric - best_f1) <= 1e-12:
            if abs(float(threshold) - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)
    return best_threshold, best_f1


def build_predictions(samples, class_names, positive_idx=None, threshold=None):
    records = []
    y_true, y_pred, softmax_vecs = [], [], []
    negative_idx = None
    if threshold is not None:
        if positive_idx is None:
            raise ValueError("positive_idx is required when threshold is provided.")
        negative_idx = 1 - int(positive_idx)

    for sample in samples:
        probs = sample["probs"]
        gt_idx = sample["gt_idx"]
        if threshold is None:
            pred_idx = int(np.argmax(probs))
        else:
            pred_idx = int(positive_idx) if probs[positive_idx] >= threshold else negative_idx

        confidence = float(probs[pred_idx])

        y_true.append(gt_idx)
        y_pred.append(pred_idx)
        softmax_vecs.append(probs)

        record = {
            "image_path": sample["image_path"],
            "gt_label": class_names[gt_idx],
            "pred_label": class_names[pred_idx],
            "confidence": confidence,
            "correct": int(gt_idx == pred_idx),
        }
        if threshold is not None:
            record["positive_probability"] = float(probs[positive_idx])
            record["decision_threshold"] = float(threshold)
        records.append(record)

    return records, y_true, y_pred, softmax_vecs


# --------------------------------------------------
# Args
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--weights", required=True)
parser.add_argument("--input", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--model-name", default="dinov3_vits16")
parser.add_argument("--out-dir", default="outputs/eval_results")
parser.add_argument("--repo-dir", default=None)
parser.add_argument(
    "--threshold-mode",
    choices=["argmax", "fixed", "tune_val_macro_f1"],
    default="argmax",
    help="prediction rule for binary classification",
)
parser.add_argument(
    "--threshold",
    type=float,
    default=0.5,
    help="fixed threshold for positive class when --threshold-mode fixed",
)
parser.add_argument(
    "--threshold-val-input",
    default=None,
    help="validation directory used to tune threshold for --threshold-mode tune_val_macro_f1",
)
parser.add_argument(
    "--threshold-positive-class",
    default=None,
    help="positive class name for thresholding; defaults to CLASS_NAMES[1]",
)
args = parser.parse_args()


# --------------------------------------------------
# Env
# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DINOV3_REPO, _ = get_dinov3_paths(
    require_repo=not bool(args.repo_dir),
    require_weights=False,
)
DINOV3_REPO = resolve_repo_path(args.repo_dir, DINOV3_REPO)


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
model_state_dict = normalize_state_dict_keys(model_state_dict)
head_cfg = infer_head_config_from_state_dict(model_state_dict)
print(f"Inferred head config: {head_cfg}")
lora_rank = infer_lora_rank_from_state_dict(model_state_dict)
if lora_rank is not None:
    print(f"Inferred LoRA config: target=attn_qkv_proj rank={lora_rank}")

model = Dinov3Classification(
    num_classes=len(CLASS_NAMES),
    head_size=head_cfg["head_size"],
    head_hidden_dim=head_cfg["head_hidden_dim"],
    enable_lora=(lora_rank is not None),
    lora_rank=lora_rank,
    lora_alpha=lora_rank,
    lora_target="attn_qkv_proj",
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
print("\nRunning inference...\n")
test_samples = collect_samples(args.input, class_to_idx, transform, model, DEVICE)

threshold_info = {
    "mode": args.threshold_mode,
    "threshold": None,
    "positive_class": None,
    "positive_class_index": None,
    "validation_dir": None,
    "validation_macro_f1_at_selected_threshold": None,
}

decision_threshold = None
positive_class_idx = None

if args.threshold_mode != "argmax":
    if len(CLASS_NAMES) != 2:
        raise ValueError(
            f"--threshold-mode {args.threshold_mode} requires binary classification, "
            f"but got {len(CLASS_NAMES)} classes."
        )
    if args.threshold_mode == "fixed":
        if args.threshold < 0.0 or args.threshold > 1.0:
            raise ValueError("--threshold must be within [0, 1].")
        positive_class_idx = resolve_positive_class_idx(
            CLASS_NAMES, class_to_idx, args.threshold_positive_class
        )
        decision_threshold = float(args.threshold)
        threshold_info["mode"] = "fixed"
        threshold_info["threshold"] = decision_threshold
        threshold_info["positive_class"] = CLASS_NAMES[positive_class_idx]
        threshold_info["positive_class_index"] = int(positive_class_idx)
        print(
            f"Using fixed decision threshold: {decision_threshold:.4f} "
            f"(positive class={CLASS_NAMES[positive_class_idx]})"
        )
    else:
        if not args.threshold_val_input:
            raise ValueError(
                "--threshold-val-input is required when "
                "--threshold-mode tune_val_macro_f1 is selected."
            )
        positive_class_idx = resolve_positive_class_idx(
            CLASS_NAMES, class_to_idx, args.threshold_positive_class
        )
        print(
            f"Tuning decision threshold on validation set: {args.threshold_val_input} "
            f"(positive class={CLASS_NAMES[positive_class_idx]})"
        )
        val_samples = collect_samples(
            args.threshold_val_input, class_to_idx, transform, model, DEVICE
        )
        decision_threshold, val_best_macro_f1 = select_best_threshold_macro_f1(
            val_samples, positive_class_idx
        )
        threshold_info["mode"] = "tune_val_macro_f1"
        threshold_info["threshold"] = float(decision_threshold)
        threshold_info["positive_class"] = CLASS_NAMES[positive_class_idx]
        threshold_info["positive_class_index"] = int(positive_class_idx)
        threshold_info["validation_dir"] = args.threshold_val_input
        threshold_info["validation_macro_f1_at_selected_threshold"] = float(
            val_best_macro_f1
        )
        print(
            f"Selected threshold={decision_threshold:.4f} "
            f"from validation macro-F1={val_best_macro_f1:.4f}"
        )

records, y_true, y_pred, softmax_vecs = build_predictions(
    samples=test_samples,
    class_names=CLASS_NAMES,
    positive_idx=positive_class_idx,
    threshold=decision_threshold,
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
    if decision_threshold is None:
        f.write("Prediction mode: argmax\n")
    else:
        f.write(f"Prediction mode: threshold ({threshold_info['mode']})\n")
        f.write(
            f"Positive class: {threshold_info['positive_class']} "
            f"(index={threshold_info['positive_class_index']})\n"
        )
        f.write(f"Decision threshold: {threshold_info['threshold']:.4f}\n")
        if threshold_info["validation_dir"] is not None:
            f.write(f"Threshold validation set: {threshold_info['validation_dir']}\n")
            f.write(
                "Validation macro-F1 at selected threshold: "
                f"{threshold_info['validation_macro_f1_at_selected_threshold']:.4f}\n"
            )
    f.write("\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(cls_report)

if decision_threshold is not None:
    with open(os.path.join(args.out_dir, f"threshold_info_{timestamp}.json"), "w") as f:
        json.dump(threshold_info, f, indent=2)


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
    decision_boundary_p0 = None
    if decision_threshold is not None and positive_class_idx is not None:
        if positive_class_idx == 0:
            decision_boundary_p0 = decision_threshold
        else:
            decision_boundary_p0 = 1.0 - decision_threshold
    plot_simplex_2class(
        df,
        softmax_arr,
        CLASS_NAMES,
        simplex_path,
        decision_boundary_p0=decision_boundary_p0,
    )

elif num_classes == 3:
    plot_simplex_3class(df, softmax_arr, CLASS_NAMES, simplex_path)

else:
    print(
        f"[WARNING] Simplex visualization skipped: "
        f"{num_classes}-class not supported (only 2 or 3)."
    )


print("\nEvaluation complete.")
