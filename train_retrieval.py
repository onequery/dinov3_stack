#!/usr/bin/env python3
"""
Train Retrieval Model with Projection Head on DINOv3 Backbone
(Supervised Contrastive Learning)
"""

import argparse
import os
import random
import re
from datetime import datetime
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.utils.common import get_dinov3_paths

# -------------------------
# Reproducibility
# -------------------------
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# -------------------------
# Utils
# -------------------------
_PATIENT_RE = re.compile(r"(^|[\\/])(\d{8})(?=([\\/]|$))")


def extract_patient_id(path: str) -> str:
    m = _PATIENT_RE.search(path)
    if not m:
        raise ValueError(f"Cannot extract patient id from: {path}")
    return m.group(2)


def l2_normalize(x):
    return F.normalize(x, dim=1)


# -------------------------
# Dataset
# -------------------------
class RetrievalDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.paths = []
        for ext in ("png", "jpg", "jpeg"):
            self.paths.extend(
                __import__("glob").glob(
                    os.path.join(root_dir, "**", f"*.{ext}"), recursive=True
                )
            )

        self.paths = sorted(self.paths)
        self.transform = transform
        self.patient_ids = [extract_patient_id(p) for p in self.paths]

        unique_ids = sorted(set(self.patient_ids))
        self.pid_to_label = {pid: i for i, pid in enumerate(unique_ids)}
        self.labels = [self.pid_to_label[pid] for pid in self.patient_ids]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


# -------------------------
# Model
# -------------------------
class Dinov3Retrieval(nn.Module):
    def __init__(self, backbone, feat_dim, proj_dim=128, fine_tune=False):
        super().__init__()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

        if not fine_tune:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x)
        z = self.projector(feat)
        z = l2_normalize(z)
        return z


# -------------------------
# Supervised Contrastive Loss
# -------------------------
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.view(-1, 1)

        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)

        sim = torch.matmul(features, features.T) / self.temperature
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        mask = mask * logits_mask
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        return -mean_log_prob_pos.mean()


# -------------------------
# Training / Validation
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        z = model(imgs)
        loss = criterion(z, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))


def save_checkpoint(epoch, model, optimizer, out_dir, name):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(out_dir, f"{name}.pth"),
    )
    torch.save(
        {"epoch": epoch, "model_state_dict": model.backbone.state_dict()},
        os.path.join(out_dir, f"backbone_{name}.pth"),
    )
    torch.save(
        {"epoch": epoch, "model_state_dict": model.projector.state_dict()},
        os.path.join(out_dir, f"head_{name}.pth"),
    )


def save_best_checkpoint(epoch, model, out_dir, name):
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict()},
        os.path.join(out_dir, f"best_{name}.pth"),
    )
    torch.save(
        {"epoch": epoch, "model_state_dict": model.backbone.state_dict()},
        os.path.join(out_dir, f"best_backbone_{name}.pth"),
    )
    torch.save(
        {"epoch": epoch, "model_state_dict": model.projector.state_dict()},
        os.path.join(out_dir, f"best_head_{name}.pth"),
    )


@torch.no_grad()
def compute_val_loss(model, loader, criterion, device):
    model.eval()
    losses = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        z = model(imgs)
        loss = criterion(z, labels)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def compute_recall_at_k(model, loader, device, ks=(1, 5)):
    model.eval()
    feats, labels = [], []

    for imgs, lbls in loader:
        imgs = imgs.to(device)
        z = model(imgs)
        feats.append(z.cpu())
        labels.append(lbls)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    sims = feats @ feats.T
    sims.fill_diagonal_(-1e9)

    sorted_idx = torch.argsort(sims, dim=1, descending=True)
    recalls = {k: [] for k in ks}

    for i in range(len(labels)):
        pos = labels == labels[i]
        pos[i] = False
        for k in ks:
            hit = pos[sorted_idx[i, :k]].any().item()
            recalls[k].append(hit)

    return {k: float(np.mean(v)) for k, v in recalls.items()}


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-epochs", dest="max_epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--out-dir", default="outputs/train_retrieval")
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--save-name", dest="save_name", default="model")
    parser.add_argument(
        "--early-stopping",
        dest="early_stopping",
        action="store_true",
        help="enable early stopping based on validation metric",
    )
    parser.add_argument(
        "--early-stopping-patience",
        dest="early_stopping_patience",
        type=int,
        default=10,
        help="number of epochs with no improvement before stopping",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        dest="early_stopping_min_delta",
        type=float,
        default=0.0,
        help="minimum improvement to reset early stopping patience",
    )
    parser.add_argument(
        "--early-stopping-monitor",
        dest="early_stopping_monitor",
        choices=["r1", "r5", "val_loss"],
        default="r1",
        help="metric to monitor for early stopping",
    )
    parser.add_argument(
        "--repo-dir",
        dest="repo_dir",
        help="path to the cloned DINOv3 repository",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((cfg["RESIZE_SIZE"], cfg["RESIZE_SIZE"])),
            transforms.CenterCrop(cfg["CENTER_CROP_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_set = RetrievalDataset(args.train_dir, transform)
    valid_set = RetrievalDataset(args.valid_dir, transform)

    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        valid_set, args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    if args.repo_dir:
        dinov3_repo = args.repo_dir
        dinov3_weights = None
    else:
        dinov3_repo, dinov3_weights = get_dinov3_paths()

    weights_path = args.weights
    if not os.path.exists(weights_path) and dinov3_weights:
        candidate = os.path.join(dinov3_weights, weights_path)
        if os.path.exists(candidate):
            weights_path = candidate
    backbone = torch.hub.load(
        dinov3_repo,
        args.model_name,
        source="local",
        weights=weights_path,
    ).to(device)

    feat_dim = backbone.norm.normalized_shape[0]
    model = Dinov3Retrieval(backbone, feat_dim, args.proj_dim, args.fine_tune).to(
        device
    )

    criterion = SupConLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    train_losses, val_losses = [], []
    r1_list, r5_list = [], []
    if args.early_stopping_monitor == "val_loss":
        best_metric = float("inf")
    else:
        best_metric = -float("inf")
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(args.max_epochs):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts}] Epoch [{epoch+1}/{args.max_epochs}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        recalls = compute_recall_at_k(model, valid_loader, device)
        val_loss = None
        if args.early_stopping_monitor == "val_loss":
            val_loss = compute_val_loss(model, valid_loader, criterion, device)

        train_losses.append(train_loss)
        if val_loss is not None:
            val_losses.append(val_loss)
        r1_list.append(recalls[1])
        r5_list.append(recalls[5])

        if args.early_stopping_monitor == "val_loss":
            current_metric = val_loss
            improved = current_metric < (best_metric - args.early_stopping_min_delta)
        elif args.early_stopping_monitor == "r5":
            current_metric = recalls[5]
            improved = current_metric > (best_metric + args.early_stopping_min_delta)
        else:
            current_metric = recalls[1]
            improved = current_metric > (best_metric + args.early_stopping_min_delta)

        if improved:
            best_metric = current_metric
            print(
                f"\nSaving best model for epoch: {epoch+1} "
                f"(monitor={args.early_stopping_monitor}, metric={current_metric:.4f})\n"
            )
            save_best_checkpoint(epoch + 1, model, args.out_dir, args.save_name)

        msg = (
            f"Train Loss: {train_loss:.4f} | "
            f"R@1: {recalls[1]:.4f} | R@5: {recalls[5]:.4f}"
        )
        if val_loss is not None:
            msg += f" | Val Loss: {val_loss:.4f}"
        print(msg)

        epochs_trained = epoch + 1
        if args.early_stopping:
            if improved:
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    "EarlyStopping counter: "
                    f"{patience_counter} of {args.early_stopping_patience}"
                )
            if patience_counter >= args.early_stopping_patience:
                print(
                    "Early stopping triggered "
                    f"(monitor={args.early_stopping_monitor}, "
                    f"patience={args.early_stopping_patience}, "
                    f"min_delta={args.early_stopping_min_delta})"
                )
                break

    save_checkpoint(epochs_trained, model, optimizer, args.out_dir, args.save_name)

    # -------------------------
    # Plot curves
    # -------------------------
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"))

    plt.figure()
    plt.plot(r1_list, label="Recall@1")
    plt.plot(r5_list, label="Recall@5")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.legend()
    plt.savefig(os.path.join(args.out_dir, "recall_curve.png"))

    end_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nTraining finished at {end_ts}.")
    print(f"[Saved] loss_curve.png, recall_curve.png")


if __name__ == "__main__":
    main()
