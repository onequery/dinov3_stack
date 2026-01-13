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
                list(
                    p
                    for p in sorted(
                        glob
                        for glob in __import__("glob").glob(
                            os.path.join(root_dir, "**", f"*.{ext}"), recursive=True
                        )
                    )
                )
            )

        self.transform = transform
        self.patient_ids = [extract_patient_id(p) for p in self.paths]

        # map patient_id -> integer label
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
        feat = self.backbone(x)  # (B, D)
        z = self.projector(feat)  # (B, d)
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

        loss = -mean_log_prob_pos.mean()
        return loss


# -------------------------
# Training
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        z = model(imgs)
        loss = criterion(z, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Valid", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)
            z = model(imgs)
            loss = criterion(z, labels)
            running_loss += loss.item()

    return running_loss / len(loader)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--valid-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument("--out-dir", default="outputs/train_retrieval")
    parser.add_argument("--model-name", default="dinov3_vits16")
    parser.add_argument("--weights", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((cfg["RESIZE_SIZE"], cfg["RESIZE_SIZE"])),
            transforms.CenterCrop(cfg["CENTER_CROP_SIZE"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_set = RetrievalDataset(args.train_dir, transform)
    valid_set = RetrievalDataset(args.valid_dir, transform)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths()
    backbone = torch.hub.load(
        DINOV3_REPO,
        args.model_name,
        source="local",
        weights=os.path.join(DINOV3_WEIGHTS, args.weights),
    ).to(device)

    feat_dim = backbone.norm.normalized_shape[0]

    model = Dinov3Retrieval(
        backbone=backbone,
        feat_dim=feat_dim,
        proj_dim=args.proj_dim,
        fine_tune=args.fine_tune,
    ).to(device)

    criterion = SupConLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    best_loss = float("inf")

    # TODO: Training visualizations
    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, valid_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {"model": model.state_dict()},
                os.path.join(args.out_dir, "best_retrieval_model.pth"),
            )
            print("✓ Saved best model")

    print("Training finished.")


if __name__ == "__main__":
    main()
