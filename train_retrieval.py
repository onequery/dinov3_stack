#!/usr/bin/env python3
"""
Train Retrieval Model with Projection Head on DINOv3 Backbone
(Supervised Contrastive Learning)
"""

import argparse
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

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

from src.utils.common import configure_backbone_trainability, get_dinov3_paths

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


def _unwrap_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return {}

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint

    for key in ("model_state_dict", "state_dict", "model", "teacher", "student", "module"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            nested = _unwrap_state_dict(value)
            if nested:
                return nested

    for value in checkpoint.values():
        if isinstance(value, dict):
            nested = _unwrap_state_dict(value)
            if nested:
                return nested

    return {}


def _extract_backbone_state_dict(
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], str]:
    prefix_candidates = (
        "module.backbone.",
        "backbone.",
        "teacher.backbone.",
        "student.backbone.",
        "model.backbone.",
    )
    for prefix in prefix_candidates:
        extracted = {
            k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
        }
        if extracted:
            return extracted, f"prefix:{prefix}"

    if any(k.startswith("module.") for k in state_dict.keys()):
        stripped = {
            k[len("module.") :]: v for k, v in state_dict.items() if k.startswith("module.")
        }
        if stripped:
            return stripped, "prefix:module."

    return state_dict, "raw"


def _find_pretrain_config(weights_path: str) -> str | None:
    path = Path(weights_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    for parent in [path.parent, *path.parents]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return str(candidate)
    return None


def _load_pretrain_config(weights_path: str) -> Tuple[str | None, Dict[str, Any]]:
    config_path = _find_pretrain_config(weights_path)
    if not config_path:
        return None, {}
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return config_path, cfg if isinstance(cfg, dict) else {}


def _build_model_kwargs_from_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    student = cfg.get("student", {}) if isinstance(cfg, dict) else {}
    if not isinstance(student, dict):
        return {}

    keys = (
        "img_size",
        "patch_size",
        "drop_path_rate",
        "ffn_layer",
        "ffn_ratio",
        "qkv_bias",
        "proj_bias",
        "ffn_bias",
        "norm_layer",
        "layerscale_init",
        "n_storage_tokens",
        "mask_k_bias",
        "untie_cls_and_patch_norms",
        "untie_global_and_local_cls_norm",
        "pos_embed_rope_base",
        "pos_embed_rope_min_period",
        "pos_embed_rope_max_period",
        "pos_embed_rope_normalize_coords",
        "pos_embed_rope_shift_coords",
        "pos_embed_rope_jitter_coords",
        "pos_embed_rope_rescale_coords",
        "pos_embed_rope_dtype",
        "in_chans",
    )

    model_kwargs: Dict[str, Any] = {}
    crops = cfg.get("crops", {}) if isinstance(cfg, dict) else {}
    global_crops = crops.get("global_crops_size") if isinstance(crops, dict) else None
    if isinstance(global_crops, list) and global_crops:
        model_kwargs["img_size"] = max(global_crops)
    elif isinstance(global_crops, int):
        model_kwargs["img_size"] = global_crops

    if "patch_size" in student and student["patch_size"] is not None:
        model_kwargs["patch_size"] = student["patch_size"]
    if "drop_path_rate" in student and student["drop_path_rate"] is not None:
        model_kwargs["drop_path_rate"] = student["drop_path_rate"]
    if "layerscale" in student and student["layerscale"] is not None:
        model_kwargs["layerscale_init"] = student["layerscale"]

    for key in keys:
        if key in student and student[key] is not None:
            model_kwargs[key] = student[key]
    return model_kwargs


def load_backbone_from_local_checkpoint(
    repo_dir: str,
    model_name: str,
    weights_path: str,
    device: str,
):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Local weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    raw_state_dict = _unwrap_state_dict(checkpoint)
    if not raw_state_dict:
        raise ValueError(
            f"Could not parse a valid state_dict from checkpoint: {weights_path}"
        )

    candidate_state_dict, state_source = _extract_backbone_state_dict(raw_state_dict)

    config_path, cfg = _load_pretrain_config(weights_path)
    model_kwargs = _build_model_kwargs_from_cfg(cfg) if cfg else {}
    student_cfg = cfg.get("student", {}) if isinstance(cfg, dict) else {}
    arch = student_cfg.get("arch") if isinstance(student_cfg, dict) else None
    if not arch:
        model_name_to_arch = {
            "dinov3_vits16": "vit_small",
            "dinov3_vits16plus": "vit_small",
            "dinov3_vitb16": "vit_base",
            "dinov3_vitl16": "vit_large",
            "dinov3_vit7b16": "vit_7b",
        }
        arch = model_name_to_arch.get(model_name)

    embed_dim_from_arch = {
        "vit_small": 384,
        "vit_base": 768,
        "vit_large": 1024,
        "vit_huge2": 1280,
        "vit_giant2": 1536,
        "vit_7b": 4096,
    }
    embed_dim = embed_dim_from_arch.get(arch, 384)

    if not cfg and isinstance(arch, str) and arch.startswith("vit_"):
        model_kwargs.setdefault("norm_layer", "layernormbf16")
        model_kwargs.setdefault("pos_embed_rope_base", 100)
        model_kwargs.setdefault("pos_embed_rope_normalize_coords", "separate")
        model_kwargs.setdefault("pos_embed_rope_rescale_coords", 2)
        model_kwargs.setdefault("pos_embed_rope_dtype", "fp32")

    # Infer key architectural flags from checkpoint to avoid mismatch.
    if any(k.endswith("attn.qkv.bias") for k in candidate_state_dict.keys()):
        model_kwargs["qkv_bias"] = True
    else:
        model_kwargs["qkv_bias"] = False

    if "storage_tokens" in candidate_state_dict and torch.is_tensor(
        candidate_state_dict["storage_tokens"]
    ):
        storage_tokens = candidate_state_dict["storage_tokens"]
        if storage_tokens.ndim == 3:
            model_kwargs["n_storage_tokens"] = int(storage_tokens.shape[1])

    if any(k.endswith("attn.qkv.bias_mask") for k in candidate_state_dict.keys()):
        model_kwargs["mask_k_bias"] = True

    if any(
        k.endswith(".ls1.gamma") or k.endswith(".ls2.gamma")
        for k in candidate_state_dict.keys()
    ):
        model_kwargs["layerscale_init"] = 1.0e-5

    if any(k.endswith("local_cls_norm.weight") for k in candidate_state_dict.keys()):
        model_kwargs["untie_global_and_local_cls_norm"] = True

    if any(
        k.endswith("cls_norm.weight") and "local_cls_norm" not in k
        for k in candidate_state_dict.keys()
    ):
        model_kwargs["untie_cls_and_patch_norms"] = True

    if any(k.endswith("mlp.w1.weight") for k in candidate_state_dict.keys()):
        model_kwargs["ffn_layer"] = "swiglu64"
        for k, v in candidate_state_dict.items():
            if k.endswith("mlp.w1.weight") and torch.is_tensor(v):
                swiglu_hidden = v.shape[0]
                hidden_features = int(swiglu_hidden * 3 / 2)
                model_kwargs["ffn_ratio"] = hidden_features / float(embed_dim)
                break
    elif any(k.endswith("mlp.fc1.weight") for k in candidate_state_dict.keys()):
        model_kwargs["ffn_layer"] = "mlp"
        for k, v in candidate_state_dict.items():
            if k.endswith("mlp.fc1.weight") and torch.is_tensor(v):
                model_kwargs["ffn_ratio"] = v.shape[0] / float(embed_dim)
                break
    if arch and isinstance(arch, str):
        repo_path = Path(repo_dir).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        from dinov3.models import convnext as convnext_models
        from dinov3.models.vision_transformer import DinoVisionTransformer

        vit_arch = {
            "vit_small": {"embed_dim": 384, "depth": 12, "num_heads": 6, "ffn_ratio": 4.0},
            "vit_base": {"embed_dim": 768, "depth": 12, "num_heads": 12, "ffn_ratio": 4.0},
            "vit_large": {"embed_dim": 1024, "depth": 24, "num_heads": 16, "ffn_ratio": 4.0},
            "vit_huge2": {"embed_dim": 1280, "depth": 32, "num_heads": 20, "ffn_ratio": 4.0},
            "vit_giant2": {"embed_dim": 1536, "depth": 40, "num_heads": 24, "ffn_ratio": 4.0},
            "vit_7b": {"embed_dim": 4096, "depth": 40, "num_heads": 32, "ffn_ratio": 3.0},
        }

        if arch in vit_arch:
            arch_cfg = vit_arch[arch]
            img_size = model_kwargs.pop("img_size", 224)
            patch_size = model_kwargs.pop("patch_size", 16)
            ffn_ratio = model_kwargs.pop("ffn_ratio", arch_cfg["ffn_ratio"])
            backbone = DinoVisionTransformer(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=arch_cfg["embed_dim"],
                depth=arch_cfg["depth"],
                num_heads=arch_cfg["num_heads"],
                ffn_ratio=ffn_ratio,
                **model_kwargs,
            ).to(device)
            builder = f"DinoVisionTransformer.{arch}"
        elif "convnext" in arch:
            convnext_cls = convnext_models.get_convnext_arch(arch)
            backbone = convnext_cls(**model_kwargs).to(device)
            builder = f"convnext.{arch}"
        else:
            backbone = torch.hub.load(
                repo_dir, model_name, source="local", pretrained=False
            ).to(device)
            builder = "hub"
    else:
        backbone = torch.hub.load(
            repo_dir, model_name, source="local", pretrained=False
        ).to(device)
        builder = "hub"

    model_state = backbone.state_dict()

    loadable_state_dict = {}
    skipped_not_found = 0
    skipped_shape = 0
    skipped_not_found_keys = []
    skipped_shape_keys = []
    for key, value in candidate_state_dict.items():
        if not torch.is_tensor(value):
            continue
        if key not in model_state:
            skipped_not_found += 1
            if len(skipped_not_found_keys) < 10:
                skipped_not_found_keys.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape += 1
            if len(skipped_shape_keys) < 10:
                skipped_shape_keys.append(
                    f"{key}: ckpt={tuple(value.shape)} model={tuple(model_state[key].shape)}"
                )
            continue
        loadable_state_dict[key] = value

    if not loadable_state_dict:
        raise ValueError(
            "No compatible backbone weights were found in the checkpoint. "
            f"model={model_name}, file={weights_path}, source={state_source}"
        )

    if skipped_not_found > 0 or skipped_shape > 0:
        raise ValueError(
            "Backbone checkpoint does not fully match the constructed architecture. "
            f"model={model_name}, file={weights_path}, source={state_source}, "
            f"skipped_key={skipped_not_found}, skipped_shape={skipped_shape}, "
            f"sample_missing={skipped_not_found_keys}, sample_shape_mismatch={skipped_shape_keys}"
        )

    strict_load = set(model_state.keys()).issubset(set(loadable_state_dict.keys()))
    incompatible = backbone.load_state_dict(loadable_state_dict, strict=strict_load)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Backbone load resulted in incompatible keys. "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}, "
            f"sample_missing={incompatible.missing_keys[:10]}, "
            f"sample_unexpected={incompatible.unexpected_keys[:10]}"
        )
    load_report = {
        "state_source": state_source,
        "config_path": config_path,
        "builder": builder,
        "strict_load": strict_load,
        "candidate_total": len(candidate_state_dict),
        "loaded_total": len(loadable_state_dict),
        "skipped_not_found": skipped_not_found,
        "skipped_shape": skipped_shape,
        "missing_after_load": len(incompatible.missing_keys),
        "unexpected_after_load": len(incompatible.unexpected_keys),
    }
    return backbone, load_report


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
    def __init__(
        self,
        backbone,
        backbone_embed_dim,
        retrieval_embedding_dim=128,
        fine_tune=False,
        unfreeze_last_n_blocks=None,
    ):
        super().__init__()
        self.backbone = backbone

        self.projector = nn.Sequential(
            nn.Linear(backbone_embed_dim, backbone_embed_dim),
            nn.BatchNorm1d(backbone_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(backbone_embed_dim, retrieval_embedding_dim),
        )

        self.backbone_trainability = configure_backbone_trainability(
            self.backbone,
            fine_tune=fine_tune,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
        )

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
    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="learning rate for backbone parameters during full fine-tuning",
    )
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--fine-tune", action="store_true")
    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=None,
        help=(
            "number of last ViT blocks to unfreeze when --fine-tune is set. "
            "Use 0 for linear probe, 12 for full fine-tune on ViT-S/16."
        ),
    )
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
    if args.unfreeze_blocks is not None and args.unfreeze_blocks < 0:
        raise ValueError("--unfreeze-blocks must be >= 0")
    if not args.fine_tune and args.unfreeze_blocks not in (None, 0):
        raise ValueError(
            "--unfreeze-blocks > 0 requires --fine-tune. "
            "For linear probe, use --unfreeze-blocks 0 without --fine-tune."
        )

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
    backbone, load_report = load_backbone_from_local_checkpoint(
        dinov3_repo,
        args.model_name,
        weights_path,
        device,
    )

    backbone_embed_dim = backbone.norm.normalized_shape[0]
    model = Dinov3Retrieval(
        backbone,
        backbone_embed_dim=backbone_embed_dim,
        retrieval_embedding_dim=args.proj_dim,
        fine_tune=args.fine_tune,
        unfreeze_last_n_blocks=args.unfreeze_blocks,
    ).to(
        device
    )

    criterion = SupConLoss()
    if args.fine_tune and args.backbone_lr is not None:
        optimizer = optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": args.backbone_lr},
                {"params": model.projector.parameters(), "lr": args.lr},
            ]
        )
    else:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    start_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{start_ts}] Setup | model={args.model_name} | fine_tune={args.fine_tune} "
        f"| backbone_embed_dim={backbone_embed_dim} "
        f"| retrieval_embedding_dim={args.proj_dim} "
        f"| proj_dim={args.proj_dim} "
        f"| batch_size={args.batch_size} "
        f"| num_workers={args.num_workers} | lr={args.lr} "
        f"| backbone_lr={args.backbone_lr if args.backbone_lr is not None else 'default'} "
        f"| unfreeze_blocks={args.unfreeze_blocks if args.unfreeze_blocks is not None else 'full'} "
        f"| max_epochs={args.max_epochs} | early_stopping={args.early_stopping} "
        f"| weights={weights_path} "
        f"| monitor={args.early_stopping_monitor} "
        f"| patience={args.early_stopping_patience}"
    )
    print(f"[{start_ts}] Backbone trainability | {model.backbone_trainability}")
    print(
        f"[{start_ts}] Backbone load | source={load_report['state_source']} "
        f"| config={load_report['config_path'] if load_report['config_path'] else 'none'} "
        f"| builder={load_report['builder']} "
        f"| strict={load_report['strict_load']} "
        f"| loaded={load_report['loaded_total']}/{load_report['candidate_total']} "
        f"| skipped_key={load_report['skipped_not_found']} "
        f"| skipped_shape={load_report['skipped_shape']} "
        f"| missing={load_report['missing_after_load']} "
        f"| unexpected={load_report['unexpected_after_load']}"
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
