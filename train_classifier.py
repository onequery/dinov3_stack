import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from tqdm.auto import tqdm

from src.img_cls.datasets import get_data_loaders, get_datasets
from src.img_cls.model import Dinov3Classification
from src.img_cls.utils import SaveBestModel, save_model, save_plots
from src.utils.lora import count_lora_params
from src.utils.common import get_dinov3_paths

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def log_with_time(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-e",
    "--max-epochs",
    type=int,
    dest="max_epochs",
    default=10,
    help="Maximum number of epochs to train for",
)
parser.add_argument(
    "-lr",
    "--learning-rate",
    type=float,
    dest="learning_rate",
    default=0.001,
    help="Learning rate for training the model",
)
parser.add_argument(
    "--backbone-lr",
    type=float,
    dest="backbone_lr",
    default=None,
    help="Learning rate for backbone parameters during full fine-tuning",
)
parser.add_argument("-b", "--batch-size", dest="batch_size", default=32, type=int)
parser.add_argument(
    "--num-workers",
    dest="num_workers",
    default=4,
    type=int,
    help="number of dataloader worker processes",
)
parser.add_argument(
    "--save-name",
    dest="save_name",
    default="model",
    help="file name of the final model to save",
)
parser.add_argument(
    "--fine-tune",
    dest="fine_tune",
    action="store_true",
    help="whether to fine-tune the model or train the classifier layer only",
)
parser.add_argument(
    "--head-size",
    dest="head_size",
    choices=["small", "big"],
    default="small",
    help="classification head size",
)
parser.add_argument(
    "--head-hidden-dim",
    dest="head_hidden_dim",
    type=int,
    default=None,
    help="hidden dim for big head MLP",
)
parser.add_argument(
    "--enable-lora",
    dest="enable_lora",
    action="store_true",
    help="enable LoRA injection on ViT attention qkv/proj",
)
parser.add_argument(
    "--lora-rank",
    dest="lora_rank",
    type=int,
    default=None,
    help="LoRA rank",
)
parser.add_argument(
    "--lora-alpha",
    dest="lora_alpha",
    type=int,
    default=None,
    help="LoRA alpha (default: rank)",
)
parser.add_argument(
    "--lora-dropout",
    dest="lora_dropout",
    type=float,
    default=0.0,
    help="LoRA dropout",
)
parser.add_argument(
    "--lora-target",
    dest="lora_target",
    choices=["attn_qkv_proj"],
    default="attn_qkv_proj",
    help="LoRA injection target",
)
parser.add_argument(
    "--save-config-json",
    dest="save_config_json",
    default=None,
    help="optional output path for resolved run config json",
)
parser.add_argument(
    "--unfreeze-blocks",
    dest="unfreeze_blocks",
    type=int,
    default=None,
    help=(
        "number of last ViT blocks to unfreeze when --fine-tune is set. "
        "Use 0 for linear probe, 12 for full fine-tune on ViT-S/16."
    ),
)
parser.add_argument(
    "--out-dir",
    dest="out_dir",
    default="img_cls",
    help="output sub-directory path inside the `outputs` directory",
)
parser.add_argument(
    "--scheduler",
    type=int,
    nargs="+",
    default=[1000],
    help="number of epochs after which learning rate scheduler is applied",
)
parser.add_argument(
    "--train-dir",
    dest="train_dir",
    required=True,
    help="path to the training directory containing class folders in \
          PyTorch ImageFolder format",
)
parser.add_argument(
    "--valid-dir",
    dest="valid_dir",
    required=True,
    help="path to the validation directory containing class folders in \
          PyTorch ImageFolder format",
)
parser.add_argument(
    "--weights", help="path to the pretrained backbone weights", required=True
)
parser.add_argument(
    "--repo-dir",
    dest="repo_dir",
    help="path to the cloned DINOv3 repository",
)
parser.add_argument(
    "--model-name",
    dest="model_name",
    help="name of the model, check: https://github.com/facebookresearch/dinov3?tab=readme-ov-file#pretrained-backbones-via-pytorch-hub",
    default="dinov3_vits16",
)
parser.add_argument(
    "--config", required=True, help="yaml file with IMAGE_SIZE and CENTER_CROP_SIZE"
)
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
    choices=["val_loss", "val_acc"],
    default="val_loss",
    help="metric to monitor for early stopping",
)
args = parser.parse_args()
log_with_time(f"Args: {args}")
if args.unfreeze_blocks is not None and args.unfreeze_blocks < 0:
    raise ValueError("--unfreeze-blocks must be >= 0")
if not args.fine_tune and args.unfreeze_blocks not in (None, 0):
    raise ValueError(
        "--unfreeze-blocks > 0 requires --fine-tune. "
        "For linear probe, use --unfreeze-blocks 0 without --fine-tune."
    )
if args.enable_lora and (args.lora_rank is None or args.lora_rank <= 0):
    raise ValueError("--enable-lora requires --lora-rank > 0")
if args.head_size == "big" and (args.head_hidden_dim is None or args.head_hidden_dim <= 0):
    raise ValueError("--head-size big requires --head-hidden-dim > 0")


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


def resolve_weights_path(weights_arg, env_weights_dir):
    candidate_path = os.path.expanduser(weights_arg)
    if os.path.isabs(candidate_path):
        if os.path.exists(candidate_path):
            return candidate_path
        raise FileNotFoundError(f"Pretrained weights not found at: {candidate_path}")

    local_candidate = os.path.abspath(candidate_path)
    if os.path.exists(local_candidate):
        return local_candidate

    if env_weights_dir:
        env_candidate = os.path.join(env_weights_dir, candidate_path)
        if os.path.exists(env_candidate):
            return env_candidate

    raise FileNotFoundError(
        "Pretrained weights not found. "
        f"Checked local path '{local_candidate}'"
        + (
            f" and '{os.path.join(env_weights_dir, candidate_path)}'."
            if env_weights_dir
            else "."
        )
    )


# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    log_with_time("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100.0 * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    log_with_time("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100.0 * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


if __name__ == "__main__":
    DINOV3_REPO, DINOV3_WEIGHTS = get_dinov3_paths(
        require_repo=not bool(args.repo_dir),
        require_weights=False,
    )
    repo_dir = resolve_repo_path(args.repo_dir, DINOV3_REPO)
    weights_path = resolve_weights_path(args.weights, DINOV3_WEIGHTS)
    log_with_time(f"DINOv3 repo: {repo_dir}")
    log_with_time(f"Backbone weights: {weights_path}")

    log_with_time("=============== CUDA Device ===============")
    log_with_time(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    cuda_device_count = torch.cuda.device_count()
    log_with_time(f"torch.cuda.device_count(): {cuda_device_count}")
    if cuda_device_count > 0:
        log_with_time(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        log_with_time(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
    else:
        log_with_time("CUDA is not available. Running on CPU.")
    log_with_time("===========================================")

    # Create a directory with the model name for outputs.
    # out_dir = os.path.join("outputs", args.out_dir)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # Load the training and validation datasets.
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_train, dataset_valid, dataset_classes = get_datasets(
        train_dir=args.train_dir,
        valid_dir=args.valid_dir,
        resize=config["RESIZE_SIZE"],
        center_crop=config["CENTER_CROP_SIZE"],
    )
    log_with_time(f"[INFO]: Number of training images: {len(dataset_train)}")
    log_with_time(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    log_with_time(f"[INFO]: Classes: {dataset_classes}")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(
        dataset_train,
        dataset_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Learning_parameters.
    lr = args.learning_rate
    epochs = args.max_epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_with_time(f"Computation device: {device}")
    if args.fine_tune and args.backbone_lr is not None:
        log_with_time(f"Learning rates: head={lr}, backbone={args.backbone_lr}")
    else:
        log_with_time(f"Learning rate: {lr}")
    log_with_time(f"Max epochs to train for: {epochs}")
    if args.early_stopping:
        log_with_time(
            "Early stopping enabled "
            f"(monitor={args.early_stopping_monitor}, "
            f"patience={args.early_stopping_patience}, "
            f"min_delta={args.early_stopping_min_delta})"
        )

    # Load the model.
    model = Dinov3Classification(
        num_classes=len(dataset_classes),
        fine_tune=args.fine_tune,
        unfreeze_last_n_blocks=args.unfreeze_blocks,
        head_size=args.head_size,
        head_hidden_dim=args.head_hidden_dim,
        enable_lora=args.enable_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
        weights=weights_path,
        model_name=args.model_name,
        repo_dir=repo_dir,
    ).to(device)
    print(model)
    log_with_time(f"Backbone trainability: {model.backbone_trainability}")

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    log_with_time(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    log_with_time(f"{total_trainable_params:,} training parameters.")
    lora_params = count_lora_params(model)
    log_with_time(f"{lora_params:,} LoRA parameters.")
    param_stats = {
        "total_params": int(total_params),
        "trainable_params": int(total_trainable_params),
        "lora_params": int(lora_params),
        "head_info": getattr(model, "head_info", {}),
        "backbone_trainability": getattr(model, "backbone_trainability", {}),
        "lora_info": getattr(model, "lora_info", None),
    }
    with open(os.path.join(out_dir, "param_stats.json"), "w") as f:
        json.dump(param_stats, f, indent=2)
    run_config = {
        "args": vars(args),
        "resolved": {
            "repo_dir": repo_dir,
            "weights_path": weights_path,
            "device": device,
            "out_dir": out_dir,
        },
    }
    run_config_path = args.save_config_json or os.path.join(out_dir, "run_config.json")
    with open(run_config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    # Optimizer.
    if args.fine_tune and args.backbone_lr is not None:
        optimizer = optim.SGD(
            [
                {"params": model.backbone_model.parameters(), "lr": args.backbone_lr},
                {"params": model.head.parameters(), "lr": lr},
            ],
            momentum=0.9,
            nesterov=True,
        )
    else:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            momentum=0.9,
            nesterov=True,
        )
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Initialize `SaveBestModel` class.
    save_best_model = SaveBestModel()

    # Scheduler.
    scheduler = MultiStepLR(optimizer, milestones=args.scheduler, gamma=0.1)

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    epochs_trained = 0
    if args.early_stopping_monitor == "val_loss":
        best_metric = float("inf")
    else:
        best_metric = -float("inf")
    patience_counter = 0
    for epoch in range(epochs):
        log_with_time(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader, optimizer, criterion
        )
        valid_epoch_loss, valid_epoch_acc = validate(
            model, valid_loader, criterion, dataset_classes
        )
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        log_with_time(
            f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}"
        )
        log_with_time(
            f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"
        )
        save_best_model(valid_epoch_loss, epoch, model, out_dir, args.save_name)
        epochs_trained = epoch + 1
        if args.early_stopping:
            if args.early_stopping_monitor == "val_loss":
                current_metric = valid_epoch_loss
                improved = current_metric < (
                    best_metric - args.early_stopping_min_delta
                )
            else:
                current_metric = valid_epoch_acc
                improved = current_metric > (
                    best_metric + args.early_stopping_min_delta
                )
            if improved:
                best_metric = current_metric
                patience_counter = 0
            else:
                patience_counter += 1
                log_with_time(
                    "EarlyStopping counter: "
                    f"{patience_counter} of {args.early_stopping_patience}"
                )
                if patience_counter >= args.early_stopping_patience:
                    log_with_time(f"Early stopping triggered at epoch {epoch+1}")
                    break
        log_with_time("-" * 50)
        scheduler.step()
        last_lr = scheduler.get_last_lr()
        log_with_time(f"LR for next epoch: {last_lr}")

    # Save the trained model weights.
    save_model(epochs_trained, model, optimizer, criterion, out_dir, args.save_name)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, out_dir)
    log_with_time("TRAINING COMPLETE")
