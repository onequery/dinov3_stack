import os
import sys

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_dinov3_paths(require_repo=True, require_weights=True):
    """
    Get DINOv3 repository and weights paths from environment variables.

    Args:
        require_repo (bool): Whether repository path must be present and valid.
        require_weights (bool): Whether weights path must be present and valid.

    Returns:
        tuple: (repo_path, weights_path) as resolved strings or None

    Raises:
        FileNotFoundError: If required paths do not exist
        ValueError: If required environment variables are not set
    """
    # Get paths from environment
    repo_path_str = os.getenv("DINOv3_REPO")
    weights_path_str = os.getenv("DINOv3_WEIGHTS")

    # Check if required environment variables are set
    if require_repo and not repo_path_str:
        raise ValueError(
            "DINOV3_REPO not found in environment variables. "
            "Please set DINOV3_REPO=/path/to/dinov3/repo in your .env file"
        )

    if require_weights and not weights_path_str:
        raise ValueError(
            "DINOV3_WEIGHTS not found in environment variables. "
            "Please set DINOV3_WEIGHTS=/path/to/weights in your .env file"
        )

    resolved_repo_path = None
    resolved_weights_path = None

    if repo_path_str:
        repo_path = Path(repo_path_str).expanduser().resolve()
        if require_repo and not repo_path.exists():
            raise FileNotFoundError(
                f"DINOv3 repository not found at: {repo_path}\n"
                f"Please check your DINOV3_REPO path in .env file"
            )
        resolved_repo_path = str(repo_path)

    if weights_path_str:
        weights_path = Path(weights_path_str).expanduser().resolve()
        if require_weights and not weights_path.exists():
            raise FileNotFoundError(
                f"DINOv3 weights not found at: {weights_path}\n"
                f"Please check your DINOV3_WEIGHTS path in .env file"
            )
        resolved_weights_path = str(weights_path)

    return resolved_repo_path, resolved_weights_path


def configure_backbone_trainability(
    backbone,
    fine_tune: bool,
    unfreeze_last_n_blocks: int | None = None,
):
    """
    Configure trainable parameters for ViT-like backbones.

    Args:
        backbone: backbone model instance.
        fine_tune: if False, freeze all backbone parameters.
        unfreeze_last_n_blocks: number of last transformer blocks to unfreeze.
            - None: full fine-tuning when fine_tune=True
            - 0: keep backbone frozen
            - N>0: unfreeze only last N blocks (+final norm if present)

    Returns:
        dict: trainability report with keys
            mode, total_blocks, unfrozen_blocks.
    """
    if unfreeze_last_n_blocks is not None and unfreeze_last_n_blocks < 0:
        raise ValueError(
            f"unfreeze_last_n_blocks must be >= 0, got {unfreeze_last_n_blocks}"
        )

    for param in backbone.parameters():
        param.requires_grad = False

    blocks = getattr(backbone, "blocks", None)
    total_blocks = len(blocks) if blocks is not None else None

    if not fine_tune:
        return {
            "mode": "head_only",
            "total_blocks": total_blocks,
            "unfrozen_blocks": 0,
        }

    if unfreeze_last_n_blocks is None:
        for param in backbone.parameters():
            param.requires_grad = True
        return {
            "mode": "full_finetune",
            "total_blocks": total_blocks,
            "unfrozen_blocks": total_blocks,
        }

    if total_blocks is None:
        if unfreeze_last_n_blocks == 0:
            return {
                "mode": "head_only",
                "total_blocks": None,
                "unfrozen_blocks": 0,
            }
        for param in backbone.parameters():
            param.requires_grad = True
        return {
            "mode": "full_finetune_non_vit_fallback",
            "total_blocks": None,
            "unfrozen_blocks": None,
        }

    if unfreeze_last_n_blocks > total_blocks:
        raise ValueError(
            f"Requested unfreeze_last_n_blocks={unfreeze_last_n_blocks}, "
            f"but backbone has only {total_blocks} blocks."
        )

    if unfreeze_last_n_blocks == total_blocks:
        for param in backbone.parameters():
            param.requires_grad = True
        return {
            "mode": "full_finetune",
            "total_blocks": total_blocks,
            "unfrozen_blocks": total_blocks,
        }

    if unfreeze_last_n_blocks == 0:
        return {
            "mode": "head_only",
            "total_blocks": total_blocks,
            "unfrozen_blocks": 0,
        }

    start_idx = total_blocks - unfreeze_last_n_blocks
    for block in blocks[start_idx:]:
        for param in block.parameters():
            param.requires_grad = True

    norm = getattr(backbone, "norm", None)
    if norm is not None and hasattr(norm, "parameters"):
        for param in norm.parameters():
            param.requires_grad = True

    return {
        "mode": "partial_finetune",
        "total_blocks": total_blocks,
        "unfrozen_blocks": unfreeze_last_n_blocks,
    }


if __name__ == "__main__":
    try:
        repo, weights = get_dinov3_paths()
        print(f"DINOv3 Repository: {repo}")
        print(f"DINOv3 Weights: {weights}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
