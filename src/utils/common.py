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
    repo_path_str = os.getenv('DINOv3_REPO')
    weights_path_str = os.getenv('DINOv3_WEIGHTS')

    # Check if required environment variables are set
    if require_repo and not repo_path_str:
        raise ValueError(
            'DINOV3_REPO not found in environment variables. '
            'Please set DINOV3_REPO=/path/to/dinov3/repo in your .env file'
        )

    if require_weights and not weights_path_str:
        raise ValueError(
            'DINOV3_WEIGHTS not found in environment variables. '
            'Please set DINOV3_WEIGHTS=/path/to/weights in your .env file'
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


if __name__ == '__main__':
    try:
        repo, weights = get_dinov3_paths()
        print(f'DINOv3 Repository: {repo}')
        print(f'DINOv3 Weights: {weights}')
    except (ValueError, FileNotFoundError) as e:
        print(f'Error: {e}')
