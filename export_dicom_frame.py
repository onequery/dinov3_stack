import argparse
import json
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
DEFAULT_DCM_ROOT = Path("input/stent_split_dcm")
DEFAULT_IMG_ROOT = Path("input/stent_split_img")
DEFAULT_LABEL_JSON = Path("input/frames_prediction.json")
DEFAULT_FRAME_INDEX_BASE = 0  # 0 if indices are 0-based, 1 if indices are 1-based


# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def normalize_to_uint8(img: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Clip to [low, high] and normalize to uint8 [0, 255]
    """
    img = np.clip(img, low, high)
    img = (img - low) / (high - low)
    img = (img * 255.0).astype(np.uint8)
    return img


def get_bits_stored_range(ds):
    bits_stored = int(ds.BitsStored)
    pixel_repr = int(ds.PixelRepresentation)  # 0=unsigned, 1=signed

    if pixel_repr == 0:
        return 0, (2**bits_stored) - 1
    return -(2 ** (bits_stored - 1)), (2 ** (bits_stored - 1)) - 1


# --------------------------------------------------
# Core: extract selected frame safely
# --------------------------------------------------
def extract_frame_uint8(pixel: np.ndarray, ds, frame_index: int) -> np.ndarray:
    """
    Return selected frame as uint8 numpy array
    following DICOM standard priority:
    VOI LUT > WC/WW > BitsStored fallback
    """
    if "PixelData" not in ds:
        raise ValueError("No PixelData")

    if pixel.ndim == 3:
        if frame_index < 0 or frame_index >= pixel.shape[0]:
            raise IndexError(f"Frame index out of range: {frame_index}")
        frame = pixel[frame_index]
    else:
        if frame_index not in (0, -1):
            raise IndexError(f"Single-frame DICOM, index {frame_index} is invalid")
        frame = pixel

    # 1. VOI LUT (best / standard)
    try:
        pixel_voi = apply_voi_lut(frame, ds)
        if pixel_voi is not None:
            low, high = pixel_voi.min(), pixel_voi.max()
            return normalize_to_uint8(pixel_voi, low, high)
    except Exception:
        pass

    # 2. Window Center / Width
    if "WindowCenter" in ds and "WindowWidth" in ds:
        wc = ds.WindowCenter
        ww = ds.WindowWidth

        # Multi-valued handling
        wc = float(wc[0] if isinstance(wc, (list, pydicom.multival.MultiValue)) else wc)
        ww = float(ww[0] if isinstance(ww, (list, pydicom.multival.MultiValue)) else ww)

        low = wc - ww / 2.0
        high = wc + ww / 2.0

        return normalize_to_uint8(frame, low, high)

    # 3. BitsStored fallback (raw faithful scaling)
    low, high = get_bits_stored_range(ds)
    return normalize_to_uint8(frame, low, high)


# --------------------------------------------------
# Save PNG
# --------------------------------------------------
def save_frame_png(dcm_path: Path, out_png_path: Path, frame_index: int) -> bool:
    try:
        ds = pydicom.dcmread(dcm_path, force=True)
        pixel = ds.pixel_array.astype(np.float32)
        img_uint8 = extract_frame_uint8(pixel, ds, frame_index)

        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_uint8, mode="L").save(out_png_path)

        return True
    except Exception as e:
        print(f"[SKIP] {dcm_path} | {e}")
        return False


# --------------------------------------------------
# Label loading
# --------------------------------------------------
def build_dcm_index(dcm_root: Path):
    dcm_files = list(dcm_root.rglob("*.dcm"))
    index = {}
    duplicates = 0
    for dcm_path in dcm_files:
        key = Path(*dcm_path.parts[-4:])
        if key in index:
            duplicates += 1
            continue
        index[key] = dcm_path
    return index, duplicates, len(dcm_files)


def iter_label_records(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                f.seek(0)
                data = json.load(f)
                if isinstance(data, list):
                    for record in data:
                        yield record
                else:
                    raise ValueError("Unsupported JSON format for labels")
                return


def load_labels(dcm_index: dict, label_json: Path, frame_index_base: int):
    labels = {}
    missing = 0
    invalid = 0

    for record in tqdm(iter_label_records(label_json), desc="Reading labels"):
        filename = record.get("filename")
        data = record.get("data", [])
        if not filename or not isinstance(data, list):
            invalid += 1
            continue

        key = Path(filename)
        dcm_path = dcm_index.get(key)
        if dcm_path is None:
            missing += 1
            continue

        for item in data:
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index")
            if raw_index is None:
                continue
            try:
                frame_index = int(raw_index) - frame_index_base
            except (TypeError, ValueError):
                continue
            if frame_index < 0:
                continue
            labels.setdefault(dcm_path, []).append(frame_index)

    return labels, missing, invalid


# --------------------------------------------------
# Main loop
# --------------------------------------------------
def build_output_path(dcm_path: Path, dcm_root: Path, img_root: Path, frame_index: int) -> Path:
    rel_path = dcm_path.relative_to(dcm_root)
    out_dir = img_root / rel_path.parent
    out_name = f"{rel_path.stem}_frame{frame_index:04d}.png"
    return out_dir / out_name


def extract_labeled_frames(dcm_root: Path, img_root: Path, label_json: Path, frame_index_base: int):
    dcm_index, duplicates, total_dcms = build_dcm_index(dcm_root)
    labels, missing, invalid = load_labels(dcm_index, label_json, frame_index_base)

    print(f"Found {total_dcms} DCM files")
    if duplicates:
        print(f"[WARN] Duplicate filename keys skipped: {duplicates}")
    if missing:
        print(f"[WARN] Label entries with missing DICOM: {missing}")
    if invalid:
        print(f"[WARN] Invalid label records skipped: {invalid}")

    success = 0
    failed = 0

    for dcm_path, frame_indices in tqdm(labels.items(), desc="Extracting frames"):
        for frame_index in sorted(set(frame_indices)):
            out_png_path = build_output_path(dcm_path, dcm_root, img_root, frame_index)
            if save_frame_png(dcm_path, out_png_path, frame_index):
                success += 1
            else:
                failed += 1

    print("\n========== DONE ==========")
    print(f"Success: {success}")
    print(f"Failed : {failed}")
    print(f"Output directory: {img_root.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract labeled frames from DICOM files.")
    parser.add_argument("--dcm-root", type=Path, default=DEFAULT_DCM_ROOT, help="Root directory of DICOM files.")
    parser.add_argument("--img-root", type=Path, default=DEFAULT_IMG_ROOT, help="Root directory to save PNG images.")
    parser.add_argument("--label-json", type=Path, default=DEFAULT_LABEL_JSON, help="Label JSON/JSONL file path.")
    parser.add_argument(
        "--frame-index-base",
        type=int,
        default=DEFAULT_FRAME_INDEX_BASE,
        choices=[0, 1],
        help="0 if label indices are 0-based, 1 if 1-based.",
    )
    return parser.parse_args()


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    extract_labeled_frames(args.dcm_root, args.img_root, args.label_json, args.frame_index_base)
