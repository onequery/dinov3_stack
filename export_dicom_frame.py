from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------
DCM_ROOT = Path("input/stent_split_dcm")
IMG_ROOT = Path("input/stent_split_img")


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
    else:
        return -(2 ** (bits_stored - 1)), (2 ** (bits_stored - 1)) - 1


# --------------------------------------------------
# Core: extract first frame safely
# --------------------------------------------------
def extract_first_frame_uint8(ds):
    """
    Return first frame as uint8 numpy array
    following DICOM standard priority:
    VOI LUT > WC/WW > BitsStored fallback
    """
    if "PixelData" not in ds:
        raise ValueError("No PixelData")

    # Load pixel array
    pixel = ds.pixel_array.astype(np.float32)

    # Multi-frame → first frame
    if pixel.ndim == 3:
        pixel = pixel[0]

    # 1. VOI LUT (best / standard)
    try:
        pixel_voi = apply_voi_lut(pixel, ds)
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

        return normalize_to_uint8(pixel, low, high)

    # 3. BitsStored fallback (raw faithful scaling)
    low, high = get_bits_stored_range(ds)
    return normalize_to_uint8(pixel, low, high)


# --------------------------------------------------
# Save PNG
# --------------------------------------------------
def save_first_frame_png(dcm_path: Path, out_png_path: Path) -> bool:
    try:
        ds = pydicom.dcmread(dcm_path, force=True)

        img_uint8 = extract_first_frame_uint8(ds)

        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_uint8, mode="L").save(out_png_path)

        return True

    except Exception as e:
        print(f"[SKIP] {dcm_path} | {e}")
        return False


# --------------------------------------------------
# Main loop
# --------------------------------------------------
def extract_all_first_frames():
    dcm_files = list(DCM_ROOT.rglob("*.dcm"))

    print(f"Found {len(dcm_files)} DCM files")

    success = 0
    failed = 0

    for dcm_path in tqdm(dcm_files, desc="Extracting frames"):
        rel_path = dcm_path.relative_to(DCM_ROOT)
        out_png_path = IMG_ROOT / rel_path.with_suffix(".png")

        if save_first_frame_png(dcm_path, out_png_path):
            success += 1
        else:
            failed += 1

    print("\n========== DONE ==========")
    print(f"Success: {success}")
    print(f"Failed : {failed}")
    print(f"Output directory: {IMG_ROOT.resolve()}")


# --------------------------------------------------
# Entry
# --------------------------------------------------
if __name__ == "__main__":
    extract_all_first_frames()
