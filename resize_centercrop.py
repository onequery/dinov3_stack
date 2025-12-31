from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

# ----------------------------------
# Input / Output paths
# ----------------------------------
# IMG_PATH = "input_example.png"   # 512x512 이미지
IMG_PATH = (
    # "input/stent_split_img/train/stent/10010415/20150121/XA/039.png"  # 512x512 이미지
    # "input/stent_split_img/train/stent/10010415/20151012/XA/006.png"  # 512x512 이미지
    "input/stent_split_img/train/stent/10010415/20231220/XA/007.png"  # 512x512 이미지
)
# OUT_DIR = Path("debug_transforms")
# OUT_DIR = Path("outputs/resize_centercrop_example/039")
# OUT_DIR = Path("outputs/resize_centercrop_example/006")
# OUT_DIR = Path("outputs/resize_centercrop_example/007")
# OUT_DIR = Path("outputs/centercrop_example/039")
# OUT_DIR = Path("outputs/centercrop_example/006")
OUT_DIR = Path("outputs/centercrop_example/007")

OUT_DIR.mkdir(exist_ok=True)

# ----------------------------------
# Load image
# ----------------------------------
img_bgr = cv2.imread(IMG_PATH)
assert img_bgr is not None, "Image not found"

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(img_rgb)

# ----------------------------------
# Define transforms (same as training)
# ----------------------------------
# resize = transforms.Resize((256, 256))
resize = transforms.Resize((512, 512))
# center_crop = transforms.CenterCrop((224, 224))
center_crop = transforms.CenterCrop((448, 448))

img_resized = resize(pil_img)
img_cropped = center_crop(img_resized)

# ----------------------------------
# Save images
# ----------------------------------
# img_resized.save(OUT_DIR / "resized_256x256.png")
# img_cropped.save(OUT_DIR / "center_crop_224x224.png")
img_resized.save(OUT_DIR / "resized_512x512.png")
img_cropped.save(OUT_DIR / "center_crop_448x448.png")

# ----------------------------------
# Visualization
# ----------------------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original (512x512)")
plt.imshow(pil_img)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Resized (256x256)")
plt.imshow(img_resized)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("CenterCrop (224x224)")
plt.imshow(img_cropped)
plt.axis("off")

plt.tight_layout()
plt.show()
