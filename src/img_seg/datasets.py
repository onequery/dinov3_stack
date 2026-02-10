import glob
import os
import albumentations as A
import cv2
import torch
import numpy as np

from src.img_seg.utils import get_label_mask, set_class_values
from torch.utils.data import Dataset, DataLoader

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_files(root_dir, extensions):
    files = glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
    return sorted(
        [
            file_path
            for file_path in files
            if os.path.isfile(file_path)
            and os.path.splitext(file_path)[1].lower() in extensions
        ]
    )


def _pair_image_and_mask_paths(image_paths, mask_paths, image_root, mask_root, split_name):
    image_map = {
        os.path.relpath(image_path, image_root): image_path for image_path in image_paths
    }
    mask_map = {os.path.relpath(mask_path, mask_root): mask_path for mask_path in mask_paths}

    shared_paths = sorted(set(image_map.keys()) & set(mask_map.keys()))
    missing_images = sorted(set(mask_map.keys()) - set(image_map.keys()))
    missing_masks = sorted(set(image_map.keys()) - set(mask_map.keys()))

    if missing_images or missing_masks:
        print(
            f"[WARN] {split_name}: paired using intersection only "
            f"(missing_images={len(missing_images)}, missing_masks={len(missing_masks)})."
        )

    if not shared_paths:
        raise ValueError(
            f"No paired image/mask files found for {split_name}. "
            f"images_root={image_root}, masks_root={mask_root}"
        )

    paired_images = [image_map[path] for path in shared_paths]
    paired_masks = [mask_map[path] for path in shared_paths]
    return paired_images, paired_masks

def get_images(train_images, train_masks, valid_images, valid_masks):
    train_image_paths = _collect_files(train_images, IMAGE_EXTENSIONS)
    train_mask_paths = _collect_files(train_masks, MASK_EXTENSIONS)
    valid_image_paths = _collect_files(valid_images, IMAGE_EXTENSIONS)
    valid_mask_paths = _collect_files(valid_masks, MASK_EXTENSIONS)

    train_images, train_masks = _pair_image_and_mask_paths(
        train_image_paths,
        train_mask_paths,
        train_images,
        train_masks,
        "train",
    )
    valid_images, valid_masks = _pair_image_and_mask_paths(
        valid_image_paths,
        valid_mask_paths,
        valid_images,
        valid_masks,
        "valid",
    )

    return train_images, train_masks, valid_images, valid_masks

# TODO: Batchwise rescaling with different image ratios.
def train_transforms(img_size):
    """
    Transforms/augmentations for training images and masks.

    :param img_size: Integer, for image resize.
    """
    train_image_transform = A.Compose([
        A.Resize(
            img_size[1], 
            img_size[0], 
            always_apply=True,
            # interpolation=cv2.INTER_CUBIC
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=25),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD, max_pixel_value=255.)
    ], is_check_shapes=False)
    return train_image_transform

# TODO: Batchwise rescaling with different image ratios.
def valid_transforms(img_size):
    """
    Transforms/augmentations for validation images and masks.

    :param img_size: Integer, for image resize.
    """
    valid_image_transform = A.Compose([
        A.Resize(
            img_size[1], img_size[0], 
            always_apply=True, 
            # interpolation=cv2.INTER_CUBIC
        ),
        A.Normalize(mean=IMG_MEAN, std=IMG_STD, max_pixel_value=255.)
    ], is_check_shapes=False)
    return valid_image_transform

class SegmentationDataset(Dataset):
    def __init__(
        self, 
        image_paths, 
        mask_paths, 
        tfms, 
        label_colors_list,
        classes_to_train,
        all_classes
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tfms = tfms
        self.label_colors_list = label_colors_list
        self.all_classes = all_classes
        self.classes_to_train = classes_to_train
        self.class_values = set_class_values(
            self.all_classes, self.classes_to_train
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
        mask = cv2.imread(self.mask_paths[index], cv2.IMREAD_COLOR)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB).astype('float32')

        # Make all pixel > 0 as 255.
        if len(self.all_classes) == 2:
            im = mask > 0
            mask[im] = 255
            mask[np.logical_not(im)] = 0

        transformed = self.tfms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Get 2D label mask.
        mask = get_label_mask(mask, self.class_values, self.label_colors_list).astype('uint8')
        # mask = Image.fromarray(mask)

        # To C, H, W.
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.LongTensor(mask)
    
def collate_fn(inputs):
    batch = dict()
    batch[0] = torch.stack([i[0] for i in inputs], dim=0)
    batch[1] = torch.stack([i[1] for i in inputs], dim=0)

    return batch

def get_dataset(
    train_image_paths, 
    train_mask_paths,
    valid_image_paths,
    valid_mask_paths,
    all_classes,
    classes_to_train,
    label_colors_list,
    img_size
):
    train_tfms = train_transforms(img_size)
    valid_tfms = valid_transforms(img_size)

    train_dataset = SegmentationDataset(
        train_image_paths,
        train_mask_paths,
        train_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    valid_dataset = SegmentationDataset(
        valid_image_paths,
        valid_mask_paths,
        valid_tfms,
        label_colors_list,
        classes_to_train,
        all_classes
    )
    return train_dataset, valid_dataset

def get_data_loaders(train_dataset, valid_dataset, batch_size, num_workers=8):
    loader_kwargs = {
        "batch_size": batch_size,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_data_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs
    )
    valid_data_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        **loader_kwargs
    )

    return train_data_loader, valid_data_loader
