"""
TOMPEI-CMMD binary lesion/normal classification dataset & dataloaders.

Reads grayscale mammography DICOMs from the pre-made train/val/test split
(fusionLearning/data/TOMPEI-CMMD/Task_classification/), pairs each with its flat
lesion/normal label file, and returns fixed-size 3-channel tensors ready for timm
classification backbones.

Unlike CUBDataset, the train/val/test split is NOT built here via random_split() -
TOMPEI-CMMD ships pre-made, patient-disjoint train/val/test directories, so each
split is loaded directly from its own folder pair.

See notes/tompei_cmmd_classification_spec.md for the full design rationale.
"""

from __future__ import annotations

import os

import numpy as np
import pydicom
from pydicom.pixels import apply_voi_lut
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import v2

from fusionLearning.models.consts import INPUT_SIZE_TOMPEI_CMMD

LABEL_MAP = {"normal": 0.0, "lesion": 1.0}


def get_dcm_paths(directory: str) -> list[str]:
    return sorted(
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".dcm")
    )


def load_dicom_image(path: str) -> tuple[np.ndarray, str]:
    """
    Loads a DICOM file, applies the VOI LUT/window (falling back to the raw pixel
    array if none is present), corrects MONOCHROME1 inversion, and per-image
    min-max normalizes to float32 [0,1].
    """
    ds = pydicom.dcmread(path)

    try:
        arr = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        arr = ds.pixel_array

    arr = arr.astype(np.float32)

    if ds.get("PhotometricInterpretation") == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    return arr, os.path.basename(path)


# Image-only augmentation for grayscale mammography - no color/hue/saturation jitter
# (source is grayscale-replicated, those ops are noise), no vertical flip or perspective
# warp (breast orientation is anatomically meaningful, mammography has no camera-perspective
# variation to simulate). Deliberately not added to data/aug.py, which is built around
# paired image+mask geometric consistency that doesn't apply to classification.
classificationTransforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=7, interpolation=InterpolationMode.BILINEAR),
        v2.RandomAdjustSharpness(sharpness_factor=1.1, p=0.1),
    ]
)


class TOMPEICMMDDataset(Dataset):
    """
    Binary lesion/normal classification over TOMPEI-CMMD mammography DICOMs.

    Returns (image_tensor [3, resolution, resolution] float32,
             label_tensor [1] float32, filename).
    """

    def __init__(self, image_dir: str, label_dir: str, resolution: int = INPUT_SIZE_TOMPEI_CMMD,
                 train: bool = False):
        self.image_paths = get_dcm_paths(image_dir)
        self.label_dir = label_dir
        self.resolution = resolution
        self.train = train

        stems = [os.path.splitext(os.path.basename(p))[0] for p in self.image_paths]
        missing = [s for s in stems if not os.path.exists(os.path.join(label_dir, s + ".txt"))]
        if missing:
            raise ValueError(
                f"{len(missing)} images in {image_dir} have no matching label file in "
                f"{label_dir} (e.g. {missing[0]}.txt)"
            )

        print(f"TOMPEICMMDDataset: {len(self.image_paths)} DICOM/label pairs from {image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        arr, filename = load_dicom_image(path)

        image_tensor = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        image_tensor = v2.functional.resize(
            image_tensor,
            [self.resolution, self.resolution],
            interpolation=InterpolationMode.BICUBIC,
        )

        if self.train:
            image_tensor = classificationTransforms(image_tensor)

        image_tensor = image_tensor.clamp(0.0, 1.0).repeat(3, 1, 1)  # [3, H, W]

        stem = os.path.splitext(filename)[0]
        with open(os.path.join(self.label_dir, stem + ".txt")) as f:
            label_str = f.read().strip().lower()
        label_tensor = torch.tensor([LABEL_MAP[label_str]], dtype=torch.float32)

        return image_tensor, label_tensor, filename


def create_tompei_cmmd_loaders_distributed(
    train_dir: str,
    train_label_dir: str,
    val_dir: str,
    val_label_dir: str,
    test_dir: str,
    test_label_dir: str,
    batch_size: int = 16,
    resolution: int = INPUT_SIZE_TOMPEI_CMMD,
    num_workers: int = 4,
):
    """
    Builds train/val/test DataLoaders directly from TOMPEI-CMMD's pre-made,
    patient-disjoint split directories. All images are resized to a fixed square
    size (per `resolution`, varies per roster variant - see roster_cls.py's
    depth x resolution grid) before batching, so (unlike the segmentation
    loaders) no pad_collate is needed - default collation is used.
    """
    train_dataset = TOMPEICMMDDataset(train_dir, train_label_dir, resolution=resolution, train=True)
    val_dataset = TOMPEICMMDDataset(val_dir, val_label_dir, resolution=resolution, train=False)
    test_dataset = TOMPEICMMDDataset(test_dir, test_label_dir, resolution=resolution, train=False)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    _pw = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=_pw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=_pw,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=_pw,
    )

    return train_loader, val_loader, test_loader
