"""fusionLearning.data.vis

Utilities for visualizing the preprocessing pipeline defined in
`fusionLearning.data.dataloaders`.

Run this module as a script to quickly sanity–check that every step of the
pipeline behaves as expected.

Example
-------
python -m fusionLearning.data.vis \
    --images /path/to/CUB_200_2011/images \
    --segmentations /path/to/CUB_200_2011/segmentations \
    --num 5
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import v2 as T

# Relative imports within the package
from .dataloaders import (
    CUBDataset,
    get_file_paths,
    load_image,
    load_segmentation_mask,
)

__all__ = [
    "visualize_preprocessing_step",
    "visualize_random_samples",
]


def _tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """Convert a CHW float-tensor (0-1) to an HWC uint8 numpy image."""
    img_np: np.ndarray = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(
        np.uint8
    )
    return img_np


def visualize_preprocessing_step(
    image_path: str,
    mask_path: str,
    *,
    g_transforms: Optional[torch.nn.Module] = None,
    p_transforms: Optional[torch.nn.Module] = None,
    show: bool = True,
):
    """Visualize every step of the preprocessing pipeline for a single sample.

    Parameters
    ----------
    image_path: str
        Path to the raw RGB image.
    mask_path: str
        Path to the corresponding segmentation mask.
    g_transforms: torch.nn.Module | None, optional
        Geometric transforms applied on the *PIL* image (and mask if desired).
    p_transforms: torch.nn.Module | None, optional
        Photometric transforms applied on the image *tensor*.
    show: bool, default True
        Whether to immediately display the matplotlib figure.  If *False* the
        caller can further modify or save the figure returned.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ---------------------------------------------------------------------
    # 1. Load raw data (PIL)
    # ---------------------------------------------------------------------
    img_pil, filename = load_image(image_path)
    mask_pil = load_segmentation_mask(mask_path)
    if img_pil is None or mask_pil is None:
        raise RuntimeError(f"Failed to load '{image_path}' or its mask.")

    # Preserve originals for display.
    img_raw = img_pil.copy()
    mask_raw = mask_pil.copy()

    # ------------------------------------------------------------------
    # 2. Geometric transforms (operate on PIL)
    # ------------------------------------------------------------------
    img_geo = g_transforms(img_pil) if g_transforms else img_pil
    # NOTE: The current dataset implementation does *not* transform the mask,
    # but you may want to apply the same transform here for completeness.

    # ------------------------------------------------------------------
    # 3. Convert to tensor & scale to 0-1
    # ------------------------------------------------------------------
    img_tensor = T.PILToTensor()(img_geo).float() / 255.0  # [C,H,W]

    # ------------------------------------------------------------------
    # 4. Photometric transforms (operate on tensor)
    # ------------------------------------------------------------------
    img_tensor_photo = p_transforms(img_tensor) if p_transforms else img_tensor

    # ------------------------------------------------------------------
    # 5. Prepare visuals
    # ------------------------------------------------------------------
    img_geo_np = np.asarray(img_geo)
    img_photo_np = _tensor_to_numpy(img_tensor_photo)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    titles = [
        "Raw image",
        "Segmentation mask",
        "After geometric",
        "Tensor + photometric",
    ]
    visuals = [img_raw, mask_raw, img_geo_np, img_photo_np]

    for ax, title, vis in zip(axes, titles, visuals):
        if vis.ndim == 2:  # grayscale mask
            ax.imshow(vis, cmap="gray")
        else:
            ax.imshow(vis)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(filename)

    if show:
        plt.show()

    return fig


def visualize_random_samples(
    image_dir: str,
    segmentation_dir: str,
    *,
    num_samples: int = 3,
    g_transforms: Optional[torch.nn.Module] = None,
    p_transforms: Optional[torch.nn.Module] = None,
):
    """Randomly choose *num_samples* from *image_dir* and visualize them."""

    image_paths = get_file_paths(image_dir)
    seg_paths = get_file_paths(segmentation_dir)
    image_paths.sort()
    seg_paths.sort()

    if len(image_paths) == 0:
        raise ValueError(f"No images found in '{image_dir}'.")
    if len(image_paths) != len(seg_paths):
        raise ValueError("Image / segmentation count mismatch.")

    indices = random.sample(range(len(image_paths)), k=min(num_samples, len(image_paths)))
    for idx in indices:
        visualize_preprocessing_step(
            image_paths[idx],
            seg_paths[idx],
            g_transforms=g_transforms,
            p_transforms=p_transforms,
        )


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visualize preprocessing steps.")
    parser.add_argument("--images", required=True, help="Directory with raw images")
    parser.add_argument(
        "--segmentations", required=True, help="Directory with segmentation masks"
    )
    parser.add_argument("--num", type=int, default=3, help="Number of samples to show")
    return parser


def _main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)
    visualize_random_samples(args.images, args.segmentations, num_samples=args.num)


if __name__ == "__main__":
    _main()
