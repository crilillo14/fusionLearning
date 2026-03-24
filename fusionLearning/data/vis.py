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
from fusionLearning.data.aug import geom_transform_pair
from fusionLearning.data.dataloaders import (
    CUBDataset,
    get_file_paths,
    load_image,
    load_segmentation_mask,
)

__all__ = [
    "visualize_preprocessing_step",
    "visualize_random_samples",
    "visualize_geo_pair",
    "visualize_geo_pair_samples",
    "visualize_dataset_samples",
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


# ── dataset-agnostic helpers ──────────────────────────────────────────────────

def _get_raw_pil(dataset, idx: int):
    """Extract (img_pil, mask_pil, filename) *before* any dataset transforms.

    Handles:
      - CUBDataset / vanillaCUBDataset  — file-path lists on the instance
      - CityscapesDataset               — torchvision inner dataset + .images list
      - ADE20KDataset                   — torchvision inner dataset + .files list
    """
    if hasattr(dataset, "_inner"):
        # CityscapesDataset or ADE20KDataset
        img_pil, mask_pil = dataset._inner[idx]
        inner = dataset._inner
        if hasattr(inner, "images") and isinstance(inner.images, list):
            fname = os.path.basename(inner.images[idx])
        elif hasattr(inner, "files") and isinstance(inner.files, list):
            fname = os.path.basename(inner.files[idx]["image"])
        else:
            fname = str(idx)
    elif hasattr(dataset, "image_paths"):
        # CUBDataset or vanillaCUBDataset
        img_pil, fname = load_image(dataset.image_paths[idx])
        mask_pil = load_segmentation_mask(dataset.segmentation_paths[idx])
        if img_pil is None or mask_pil is None:
            raise RuntimeError(f"Failed to load sample {idx} from dataset.")
    else:
        raise TypeError(
            f"Cannot extract raw PIL from {type(dataset).__name__}. "
            "Expected a CUBDataset, CityscapesDataset, or ADE20KDataset instance."
        )
    return img_pil, mask_pil, fname


def _render_mask(ax, mask_arr: np.ndarray, title: str) -> None:
    """Render a mask with an appropriate colormap.

    - Binary / float [0,1]  (CUB)          → grayscale
    - Integer class IDs     (Cityscapes / ADE20K) → tab20; ignore index 255 → black
    """
    unique_vals = np.unique(mask_arr)
    is_float_binary = mask_arr.dtype.kind == "f" or int(unique_vals.max()) <= 1

    if is_float_binary:
        ax.imshow(mask_arr, cmap="gray", vmin=0, vmax=1)
    else:
        # Replace ignore index (255) with NaN so matplotlib renders it as the
        # background colour; use tab20 for the valid class IDs.
        display = mask_arr.astype(float)
        display[display == 255] = np.nan
        ax.imshow(display, cmap="tab20", interpolation="nearest")

    ax.set_title(title)
    ax.axis("off")


# ── geo-pair visualization ────────────────────────────────────────────────────

def visualize_geo_pair(
    dataset_or_path,
    mask_path: Optional[str] = None,
    *,
    idx: int = 0,
    show: bool = True,
):
    """Show image + mask before and after geom_transform_pair.

    Layout (1 row × 4 cols):
        Raw image | Raw mask | Transformed image | Transformed mask

    Can be called two ways:

    1. With a dataset object (CUBDataset, CityscapesDataset, ADE20KDataset)::

        visualize_geo_pair(dataset, idx=42)

    2. With raw file paths (CUB-style directory layout)::

        visualize_geo_pair("/path/img.jpg", "/path/mask.png")

    Returns
    -------
    matplotlib.figure.Figure
    """
    if isinstance(dataset_or_path, str):
        # file-path mode — CUB backward compat
        img_pil, fname = load_image(dataset_or_path)
        mask_pil = load_segmentation_mask(mask_path)
        if img_pil is None or mask_pil is None:
            raise RuntimeError(f"Failed to load '{dataset_or_path}' or its mask.")
    else:
        img_pil, mask_pil, fname = _get_raw_pil(dataset_or_path, idx)

    img_t, mask_t = geom_transform_pair(img_pil.copy(), mask_pil.copy())

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    axes[0].imshow(np.asarray(img_pil))
    axes[0].set_title("Raw image")
    axes[0].axis("off")
    _render_mask(axes[1], np.asarray(mask_pil), "Raw mask")
    axes[2].imshow(np.asarray(img_t))
    axes[2].set_title("Transformed image")
    axes[2].axis("off")
    _render_mask(axes[3], np.asarray(mask_t), "Transformed mask")

    fig.suptitle(fname)
    if show:
        plt.show()
    return fig


def visualize_geo_pair_samples(
    dataset_or_image_dir,
    segmentation_dir: Optional[str] = None,
    *,
    num_samples: int = 3,
    show: bool = True,
):
    """Randomly pick *num_samples* and call :func:`visualize_geo_pair` on each.

    Accepts either a dataset object or CUB-style directory paths.
    """
    if isinstance(dataset_or_image_dir, str):
        # file-path mode
        image_paths = sorted(get_file_paths(dataset_or_image_dir))
        seg_paths = sorted(get_file_paths(segmentation_dir))
        if not image_paths:
            raise ValueError(f"No images found in '{dataset_or_image_dir}'.")
        if len(image_paths) != len(seg_paths):
            raise ValueError("Image / segmentation count mismatch.")
        indices = random.sample(range(len(image_paths)), k=min(num_samples, len(image_paths)))
        for i in indices:
            visualize_geo_pair(image_paths[i], seg_paths[i], show=show)
    else:
        dataset = dataset_or_image_dir
        indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
        for i in indices:
            visualize_geo_pair(dataset, idx=i, show=show)


def visualize_dataset_samples(
    dataset,
    *,
    num_samples: int = 4,
    show: bool = True,
):
    """Show (image, mask) pairs from any dataset using its own transform output.

    Works for CUBDataset, CityscapesDataset, ADE20KDataset — anything that
    returns ``(image_tensor [3,H,W], mask_tensor, filename)`` from ``__getitem__``.

    Layout: num_samples rows × 2 cols  (image | mask)
    """
    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples),
                             constrained_layout=True)
    if num_samples == 1:
        axes = [axes]

    for row, idx in enumerate(indices):
        img_tensor, mask_tensor, fname = dataset[idx]
        axes[row][0].imshow(_tensor_to_numpy(img_tensor))
        axes[row][0].set_title(f"Image — {fname}")
        axes[row][0].axis("off")
        _render_mask(axes[row][1], mask_tensor.numpy(), f"Mask — {fname}")

    if show:
        plt.show()
    return fig


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
    parser.add_argument(
        "--geo-pair",
        action="store_true",
        help="Show image+mask before/after geom_transform_pair instead of the full pipeline",
    )
    return parser


def _main(argv: list[str] | None = None):
    args = _build_argparser().parse_args(argv)
    if args.geo_pair:
        visualize_geo_pair_samples(args.images, args.segmentations, num_samples=args.num)
    else:
        visualize_random_samples(args.images, args.segmentations, num_samples=args.num)
    plt.show()


if __name__ == "__main__":
    _main()
