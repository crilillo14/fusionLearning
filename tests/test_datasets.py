"""
Unit tests for CityscapesDataset and ADE20KDataset.

Mocks torchvision inner datasets so no actual downloaded data is needed.
Run with: python -m pytest tests/test_datasets.py -v
"""

from __future__ import annotations

import functools
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusionLearning.data.dataloaders import (
    ADE20KDataset,
    CityscapesDataset,
    _CS_LABEL_MAP,
    create_ade20k_loaders_distributed,
    create_cityscapes_loaders_distributed,
    pad_collate,
)


# ── synthetic data helpers ────────────────────────────────────────────────────

def _rgb_pil(h: int = 64, w: int = 64) -> Image.Image:
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _mask_pil(h: int = 64, w: int = 64, fill: int = 7) -> Image.Image:
    arr = np.full((h, w), fill, dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


def _make_cs_inner(n: int = 4, h: int = 64, w: int = 64, fill: int = 7) -> MagicMock:
    mock = MagicMock()
    mock.__len__ = MagicMock(return_value=n)
    mock.__getitem__ = MagicMock(return_value=(_rgb_pil(h, w), _mask_pil(h, w, fill)))
    mock.images = [f"/fake/img_{i}.png" for i in range(n)]
    return mock


def _ade_dataset(n: int = 4, h: int = 64, w: int = 64, fill: int = 1,
                 gTransforms=None, pTransforms=None) -> ADE20KDataset:
    """Build an ADE20KDataset bypassing __init__ file scanning."""
    ds = ADE20KDataset.__new__(ADE20KDataset)
    ds.image_paths        = [f"/fake/ADE_train_{i:08d}.jpg" for i in range(n)]
    ds.segmentation_paths = [f"/fake/ADE_train_{i:08d}.png" for i in range(n)]
    ds.gTransforms = gTransforms
    ds.pTransforms = pTransforms
    ds._h, ds._w, ds._fill = h, w, fill  # stored for use in open mock
    return ds


def _ade_open_mock(h: int = 64, w: int = 64, fill: int = 1):
    """Return a side_effect for Image.open that serves RGB or mask by extension."""
    def _open(path):
        if str(path).endswith(".jpg"):
            return _rgb_pil(h, w)
        return _mask_pil(h, w, fill)
    return _open


def _cs_dataset(inner: MagicMock, **kwargs) -> CityscapesDataset:
    with patch("torchvision.datasets.Cityscapes", return_value=inner):
        ds = CityscapesDataset("/fake", **kwargs)
    ds._inner = inner
    return ds



# ── _CS_LABEL_MAP ─────────────────────────────────────────────────────────────

class TestCityscapesLabelMap(unittest.TestCase):

    KNOWN = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
        21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18,
    }

    def test_all_known_ids_map_correctly(self):
        for raw, train in self.KNOWN.items():
            self.assertEqual(int(_CS_LABEL_MAP[raw]), train,
                             f"raw {raw} → expected train {train}")

    def test_unknown_ids_map_to_255(self):
        unknown = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, 100, 200]
        for raw in unknown:
            self.assertEqual(int(_CS_LABEL_MAP[raw]), 255,
                             f"raw {raw} should be ignore (255)")

    def test_map_length_is_256(self):
        self.assertEqual(len(_CS_LABEL_MAP), 256)


# ── CityscapesDataset ─────────────────────────────────────────────────────────

class TestCityscapesDataset(unittest.TestCase):

    def test_len(self):
        ds = _cs_dataset(_make_cs_inner(n=5))
        self.assertEqual(len(ds), 5)

    def test_getitem_returns_three_tuple(self):
        ds = _cs_dataset(_make_cs_inner())
        out = ds[0]
        self.assertEqual(len(out), 3)

    def test_image_shape(self):
        ds = _cs_dataset(_make_cs_inner(h=64, w=64))
        img, _, _ = ds[0]
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[0], 3)  # RGB channels

    def test_image_dtype_float32(self):
        ds = _cs_dataset(_make_cs_inner())
        img, _, _ = ds[0]
        self.assertEqual(img.dtype, torch.float32)

    def test_image_range_0_to_1(self):
        ds = _cs_dataset(_make_cs_inner())
        img, _, _ = ds[0]
        self.assertGreaterEqual(img.min().item(), 0.0)
        self.assertLessEqual(img.max().item(), 1.0)

    def test_mask_is_2d(self):
        ds = _cs_dataset(_make_cs_inner(h=64, w=64))
        _, mask, _ = ds[0]
        self.assertEqual(mask.ndim, 2)

    def test_mask_dtype_long(self):
        ds = _cs_dataset(_make_cs_inner())
        _, mask, _ = ds[0]
        self.assertEqual(mask.dtype, torch.long)

    def test_image_and_mask_spatial_dims_match(self):
        ds = _cs_dataset(_make_cs_inner(h=64, w=64))
        img, mask, _ = ds[0]
        self.assertEqual(img.shape[1], mask.shape[0])  # H
        self.assertEqual(img.shape[2], mask.shape[1])  # W

    def test_mask_dims_multiple_of_32(self):
        # 65x97 → crop_to_multiple → 64x96
        ds = _cs_dataset(_make_cs_inner(h=65, w=97))
        _, mask, _ = ds[0]
        self.assertEqual(mask.shape[0] % 32, 0, "H not divisible by 32")
        self.assertEqual(mask.shape[1] % 32, 0, "W not divisible by 32")

    def test_label_raw7_maps_to_train0(self):
        ds = _cs_dataset(_make_cs_inner(fill=7))
        _, mask, _ = ds[0]
        self.assertTrue((mask == 0).all(), "raw 7 (road) → train 0")

    def test_label_raw26_maps_to_train13(self):
        ds = _cs_dataset(_make_cs_inner(fill=26))
        _, mask, _ = ds[0]
        self.assertTrue((mask == 13).all(), "raw 26 (car) → train 13")

    def test_label_raw33_maps_to_train18(self):
        ds = _cs_dataset(_make_cs_inner(fill=33))
        _, mask, _ = ds[0]
        self.assertTrue((mask == 18).all(), "raw 33 → train 18")

    def test_unknown_raw_label_maps_to_ignore_255(self):
        ds = _cs_dataset(_make_cs_inner(fill=0))  # raw 0 not in train set
        _, mask, _ = ds[0]
        self.assertTrue((mask == 255).all(), "raw 0 → ignore (255)")

    def test_filename_is_string(self):
        ds = _cs_dataset(_make_cs_inner())
        _, _, fname = ds[0]
        self.assertIsInstance(fname, str)

    def test_filename_is_basename_only(self):
        ds = _cs_dataset(_make_cs_inner())
        _, _, fname = ds[0]
        self.assertEqual(fname, os.path.basename(fname))

    def test_mask_values_in_valid_range(self):
        # all values must be 0-18 or 255
        ds = _cs_dataset(_make_cs_inner(fill=11))  # raw 11 → train 2
        _, mask, _ = ds[0]
        valid = ((mask >= 0) & (mask <= 18)) | (mask == 255)
        self.assertTrue(valid.all())


# ── ADE20KDataset ─────────────────────────────────────────────────────────────

class TestADE20KDataset(unittest.TestCase):

    def _item(self, ds, idx=0):
        with patch("fusionLearning.data.dataloaders.Image.open",
                   side_effect=_ade_open_mock(ds._h, ds._w, ds._fill)):
            return ds[idx]

    def test_len(self):
        self.assertEqual(len(_ade_dataset(n=6)), 6)

    def test_getitem_returns_three_tuple(self):
        ds = _ade_dataset()
        self.assertEqual(len(self._item(ds)), 3)

    def test_image_shape(self):
        ds = _ade_dataset(h=64, w=64)
        img, _, _ = self._item(ds)
        self.assertEqual(img.ndim, 3)
        self.assertEqual(img.shape[0], 3)

    def test_image_dtype_float32(self):
        img, _, _ = self._item(_ade_dataset())
        self.assertEqual(img.dtype, torch.float32)

    def test_image_range_0_to_1(self):
        img, _, _ = self._item(_ade_dataset())
        self.assertGreaterEqual(img.min().item(), 0.0)
        self.assertLessEqual(img.max().item(), 1.0)

    def test_mask_is_2d(self):
        _, mask, _ = self._item(_ade_dataset())
        self.assertEqual(mask.ndim, 2)

    def test_mask_dtype_long(self):
        _, mask, _ = self._item(_ade_dataset())
        self.assertEqual(mask.dtype, torch.long)

    def test_image_and_mask_spatial_dims_match(self):
        ds = _ade_dataset(h=64, w=64)
        img, mask, _ = self._item(ds)
        self.assertEqual(img.shape[1], mask.shape[0])
        self.assertEqual(img.shape[2], mask.shape[1])

    def test_mask_dims_multiple_of_32(self):
        ds = _ade_dataset(h=65, w=97)
        _, mask, _ = self._item(ds)
        self.assertEqual(mask.shape[0] % 32, 0, "H not divisible by 32")
        self.assertEqual(mask.shape[1] % 32, 0, "W not divisible by 32")

    def test_remap_background_raw0_to_255(self):
        ds = _ade_dataset(fill=0)
        _, mask, _ = self._item(ds)
        self.assertTrue((mask == 255).all(), "raw 0 (background) → 255")

    def test_remap_raw1_to_class0(self):
        ds = _ade_dataset(fill=1)
        _, mask, _ = self._item(ds)
        self.assertTrue((mask == 0).all(), "raw 1 → class 0")

    def test_remap_raw150_to_class149(self):
        ds = _ade_dataset(fill=150)
        _, mask, _ = self._item(ds)
        self.assertTrue((mask == 149).all(), "raw 150 → class 149")

    def test_remap_midrange_class(self):
        ds = _ade_dataset(fill=75)
        _, mask, _ = self._item(ds)
        self.assertTrue((mask == 74).all(), "raw 75 → class 74")

    def test_filename_is_string(self):
        _, _, fname = self._item(_ade_dataset())
        self.assertIsInstance(fname, str)

    def test_filename_is_basename_only(self):
        _, _, fname = self._item(_ade_dataset())
        self.assertEqual(fname, os.path.basename(fname))

    def test_mask_values_in_valid_range(self):
        ds = _ade_dataset(fill=50)
        _, mask, _ = self._item(ds)
        valid = ((mask >= 0) & (mask <= 149)) | (mask == 255)
        self.assertTrue(valid.all())


# ── pad_collate (shared by both datasets) ────────────────────────────────────

class TestPadCollate(unittest.TestCase):
    """Test the padding collate function that both datasets rely on."""

    def _make_items(self, sizes, mask_fill=0, mask_dtype=torch.long):
        """Build a list of (image, mask, fname) tuples of given (H,W) sizes."""
        items = []
        for h, w in sizes:
            img = torch.zeros(3, h, w)
            mask = torch.full((h, w), mask_fill, dtype=mask_dtype)
            items.append((img, mask, "fname.png"))
        return items

    def test_uniform_sizes_no_padding_needed(self):
        items = self._make_items([(64, 64), (64, 64)])
        imgs, masks, _ = pad_collate(items, mask_pad_value=255)
        self.assertEqual(imgs.shape, torch.Size([2, 3, 64, 64]))
        self.assertEqual(masks.shape, torch.Size([2, 64, 64]))

    def test_different_heights_pads_to_max(self):
        items = self._make_items([(32, 64), (64, 64)])
        imgs, masks, _ = pad_collate(items, mask_pad_value=255)
        self.assertEqual(imgs.shape[2], 64)

    def test_different_widths_pads_to_max(self):
        items = self._make_items([(64, 32), (64, 64)])
        imgs, masks, _ = pad_collate(items, mask_pad_value=255)
        self.assertEqual(imgs.shape[3], 64)

    def test_pad_region_uses_mask_pad_value_255(self):
        # smaller item: 32x32 filled with 0; padded to 64x64 with 255
        items = self._make_items([(64, 64), (32, 32)], mask_fill=0)
        _, masks, _ = pad_collate(items, mask_pad_value=255)
        # smaller item (index 1): rows 32: and cols 32: are padding
        self.assertTrue((masks[1, 32:, :] == 255).all(), "padded rows should be 255")
        self.assertTrue((masks[1, :, 32:] == 255).all(), "padded cols should be 255")

    def test_pad_region_uses_mask_pad_value_0(self):
        items = self._make_items([(64, 64), (32, 32)], mask_fill=1)
        _, masks, _ = pad_collate(items, mask_pad_value=0)
        self.assertTrue((masks[1, 32:, :] == 0).all())

    def test_image_pad_region_is_zero(self):
        items = self._make_items([(64, 64), (32, 32)])
        imgs, _, _ = pad_collate(items, mask_pad_value=255)
        # image padding is always 0
        self.assertTrue((imgs[1, :, 32:, :] == 0).all())
        self.assertTrue((imgs[1, :, :, 32:] == 0).all())

    def test_batch_size_preserved(self):
        items = self._make_items([(64, 64)] * 3)
        imgs, masks, names = pad_collate(items, mask_pad_value=255)
        self.assertEqual(imgs.shape[0], 3)
        self.assertEqual(masks.shape[0], 3)
        self.assertEqual(len(names), 3)


# ── distributed loader creation ───────────────────────────────────────────────

def _seq_sampler(dataset, **kwargs):
    """Drop-in for DistributedSampler that doesn't need a process group."""
    from torch.utils.data import SequentialSampler
    return SequentialSampler(dataset)


class TestCityscapesLoadersDistributed(unittest.TestCase):

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("torchvision.datasets.Cityscapes")
    def test_returns_three_loaders(self, mock_cls, _):
        mock_cls.return_value = _make_cs_inner(n=4)
        train, val, test = create_cityscapes_loaders_distributed("/fake", batch_size=2, num_workers=0)
        self.assertIsNotNone(train)
        self.assertIsNotNone(val)
        self.assertIsNotNone(test)

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("torchvision.datasets.Cityscapes")
    def test_batch_image_shape(self, mock_cls, _):
        mock_cls.return_value = _make_cs_inner(n=4)
        train, _, _ = create_cityscapes_loaders_distributed("/fake", batch_size=2, num_workers=0)
        imgs, masks, fnames = next(iter(train))
        self.assertEqual(imgs.shape[0], 2)   # batch size
        self.assertEqual(imgs.shape[1], 3)   # RGB
        self.assertEqual(imgs.dtype, torch.float32)

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("torchvision.datasets.Cityscapes")
    def test_batch_mask_dtype_and_ndim(self, mock_cls, _):
        mock_cls.return_value = _make_cs_inner(n=4)
        train, _, _ = create_cityscapes_loaders_distributed("/fake", batch_size=2, num_workers=0)
        _, masks, _ = next(iter(train))
        self.assertEqual(masks.ndim, 3)       # [B, H, W]
        self.assertEqual(masks.dtype, torch.long)

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("torchvision.datasets.Cityscapes")
    def test_label_values_in_batch(self, mock_cls, _):
        # raw 7 → train 0
        mock_cls.return_value = _make_cs_inner(n=4, fill=7)
        train, _, _ = create_cityscapes_loaders_distributed("/fake", batch_size=2, num_workers=0)
        _, masks, _ = next(iter(train))
        self.assertTrue((masks == 0).all())

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("torchvision.datasets.Cityscapes")
    def test_train_val_test_are_independent_loaders(self, mock_cls, _):
        mock_cls.return_value = _make_cs_inner(n=4)
        train, val, test = create_cityscapes_loaders_distributed("/fake", batch_size=2, num_workers=0)
        self.assertIsNot(train, val)
        self.assertIsNot(val, test)


class TestADE20KLoadersDistributed(unittest.TestCase):
    """
    get_file_paths is mocked to return fake paths (no disk needed).
    Image.open is mocked to return synthetic PIL images.
    DistributedSampler is replaced with SequentialSampler.
    """

    N = 4

    def _fake_gfp(self, fill=1, h=64, w=64):
        """side_effect for get_file_paths: returns .jpg paths for image dirs, .png for annotation dirs."""
        def _gfp(path):
            if "images" in path:
                return [f"/fake/ADE_train_{i:08d}.jpg" for i in range(self.N)]
            return [f"/fake/ADE_train_{i:08d}.png" for i in range(self.N)]
        return _gfp

    def _fake_open(self, fill=1, h=64, w=64):
        return _ade_open_mock(h, w, fill)

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("fusionLearning.data.dataloaders.Image.open")
    @patch("fusionLearning.data.dataloaders.get_file_paths")
    def test_returns_three_loaders(self, mock_gfp, mock_open, _):
        mock_gfp.side_effect = self._fake_gfp()
        mock_open.side_effect = self._fake_open()
        train, val, test = create_ade20k_loaders_distributed("/fake", batch_size=2, num_workers=0)
        self.assertIsNotNone(train)
        self.assertIsNotNone(val)
        self.assertIsNotNone(test)

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("fusionLearning.data.dataloaders.Image.open")
    @patch("fusionLearning.data.dataloaders.get_file_paths")
    def test_batch_image_shape(self, mock_gfp, mock_open, _):
        mock_gfp.side_effect = self._fake_gfp()
        mock_open.side_effect = self._fake_open()
        train, _, _ = create_ade20k_loaders_distributed("/fake", batch_size=2, num_workers=0)
        imgs, masks, _ = next(iter(train))
        self.assertEqual(imgs.shape[0], 2)
        self.assertEqual(imgs.shape[1], 3)
        self.assertEqual(imgs.dtype, torch.float32)

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("fusionLearning.data.dataloaders.Image.open")
    @patch("fusionLearning.data.dataloaders.get_file_paths")
    def test_batch_mask_dtype_and_ndim(self, mock_gfp, mock_open, _):
        mock_gfp.side_effect = self._fake_gfp()
        mock_open.side_effect = self._fake_open()
        train, _, _ = create_ade20k_loaders_distributed("/fake", batch_size=2, num_workers=0)
        _, masks, _ = next(iter(train))
        self.assertEqual(masks.ndim, 3)
        self.assertEqual(masks.dtype, torch.long)

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("fusionLearning.data.dataloaders.Image.open")
    @patch("fusionLearning.data.dataloaders.get_file_paths")
    def test_label_values_in_batch(self, mock_gfp, mock_open, _):
        mock_gfp.side_effect = self._fake_gfp(fill=1)
        mock_open.side_effect = self._fake_open(fill=1)
        train, _, _ = create_ade20k_loaders_distributed("/fake", batch_size=2, num_workers=0)
        _, masks, _ = next(iter(train))
        self.assertTrue((masks == 0).all())  # raw 1 → class 0

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("fusionLearning.data.dataloaders.Image.open")
    @patch("fusionLearning.data.dataloaders.get_file_paths")
    def test_background_label_in_batch(self, mock_gfp, mock_open, _):
        mock_gfp.side_effect = self._fake_gfp(fill=0)
        mock_open.side_effect = self._fake_open(fill=0)
        train, _, _ = create_ade20k_loaders_distributed("/fake", batch_size=2, num_workers=0)
        _, masks, _ = next(iter(train))
        self.assertTrue((masks == 255).all())  # raw 0 → ignore 255

    @patch("fusionLearning.data.dataloaders.DistributedSampler", side_effect=_seq_sampler)
    @patch("fusionLearning.data.dataloaders.Image.open")
    @patch("fusionLearning.data.dataloaders.get_file_paths")
    def test_val_and_test_loaders_are_not_none(self, mock_gfp, mock_open, _):
        mock_gfp.side_effect = self._fake_gfp()
        mock_open.side_effect = self._fake_open()
        _, val, test = create_ade20k_loaders_distributed("/fake", batch_size=2, num_workers=0)
        self.assertIsNotNone(val)
        self.assertIsNotNone(test)


if __name__ == "__main__":
    unittest.main(verbosity=2)