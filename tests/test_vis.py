"""
Unit tests for fusionLearning.data.vis dataset-agnostic visualization helpers.
Run with: python -m unittest tests/test_vis.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusionLearning.data.vis import (
    _get_raw_pil,
    _render_mask,
    visualize_geo_pair,
    visualize_geo_pair_samples,
    visualize_dataset_samples,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _rgb(h=64, w=64):
    return Image.fromarray(np.full((h, w, 3), 100, dtype=np.uint8), "RGB")


def _mask(h=64, w=64, fill=7):
    return Image.fromarray(np.full((h, w), fill, dtype=np.uint8), "L")


def _cs_dataset(n=4, fill=7):
    inner = MagicMock()
    inner.__len__ = MagicMock(return_value=n)
    inner.__getitem__ = MagicMock(return_value=(_rgb(), _mask(fill=fill)))
    inner.images = [f"/fake/img_{i}.png" for i in range(n)]
    with patch("torchvision.datasets.Cityscapes", return_value=inner):
        from fusionLearning.data.dataloaders import CityscapesDataset
        ds = CityscapesDataset("/fake")
    ds._inner = inner
    return ds


def _ade_dataset(n=4, h=64, w=64, fill=1):
    from fusionLearning.data.dataloaders import ADE20KDataset
    ds = ADE20KDataset.__new__(ADE20KDataset)
    ds.image_paths        = [f"/fake/ADE_train_{i:08d}.jpg" for i in range(n)]
    ds.segmentation_paths = [f"/fake/ADE_train_{i:08d}.png" for i in range(n)]
    ds.gTransforms = None
    ds.pTransforms = None
    ds._h, ds._w, ds._fill = h, w, fill
    return ds


def _load_image_mock(h=64, w=64):
    def _load(path):
        return _rgb(h, w), os.path.basename(path)
    return _load


def _load_seg_mock(h=64, w=64, fill=1):
    def _load(path):
        return _mask(h, w, fill)
    return _load


def _cub_dataset(n=4):
    from fusionLearning.data.dataloaders import CUBDataset
    ds = CUBDataset.__new__(CUBDataset)
    ds.image_paths = [f"/fake/img_{i}.png" for i in range(n)]
    ds.segmentation_paths = [f"/fake/seg_{i}.png" for i in range(n)]
    ds.use_geo = False
    ds.photometricTransforms = None
    return ds


# ── _get_raw_pil ──────────────────────────────────────────────────────────────

class TestGetRawPil(unittest.TestCase):

    def test_cityscapes_returns_pil_images(self):
        ds = _cs_dataset()
        img, msk, fname = _get_raw_pil(ds, 0)
        self.assertIsInstance(img, Image.Image)
        self.assertIsInstance(msk, Image.Image)

    def test_cityscapes_fname_is_basename(self):
        ds = _cs_dataset()
        _, _, fname = _get_raw_pil(ds, 0)
        self.assertEqual(fname, os.path.basename(fname))
        self.assertTrue(fname.endswith(".png"))

    def test_cityscapes_correct_index(self):
        ds = _cs_dataset(n=3)
        _get_raw_pil(ds, 2)
        ds._inner.__getitem__.assert_called_with(2)

    def test_ade20k_returns_pil_images(self):
        ds = _ade_dataset()
        with patch("fusionLearning.data.vis.load_image", side_effect=_load_image_mock()), \
             patch("fusionLearning.data.vis.load_segmentation_mask", side_effect=_load_seg_mock()):
            img, msk, fname = _get_raw_pil(ds, 0)
        self.assertIsInstance(img, Image.Image)
        self.assertIsInstance(msk, Image.Image)

    def test_ade20k_fname_is_basename(self):
        ds = _ade_dataset()
        with patch("fusionLearning.data.vis.load_image", side_effect=_load_image_mock()), \
             patch("fusionLearning.data.vis.load_segmentation_mask", side_effect=_load_seg_mock()):
            _, _, fname = _get_raw_pil(ds, 0)
        self.assertEqual(fname, os.path.basename(fname))
        self.assertTrue(fname.endswith(".jpg"))

    def test_ade20k_correct_path_used(self):
        ds = _ade_dataset(n=3)
        mock_load = MagicMock(side_effect=_load_image_mock())
        mock_seg  = MagicMock(side_effect=_load_seg_mock())
        with patch("fusionLearning.data.vis.load_image", mock_load), \
             patch("fusionLearning.data.vis.load_segmentation_mask", mock_seg):
            _get_raw_pil(ds, 1)
        mock_load.assert_called_once_with(ds.image_paths[1])
        mock_seg.assert_called_once_with(ds.segmentation_paths[1])

    def test_unsupported_dataset_raises(self):
        class WeirdDataset:
            pass
        with self.assertRaises(TypeError):
            _get_raw_pil(WeirdDataset(), 0)


# ── _render_mask ──────────────────────────────────────────────────────────────

class TestRenderMask(unittest.TestCase):

    def _make_ax(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        return fig, ax

    def test_binary_float_uses_gray(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        arr = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)
        _render_mask(ax, arr, "binary")
        im = ax.get_images()[0]
        self.assertEqual(im.cmap.name, "gray")
        plt.close(fig)

    def test_binary_int_uses_gray(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        arr = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        _render_mask(ax, arr, "binary int")
        im = ax.get_images()[0]
        self.assertEqual(im.cmap.name, "gray")
        plt.close(fig)

    def test_multiclass_uses_tab20(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        arr = np.array([[0, 5], [12, 255]], dtype=np.int64)
        _render_mask(ax, arr, "multiclass")
        im = ax.get_images()[0]
        self.assertEqual(im.cmap.name, "tab20")
        plt.close(fig)

    def test_title_set(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        arr = np.zeros((4, 4), dtype=np.uint8)
        _render_mask(ax, arr, "my title")
        self.assertEqual(ax.get_title(), "my title")
        plt.close(fig)


# ── visualize_geo_pair ────────────────────────────────────────────────────────

class TestVisualizeGeoPair(unittest.TestCase):

    def test_returns_figure_cityscapes(self):
        import matplotlib
        matplotlib.use("Agg")
        ds = _cs_dataset()
        fig = visualize_geo_pair(ds, idx=0, show=False)
        import matplotlib.pyplot as plt
        self.assertEqual(type(fig).__name__, "Figure")
        plt.close(fig)

    def test_returns_figure_ade20k(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _ade_dataset()
        with patch("fusionLearning.data.vis.load_image", side_effect=_load_image_mock()), \
             patch("fusionLearning.data.vis.load_segmentation_mask", side_effect=_load_seg_mock()):
            fig = visualize_geo_pair(ds, idx=0, show=False)
        self.assertEqual(type(fig).__name__, "Figure")
        plt.close(fig)

    def test_four_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _cs_dataset()
        fig = visualize_geo_pair(ds, idx=0, show=False)
        self.assertEqual(len(fig.axes), 4)
        plt.close(fig)

    def test_suptitle_is_filename(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _cs_dataset()
        fig = visualize_geo_pair(ds, idx=0, show=False)
        self.assertIn("img_0", fig.texts[0].get_text())
        plt.close(fig)

    def test_file_path_mode_raises_on_missing_file(self):
        with self.assertRaises(Exception):
            visualize_geo_pair("/nonexistent/img.png", "/nonexistent/mask.png", show=False)


# ── visualize_geo_pair_samples ────────────────────────────────────────────────

class TestVisualizeGeoPairSamples(unittest.TestCase):

    def test_produces_n_figures_cityscapes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _cs_dataset(n=6)
        plt.close("all")
        visualize_geo_pair_samples(ds, num_samples=3, show=False)
        self.assertEqual(plt.get_fignums().__len__(), 3)
        plt.close("all")

    def test_produces_n_figures_ade20k(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _ade_dataset(n=6)
        plt.close("all")
        with patch("fusionLearning.data.vis.load_image", side_effect=_load_image_mock()), \
             patch("fusionLearning.data.vis.load_segmentation_mask", side_effect=_load_seg_mock()):
            visualize_geo_pair_samples(ds, num_samples=2, show=False)
        self.assertEqual(len(plt.get_fignums()), 2)
        plt.close("all")

    def test_clamps_to_dataset_size(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _cs_dataset(n=2)
        plt.close("all")
        visualize_geo_pair_samples(ds, num_samples=10, show=False)
        self.assertEqual(len(plt.get_fignums()), 2)
        plt.close("all")


# ── visualize_dataset_samples ─────────────────────────────────────────────────

class TestVisualizeDatasetSamples(unittest.TestCase):

    def test_returns_figure_cityscapes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _cs_dataset(n=4)
        fig = visualize_dataset_samples(ds, num_samples=2, show=False)
        self.assertEqual(type(fig).__name__, "Figure")
        plt.close(fig)

    def test_returns_figure_ade20k(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _ade_dataset(n=4)
        with patch("fusionLearning.data.dataloaders.Image.open",
                   side_effect=lambda p: _rgb() if p.endswith(".jpg") else _mask(fill=ds._fill)):
            fig = visualize_dataset_samples(ds, num_samples=2, show=False)
        self.assertEqual(type(fig).__name__, "Figure")
        plt.close(fig)

    def test_correct_number_of_rows(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ds = _cs_dataset(n=6)
        fig = visualize_dataset_samples(ds, num_samples=3, show=False)
        # 3 rows × 2 cols = 6 axes
        self.assertEqual(len(fig.axes), 6)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main(verbosity=2)
