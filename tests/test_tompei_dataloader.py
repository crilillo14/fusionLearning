"""
Unit tests for TOMPEICMMDDataset / tompei_dataloader.py.

Mocks pydicom.dcmread (and the file system / label file reads) so no real .dcm
fixtures or dataset download are needed.
Run with: python -m pytest tests/test_tompei_dataloader.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fusionLearning.data.tompei_dataloader import (
    TOMPEICMMDDataset,
    create_tompei_cmmd_loaders_distributed,
    get_dcm_paths,
    load_dicom_image,
)

MODULE = "fusionLearning.data.tompei_dataloader"


def _fake_ds(photometric: str = "MONOCHROME2") -> MagicMock:
    ds = MagicMock()
    ds.get.side_effect = lambda key, default=None: (
        photometric if key == "PhotometricInterpretation" else default
    )
    return ds


def _seq_sampler(dataset, **kwargs):
    from torch.utils.data import SequentialSampler
    return SequentialSampler(dataset)


# ── get_dcm_paths ─────────────────────────────────────────────────────────────

class TestGetDcmPaths(unittest.TestCase):

    @patch(f"{MODULE}.os.listdir")
    def test_filters_to_dcm_only(self, mock_listdir):
        mock_listdir.return_value = ["a.dcm", "b.txt", "c.DCM", "d.dcm"]
        paths = get_dcm_paths("/fake/dir")
        self.assertEqual(len(paths), 2)

    @patch(f"{MODULE}.os.listdir")
    def test_returns_sorted(self, mock_listdir):
        mock_listdir.return_value = ["b.dcm", "a.dcm"]
        paths = get_dcm_paths("/fake/dir")
        self.assertEqual(os.path.basename(paths[0]), "a.dcm")
        self.assertEqual(os.path.basename(paths[1]), "b.dcm")


# ── load_dicom_image ───────────────────────────────────────────────────────────

class TestLoadDicomImage(unittest.TestCase):

    @patch(f"{MODULE}.apply_voi_lut")
    @patch(f"{MODULE}.pydicom.dcmread")
    def test_min_max_normalization_range(self, mock_dcmread, mock_avl):
        mock_dcmread.return_value = _fake_ds()
        mock_avl.return_value = np.array([[0, 128], [255, 64]], dtype=np.uint8)
        arr, _ = load_dicom_image("/fake/img.dcm")
        self.assertAlmostEqual(float(arr.min()), 0.0)
        self.assertAlmostEqual(float(arr.max()), 1.0)

    @patch(f"{MODULE}.apply_voi_lut")
    @patch(f"{MODULE}.pydicom.dcmread")
    def test_constant_image_normalizes_to_zero(self, mock_dcmread, mock_avl):
        mock_dcmread.return_value = _fake_ds()
        mock_avl.return_value = np.full((8, 8), 100, dtype=np.uint8)
        arr, _ = load_dicom_image("/fake/img.dcm")
        self.assertTrue((arr == 0).all())

    @patch(f"{MODULE}.apply_voi_lut")
    @patch(f"{MODULE}.pydicom.dcmread")
    def test_monochrome1_inverts_before_normalizing(self, mock_dcmread, mock_avl):
        mock_dcmread.return_value = _fake_ds(photometric="MONOCHROME1")
        mock_avl.return_value = np.array([[0, 255]], dtype=np.uint8)
        arr, _ = load_dicom_image("/fake/img.dcm")
        # MONOCHROME1: raw 0 is brightest -> after inversion+normalize it's 1.0
        self.assertAlmostEqual(float(arr[0, 0]), 1.0)
        self.assertAlmostEqual(float(arr[0, 1]), 0.0)

    @patch(f"{MODULE}.apply_voi_lut", side_effect=Exception("no VOI LUT in this file"))
    @patch(f"{MODULE}.pydicom.dcmread")
    def test_falls_back_to_raw_pixel_array_on_voi_lut_failure(self, mock_dcmread, mock_avl):
        ds = _fake_ds()
        ds.pixel_array = np.full((8, 8), 200, dtype=np.uint8)
        mock_dcmread.return_value = ds
        arr, fname = load_dicom_image("/fake/img.dcm")
        self.assertEqual(arr.shape, (8, 8))
        self.assertEqual(fname, "img.dcm")

    @patch(f"{MODULE}.apply_voi_lut")
    @patch(f"{MODULE}.pydicom.dcmread")
    def test_filename_is_basename(self, mock_dcmread, mock_avl):
        mock_dcmread.return_value = _fake_ds()
        mock_avl.return_value = np.zeros((4, 4), dtype=np.uint8)
        _, fname = load_dicom_image("/fake/dir/scan.dcm")
        self.assertEqual(fname, "scan.dcm")


# ── TOMPEICMMDDataset ────────────────────────────────────────────────────────

def _make_dataset(n=3, train=False):
    fnames = [f"D1-000{i}_study_series_1-{i}.dcm" for i in range(n)]
    with patch(f"{MODULE}.os.listdir", return_value=fnames), \
         patch(f"{MODULE}.os.path.exists", return_value=True):
        ds = TOMPEICMMDDataset("/fake/train", "/fake/train_label", train=train)
    return ds, fnames


class TestTOMPEICMMDDataset(unittest.TestCase):

    def test_len(self):
        ds, _ = _make_dataset(n=5)
        self.assertEqual(len(ds), 5)

    @patch(f"{MODULE}.load_dicom_image")
    def test_getitem_returns_three_tuple(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.full((64, 64), 0.5, dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="lesion")):
            out = ds[0]
        self.assertEqual(len(out), 3)

    @patch(f"{MODULE}.load_dicom_image")
    def test_image_shape_and_dtype(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.full((64, 64), 0.5, dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="normal")):
            img, _, _ = ds[0]
        self.assertEqual(img.shape, (3, 512, 512))
        self.assertEqual(img.dtype, torch.float32)

    @patch(f"{MODULE}.load_dicom_image")
    def test_image_range_0_to_1(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.full((64, 64), 0.5, dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="lesion")):
            img, _, _ = ds[0]
        self.assertGreaterEqual(img.min().item(), 0.0)
        self.assertLessEqual(img.max().item(), 1.0)

    @patch(f"{MODULE}.load_dicom_image")
    def test_channels_are_replicated_grayscale(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.full((64, 64), 0.3, dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="lesion")):
            img, _, _ = ds[0]
        self.assertTrue(torch.allclose(img[0], img[1]))
        self.assertTrue(torch.allclose(img[1], img[2]))

    @patch(f"{MODULE}.load_dicom_image")
    def test_label_lesion_maps_to_1(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.zeros((64, 64), dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="lesion")):
            _, label, _ = ds[0]
        self.assertEqual(label.item(), 1.0)
        self.assertEqual(label.dtype, torch.float32)
        self.assertEqual(label.shape, (1,))

    @patch(f"{MODULE}.load_dicom_image")
    def test_label_normal_maps_to_0(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.zeros((64, 64), dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="normal\n")):
            _, label, _ = ds[0]
        self.assertEqual(label.item(), 0.0)

    @patch(f"{MODULE}.load_dicom_image")
    def test_label_string_is_case_and_whitespace_insensitive(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.zeros((64, 64), dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="  LESION  \n")):
            _, label, _ = ds[0]
        self.assertEqual(label.item(), 1.0)

    @patch(f"{MODULE}.load_dicom_image")
    def test_train_mode_output_shape_matches_eval_mode(self, mock_load):
        ds_train, fnames = _make_dataset(n=1, train=True)
        mock_load.return_value = (np.full((64, 64), 0.5, dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="lesion")):
            img, _, _ = ds_train[0]
        self.assertEqual(img.shape, (3, 512, 512))

    @patch(f"{MODULE}.load_dicom_image")
    def test_filename_matches_source_file(self, mock_load):
        ds, fnames = _make_dataset(n=1)
        mock_load.return_value = (np.zeros((64, 64), dtype=np.float32), fnames[0])
        with patch(f"{MODULE}.open", mock_open(read_data="lesion")):
            _, _, fname = ds[0]
        self.assertEqual(fname, fnames[0])

    def test_missing_label_file_raises_value_error(self):
        fnames = ["D1-0001_a_b_1-1.dcm"]
        with patch(f"{MODULE}.os.listdir", return_value=fnames), \
             patch(f"{MODULE}.os.path.exists", return_value=False):
            with self.assertRaises(ValueError):
                TOMPEICMMDDataset("/fake/train", "/fake/train_label")


# ── create_tompei_cmmd_loaders_distributed ─────────────────────────────────────

class TestCreateTompeiCmmdLoadersDistributed(unittest.TestCase):

    def _patched(self, n=4):
        fnames = [f"D1-000{i}_a_b_1-{i}.dcm" for i in range(n)]
        return (
            patch(f"{MODULE}.load_dicom_image",
                  return_value=(np.zeros((64, 64), dtype=np.float32), fnames[0])),
            patch(f"{MODULE}.DistributedSampler", side_effect=_seq_sampler),
            patch(f"{MODULE}.os.path.exists", return_value=True),
            patch(f"{MODULE}.os.listdir", return_value=fnames),
        )

    def test_returns_three_loaders(self):
        p_load, p_sampler, p_exists, p_listdir = self._patched()
        with p_load, p_sampler, p_exists, p_listdir, patch(f"{MODULE}.open", mock_open(read_data="lesion")):
            train, val, test = create_tompei_cmmd_loaders_distributed(
                "/fake/train", "/fake/train_label",
                "/fake/val", "/fake/val_label",
                "/fake/test", "/fake/test_label",
                batch_size=2, num_workers=0,
            )
        self.assertIsNotNone(train)
        self.assertIsNotNone(val)
        self.assertIsNotNone(test)
        self.assertIsNot(train, val)
        self.assertIsNot(val, test)

    def test_batch_shapes(self):
        p_load, p_sampler, p_exists, p_listdir = self._patched()
        with p_load, p_sampler, p_exists, p_listdir, patch(f"{MODULE}.open", mock_open(read_data="normal")):
            train, _, _ = create_tompei_cmmd_loaders_distributed(
                "/fake/train", "/fake/train_label",
                "/fake/val", "/fake/val_label",
                "/fake/test", "/fake/test_label",
                batch_size=2, num_workers=0,
            )
            imgs, labels, fnames = next(iter(train))
        self.assertEqual(imgs.shape, (2, 3, 512, 512))
        self.assertEqual(imgs.dtype, torch.float32)
        self.assertEqual(labels.shape, (2, 1))
        self.assertEqual(len(fnames), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
