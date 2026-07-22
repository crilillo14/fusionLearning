# TOMPEI-CMMD Binary Classification Pipeline — Engineering Spec

Second parallel task track alongside the existing CUB/Cityscapes/ADE20K/VOC **segmentation**
pipeline. This is **classification**, on grayscale mammography DICOMs, using **timm** instead
of `segmentation_models_pytorch`. Nothing below touches `fusion/` (out of scope per CLAUDE.md).

## 0. Scope guardrail

Every file listed in §2 is **new**. `train.py`, `test.py`, `distributed.py`, `dataloaders.py`,
`vis.py`, `inference.py`, `orchestrator.py`, `aug.py` are not edited or refactored — only
imported from where genuinely reusable (`ddp_setup`, `copy_best_model_to_weights`). The only
existing files touched are `config.py`, `models/consts.py`, and `pyproject.toml`, via small
additive blocks only.

## 1. Dataset facts (verified on disk)

`fusionLearning/data/TOMPEI-CMMD/Task_classification/`
- `{train,val,test}/` — flat dirs of raw `.dcm`, ~2000×2000, single-channel grayscale.
- `{train,val,test}_label/` — paired dirs, same filename stem, each a `.txt` containing
  `lesion` or `normal`. 1:1 correspondence by stem.
- Counts: train 2892 (1934 lesion / 958 normal), val 412 (276/136), test 828 (554/274) —
  ~2:1 class imbalance (lesion:normal) in every split.
- Splits are **pre-made and patient-disjoint** — use the three folders directly, do **not**
  `random_split()` like `CUBDataset` does.
- `split_manifest.csv` has a richer `raw_tompei_classification` field (Normal/Benign/Malignant)
  — **not used**; confirmed binary target uses the flat label files as-shipped.
- `exclusion/`, `invisible/`, `Segmentations/*.json`, nested `CMMD/{patient}/{study}/{series}/`
  raw DICOMs, and the clinical `.xlsx` — all **out of scope**.

## 2. New files to create

```
fusionLearning/data/tompei_dataloader.py    # TOMPEICMMDDataset + create_tompei_cmmd_loaders_distributed
fusionLearning/models/train_cls.py          # train_dist_cls()
fusionLearning/models/test_cls.py           # test_dist_cls()
fusionLearning/models/distributed_cls.py    # main(), TIMM_MODELS roster, ddp entry point
fusionLearning/models/orchestrator_cls.py   # mass-training entry point over TIMM_MODELS
fusionLearning/models/vis_cls.py            # plot_metrics_cls() — loss/acc curves + CM heatmap
fusionLearning/models/inference_cls.py      # inference_from_paths_cls() + gradcam_from_paths_cls()
tests/test_tompei_dataloader.py             # unit tests, mocked pydicom.dcmread (no real .dcm needed)
```

## 3. Decision log

| # | Decision | Choice | Why |
|---|---|---|---|
| 1 | Label tensor / loss | `float32` scalar `[1]` (0.0=normal, 1.0=lesion) + `BCEWithLogitsLoss`, `timm.create_model(..., num_classes=1)` | Mirrors the codebase's existing binary convention exactly (`NUM_CLASSES_CUB=1` + `BCEWithLogitsLoss`). |
| 2 | Resize | Direct resize (no crop) to **512×512**, **bicubic** interpolation | User-specified range (500–800px) and interpolation mode. Plain resize (not center-crop) preserves full field of view — mammograms have no consistent object-centering, so cropping risks cutting off tissue at the periphery. 512 is divisible by 32 (standard conv-stride friendliness) and still lets the full timm roster (incl. ViT/Swin) run without becoming compute-impractical, unlike native ~2000×2000. |
| 3 | DICOM normalization | `pydicom.dcmread` → `pydicom.pixels.apply_voi_lut` (fallback to raw `pixel_array` if absent) → invert if `PhotometricInterpretation == "MONOCHROME1"` → **per-image min-max normalize to [0,1]** | Standard mammography-CV normalization; DICOM intensity ranges vary by manufacturer/exposure, so per-image (not dataset-wide) min-max is the right default. MONOCHROME1 inversion avoids silently feeding inverted-brightness images. |
| 4 | Channel replication | Normalize+resize as 1-channel, then `.repeat(3,1,1)` → `[3,512,512]` | Confirmed: lets standard ImageNet-pretrained timm weights load unmodified. Done after resize for cheapest compute. |
| 5 | Train-time augmentation | New **image-only** transform (not in `aug.py`): `RandomHorizontalFlip(0.5)`, `RandomRotation(7°, bicubic)`, `RandomAdjustSharpness(1.1, p=0.1)`. No vertical flip (breast orientation is anatomically meaningful), no color/hue/saturation jitter (source is grayscale-replicated, these ops are noise), no perspective warp (no camera-perspective variation exists in mammography acquisition). | `aug.py`'s existing pipelines are built around paired image+mask geometric consistency and color-photo assumptions — neither applies here. |
| 6 | Best-epoch criterion | **Lowest val loss** (not accuracy) | With ~2:1 class imbalance, raw accuracy is gameable by a majority-class predictor; loss is smoother and imbalance-aware. Balanced accuracy is still logged/plotted from the confusion matrix, just not used for model selection. |
| 7 | Confusion matrix DDP reduction | `dist.all_reduce(cm_tensor, op=ReduceOp.SUM)` on the raw `[2,2]` count tensor from `torchmetrics.classification.BinaryConfusionMatrix` | Counts must be summed, not averaged, unlike loss (which stays `ReduceOp.AVG`, matching existing `train_dist` behavior). |
| 8 | Model roster | Flat `TIMM_MODELS` list (no arch/encoder split — a timm name is already architecture+encoder fused): `resnet18, resnet50, resnet101, densenet121, efficientnet_b0, efficientnet_b3, convnext_tiny, convnext_base, vit_base_patch16_224, swin_tiny_patch4_window7_224, mobilenetv3_large_100, regnety_032, tf_efficientnetv2_s` | Mirrors the breadth of the existing `flat_encoders` list. All verified importable via `timm.list_models()`. At 512×512, ViT/Swin are computationally reasonable (unlike at native ~2000×2000), so the roster doesn't need to be CNN-only. |
| 9 | Batch size | New `BATCHSIZE_CLS = 16` | Flagged as a remote-GPU tuning knob (this dev machine has no GPU to profile against) — best-guess default, not measured. |
| 10 | Epoch horizon / LR schedule | `MAXEPOCHS_CLS = 20`, `LEARNING_RATE_CLS = 0.01`, `LR_MIN_CLS = 1e-6`, `CosineAnnealingLR(optimizer, T_max=MAXEPOCHS_CLS, eta_min=LR_MIN_CLS)`, fixed horizon (no early stopping) | Confirmed: "vanilla, standard" LR scheduling, fixed epoch horizon. Reuses existing hyperparameter values where no reason to diverge. |
| 11 | Optimizer | `SGD(model.parameters(), lr=LEARNING_RATE_CLS, momentum=MOMENTUM)`, `MOMENTUM` shared from existing `consts.py` | No stated reason to diverge from the segmentation pipeline's optimizer choice. |
| 12 | Naming convention | `_cls` suffix for 1:1 structural parallels of an existing file (`train_cls.py`, `test_cls.py`, `distributed_cls.py`, `orchestrator_cls.py`, `vis_cls.py`, `inference_cls.py`); dataloader named after its domain (`tompei_dataloader.py`), matching the existing `fusion_dataloader.py` sibling pattern rather than `dataloaders_cls.py` | Disambiguates instantly against segmentation counterparts; matches existing repo naming precedent for domain-named dataloaders. |
| 13 | Results dir | `results/TOMPEI-CMMD/{timm_model_name}/{weights,metrics,figures}` (e.g. `results/TOMPEI-CMMD/resnet50/weights/best_model.pth`) | Matches `results/{dataset}/{arch}_{encoder}` convention minus the encoder segment (timm name is already the full spec). |
| 14 | Collation | Default `DataLoader` collation, no custom `collate_fn` | All images already fixed at 512×512 before batching — segmentation's `pad_collate` (for variable native sizes) is dead weight here. |
| 15 | Infra reuse | `ddp_setup` and `copy_best_model_to_weights` **imported as-is** from `distributed.py`/`inference.py`; skip-pair tracking **reimplemented locally** in `distributed_cls.py` (`skip_models_cls.json`, keyed by model name only, no arch/encoder pair) | Both reused functions are genuinely dataset/task-agnostic infra. Skip-pair tracking needs a different key shape (no encoder axis), so it's a small (~15 line) local reimplementation rather than repurposing the pair-keyed original. |
| 16 | New dependencies | `pydicom>=2.4.0,<4.0`, `timm>=1.0.0` (already installed locally but missing from `pyproject.toml`), `grad-cam>=1.5.0` | `apply_voi_lut` needs pydicom 2.4+. `grad-cam` (imports as `pytorch_grad_cam`) is the standard, maintained library for Grad-CAM across both CNN and transformer backbones — avoids hand-rolling hook-based activation/gradient capture. |
| 17 | pandas/openpyxl | Not added | `split_manifest.csv`/clinical `.xlsx` aren't read by this pipeline per confirmed scope. |
| 18 | Grad-CAM integration | `pytorch_grad_cam.GradCAM`, **per-stage for CNN families, final-block-only for transformer families** — see §5 | User-requested: "gradCAM explainability to inference per layer when applicable." CNN backbones (resnet/densenet/efficientnet/convnext/mobilenet/regnet) have spatially-meaningful intermediate feature maps at each stage — genuine per-layer progression is possible. ViT/Swin's intermediate blocks operate on token sequences, not spatial grids, until reshaped — Grad-CAM is only meaningful at the final block there (via `pytorch_grad_cam`'s `reshape_transform`), so "per-layer" degrades gracefully to "final-layer" for those two roster entries. This is the "when applicable" boundary. |

## 4. Data pipeline (`tompei_dataloader.py`)

- `load_dicom_image(path) -> (np.float32 [0,1] HxW array, filename)` — per decision #3.
- `classificationTransforms` — image-only `v2.Compose`, per decision #5, defined locally
  (not added to `aug.py`, which is explicitly built around paired image+mask geometric
  consistency).
- `TOMPEICMMDDataset(Dataset)`:
  - `__init__(image_dir, label_dir, train=False)` — `sorted()` `.dcm` paths, label paths
    matched by stem, fail-fast length check (mirrors `CUBDataset`).
  - `__getitem__` → load DICOM → normalize → resize 512×512 bicubic → augment if `train` →
    replicate to 3ch → label `"lesion"→1.0 / "normal"→0.0` → returns
    `(image_tensor [3,512,512] float32, label_tensor [1] float32, filename)`.
- `create_tompei_cmmd_loaders_distributed(...)` — **no `random_split`**; three
  `TOMPEICMMDDataset` instances built directly from the pre-split directory pairs, each with
  its own `DistributedSampler` (train `shuffle=True`, val/test `shuffle=False`), default
  collate (decision #14).

## 5. Grad-CAM spec (`inference_cls.py`)

```python
GRADCAM_FAMILY_BY_PREFIX = {
    "resnet": "cnn_staged", "densenet": "cnn_staged", "efficientnet": "cnn_staged",
    "tf_efficientnetv2": "cnn_staged", "convnext": "cnn_staged", "regnety": "cnn_staged",
    "mobilenetv3": "cnn_final",       # block structure too irregular for clean staging
    "vit_": "transformer_final", "swin_": "transformer_final",
}
```

- **`cnn_staged`** families: target layers = last block of each major stage (e.g. ResNet's
  `layer1[-1]..layer4[-1]`, ConvNeXt's `stages[0..3]`, DenseNet's `denseblock1..4`,
  EfficientNet's stage-boundary blocks). Run `pytorch_grad_cam.GradCAM` once per stage,
  producing one heatmap per stage → saved as a grid (rows=samples, cols=stages) at
  `figures/gradcam_{model_name}.png`. This is the genuine "per layer" case.
- **`cnn_final`** / **`transformer_final`**: single target layer (final feature block; for
  ViT/Swin, resolved via `pytorch_grad_cam`'s `reshape_transform` to unflatten tokens back to
  a spatial grid) → single-column CAM grid. This is the "when applicable" fallback.
- Must run with `torch.enable_grad()` even though the model is in `.eval()` mode — Grad-CAM
  needs gradients w.r.t. activations, so it must **not** be wrapped in `torch.no_grad()`
  (unlike the rest of inference).
- Called from `distributed_cls.py`'s `main()` alongside `inference_from_paths_cls(...)`, same
  `modelDir/figures/` convention.

## 6. `train_cls.py` / `test_cls.py` (mirrors `train_dist`/`test_dist` shape)

- Fresh `BinaryConfusionMatrix().to(device)` per epoch (same instantiate-fresh-per-epoch
  pattern as the existing `_make_iou`), threshold `torch.sigmoid(logits)` at 0.5.
- `scheduler.step()` once per epoch (cosine, decision #10).
- Loss reduced via `ReduceOp.AVG` (existing pattern); confusion matrix via `ReduceOp.SUM`
  (decision #7).
- `is_best = avg_val_loss < best_val_loss` (decision #6).
- `metrics/epoch_metrics.json` — same envelope shape as segmentation
  (`meta`/`epochs`/`best`), each epoch record replaces `train_miou`/`val_miou` with
  `train_acc`/`val_acc` (derived from the CM) and adds `train_cm`/`val_cm` as nested
  `[[tn,fp],[fn,tp]]` lists; `lr` is now `round(scheduler.get_last_lr()[0], 8)` instead of a
  static constant (the segmentation pipeline currently logs a static LR because it has no
  scheduler — this pipeline fixes that gap for itself only).
- `test_dist_cls` writes `metrics/test_metrics.json` with `test_loss`, `test_acc`, `test_cm`
  (summed across ranks), `tested_at`.

## 7. `distributed_cls.py` / `orchestrator_cls.py`

- `main(rank, world_size, model_name, dset="TOMPEI-CMMD")`: `ddp_setup` (imported) →
  `timm.create_model(model_name, pretrained=True, num_classes=1, in_chans=3)` → DDP wrap →
  `SGD` + `CosineAnnealingLR` + `BCEWithLogitsLoss` → dataloaders from
  `create_tompei_cmmd_loaders_distributed` → `results/TOMPEI-CMMD/{model_name}/` → same
  trained-check/skip-resume pattern as `distributed.py` → `train_dist_cls`/`test_dist_cls` →
  `plot_metrics_cls` → `inference_from_paths_cls` + `gradcam_from_paths_cls` →
  `copy_best_model_to_weights` (imported, reused).
- `launch_training_cls(model_name, dataset="TOMPEI-CMMD")` — `mp.spawn` wrapper, same shape
  as `launch_training`.
- CLI: `python distributed_cls.py TOMPEI-CMMD <model_name>` (`choices=TIMM_MODELS`).
- `orchestrator_cls.py` iterates the flat `TIMM_MODELS` list (`--all` / `--model <name>`),
  same skip/success/fail bucket reporting as `orchestrator.py`.

## 8. Config / consts / dependency additions

**`config.py`** (append):
```python
TOMPEI_CMMD = os.path.join(DATA_DIR, "TOMPEI-CMMD")
TOMPEI_CMMD_TASK = os.path.join(TOMPEI_CMMD, "Task_classification")
TOMPEI_CMMD_TRAIN = os.path.join(TOMPEI_CMMD_TASK, "train")
TOMPEI_CMMD_TRAIN_LABEL = os.path.join(TOMPEI_CMMD_TASK, "train_label")
TOMPEI_CMMD_VAL = os.path.join(TOMPEI_CMMD_TASK, "val")
TOMPEI_CMMD_VAL_LABEL = os.path.join(TOMPEI_CMMD_TASK, "val_label")
TOMPEI_CMMD_TEST = os.path.join(TOMPEI_CMMD_TASK, "test")
TOMPEI_CMMD_TEST_LABEL = os.path.join(TOMPEI_CMMD_TASK, "test_label")
```

**`models/consts.py`** (append):
```python
MAXEPOCHS_CLS : int = 20
BATCHSIZE_CLS : int = 16
LEARNING_RATE_CLS : float = 0.01
LR_MIN_CLS : float = 1e-6
NUM_CLASSES_TOMPEI_CMMD : int = 1
INPUT_SIZE_TOMPEI_CMMD : int = 512
```

**`pyproject.toml`** (append to `dependencies`):
```
"pydicom>=2.4.0,<4.0",
"timm>=1.0.0",
"grad-cam>=1.5.0",
```

## 9. Testing

`tests/test_tompei_dataloader.py` mocks `pydicom.dcmread` (returns a `MagicMock` with
`.pixel_array` = synthetic `np.uint16` array, `PhotometricInterpretation="MONOCHROME2"`) —
same style as existing dataset tests mock `torchvision.datasets.Cityscapes`. No real `.dcm`
fixtures needed, keeps this fully runnable on the GPU-less dev machine.

## Open items deferred to implementation time (flagged, not blocking)

- `BATCHSIZE_CLS = 16` is an unmeasured guess — tune against actual remote GPU memory once
  training starts.
- Resize is plain (aspect-distorting); revisit to pad-to-square-then-resize if accuracy is
  poor and distortion looks like the cause.
