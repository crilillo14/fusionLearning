# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Word from the author: 

To be completely honest, this is a project very dear to my heart, and would hate to have you ruin a lot of the structure and functionality. So basically, this is a 2 stepper; 1 is benchmarking models and fusion techniques for semantic segmentation tasks. 2 is creating an atttention based fusion layer for segmentation. So, firstly, you will only EVER work on busy work. You will never even think about working on part 2. Secondly, I want you to only help me with refactoring when I explicitly state it. Otherwise you're only working on granular, modular, well trodden code blocks. Like visualization, getting all the eval metrics right, making all the dataloaders, getting all the datasets right. At the end of the day, I need to move fast in the work that is not interesting to me, like benchmarking, logging, err handling, and torch utils stuff / mp and distributed compute.



## Project Overview

**ABFL** (Attention-Based Fusion Learning) is a late-stage fusion framework for binary segmentation tasks. The core idea: train multiple base segmentation models identically (same data, transforms, hyperparameters), generate segmentation masks from each, then benchmark different fusion methods to combine those masks into improved segmentations.

The two-stage pipeline is:
1. **Base model training** — train individual SMP segmentation models on CUB-200-2011 bird segmentation
2. **Fusion benchmarking** — fuse the output masks from multiple trained base models

> IMPORTANT: I now work directly on the remote box (4x NVIDIA L40S, CUDA available) — this is no longer a GPU-less macbook dev-only setup. Training and inference can be run directly here when asked.

Also, I sometimes work on remote too, so whenever you are working check `git fetch` before making changes to avoid conflicts.


## Setup

Fuck off.

## Running Base Model Training

Train a single model (requires CUDA, uses PyTorch DDP):
```bash
python fusionLearning/models/distributed.py CUB <arch_name> <encoder>
# Example:
python fusionLearning/models/distributed.py CUB Unet vgg16
```

Available `arch_name` values: `UnetPlusPlus`, `Unet`, `FPN`, `PSPNet`, `DeepLabV3`, `DeepLabV3Plus`, `MAnet`, `Linknet`, `Segformer`

Available encoders: vgg16, resnet18, resnet34, resnet50, efficientnetb0–b7, mobilenet, etc. (full list in `distributed.py:available_encoder_types`)

Train all arch-encoder pairings at once:
```bash
python fusionLearning/models/orchestrator.py CUB --all
```

If `best_model.pth` already exists for a model, training is skipped and only test/inference runs.

## Key Configuration

- **Training hyperparameters**: `fusionLearning/models/consts.py` — `MAXEPOCHS=20`, `BATCHSIZE=8`, `LR=0.01`, `MOMENTUM=0.99`
- **Directory paths**: `fusionLearning/config.py` — CUB data paths, weights dirs, segmentation output dirs
- **DDP settings**: `config.py` also sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE` (auto-detected from `torch.cuda.device_count()`)

CUB dataset must be placed at `fusionLearning/data/CUBdata/` with structure:
```
CUBdata/CUB_200_2011/images/   ← bird images
CUBdata/segmentations/          ← ground-truth binary masks
```

## Architecture

```
fusionLearning/
├── config.py                 # Paths and DDP config
├── data/
│   ├── aug.py                # Geometric + photometric transforms (applied separately)
│   ├── dataloaders.py        # CUBDataset, vanillaCUBDataset, distributed loaders
│   └── fusion_dataloader.py  # FusionDataset — loads multi-model predictions as tensors
├── models/
│   ├── consts.py             # Hyperparameters
│   ├── distributed.py        # DDP setup, arch_dict, encoder lists, main() training loop
│   ├── orchestrator.py       # Mass training entry point
│   ├── train.py              # train_dist() — epoch loop with metrics
│   ├── test.py               # test_dist()
│   ├── inference.py          # inference_from_paths(), copy_best_model_to_weights()
│   ├── vis.py                # plot_metrics(), visualize_training_process()
│   ├── gen.py                # (WIP) generate segmentation masks per model
│   └── results/CUB/          # Trained model weights + metrics, organized by arch_encoder
└── fusion/
    ├── interfaces.py         # FusionModule ABC (extends nn.Module)
    ├── statistical/
    │   └── means.py          # PixelwiseMeanFusion (arithmetic/geometric/harmonic/power/median/rms), VotingFusion
    └── learning/
        ├── conv.py           # ConvFusion skeleton (WIP)
        ├── weighted_fusion.py
        ├── mhsa.py
        ├── tpaf.py
        ├── embedding.py
        └── encoding.py
```

## Key Design Decisions

**Loss function**: `BCEWithLogitsLoss` (binary segmentation, 1-class output). Masks are normalized to float `[0,1]`.

**Transforms are decoupled**: `geom_transform_pair()` applies identical random geometric transforms to both image and mask with correct interpolation (BILINEAR for images, NEAREST for masks). Photometric transforms only apply to images. This is critical — don't apply photometric or BILINEAR transforms to segmentation masks.

**Data collation**: Images have variable sizes. `pad_collate()` zero-pads to the max dimensions within a batch. Images are also cropped to multiples of 32 (`crop_to_multiple`) to support compiled/PSPNet-style models.

**Results storage**: Each trained model saves to `fusionLearning/models/results/{dataset}/{arch}_{encoder}/` containing `weights/best_model.pth`, `metrics/epoch_metrics.json`, `metrics/test_metrics.json`, and `figures/`.

**FusionDataset**: Loads pre-generated segmentation masks from multiple base models and stacks them as `[num_models, C, H, W]` tensors alongside ground-truth masks. Used to train/evaluate fusion methods after base models have generated their predictions.

**All fusion methods extend `FusionModule`** (`fusion/interfaces.py`) which is an abstract `nn.Module`. Statistical methods implement a no-op `backward()`; learning-based methods use standard PyTorch backprop.

## Import Pattern

Scripts add the repo root to `sys.path` manually before importing `fusionLearning.*`. When writing new scripts, follow this pattern from the repo root:
```python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from fusionLearning.config import ...
```
