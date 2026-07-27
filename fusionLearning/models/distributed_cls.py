from __future__ import annotations

import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

import json
import traceback

import timm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from fusionLearning.config import (
    TOMPEI_CMMD_TRAIN, TOMPEI_CMMD_TRAIN_LABEL,
    TOMPEI_CMMD_VAL, TOMPEI_CMMD_VAL_LABEL,
    TOMPEI_CMMD_TEST, TOMPEI_CMMD_TEST_LABEL,
    WORLD_SIZE,
)
from fusionLearning.models.consts import (
    MAXEPOCHS_CLS, LR_MIN_CLS, MOMENTUM,
    NUM_CLASSES_TOMPEI_CMMD, INPUT_SIZE_TOMPEI_CMMD,
)
from fusionLearning.data.tompei_dataloader import create_tompei_cmmd_loaders_distributed
from fusionLearning.models.roster_cls import MODEL_CONFIGS, get_config
from fusionLearning.models.train_cls import train_dist_cls
from fusionLearning.models.test_cls import test_dist_cls
from fusionLearning.models.vis_cls import plot_metrics_cls, plot_extended_metrics_cls
from fusionLearning.models.inference_cls import (
    inference_from_paths_cls, gradcam_from_paths_cls, gradcam_summary_cls,
)
import fusionLearning.models.distributed as _seg_distributed
from fusionLearning.models.distributed import ddp_setup
from fusionLearning.models.inference import copy_best_model_to_weights

# Classification jobs use a separate rendezvous port from segmentation (config.MASTER_PORT)
# so both DDP process groups can run concurrently on the same GPUs without EADDRINUSE.
_seg_distributed.MASTER_PORT = "12356"

BASE_MODELS = os.path.abspath(os.path.dirname(__file__))

dataset_metadata_cls = {
    "TOMPEI-CMMD": {"num_classes": NUM_CLASSES_TOMPEI_CMMD, "input_size": INPUT_SIZE_TOMPEI_CMMD},
}

VARIANT_IDS = [c["variant_id"] for c in MODEL_CONFIGS]

# ── skip-variant tracking (scoped separately from segmentation's arch/encoder-keyed
#    skip_pairs.json, since there's no encoder axis here - see notes/tompei_cmmd_classification_spec.md #20)
SKIP_FILE_CLS = os.path.join(BASE_MODELS, "skip_models_cls.json")


def load_skip_set_cls() -> set:
    if os.path.exists(SKIP_FILE_CLS):
        with open(SKIP_FILE_CLS) as f:
            return set(json.load(f))
    return set()


def record_failure_cls(variant_id: str, error: str) -> None:
    err_dir = os.path.join(BASE_MODELS, "_errors")
    os.makedirs(err_dir, exist_ok=True)
    with open(os.path.join(err_dir, f"cls_{variant_id}.txt"), "a") as f:
        f.write(f"\n{'='*60}\n{error}\n{traceback.format_exc()}\n")
    skip = load_skip_set_cls()
    skip.add(variant_id)
    with open(SKIP_FILE_CLS, "w") as f:
        json.dump(sorted(skip), f, indent=2)


def create_classifier(timm_name: str, family: str, resolution: int, num_classes: int = 1) -> torch.nn.Module:
    """
    Instantiates a timm classifier for `resolution` x `resolution` input (per-variant,
    from roster_cls.py's depth x resolution grid). ViT accepts non-native resolution via
    dynamic_img_size (position embeddings interpolated at forward time); a handful of ViT
    variants (e.g. relpos family) don't support that kwarg and need img_size fixed at
    construction instead. Swin/MaxViT/CoAtNet don't support dynamic_img_size at all, but
    (confirmed empirically against timm==1.0.28) accept an explicit img_size= kwarg at
    construction for arbitrary resolutions - including ones not evenly divisible by
    patch_size*window_size - without erroring, so no manual window-size arithmetic is
    needed for these windowed-attention families.
    """
    kwargs = dict(pretrained=True, num_classes=num_classes, in_chans=3)
    if family == "vit":
        try:
            return timm.create_model(timm_name, dynamic_img_size=True, **kwargs)
        except TypeError:
            return timm.create_model(timm_name, img_size=resolution, **kwargs)
    if family in ("swin", "maxvit", "coatnet"):
        return timm.create_model(timm_name, img_size=resolution, **kwargs)
    return timm.create_model(timm_name, **kwargs)


def main(rank, world_size, variant_id, dset="TOMPEI-CMMD"):
    global torch

    try:
        ddp_setup(rank, world_size)

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        cfg = get_config(variant_id)
        timm_name, family = cfg["timm_name"], cfg["family"]
        lr, batch_size, resolution = cfg["lr"], cfg["batch_size"], cfg["resolution_px"]

        num_classes = dataset_metadata_cls[dset]["num_classes"]

        # Pretrained weights must land in the shared hub cache before any other rank
        # calls create_classifier - all WORLD_SIZE ranks hitting a cold cache at once
        # race each other over the network and reliably crash the download (see
        # _errors/cls_vit_M1.txt, cls_xception_T1.txt: httpx client closed mid-retry /
        # connection timeout). Rank 0 downloads alone, then the barrier ensures every
        # other rank's create_classifier call below is a pure local-disk cache hit.
        if rank == 0:
            _prefetch = create_classifier(timm_name, family, resolution, num_classes=num_classes)
            del _prefetch
        dist.barrier()

        model = create_classifier(timm_name, family, resolution, num_classes=num_classes).to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
        scheduler = CosineAnnealingLR(optimizer, T_max=MAXEPOCHS_CLS, eta_min=LR_MIN_CLS)
        lossFunc = torch.nn.BCEWithLogitsLoss()

        training_dataloader, validation_dataloader, test_dataloader = create_tompei_cmmd_loaders_distributed(
            TOMPEI_CMMD_TRAIN, TOMPEI_CMMD_TRAIN_LABEL,
            TOMPEI_CMMD_VAL, TOMPEI_CMMD_VAL_LABEL,
            TOMPEI_CMMD_TEST, TOMPEI_CMMD_TEST_LABEL,
            batch_size=batch_size,
            resolution=resolution,
            num_workers=4,
        )

        modelName = f"{variant_id}_{timm_name}"
        modelDir = os.path.join(BASE_MODELS, "results", dset, modelName) + os.sep

        os.makedirs(modelDir + "metrics", exist_ok=True)
        os.makedirs(modelDir + "figures", exist_ok=True)
        os.makedirs(modelDir + "weights", exist_ok=True)

        if rank == 0:
            # Full roster config saved standalone (not only inside epoch_metrics.json's
            # meta block) so it's readable without a training run ever happening -
            # e.g. for the aggregation script, or a model reloaded from an already-trained
            # checkpoint (skips train_dist_cls entirely, see `trained` branch below).
            with open(os.path.join(modelDir, "metrics", "config.json"), "w") as f:
                json.dump(cfg, f, indent=2)

        trained = os.path.exists(modelDir + "weights/best_model.pth")

        if trained:
            if rank == 0:
                print("Model already trained. To retrain, delete 'best_model.pth' or move it elsewhere.\n Going ahead with testing...")
            state_dict = torch.load(modelDir + "weights/best_model.pth", map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            test_dist_cls(modelDir, model, test_dataloader, lossFunc, rank)
        else:
            if rank == 0:
                print("Starting training process...")
            train_dist_cls(modelDir, model, optimizer, scheduler, lossFunc,
                            training_dataloader, validation_dataloader, rank,
                            maxepochs=MAXEPOCHS_CLS, arch=timm_name, dataset=dset,
                            config=cfg)
            if rank == 0:
                print("\nStarting testing process...")
            test_loss = test_dist_cls(modelDir, model, test_dataloader, lossFunc, rank)
            if rank == 0:
                print(f"\nFinal test loss: {test_loss:.4f}")

        if rank == 0:
            plot_metrics_cls(modelDir)
            plot_extended_metrics_cls(modelDir)
            eval_model = model.module if isinstance(model, DDP) else model
            inference_from_paths_cls(eval_model, modelDir, test_dataloader, n=20)
            gradcam_from_paths_cls(eval_model, modelDir, test_dataloader, timm_name, family, n=8)
            gradcam_summary_cls(eval_model, modelDir, test_dataloader, timm_name, family, n=20)
            print(f"\n\t * Results and visualizations saved under {modelDir} * ")

        copy_best_model_to_weights(modelDir)

    except Exception as e:
        if rank == 0:
            record_failure_cls(variant_id, str(e))
        print(f"[rank {rank}] Error: {e}")
        traceback.print_exc()
        # Re-raise (every rank, not just rank 0) so mp.spawn's join=True actually
        # notices the failure - previously this was swallowed here, so mp.spawn saw
        # every rank "return normally" regardless of what happened, and callers
        # (launch_training_cls / orchestrator_cls.run_variant) had no way to tell a
        # totally-failed run from a real success. mp.spawn already terminates the
        # sibling processes when any one of them raises, so this doesn't introduce a
        # new hang risk - it makes the existing fail-fast behavior actually reachable.
        raise

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Distributed Classification Training (TOMPEI-CMMD)')
    parser.add_argument('dataset', type=str, choices=list(dataset_metadata_cls.keys()))
    parser.add_argument('variant_id', type=str, choices=VARIANT_IDS)
    args = parser.parse_args()

    mp.spawn(main, args=(WORLD_SIZE, args.variant_id, args.dataset), nprocs=WORLD_SIZE, join=True)


def launch_training_cls(variant_id: str, dataset: str = "TOMPEI-CMMD"):
    world_size = WORLD_SIZE
    mp.spawn(main, args=(world_size, variant_id, dataset), nprocs=world_size, join=True)
