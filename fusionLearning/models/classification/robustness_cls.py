"""
Ensemble size (K) x noise-intensity robustness sweep for TOMPEI-CMMD
classifier ensembles: does ensembling more base models buy back AUC lost to
input corruption, and how does that tradeoff move as corruption gets worse?

Two continuously-parametrized transforms (NOISE_TRANSFORMS below) - Gaussian
noise (intensity = additive std) and Gaussian blur (intensity = sigma) - each
swept over a small intensity grid including 0.0 (clean). For every
(transform, intensity, K) cell, a prediction matrix P is assembled by running
every qualifying model (test_auc >= --auc-threshold, same filter as
bias_variance_cls.py) once on the corrupted test set, then K-sized subsets are
drawn and fused via bias_variance_cls.decompose() - the same Bregman
primal/dual bias-variance machinery already used elsewhere in this repo, so
"the evaluation metric" isn't a single arbitrary pick: every row carries
bias_primal/var_primal/bias_dual/var_dual/auc_primal/auc_dual, and --metric
just picks which one gets plotted.

At intensity=0.0 no inference runs - each model's existing
results/{dataset}/{model}/metrics/test_predictions.json is reused directly
via bias_variance_cls.load_prediction_matrix(), so the clean baseline is
guaranteed identical to already-published numbers.

Loop order per (transform, intensity) is tier-outer, model-inner: each
resolution tier's 828 test DICOMs are decoded+resized once, corrupted once,
and the batch is shared across every qualifying model at that tier - avoids
reloading/re-corrupting per model.

Corruptions are built from `albumentations` (uv add albumentations) at fixed
(p=1.0, degenerate (v,v) range) magnitude per intensity - deterministic given
a seed, not the library's usual randomized-range training augmentation.
Requires numpy/torch/albumentations only, no extra system dependencies.

Usage:
    python robustness_cls.py [--dataset TOMPEI-CMMD] [--auc-threshold 0.75]
                              [--k-values 1,2,4,8,16,32,64] [--draws-per-k 30] [--seed 42]
                              [--batch-size 32] [--metric auc_dual] [--devices 0,1,2,3]
"""

from __future__ import annotations

import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

import argparse
import csv
import json
import random

import albumentations as A
import matplotlib
import numpy as np
import torch
from torchvision.transforms import InterpolationMode, v2

from fusionLearning.config import TOMPEI_CMMD_TEST, TOMPEI_CMMD_TEST_LABEL
from fusionLearning.data.tompei_dataloader import LABEL_MAP, get_dcm_paths, load_dicom_image
from fusionLearning.models.classification.bias_variance_cls import decompose, load_prediction_matrix, load_qualifying_models
from fusionLearning.models.consts import RESOLUTION_TIERS_CLS
from fusionLearning.models.classification.distributed_cls import BASE_MODELS, create_classifier
from fusionLearning.models.classification.roster_cls import get_config

# Each transform is a (intensity -> albumentations transform) builder plus an
# intensity grid. 0.0 always means "clean, skip corruption entirely" (handled
# specially - never passed to `build`). Nonzero values were calibrated by
# measuring mean-abs-pixel-diff-from-clean on a real TOMPEI-CMMD test image
# before picking them (both are monotonically increasing in that diff).
NOISE_TRANSFORMS: dict[str, dict] = {
    "gaussian_noise": {
        "build": lambda s: A.GaussNoise(std_range=(s, s), mean_range=(0.0, 0.0), per_channel=True, p=1.0),
        "intensities": [0.0, 0.01, 0.03, 0.06, 0.10, 0.16],
        "label": "Gaussian noise (additive std, [0,1] scale)",
    },
    "gaussian_blur": {
        "build": lambda s: A.GaussianBlur(blur_limit=0, sigma_limit=(s, s), p=1.0),
        "intensities": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
        "label": "Gaussian blur (sigma, px)",
    },
}

METRIC_CHOICES = ["auc_dual", "auc_primal", "bias_dual", "var_dual", "bias_primal", "var_primal"]

RESOLUTION_TIERS_LIST = sorted(set(RESOLUTION_TIERS_CLS.values()))


def smoke_test_transforms() -> None:
    dummy = np.full((384, 384, 3), 128, dtype=np.uint8)
    for name, spec in NOISE_TRANSFORMS.items():
        for s in spec["intensities"]:
            if s == 0.0:
                continue
            try:
                out = spec["build"](s)(image=dummy)["image"]
                assert out.shape == dummy.shape and out.dtype == np.uint8
            except Exception as e:
                raise RuntimeError(f"Transform smoke test failed for '{name}' intensity={s}: {e}") from e
    print("Smoke test passed for all transform/intensity combinations.")


def load_qualifying_configs(dataset: str, auc_threshold: float) -> list[dict]:
    """
    Composes with (not duplicates) load_qualifying_models: the AUC filter is
    delegated entirely to it, summary.csv is only re-read here to recover
    variant_id/family/resolution_px, which load_qualifying_models discards.
    """
    qualifying_names = set(load_qualifying_models(dataset, auc_threshold))
    summary_path = os.path.join(BASE_MODELS, "results", dataset, "summary.csv")
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    configs = []
    for row in rows:
        model_name = f"{row['variant_id']}_{row['timm_name']}"
        if model_name in qualifying_names:
            configs.append({
                "model_name": model_name,
                "variant_id": row["variant_id"],
                "family": row["family"],
                "timm_name": row["timm_name"],
                "resolution_px": int(row["resolution_px"]),
            })
    return configs


def load_and_resize_tier(resolution: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Decodes+resizes every test DICOM once at `resolution`, replicating
    TOMPEICMMDDataset.__getitem__'s eval-time path (train=False, no
    classificationTransforms) exactly. Returns (images_u8 [N,H,W,3] uint8 -
    albumentations' expected format, y [N] float64, filenames sorted, matching
    bias_variance_cls.load_prediction_matrix's column ordering convention).
    """
    paths = get_dcm_paths(TOMPEI_CMMD_TEST)
    images, labels, filenames = [], [], []
    for path in paths:
        arr, filename = load_dicom_image(path)
        t = torch.from_numpy(arr).unsqueeze(0)
        t = v2.functional.resize(t, [resolution, resolution], interpolation=InterpolationMode.BICUBIC)
        t = t.clamp(0.0, 1.0).repeat(3, 1, 1)
        images.append((t * 255.0).round().clamp(0, 255).byte().permute(1, 2, 0).numpy())

        stem = os.path.splitext(filename)[0]
        with open(os.path.join(TOMPEI_CMMD_TEST_LABEL, stem + ".txt")) as f:
            label_str = f.read().strip().lower()
        labels.append(LABEL_MAP[label_str])
        filenames.append(filename)

    return np.stack(images), np.array(labels, dtype=np.float64), filenames


def apply_transform_batch(images_u8: np.ndarray, transform_name: str, intensity: float) -> np.ndarray:
    transform = NOISE_TRANSFORMS[transform_name]["build"](intensity)
    out = np.empty_like(images_u8)
    for i in range(images_u8.shape[0]):
        out[i] = transform(image=images_u8[i])["image"]
    return out


def batch_to_chw_float(images_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images_u8).float().div(255.0).permute(0, 3, 1, 2).contiguous()


def load_model(cfg: dict, dataset: str, device: torch.device) -> torch.nn.Module:
    roster_cfg = get_config(cfg["variant_id"])
    model = create_classifier(roster_cfg["timm_name"], roster_cfg["family"], roster_cfg["resolution_px"], num_classes=1)

    weights_path = os.path.join(BASE_MODELS, "results", dataset, cfg["model_name"], "weights", "best_model.pth")
    sd = torch.load(weights_path, map_location=device)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


def run_inference_batched(model: torch.nn.Module, images_chw: torch.Tensor, device: torch.device,
                           batch_size: int = 32) -> np.ndarray:
    probs = []
    with torch.no_grad():
        for i in range(0, images_chw.shape[0], batch_size):
            batch = images_chw[i:i + batch_size].to(device, non_blocking=True)
            logits = model(batch)
            probs.append(torch.sigmoid(logits).reshape(-1).cpu().numpy())
    return np.concatenate(probs)


def _infer_one_config_worker(args: tuple) -> np.ndarray:
    """Module-level (picklable) worker for the multi-GPU pool path."""
    cfg, images_chw, device_id, dataset, batch_size = args
    device = torch.device(f"cuda:{device_id}")
    model = load_model(cfg, dataset, device)
    probs = run_inference_batched(model, images_chw, device, batch_size)
    del model
    torch.cuda.empty_cache()
    return probs


def build_prediction_matrix(dataset: str, configs: list[dict], model_names: list[str],
                             transform_name: str, intensity: float,
                             batch_size: int, devices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y [N], P [pool_size, N]), P's row order matching `model_names`.
    intensity==0.0 reuses existing test_predictions.json (no inference, exact
    reuse of bias_variance_cls.load_prediction_matrix). Otherwise runs every
    qualifying model once per resolution tier on a corrupted batch shared
    across that tier's models.
    """
    if intensity == 0.0:
        y, P, _ = load_prediction_matrix(dataset, model_names)
        return y, P

    pool_size = len(configs)
    idx_by_name = {c["model_name"]: i for i, c in enumerate(configs)}
    P = None
    y_ref = None

    use_mp = len(devices) > 1
    pool = None
    if use_mp:
        import torch.multiprocessing as tmp
        tmp.set_start_method("spawn", force=True)  # CUDA contexts don't survive fork()
        pool = tmp.Pool(processes=len(devices))

    try:
        for tier_res in RESOLUTION_TIERS_LIST:
            configs_at_tier = [c for c in configs if c["resolution_px"] == tier_res]
            if not configs_at_tier:
                continue

            images_u8_clean, y, filenames = load_and_resize_tier(tier_res)
            if P is None:
                P = np.zeros((pool_size, len(filenames)), dtype=np.float64)
                y_ref = y

            corrupted_u8 = apply_transform_batch(images_u8_clean, transform_name, intensity)
            images_chw = batch_to_chw_float(corrupted_u8)
            del corrupted_u8, images_u8_clean

            if use_mp:
                images_chw.share_memory_()
                args_list = [
                    (cfg, images_chw, devices[i % len(devices)], dataset, batch_size)
                    for i, cfg in enumerate(configs_at_tier)
                ]
                results = pool.map(_infer_one_config_worker, args_list)
                for cfg, probs in zip(configs_at_tier, results):
                    P[idx_by_name[cfg["model_name"]]] = probs
            else:
                device = torch.device(f"cuda:{devices[0]}") if devices else torch.device("cpu")
                for cfg in configs_at_tier:
                    model = load_model(cfg, dataset, device)
                    probs = run_inference_batched(model, images_chw, device, batch_size)
                    P[idx_by_name[cfg["model_name"]]] = probs
                    del model
                    torch.cuda.empty_cache()

            del images_chw
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return y_ref, P


def run_k_sweep_for_matrix(y: np.ndarray, P: np.ndarray, k_values: list[int],
                            draws_per_k: int, seed: int) -> list[dict]:
    pool_size = P.shape[0]
    valid_k = [k for k in k_values if k <= pool_size]
    rng = np.random.default_rng(seed)
    rows = []
    for k in valid_k:
        n_draws = 1 if k == pool_size else draws_per_k
        for draw_idx in range(n_draws):
            idx = rng.choice(pool_size, size=k, replace=False)
            result = decompose(y, P[idx])
            result["draw_idx"] = draw_idx
            rows.append(result)
    return rows


def run_full_sweep(dataset: str, configs: list[dict], k_values: list[int], draws_per_k: int, seed: int,
                    batch_size: int, devices: list[int]) -> list[dict]:
    np.random.seed(seed)  # albumentations draws off the global numpy/python RNG (no per-instance seed
    random.seed(seed)     # kwarg in this version) - reproducibility is a property of call ordering

    model_names = [c["model_name"] for c in configs]
    rows = []
    for transform_name, spec in NOISE_TRANSFORMS.items():
        for intensity in spec["intensities"]:
            print(f"[{transform_name} intensity={intensity}] building prediction matrix "
                  f"over {len(configs)} qualifying model(s)...")
            y, P = build_prediction_matrix(dataset, configs, model_names, transform_name, intensity,
                                            batch_size, devices)
            k_rows = run_k_sweep_for_matrix(y, P, k_values, draws_per_k, seed)
            for r in k_rows:
                r["transform"] = transform_name
                r["intensity"] = intensity
            rows.extend(k_rows)
    return rows


def write_outputs(pool_size: int, rows: list[dict], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    fields = ["transform", "intensity", "k", "draw_idx", "L_indiv",
              "bias_primal", "var_primal", "gap_primal", "bias_dual", "var_dual",
              "auc_primal", "auc_dual"]
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_fields = fields[4:]
    by_combo: dict[str, dict[str, dict[str, list[dict]]]] = {}
    for row in rows:
        by_combo.setdefault(row["transform"], {}).setdefault(str(row["intensity"]), {}) \
                 .setdefault(str(row["k"]), []).append(row)

    by_transform_summary = {}
    for transform, by_intensity in by_combo.items():
        by_transform_summary[transform] = {}
        for intensity, by_k in by_intensity.items():
            by_transform_summary[transform][intensity] = {}
            for k, k_rows in by_k.items():
                by_transform_summary[transform][intensity][k] = {
                    "n_draws": len(k_rows),
                    **{f"{m}_mean": float(np.mean([r[m] for r in k_rows])) for m in metric_fields},
                    **{f"{m}_std": float(np.std([r[m] for r in k_rows])) for m in metric_fields},
                }

    summary = {"pool_size": pool_size, "by_transform": by_transform_summary}
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote summary to {json_path}")
    return summary


def plot_heatmaps(summary: dict, metric: str, out_path: str) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    transforms = sorted(summary["by_transform"].keys())
    fig, axes = plt.subplots(1, len(transforms), figsize=(6.5 * len(transforms), 5.5), squeeze=False)
    axes = axes[0]

    for ax, transform in zip(axes, transforms):
        by_intensity = summary["by_transform"][transform]
        intensities = sorted(by_intensity.keys(), key=float)
        ks = sorted({int(k) for by_k in by_intensity.values() for k in by_k.keys()})

        grid = np.full((len(ks), len(intensities)), np.nan)
        for j, intensity in enumerate(intensities):
            for i, k in enumerate(ks):
                cell = by_intensity[intensity].get(str(k))
                if cell:
                    grid[i, j] = cell[f"{metric}_mean"]

        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(intensities)))
        ax.set_xticklabels(intensities, rotation=45)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks)
        ax.set_xlabel(NOISE_TRANSFORMS[transform]["label"])
        ax.set_ylabel("Ensemble size K")
        ax.set_title(f"{transform}\n{metric}")
        fig.colorbar(im, ax=ax)

        vmid = np.nanmean(grid)
        for i in range(len(ks)):
            for j in range(len(intensities)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7,
                            color="white" if grid[i, j] < vmid else "black")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble size (K) x noise-intensity robustness sweep over TOMPEI-CMMD classifier ensembles."
    )
    parser.add_argument("--dataset", type=str, default="TOMPEI-CMMD")
    parser.add_argument("--auc-threshold", type=float, default=0.75)
    parser.add_argument("--k-values", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--draws-per-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--metric", type=str, default="auc_dual", choices=METRIC_CHOICES,
                         help="Which decompose() field to plot in the heatmap - all of them are "
                              "in sweep_results.csv regardless of this choice.")
    parser.add_argument("--devices", type=str, default=None,
                         help="Comma-separated CUDA device indices to shard model inference across "
                              "(e.g. '0,1,2,3'). Defaults to all visible GPUs; a single device runs "
                              "sequentially in-process, more than one uses a persistent worker pool.")
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    devices = [int(d) for d in args.devices.split(",")] if args.devices else (list(range(torch.cuda.device_count())) or [0])

    smoke_test_transforms()

    configs = load_qualifying_configs(args.dataset, args.auc_threshold)
    print(f"Qualifying pool: {len(configs)} models (test_auc >= {args.auc_threshold})")
    if not configs:
        raise ValueError(f"No models qualify at auc_threshold={args.auc_threshold}.")

    rows = run_full_sweep(args.dataset, configs, k_values, args.draws_per_k, args.seed, args.batch_size, devices)

    out_dir = os.path.join(BASE_MODELS, "results", args.dataset, "robustness")
    summary = write_outputs(len(configs), rows, out_dir)
    plot_heatmaps(summary, args.metric, os.path.join(out_dir, "figures", f"robustness_{args.metric}_heatmap.png"))
