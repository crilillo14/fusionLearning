"""
Noise-robustness sweep for TOMPEI-CMMD classifiers and their ensembles: how
much do predictions move under acquisition-style noise, and does ensembling
(as in bias_variance_cls.py: random K-subsets of the test_auc >= 0.75 pool,
fused in primal/prob or dual/logit space) buy back robustness?

Robustness is measured two ways for every (noise, severity, K-subset):
  - performance:  decompose() fields (auc / bias / var, primal & dual) - same
                  machinery as the bias-variance experiments.
  - invariance:   how far the ensemble's prediction on the noisy image moves
                  from ITS OWN clean prediction - flip rate at the 0.5 decision
                  threshold (the same threshold test_cls.py uses for pred_label)
                  and mean |delta p|. Each subset's flip rate is reported next to
                  the mean and best (by clean AUC) flip rate of its own members,
                  on identical noisy images, so ensemble-vs-individual is a
                  paired comparison.

Noise models (NOISE_MODELS) are applied at NATIVE resolution in [0,1]
intensity space - after load_dicom_image's VOI/min-max, before the bicubic
resize - so every model at every resolution tier sees the same corrupted
acquisition. All three share one severity axis sigma:
  - gaussian: x + sigma*z                                  (additive white noise)
  - rician:   |x + sigma*(z1 + i*z2)|                       (MRI magnitude-image noise)
  - poisson:  Poisson(x*N)/N, N = 0.5/sigma^2               (photon/shot noise - the
              physically closest model for X-ray mammography; std = sigma at x=0.5)
Results are clipped to [0,1]. Noise is seeded per (seed, noise, severity, image)
so it is identical across models and independent of worker scheduling. Because
the downstream resize averages over ~3-6 native pixels per axis, native sigma is
attenuated several-fold by the time a model sees it - the grid is calibrated in
native units on purpose (see the preview figure).

Stages (--stage, default all; each is resumable / skippable):
  preview  - sample-image grid at every severity (CPU only)
  corrupt  - builds results/{dataset}/robustness/_cache/{tier}/{condition}.npy
             (float32 [N,H,W], memmapped) via a CPU process pool
  infer    - every pool model x every condition, sharded over --devices,
             logits saved to predictions/{model}.npz; clean predictions are
             checked against the model's existing test_predictions.json
  analyze  - K-sweep + invariance metrics, CSV/JSON + figures (no GPU)

Usage:
    uv run python robustness_cls.py --stage preview
    uv run python robustness_cls.py --stage corrupt
    uv run python robustness_cls.py --stage infer --devices 0,1,2,3
    uv run python robustness_cls.py --stage analyze
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
import subprocess
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import matplotlib
import numpy as np
import torch
from torchvision.transforms import InterpolationMode, v2

from fusionLearning.config import TOMPEI_CMMD_TEST, TOMPEI_CMMD_TEST_LABEL
from fusionLearning.data.tompei_dataloader import LABEL_MAP, get_dcm_paths, load_dicom_image
from fusionLearning.models.classification.bias_variance_cls import EPS, decompose, load_qualifying_models
from fusionLearning.models.classification.distributed_cls import BASE_MODELS, create_classifier
from fusionLearning.models.classification.roster_cls import get_config
from fusionLearning.models.consts import RESOLUTION_TIERS_CLS

matplotlib.use("Agg")

# ── noise models ─────────────────────────────────────────────────────────────

NoiseFn = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


def gaussian_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return x + sigma * rng.standard_normal(x.shape, dtype=np.float32)


def rician_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    re = x + sigma * rng.standard_normal(x.shape, dtype=np.float32)
    im = sigma * rng.standard_normal(x.shape, dtype=np.float32)
    return np.sqrt(re * re + im * im)


def poisson_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    n_photons = 0.5 / sigma ** 2  # matches gaussian std at mid-gray
    return (rng.poisson(np.clip(x, 0.0, None) * n_photons) / n_photons).astype(np.float32)


NOISE_MODELS: dict[str, dict] = {
    "gaussian": {"fn": gaussian_noise, "label": "Gaussian (white)"},
    "rician":   {"fn": rician_noise,   "label": "Rician (MRI magnitude)"},
    "poisson":  {"fn": poisson_noise,  "label": "Poisson (shot / dose)"},
}

# Native-resolution sigma, [0,1] intensity units. Attenuated ~3-6x by the resize.
SEVERITIES: list[float] = [0.02, 0.05, 0.10, 0.20, 0.35]

CLEAN = "clean"
DECISION_THRESHOLD = 0.5
TIERS: list[int] = sorted(set(RESOLUTION_TIERS_CLS.values()))


def condition_keys(noises: list[str], severities: list[float]) -> list[str]:
    return [CLEAN] + [f"{n}@{s:g}" for n in noises for s in severities]


def parse_condition(key: str) -> tuple[str, float]:
    if key == CLEAN:
        return CLEAN, 0.0
    noise, sev = key.split("@")
    return noise, float(sev)


def corrupt(x: np.ndarray, key: str, seed: int, image_idx: int) -> np.ndarray:
    """Deterministic corruption of one native [0,1] image for condition `key`."""
    if key == CLEAN:
        return x
    noise, sigma = parse_condition(key)
    rng = np.random.default_rng([seed, zlib.crc32(key.encode()), image_idx])
    return np.clip(NOISE_MODELS[noise]["fn"](x, sigma, rng), 0.0, 1.0).astype(np.float32)


def resize_like_dataset(x: np.ndarray, resolution: int) -> np.ndarray:
    """Exactly TOMPEICMMDDataset.__getitem__'s eval path (minus channel repeat)."""
    t = torch.from_numpy(x).unsqueeze(0)
    t = v2.functional.resize(t, [resolution, resolution], interpolation=InterpolationMode.BICUBIC)
    return t.clamp(0.0, 1.0)[0].numpy()


# ── paths / bookkeeping ──────────────────────────────────────────────────────

def out_dir_for(dataset: str) -> str:
    return os.path.join(BASE_MODELS, "results", dataset, "robustness")


def cache_path(out_dir: str, tier: int, key: str) -> str:
    return os.path.join(out_dir, "_cache", str(tier), f"{key}.npy")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_MODELS, text=True).strip()
    except Exception:
        return "unknown"


def append_run_log(out_dir: str, entry: dict) -> None:
    """One JSON line per stage invocation - the reproducibility trail."""
    os.makedirs(out_dir, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": git_commit(), **entry}
    with open(os.path.join(out_dir, "run_log.jsonl"), "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_test_index() -> tuple[list[str], list[str], np.ndarray]:
    """(paths, filenames, y) in sorted-filename order - bias_variance_cls's column convention."""
    paths = get_dcm_paths(TOMPEI_CMMD_TEST)
    filenames = [os.path.basename(p) for p in paths]
    y = []
    for fn in filenames:
        with open(os.path.join(TOMPEI_CMMD_TEST_LABEL, os.path.splitext(fn)[0] + ".txt")) as f:
            y.append(LABEL_MAP[f.read().strip().lower()])
    return paths, filenames, np.array(y, dtype=np.float64)


def load_pool(dataset: str, auc_threshold: float) -> list[dict]:
    """Roster configs for the qualifying pool (same filter as bias_variance_cls.py)."""
    pool = []
    for name in load_qualifying_models(dataset, auc_threshold):
        variant_id = "_".join(name.split("_")[:2])
        pool.append({"model_name": name, **get_config(variant_id)})
    return pool


# ── stage: preview ───────────────────────────────────────────────────────────

def stage_preview(out_dir: str, noises: list[str], severities: list[float], seed: int,
                  image_idx: int, crop: int = 384, preview_tier: int = 512) -> str:
    """
    Per noise model, two rows across [clean, *severities]: a native-resolution
    crop (what the noise looks like at acquisition) and the full image at
    `preview_tier` px (what a mid-res model actually sees).
    """
    import matplotlib.pyplot as plt

    paths, filenames, y = load_test_index()
    x, _ = load_dicom_image(paths[image_idx])
    # crop centred on the tissue's centre of mass so it isn't background
    rr, cc = np.nonzero(x > 0.1)
    r0 = int(np.clip(rr.mean() - crop // 2, 0, x.shape[0] - crop))
    c0 = int(np.clip(cc.mean() - crop // 2, 0, x.shape[1] - crop))

    keys_per_noise = [[CLEAN] + [f"{n}@{s:g}" for s in severities] for n in noises]
    n_cols = len(severities) + 1
    fig, axes = plt.subplots(2 * len(noises), n_cols, figsize=(2.2 * n_cols, 2.4 * 2 * len(noises)))
    for i, (noise, keys) in enumerate(zip(noises, keys_per_noise)):
        for j, key in enumerate(keys):
            xc = corrupt(x, key, seed, image_idx)
            ax_crop, ax_full = axes[2 * i, j], axes[2 * i + 1, j]
            ax_crop.imshow(xc[r0:r0 + crop, c0:c0 + crop], cmap="gray", vmin=0, vmax=1)
            ax_full.imshow(resize_like_dataset(xc, preview_tier), cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax_crop.set_title(CLEAN if key == CLEAN else f"σ={parse_condition(key)[1]:g}", fontsize=9)
            for ax in (ax_crop, ax_full):
                ax.set_xticks([]); ax.set_yticks([])
        axes[2 * i, 0].set_ylabel(f"{NOISE_MODELS[noise]['label']}\nnative crop", fontsize=8)
        axes[2 * i + 1, 0].set_ylabel(f"{preview_tier}px input", fontsize=8)
    fig.suptitle(f"Noise severities - {filenames[image_idx]} (label={int(y[image_idx])}, seed={seed})", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "figures", "noise_preview.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ── stage: corrupt ───────────────────────────────────────────────────────────

def _corrupt_image_worker(args: tuple) -> int:
    """Pool worker: decode one DICOM, write every (condition, tier) row into the memmaps."""
    idx, path, keys, seed, out_dir = args
    torch.set_num_threads(1)
    x, _ = load_dicom_image(path)
    for key in keys:
        xc = corrupt(x, key, seed, idx)
        for tier in TIERS:
            mm = np.load(cache_path(out_dir, tier, key), mmap_mode="r+")
            mm[idx] = resize_like_dataset(xc, tier)
            mm.flush()
            del mm
    return idx


def stage_corrupt(out_dir: str, keys: list[str], seed: int, workers: int) -> None:
    paths, filenames, y = load_test_index()
    meta_path = os.path.join(out_dir, "_cache", "meta.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if meta["seed"] == seed and set(keys) <= set(meta["conditions"]) and meta["filenames"] == filenames:
            print(f"Cache already complete for {len(keys)} conditions - skipping.")
            return
    for tier in TIERS:
        os.makedirs(os.path.join(out_dir, "_cache", str(tier)), exist_ok=True)
        for key in keys:
            np.lib.format.open_memmap(cache_path(out_dir, tier, key), mode="w+",
                                      dtype=np.float32, shape=(len(paths), tier, tier))

    t0 = time.time()
    jobs = [(i, p, keys, seed, out_dir) for i, p in enumerate(paths)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for n_done, _ in enumerate(ex.map(_corrupt_image_worker, jobs, chunksize=4), 1):
            if n_done % 100 == 0:
                print(f"  corrupted {n_done}/{len(paths)} images ({time.time() - t0:.0f}s)")

    with open(meta_path, "w") as f:
        json.dump({"seed": seed, "conditions": keys, "tiers": TIERS, "filenames": filenames,
                   "y": y.tolist()}, f)
    print(f"Cache built in {time.time() - t0:.0f}s")


# ── stage: infer ─────────────────────────────────────────────────────────────

def load_model(cfg: dict, dataset: str, device: torch.device) -> torch.nn.Module:
    model = create_classifier(cfg["timm_name"], cfg["family"], cfg["resolution_px"], num_classes=1)
    weights_path = os.path.join(BASE_MODELS, "results", dataset, cfg["model_name"], "weights", "best_model.pth")
    sd = torch.load(weights_path, map_location=device)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.to(device).eval()


@torch.no_grad()
def predict_logits(model: torch.nn.Module, images: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    out = []
    for i in range(0, images.shape[0], batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(images[i:i + batch_size])).to(device)
        out.append(model(batch.unsqueeze(1).repeat(1, 3, 1, 1)).reshape(-1).float().cpu().numpy())
    return np.concatenate(out)


def clean_reference_check(cfg: dict, dataset: str, filenames: list[str], logits_clean: np.ndarray) -> float:
    """Max |p - p_published| on the clean condition vs the model's test_predictions.json."""
    with open(os.path.join(BASE_MODELS, "results", dataset, cfg["model_name"], "metrics", "test_predictions.json")) as f:
        ref = {r["filename"]: r["pred_prob"] for r in json.load(f)["predictions"]}
    p_ref = np.array([ref[fn] for fn in filenames])
    return float(np.abs(1.0 / (1.0 + np.exp(-logits_clean)) - p_ref).max())


def _infer_worker(rank: int, device_ids: list[int], cfgs: list[dict], keys: list[str],
                  dataset: str, out_dir: str, filenames: list[str], pred_subdir: str = "predictions",
                  n_images: int | None = None) -> None:
    device = torch.device(f"cuda:{device_ids[rank]}")
    torch.cuda.set_device(device)
    pred_dir = os.path.join(out_dir, pred_subdir)
    filenames = filenames[:n_images]
    for cfg in cfgs[rank::len(device_ids)]:
        dst = os.path.join(pred_dir, f"{cfg['model_name']}.npz")
        if os.path.exists(dst) and set(keys) <= set(np.load(dst).files):
            continue
        t0 = time.time()
        model = load_model(cfg, dataset, device)
        bs = cfg["batch_size"] * 2  # no activations kept for backward
        logits = {}
        for key in keys:
            images = np.load(cache_path(out_dir, cfg["resolution_px"], key), mmap_mode="r")[:n_images]
            logits[key] = predict_logits(model, images, device, bs)
        err = clean_reference_check(cfg, dataset, filenames, logits[CLEAN])
        np.savez(dst + ".tmp.npz", **logits)
        os.replace(dst + ".tmp.npz", dst)
        with open(os.path.join(pred_dir, "clean_check.jsonl"), "a") as f:
            f.write(json.dumps({"model": cfg["model_name"], "max_abs_dp_vs_published": err}) + "\n")
        print(f"[gpu{device_ids[rank]}] {cfg['model_name']}: {len(keys)} conditions in "
              f"{time.time() - t0:.0f}s (clean max|Δp| vs published = {err:.2e})", flush=True)
        del model
        torch.cuda.empty_cache()


def stage_infer(out_dir: str, dataset: str, pool: list[dict], keys: list[str], device_ids: list[int],
                smoke: bool = False) -> None:
    """
    smoke=True exercises the exact same path (model build, weight load, spawn,
    memmap reads, npz write, clean check) on the 2 smallest pool models x 32
    images x (clean + highest severity per noise), written to predictions_smoke/
    so it never contaminates the real run's resumable predictions/.
    """
    import torch.multiprocessing as tmp
    filenames = json.load(open(os.path.join(out_dir, "_cache", "meta.json")))["filenames"]
    pred_subdir, n_images = "predictions", None
    if smoke:
        pred_subdir, n_images = "predictions_smoke", 32
        pool = sorted(pool, key=lambda c: c["params_m"] * c["resolution_px"] ** 2)[:2]
        top = {}
        for key in keys[1:]:
            noise, sev = parse_condition(key)
            top[noise] = key if sev >= parse_condition(top.get(noise, key))[1] else top[noise]
        keys = [CLEAN, *top.values()]
        device_ids = device_ids[:2]
        print(f"SMOKE: models={[c['model_name'] for c in pool]} conditions={keys} images={n_images} gpus={device_ids}")
    os.makedirs(os.path.join(out_dir, pred_subdir), exist_ok=True)
    # largest models first so the tail of the run isn't one GPU grinding a 768px swin
    pool = sorted(pool, key=lambda c: c["params_m"] * c["resolution_px"] ** 2, reverse=True)
    tmp.spawn(_infer_worker, args=(device_ids, pool, keys, dataset, out_dir, filenames, pred_subdir, n_images),
              nprocs=len(device_ids), join=True)
    if smoke:
        for c in pool:
            d = np.load(os.path.join(out_dir, pred_subdir, f"{c['model_name']}.npz"))
            assert set(d.files) == set(keys) and all(d[k].shape == (n_images,) and np.isfinite(d[k]).all() for k in keys)
        print("SMOKE OK - see", os.path.join(out_dir, pred_subdir, "clean_check.jsonl"))


# ── stage: analyze ───────────────────────────────────────────────────────────

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def load_logit_tensor(out_dir: str, model_names: list[str], keys: list[str]) -> np.ndarray:
    """[n_conditions, n_models, N] logits, axes aligned to (keys, model_names)."""
    per_model = [np.load(os.path.join(out_dir, "predictions", f"{m}.npz")) for m in model_names]
    return np.stack([np.stack([d[k] for d in per_model]) for k in keys])


def fuse(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[k, N] member logits -> (p_primal, p_dual) [N]."""
    return sigmoid(logits).mean(axis=0), sigmoid(logits.mean(axis=0))


def invariance(p_clean: np.ndarray, p_noisy: np.ndarray) -> tuple[float, float]:
    """(flip rate at DECISION_THRESHOLD, mean |Δp|) of p_noisy relative to p_clean."""
    flips = (p_clean >= DECISION_THRESHOLD) != (p_noisy >= DECISION_THRESHOLD)
    return float(flips.mean()), float(np.abs(p_noisy - p_clean).mean())


def draw_subsets(pool_size: int, k_values: list[int], draws_per_k: int, seed: int) -> list[tuple[int, int, np.ndarray]]:
    """Fixed (k, draw_idx, member_idx) list - reused across every condition so curves are paired."""
    rng = np.random.default_rng(seed)
    subsets = []
    for k in sorted(set(k for k in k_values if k <= pool_size) | {pool_size}):
        n = pool_size if k == 1 else (1 if k == pool_size else draws_per_k)  # k=1 enumerates every model
        for d in range(n):
            idx = np.array([d]) if k == 1 else rng.choice(pool_size, size=k, replace=False)
            subsets.append((k, d, idx))
    return subsets


def analyze(y: np.ndarray, L: np.ndarray, keys: list[str], subsets: list, clean_auc: np.ndarray) -> list[dict]:
    """
    L: [n_conditions, n_models, N] logits, L[0] = clean. One row per
    (condition, subset): decompose() fields + ensemble invariance (primal and
    dual fusion) + the subset's own members' mean and best-member invariance.
    """
    P = sigmoid(L)
    p_member_flip = np.empty(L.shape[:2]); p_member_dp = np.empty(L.shape[:2])
    for c in range(len(keys)):
        for m in range(L.shape[1]):
            p_member_flip[c, m], p_member_dp[c, m] = invariance(P[0, m], P[c, m])

    rows = []
    for c, key in enumerate(keys):
        noise, sev = parse_condition(key)
        for k, d, idx in subsets:
            row = {"condition": key, "noise": noise, "severity": sev, "draw_idx": d,
                   **decompose(y, np.clip(P[c, idx], EPS, 1 - EPS))}
            pp0, pd0 = fuse(L[0, idx])
            ppc, pdc = fuse(L[c, idx])
            row["flip_primal"], row["dp_primal"] = invariance(pp0, ppc)
            row["flip_dual"], row["dp_dual"] = invariance(pd0, pdc)
            row["flip_members_mean"] = float(p_member_flip[c, idx].mean())
            row["dp_members_mean"] = float(p_member_dp[c, idx].mean())
            best = idx[np.argmax(clean_auc[idx])]
            row["flip_best_member"], row["dp_best_member"] = float(p_member_flip[c, best]), float(p_member_dp[c, best])
            row["members"] = ";".join(map(str, idx.tolist())) if k <= 8 else ""
            rows.append(row)
    return rows


METRICS = ["L_indiv", "bias_primal", "var_primal", "gap_primal", "bias_dual", "var_dual", "auc_primal", "auc_dual",
           "flip_primal", "dp_primal", "flip_dual", "dp_dual", "flip_members_mean", "dp_members_mean",
           "flip_best_member", "dp_best_member"]


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["condition"], r["k"]), []).append(r)
    summary: dict = {}
    for (cond, k), rs in groups.items():
        summary.setdefault(cond, {})[str(k)] = {
            "n_draws": len(rs),
            **{f"{m}_mean": float(np.mean([r[m] for r in rs])) for m in METRICS},
            **{f"{m}_std": float(np.std([r[m] for r in rs])) for m in METRICS},
        }
    return summary


# ── figures ──────────────────────────────────────────────────────────────────
# Palette: dataviz reference instance. K is ordinal -> one-hue blue ramp
# (steps 250..700); noise models are categorical -> fixed slots 1..3.
K_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
NOISE_COLORS = {"gaussian": "#2a78d6", "rician": "#eb6834", "poisson": "#1baf7a"}
INK, INK_MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"


def _style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def _series(summary: dict, noise: str, severities: list[float], k: int, metric: str) -> tuple[np.ndarray, np.ndarray]:
    keys = [CLEAN] + [f"{noise}@{s:g}" for s in severities]
    cells = [summary[key][str(k)] for key in keys]
    return np.array([c[f"{metric}_mean"] for c in cells]), np.array([c[f"{metric}_std"] for c in cells])


def plot_k_curves(summary: dict, noises: list[str], severities: list[float], ks: list[int], path: str,
                  fusion: str = "dual") -> None:
    """Rows: AUC, flip rate, mean |Δp|; cols: noise model; one line per K (k=1 = average individual)."""
    import matplotlib.pyplot as plt
    rows = [(f"auc_{fusion}", "AUROC"), (f"flip_{fusion}", "flip rate vs own clean"), (f"dp_{fusion}", "mean |Δp| vs own clean")]
    x = np.array([0.0] + severities)
    fig, axes = plt.subplots(len(rows), len(noises), figsize=(4.2 * len(noises), 3.1 * len(rows)), sharex=True)
    colors = K_RAMP[-len(ks):] if len(ks) <= len(K_RAMP) else K_RAMP
    for j, noise in enumerate(noises):
        for i, (metric, label) in enumerate(rows):
            ax = axes[i, j]; _style(ax)
            for k, col in zip(ks, colors):
                mu, sd = _series(summary, noise, severities, k, metric)
                ax.plot(x, mu, color=col, lw=2, marker="o", ms=4, label=f"K={k}")
                ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
            if i == 0:
                ax.set_title(NOISE_MODELS[noise]["label"], fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, color=INK)
            if i == len(rows) - 1:
                ax.set_xlabel("native σ", fontsize=9, color=INK)
    axes[0, -1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"Noise robustness vs ensemble size ({fusion} fusion; mean ± sd over random subsets)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_ensemble_vs_members(summary: dict, noises: list[str], severities: list[float], k: int, path: str) -> None:
    """Flip rate of K-ensembles vs the mean and best of their own members, per noise model."""
    import matplotlib.pyplot as plt
    x = np.array([0.0] + severities)
    fig, axes = plt.subplots(1, len(noises), figsize=(4.2 * len(noises), 3.4), sharey=True)
    for ax, noise in zip(np.atleast_1d(axes), noises):
        _style(ax)
        for metric, label, ls in [("flip_dual", f"K={k} ensemble (dual)", "-"),
                                  ("flip_members_mean", "its members, mean", "--"),
                                  ("flip_best_member", "its best member (clean AUC)", ":")]:
            mu, sd = _series(summary, noise, severities, k, metric)
            ax.plot(x, mu, color=NOISE_COLORS[noise], ls=ls, lw=2, marker="o", ms=4, label=label)
            ax.fill_between(x, mu - sd, mu + sd, color=NOISE_COLORS[noise], alpha=0.10, lw=0)
        ax.set_title(NOISE_MODELS[noise]["label"], fontsize=10, color=INK)
        ax.set_xlabel("native σ", fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
    np.atleast_1d(axes)[0].set_ylabel("flip rate vs own clean prediction", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def plot_heatmaps(summary: dict, noises: list[str], severities: list[float], ks: list[int], metric: str, path: str) -> None:
    """K x severity grid per noise model for one metric (sequential blue, annotated)."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    fig, axes = plt.subplots(1, len(noises), figsize=(4.6 * len(noises), 0.45 * len(ks) + 1.8))
    for ax, noise in zip(np.atleast_1d(axes), noises):
        grid = np.array([_series(summary, noise, severities, k, metric)[0] for k in ks])
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(severities) + 1), ["clean"] + [f"{s:g}" for s in severities], fontsize=8)
        ax.set_yticks(range(len(ks)), [str(k) for k in ks], fontsize=8)
        ax.set_xlabel("native σ", fontsize=9); ax.set_ylabel("K", fontsize=9)
        ax.set_title(f"{NOISE_MODELS[noise]['label']}\n{metric}", fontsize=9)
        mid = np.nanmean(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=6.5,
                        color="white" if grid[i, j] > mid else INK)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def stage_analyze(out_dir: str, dataset: str, pool: list[dict], keys: list[str], noises: list[str],
                  severities: list[float], k_values: list[int], draws_per_k: int, seed: int) -> None:
    meta = json.load(open(os.path.join(out_dir, "_cache", "meta.json")))
    y = np.array(meta["y"])
    names = [c["model_name"] for c in pool]
    L = load_logit_tensor(out_dir, names, keys)

    from sklearn.metrics import roc_auc_score
    clean_auc = np.array([roc_auc_score(y, L[0, m]) for m in range(len(names))])
    subsets = draw_subsets(len(names), k_values, draws_per_k, seed)
    rows = analyze(y, L, keys, subsets, clean_auc)
    summary = summarize(rows)

    fields = ["condition", "noise", "severity", "k", "draw_idx"] + METRICS + ["members"]
    with open(os.path.join(out_dir, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({"pool": names, "clean_auc": dict(zip(names, clean_auc.round(6).tolist())),
                   "decision_threshold": DECISION_THRESHOLD, "by_condition": summary}, f, indent=2)

    ks = sorted({k for k, _, _ in subsets})
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for fusion in ("dual", "primal"):
        plot_k_curves(summary, noises, severities, ks, os.path.join(fig_dir, f"k_curves_{fusion}.png"), fusion)
    k_mid = max(k for k in ks if k <= 8)
    plot_ensemble_vs_members(summary, noises, severities, k_mid, os.path.join(fig_dir, f"ensemble_vs_members_k{k_mid}.png"))
    for metric in ("auc_dual", "flip_dual", "var_dual"):
        plot_heatmaps(summary, noises, severities, ks, metric, os.path.join(fig_dir, f"heatmap_{metric}.png"))
    print(f"Wrote {len(rows)} rows, summary and figures to {out_dir}")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noise-robustness sweep over TOMPEI-CMMD classifiers and ensembles.")
    parser.add_argument("--stage", default="all", choices=["preview", "corrupt", "infer", "analyze", "all"])
    parser.add_argument("--dataset", default="TOMPEI-CMMD")
    parser.add_argument("--auc-threshold", type=float, default=0.75)
    parser.add_argument("--noises", default=",".join(NOISE_MODELS))
    parser.add_argument("--severities", default=",".join(f"{s:g}" for s in SEVERITIES))
    parser.add_argument("--k-values", default="1,2,4,8,16,32,64")
    parser.add_argument("--draws-per-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview-image", type=int, default=0, help="test-set index for the preview grid")
    parser.add_argument("--workers", type=int, default=64, help="CPU processes for the corrupt stage")
    parser.add_argument("--smoke", action="store_true", help="infer stage: tiny end-to-end check, writes predictions_smoke/")
    parser.add_argument("--devices", default=None, help="comma-separated CUDA ids (default: all visible)")
    args = parser.parse_args()

    noises = args.noises.split(",")
    severities = [float(s) for s in args.severities.split(",")]
    keys = condition_keys(noises, severities)
    out_dir = out_dir_for(args.dataset)
    stages = ["preview", "corrupt", "infer", "analyze"] if args.stage == "all" else [args.stage]
    pool = load_pool(args.dataset, args.auc_threshold)
    print(f"Pool: {len(pool)} models (test_auc >= {args.auc_threshold}); {len(keys)} conditions")

    for stage in stages:
        t0 = time.time()
        if stage == "preview":
            print("Preview grid ->", stage_preview(out_dir, noises, severities, args.seed, args.preview_image))
        elif stage == "corrupt":
            stage_corrupt(out_dir, keys, args.seed, args.workers)
        elif stage == "infer":
            devices = [int(d) for d in args.devices.split(",")] if args.devices else list(range(torch.cuda.device_count()))
            stage_infer(out_dir, args.dataset, pool, keys, devices, smoke=args.smoke)
        elif stage == "analyze":
            stage_analyze(out_dir, args.dataset, pool, keys, noises, severities,
                          [int(k) for k in args.k_values.split(",")], args.draws_per_k, args.seed)
        append_run_log(out_dir, {"stage": stage, "args": vars(args), "pool_size": len(pool),
                                 "conditions": keys, "elapsed_s": round(time.time() - t0, 1)})
