"""
Pairwise "orthogonality"/diversity diagnostic for TOMPEI-CMMD classifier
ensembles, built on top of bias_variance_cls.py's Bregman decomposition.

Tests a hypothesis under scrutiny (not assumed true): that ensembling's bias
reduction comes from member predictors' deviations being mutually orthogonal,
rather than (per the classical Krogh & Vedelsby 1994 ambiguity decomposition)
orthogonality/decorrelation being a VARIANCE-reduction lever with no direct
bearing on bias. bias_dual = D(y, p_dual) depends only on the ensemble's
central tendency (mean logit); it has no term for how members' errors relate
to each other. That relational structure lives entirely in var_dual. This
script produces the correlation evidence to check that directly instead of
arguing about it in the abstract.

Two deviation conventions are computed per drawn subset, since "orthogonal to
what" is itself ambiguous in the original hypothesis:
  - "truth":  d_i = p_i - y            (deviation from ground truth)
  - "center": d_i = logit(p_i) - eta_bar  (deviation from the ensemble's own
              dual/logit center - the same eta_bar decompose() uses internally)
Each d_i is a vector across the N-sample test set (cosine between two scalars
is degenerate, so "per-sample orthogonality" isn't a coherent per-sample
quantity - this is necessarily a population-level, per-model-pair statistic).

Reuses load_qualifying_models/load_prediction_matrix/decompose from
bias_variance_cls.py - same pool, same draw sequence (same seed => identical
subsets), no training or inference, reads only existing
metrics/test_predictions.json files.

Usage:
    python ensemble_diversity_cls.py [--dataset TOMPEI-CMMD] [--auc-threshold 0.75]
                                      [--k-values 2,4,8,16,32,64] [--draws-per-k 30] [--seed 42]
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

import matplotlib
import numpy as np
from scipy.stats import pearsonr

from fusionLearning.models.classification.distributed_cls import BASE_MODELS
from fusionLearning.models.classification.bias_variance_cls import (
    EPS, decompose, load_prediction_matrix, load_qualifying_models,
)

NORM_EPS = 1e-8  # degenerate-pair guard for cosine's denominator - a distinct
                  # scale of quantity from EPS (probability clipping), not reused


def _logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def pairwise_cosine_stats(D: np.ndarray, norm_eps: float = NORM_EPS) -> tuple[float, float, int]:
    """
    D: [k, N] deviation vectors, one per model. Returns (mean, std, n_degenerate)
    of cos(theta) over all C(k,2) pairs. A pair is degenerate (near-zero-norm
    deviation vector on one or both sides) and excluded rather than coerced to
    0/1, which would silently bias the mean in a specific direction.
    """
    k = D.shape[0]
    norms = np.linalg.norm(D, axis=1)
    cosines = []
    n_degenerate = 0
    for i in range(k):
        for j in range(i + 1, k):
            if norms[i] < norm_eps or norms[j] < norm_eps:
                n_degenerate += 1
                continue
            cosines.append(float(np.dot(D[i], D[j]) / (norms[i] * norms[j])))
    if not cosines:
        return float("nan"), float("nan"), n_degenerate
    return float(np.mean(cosines)), float(np.std(cosines)), n_degenerate


def compute_diversity_for_subset(y: np.ndarray, P_subset: np.ndarray) -> dict:
    eta = _logit(P_subset)
    eta_bar = eta.mean(axis=0)

    D_truth = P_subset - y[None, :]
    D_center = eta - eta_bar[None, :]

    mean_cos_truth, std_cos_truth, _ = pairwise_cosine_stats(D_truth)
    mean_cos_center, std_cos_center, _ = pairwise_cosine_stats(D_center)

    return {
        "mean_cos_truth": mean_cos_truth,
        "std_cos_truth": std_cos_truth,
        "mean_cos_center": mean_cos_center,
        "std_cos_center": std_cos_center,
    }


def run_sweep(dataset: str, auc_threshold: float, k_values: list[int], draws_per_k: int, seed: int):
    """
    Mirrors bias_variance_cls.run_sweep's draw loop exactly (same rng seeding,
    same rng.choice call sequence) so that, given identical CLI defaults, the
    (k, draw_idx) subsets here line up 1:1 with bias_variance_cls.py's own
    sweep_results.csv rows - this only holds if summary.csv hasn't changed
    between the two runs, the same caveat bias_variance_cls.py itself doesn't
    guard against. The loop is reimplemented (not called into) because
    bias_variance_cls.run_sweep doesn't expose P[idx] itself, which the
    diversity computation needs alongside decompose()'s output.
    """
    model_names = load_qualifying_models(dataset, auc_threshold)
    pool_size = len(model_names)
    if pool_size < max(k_values):
        raise ValueError(f"Pool has only {pool_size} qualifying models, can't draw K={max(k_values)}.")

    y, P, _ = load_prediction_matrix(dataset, model_names)

    rng = np.random.default_rng(seed)
    rows = []
    for k in k_values:
        n_draws = 1 if k == pool_size else draws_per_k
        for draw_idx in range(n_draws):
            idx = rng.choice(pool_size, size=k, replace=False)
            P_subset = P[idx]
            result = decompose(y, P_subset)
            result.update(compute_diversity_for_subset(y, P_subset))
            result["draw_idx"] = draw_idx
            rows.append(result)

    for row in rows:
        assert abs((row["bias_dual"] + row["var_dual"]) - row["L_indiv"]) < 1e-6, (
            f"Bregman identity violated at k={row['k']}, draw={row['draw_idx']} - "
            f"bias_dual + var_dual should exactly equal L_indiv"
        )

    return pool_size, rows


def pearson_or_nan(xs: list[float], ys: list[float]) -> tuple[float, float, int]:
    pairs = [(x, y) for x, y in zip(xs, ys) if not (np.isnan(x) or np.isnan(y))]
    if len(pairs) < 2:
        return float("nan"), float("nan"), len(pairs)
    x_valid, y_valid = zip(*pairs)
    r, p = pearsonr(x_valid, y_valid)
    return float(r), float(p), len(pairs)


CORRELATION_PAIRS = [
    ("mean_cos_center", "var_dual"),
    ("mean_cos_center", "bias_dual"),
    ("mean_cos_truth", "var_primal"),
    ("mean_cos_truth", "bias_primal"),
]


def write_outputs(pool_size: int, rows: list[dict], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    fields = ["k", "draw_idx", "mean_cos_truth", "std_cos_truth", "mean_cos_center", "std_cos_center",
              "L_indiv", "bias_primal", "var_primal", "gap_primal", "bias_dual", "var_dual",
              "auc_primal", "auc_dual"]
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    by_k: dict[int, list[dict]] = {}
    for row in rows:
        by_k.setdefault(row["k"], []).append(row)

    summary = {"pool_size": pool_size, "by_k": {}}
    for k, k_rows in sorted(by_k.items()):
        summary["by_k"][str(k)] = {
            "n_draws": len(k_rows),
            **{
                f"{metric}_mean": float(np.nanmean([r[metric] for r in k_rows]))
                for metric in fields[2:]
            },
            **{
                f"{metric}_std": float(np.nanstd([r[metric] for r in k_rows]))
                for metric in fields[2:]
            },
        }

    # Pooled across all k. WARNING (confirmed, not hypothetical - see
    # correlations_within_k below): this pools over a shared k-trend in both
    # axes and can flip sign relative to the true within-k relationship -
    # e.g. mean_cos_center trends toward 0 as k grows (the -1/(k-1) baseline
    # of pairwise cosine among vectors constrained to sum to zero) while
    # var_dual trends down as k grows (more averaging), producing a positive
    # pooled r even where every single k shows a negative one. Do not read
    # this pooled number as "the" relationship - use correlations_within_k.
    summary["correlations_pooled"] = {
        f"{a}_vs_{b}": dict(zip(
            ("pearson_r", "p_value", "n"),
            pearson_or_nan([r[a] for r in rows], [r[b] for r in rows]),
        ))
        for a, b in CORRELATION_PAIRS
    }

    # k=2 is excluded here: mean_cos_center is exactly -1.0 for every k=2 draw
    # by construction (two deviations from their own 2-point mean are always
    # exact negatives of each other), a structural fact carrying no
    # information about the models drawn - including it doesn't bias
    # within-k correlations at other k (they're computed separately per k
    # regardless), but it would itself be a degenerate all-identical-x column
    # if reported as its own "k=2 correlation" (undefined/zero variance).
    summary["correlations_within_k"] = {}
    for k in sorted(by_k.keys()):
        if k == 2:
            continue
        k_rows = by_k[k]
        summary["correlations_within_k"][str(k)] = {
            f"{a}_vs_{b}": dict(zip(
                ("pearson_r", "p_value", "n"),
                pearson_or_nan([r[a] for r in k_rows], [r[b] for r in k_rows]),
            ))
            for a, b in CORRELATION_PAIRS
        }

    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote per-K summary + correlations to {json_path}")
    return summary


def plot_diversity_correlations(rows: list[dict], correlations_pooled: dict, out_path: str) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted({row["k"] for row in rows})
    cmap = plt.get_cmap("tab10")
    k_color = {k: cmap(i % 10) for i, k in enumerate(ks)}

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (a, b) in zip(axes.flat, CORRELATION_PAIRS):
        for k in ks:
            xs = [r[a] for r in rows if r["k"] == k]
            ys = [r[b] for r in rows if r["k"] == k]
            ax.scatter(xs, ys, color=k_color[k], label=f"k={k}", alpha=0.7, s=18)
        stats = correlations_pooled[f"{a}_vs_{b}"]
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title(f"{a} vs {b}\npooled r={stats['pearson_r']:.3f} (p={stats['p_value']:.3g}, n={stats['n']}) "
                      f"- confounded by k, see within-k figure")

    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote figure to {out_path}")


def plot_within_k_correlations(correlations_within_k: dict, out_path: str) -> None:
    """
    The real signal: how each pair's Pearson r moves with k, holding k fixed
    per point - the pooled scatter figure's headline r can (and here does)
    flip sign relative to every one of these, since mean_cos_center and the
    bias/var terms both trend with k independently (see write_outputs's
    correlations_pooled docstring note).
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(int(k) for k in correlations_within_k.keys())
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (a, b) in zip(axes.flat, CORRELATION_PAIRS):
        key = f"{a}_vs_{b}"
        rs = [correlations_within_k[str(k)][key]["pearson_r"] for k in ks]
        ax.plot(ks, rs, marker="o")
        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("k (ensemble size)")
        ax.set_ylabel("Pearson r")
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"{a} vs {b}\nwithin-k correlation")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pairwise deviation-orthogonality diagnostic vs. the Bregman bias/variance "
                     "decomposition, over TOMPEI-CMMD classifier ensembles."
    )
    parser.add_argument("--dataset", type=str, default="TOMPEI-CMMD")
    parser.add_argument("--auc-threshold", type=float, default=0.75)
    parser.add_argument("--k-values", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--draws-per-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    pool_size, rows = run_sweep(args.dataset, args.auc_threshold, k_values, args.draws_per_k, args.seed)
    if pool_size not in k_values:
        k_values_with_pool = k_values + [pool_size]
        pool_size, rows = run_sweep(args.dataset, args.auc_threshold, k_values_with_pool, args.draws_per_k, args.seed)

    print(f"Qualifying pool: {pool_size} models (test_auc >= {args.auc_threshold})")

    out_dir = os.path.join(BASE_MODELS, "results", args.dataset, "ensemble_diversity")
    summary = write_outputs(pool_size, rows, out_dir)
    plot_diversity_correlations(rows, summary["correlations_pooled"],
                                 os.path.join(out_dir, "figures", "ensemble_diversity.png"))
    plot_within_k_correlations(summary["correlations_within_k"],
                                os.path.join(out_dir, "figures", "ensemble_diversity_within_k.png"))

    print("\nPooled correlations (confounded by k - see within-k below for the real relationship):")
    for pair_key, stats in summary["correlations_pooled"].items():
        print(f"  {pair_key:32s} r={stats['pearson_r']:+.3f}  p={stats['p_value']:.3g}  n={stats['n']}")

    print("\nWithin-k correlations (k=2 excluded - mean_cos_center is exactly -1.0 there by construction):")
    for k in sorted(summary["correlations_within_k"].keys(), key=int):
        line = f"  k={k:>3s}: "
        for pair_key, stats in summary["correlations_within_k"][k].items():
            line += f"{pair_key}: r={stats['pearson_r']:+.2f}  "
        print(line)
