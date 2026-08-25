"""
Within-architecture-family variant of bias_variance_cls.py (Gupta et al. 2022
Bregman bias-variance decomposition): instead of drawing K-subsets from the
whole heterogeneous qualifying pool, restrict each sweep to a single
architecture family (e.g. only the maxvit variants) and vary K from 1 up to
however many members of that family passed the test_auc filter.

Purpose: the cross-family sweep in bias_variance_cls.py confounds two sources
of ensemble diversity - different depth/resolution tiers within a family, and
genuinely different architectures across families. This isolates the former:
does stacking checkpoints that only differ in depth/resolution tier (same
family) still buy a meaningful bias reduction, or is cross-architecture
diversity doing most of the work in the full-pool sweep?

Reuses decompose()/load_prediction_matrix() from bias_variance_cls.py - same
Bregman-divergence machinery, no training or inference, reads only existing
metrics/test_predictions.json files.

Usage:
    python bias_variance_within_family_cls.py [--dataset TOMPEI-CMMD] [--auc-threshold 0.75]
                                               [--draws-per-k 30] [--seed 42] [--min-family-size 2]
"""

from __future__ import annotations

import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

import argparse
import csv
import json
import math

import matplotlib
import numpy as np

from fusionLearning.config import BASE_MODELS
from fusionLearning.models.bias_variance_cls import decompose, load_prediction_matrix


def load_qualifying_models_by_family(dataset: str, auc_threshold: float) -> dict[str, list[str]]:
    summary_path = os.path.join(BASE_MODELS, "results", dataset, "summary.csv")
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    by_family: dict[str, list[str]] = {}
    for row in rows:
        if row["status"] != "done" or not row["test_auc"]:
            continue
        if float(row["test_auc"]) < auc_threshold:
            continue
        by_family.setdefault(row["family"], []).append(f"{row['variant_id']}_{row['timm_name']}")
    return by_family


def run_family_sweep(dataset: str, family: str, model_names: list[str], draws_per_k: int, seed: int) -> list[dict]:
    n = len(model_names)
    y, P, _ = load_prediction_matrix(dataset, model_names)

    rng = np.random.default_rng(seed)
    rows = []
    for k in range(1, n + 1):
        n_draws = 1 if k == n else min(draws_per_k, math.comb(n, k))
        seen_draws = set()
        draw_idx = 0
        attempts = 0
        while draw_idx < n_draws and attempts < n_draws * 20:
            attempts += 1
            idx = tuple(sorted(rng.choice(n, size=k, replace=False).tolist()))
            if idx in seen_draws:
                continue
            seen_draws.add(idx)
            result = decompose(y, P[list(idx)])
            result["family"] = family
            result["draw_idx"] = draw_idx
            rows.append(result)
            draw_idx += 1
    return rows


def write_family_outputs(all_rows: dict[str, list[dict]], out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    fields = ["family", "k", "draw_idx", "L_indiv", "bias_primal", "var_primal", "gap_primal",
              "bias_dual", "var_dual", "auc_primal", "auc_dual"]
    csv_path = os.path.join(out_dir, "sweep_results_by_family.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rows in all_rows.values():
            writer.writerows(rows)

    summary = {}
    for family, rows in all_rows.items():
        by_k: dict[int, list[dict]] = {}
        for row in rows:
            by_k.setdefault(row["k"], []).append(row)
        n = max(by_k.keys())

        by_k_summary = {}
        for k, k_rows in sorted(by_k.items()):
            by_k_summary[str(k)] = {
                "n_draws": len(k_rows),
                **{f"{m}_mean": float(np.mean([r[m] for r in k_rows])) for m in fields[3:]},
                **{f"{m}_std": float(np.std([r[m] for r in k_rows])) for m in fields[3:]},
            }

        bias_dual_k1 = by_k_summary["1"]["bias_dual_mean"]
        bias_dual_kn = by_k_summary[str(n)]["bias_dual_mean"]
        bias_primal_k1 = by_k_summary["1"]["bias_primal_mean"]
        bias_primal_kn = by_k_summary[str(n)]["bias_primal_mean"]

        summary[family] = {
            "n": n,
            "bias_dual_reduction_pct": 100.0 * (bias_dual_k1 - bias_dual_kn) / bias_dual_k1,
            "bias_primal_reduction_pct": 100.0 * (bias_primal_k1 - bias_primal_kn) / bias_primal_k1,
            "auc_dual_k1": by_k_summary["1"]["auc_dual_mean"],
            "auc_dual_kn": by_k_summary[str(n)]["auc_dual_mean"],
            "by_k": by_k_summary,
        }

    json_path = os.path.join(out_dir, "summary_by_family.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {sum(len(r) for r in all_rows.values())} rows to {csv_path}")
    print(f"Wrote per-family summary to {json_path}")
    return summary


def plot_bias_vs_k_by_family(summary: dict, out_path: str) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_dual, ax_primal) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")

    for i, (family, fam_summary) in enumerate(sorted(summary.items())):
        ks = sorted(int(k) for k in fam_summary["by_k"].keys())
        dual_means = [fam_summary["by_k"][str(k)]["bias_dual_mean"] for k in ks]
        primal_means = [fam_summary["by_k"][str(k)]["bias_primal_mean"] for k in ks]
        color = cmap(i % 10)
        ax_dual.plot(ks, dual_means, marker="o", label=f"{family} (n={fam_summary['n']})", color=color)
        ax_primal.plot(ks, primal_means, marker="o", label=f"{family} (n={fam_summary['n']})", color=color)

    ax_dual.set_xlabel("Ensemble size K (within family)"); ax_dual.set_ylabel("Bias (log-loss units)")
    ax_dual.set_title("Dual-space bias vs. K, within architecture family"); ax_dual.legend(fontsize=8)
    ax_primal.set_xlabel("Ensemble size K (within family)"); ax_primal.set_ylabel("Bias (log-loss units)")
    ax_primal.set_title("Primal-space bias vs. K, within architecture family"); ax_primal.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Within-architecture-family Bregman bias-variance sweep.")
    parser.add_argument("--dataset", type=str, default="TOMPEI-CMMD")
    parser.add_argument("--auc-threshold", type=float, default=0.75)
    parser.add_argument("--draws-per-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-family-size", type=int, default=2)
    args = parser.parse_args()

    by_family = load_qualifying_models_by_family(args.dataset, args.auc_threshold)

    skipped = {fam: names for fam, names in by_family.items() if len(names) < args.min_family_size}
    families = {fam: names for fam, names in by_family.items() if len(names) >= args.min_family_size}

    for fam, names in skipped.items():
        print(f"Skipping family '{fam}': only {len(names)} qualifying model(s) (need >= {args.min_family_size})")

    all_rows = {}
    for family, model_names in sorted(families.items()):
        print(f"Sweeping family '{family}': n={len(model_names)} qualifying models")
        all_rows[family] = run_family_sweep(args.dataset, family, model_names, args.draws_per_k, args.seed)

    out_dir = os.path.join(BASE_MODELS, "results", args.dataset, "bias_variance_within_family")
    summary = write_family_outputs(all_rows, out_dir)
    plot_bias_vs_k_by_family(summary, os.path.join(out_dir, "figures", "bias_vs_k_by_family.png"))

    print("\nBias reduction from K=1 to K=n, dual space (the exact-decomposition space):")
    for family, fam_summary in sorted(summary.items(), key=lambda kv: -kv[1]["bias_dual_reduction_pct"]):
        print(f"  {family:14s} n={fam_summary['n']:2d}  "
              f"bias_dual: {fam_summary['bias_dual_reduction_pct']:+6.1f}%  "
              f"bias_primal: {fam_summary['bias_primal_reduction_pct']:+6.1f}%  "
              f"auc_dual: {fam_summary['auc_dual_k1']:.4f} -> {fam_summary['auc_dual_kn']:.4f}")
