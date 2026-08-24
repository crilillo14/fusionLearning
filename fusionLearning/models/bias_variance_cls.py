"""
Bregman-divergence bias-variance decomposition for TOMPEI-CMMD classifier
ensembles (Gupta et al. 2022): checks whether averaging in the dual
(logit) space holds bias fixed while shrinking variance, versus averaging
directly in primal (probability) space where bias empirically also drops.

Pool is filtered to the PI's stability bar (test_auc >= 0.75) from
results/{dataset}/summary.csv, then swept over ensemble size K using random
subsets drawn from that pool. Reads only existing per-model
metrics/test_predictions.json files - no training or inference happens here.

Usage:
    python bias_variance_cls.py [--dataset TOMPEI-CMMD] [--auc-threshold 0.75]
                                 [--k-values 2,4,8,16,32,64] [--draws-per-k 30] [--seed 42]
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

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

from fusionLearning.config import BASE_MODELS

EPS = 1e-6


def load_qualifying_models(dataset: str = "TOMPEI-CMMD", auc_threshold: float = 0.75) -> list[str]:
    summary_path = os.path.join(BASE_MODELS, "results", dataset, "summary.csv")
    with open(summary_path, newline="") as f:
        rows = list(csv.DictReader(f))

    qualifying = []
    for row in rows:
        if row["status"] != "done" or not row["test_auc"]:
            continue
        if float(row["test_auc"]) >= auc_threshold:
            qualifying.append(f"{row['variant_id']}_{row['timm_name']}")
    return qualifying


def load_prediction_matrix(dataset: str, model_names: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (y, P, filenames):
      y: [N] float array of true labels (0/1)
      P: [K, N] float array of pred_prob, rows aligned to model_names, columns to filenames
    Raises if any model's test_predictions.json doesn't cover the exact same filename set
    as the first model - alignment must be exact, not just same length.
    """
    per_model_preds: list[dict[str, float]] = []
    per_model_labels: list[dict[str, int]] = []

    for model_name in model_names:
        pred_path = os.path.join(BASE_MODELS, "results", dataset, model_name, "metrics", "test_predictions.json")
        with open(pred_path) as f:
            data = json.load(f)
        preds = {r["filename"]: r["pred_prob"] for r in data["predictions"]}
        labels = {r["filename"]: r["true_label"] for r in data["predictions"]}
        per_model_preds.append(preds)
        per_model_labels.append(labels)

    reference_filenames = set(per_model_preds[0].keys())
    for model_name, preds in zip(model_names, per_model_preds):
        if set(preds.keys()) != reference_filenames:
            raise ValueError(
                f"{model_name}'s test_predictions.json filename set doesn't match "
                f"{model_names[0]}'s - test sets must be identical across the ensemble pool."
            )

    filenames = sorted(reference_filenames)
    y = np.array([per_model_labels[0][fn] for fn in filenames], dtype=np.float64)
    P = np.array([[preds[fn] for fn in filenames] for preds in per_model_preds], dtype=np.float64)
    return y, P, filenames


def bregman_div(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """D(a,b) = a*log(a/b) + (1-a)*log((1-a)/(1-b)), the Bregman divergence of
    negative binary entropy. Reduces to standard BCE when a in {0,1}."""
    b = np.clip(b, eps, 1 - eps)
    a_safe = np.clip(a, eps, 1 - eps)  # only affects the a*log(a) self-entropy term when a is continuous
    return a * np.log(a_safe / b) + (1 - a) * np.log((1 - a_safe) / (1 - b))


def decompose(y: np.ndarray, P_subset: np.ndarray) -> dict:
    """
    y: [N], P_subset: [k, N] probabilities from k selected models.

    The exact zero-residual Bregman identity
        mean_k D(y, p_k) = D(y, c) + mean_k D(c, p_k)
    holds only when the center c is the DUAL mean (arithmetic mean of the
    natural parameters/logits, mapped back through sigmoid) - NOT the primal
    arithmetic mean of probabilities. This falls out of the Bregman-divergence
    algebra: expanding both sides leaves a residual (y - c)*(mean_k[phi'(p_k)]
    - phi'(c)), which vanishes exactly iff phi'(c) = mean_k[phi'(p_k)], i.e.
    c = phi'^{-1}(mean_k[phi'(p_k)]) = the dual mean. Verified numerically
    before wiring this up (see conversation) - do not "fix" this by swapping
    which side gets the identity assertion without re-deriving.
    """
    k = P_subset.shape[0]

    indiv_losses = np.array([bregman_div(y, P_subset[i]).mean() for i in range(k)])
    L_indiv = indiv_losses.mean()

    p_primal = P_subset.mean(axis=0)
    bias_primal = bregman_div(y, p_primal).mean()
    var_primal = np.mean([bregman_div(p_primal, P_subset[i]) for i in range(k)])
    gap_primal = L_indiv - (bias_primal + var_primal)

    logits = np.log(np.clip(P_subset, EPS, 1 - EPS) / (1 - np.clip(P_subset, EPS, 1 - EPS)))
    eta_bar = logits.mean(axis=0)
    p_dual = 1.0 / (1.0 + np.exp(-eta_bar))
    bias_dual = bregman_div(y, p_dual).mean()
    var_dual = np.mean([bregman_div(p_dual, P_subset[i]) for i in range(k)])

    auc_primal = roc_auc_score(y, p_primal)
    auc_dual = roc_auc_score(y, p_dual)

    return {
        "k": k,
        "L_indiv": L_indiv,
        "bias_primal": bias_primal,
        "var_primal": var_primal,
        "gap_primal": gap_primal,
        "bias_dual": bias_dual,
        "var_dual": var_dual,
        "auc_primal": auc_primal,
        "auc_dual": auc_dual,
    }


def run_sweep(dataset: str, auc_threshold: float, k_values: list[int], draws_per_k: int, seed: int):
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
            result = decompose(y, P[idx])
            result["draw_idx"] = draw_idx
            rows.append(result)

    # sanity check: exact Bregman identity holds for dual-space (logit) averaging
    for row in rows:
        assert abs((row["bias_dual"] + row["var_dual"]) - row["L_indiv"]) < 1e-6, (
            f"Bregman identity violated at k={row['k']}, draw={row['draw_idx']} - "
            f"bias_dual + var_dual should exactly equal L_indiv"
        )

    return pool_size, rows


def write_outputs(pool_size: int, rows: list[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    fields = ["k", "draw_idx", "L_indiv", "bias_primal", "var_primal", "gap_primal",
              "bias_dual", "var_dual", "auc_primal", "auc_dual"]
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
                f"{metric}_mean": float(np.mean([r[metric] for r in k_rows]))
                for metric in fields[2:]
            },
            **{
                f"{metric}_std": float(np.std([r[metric] for r in k_rows]))
                for metric in fields[2:]
            },
        }
    json_path = os.path.join(out_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote per-K summary to {json_path}")
    return summary


def plot_bias_variance_vs_k(summary: dict, out_path: str) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = sorted(int(k) for k in summary["by_k"].keys())
    def series(metric):
        means = [summary["by_k"][str(k)][f"{metric}_mean"] for k in ks]
        stds = [summary["by_k"][str(k)][f"{metric}_std"] for k in ks]
        return np.array(means), np.array(stds)

    fig, (ax_bias, ax_var) = plt.subplots(1, 2, figsize=(12, 5))

    for metric, label, color in [("bias_primal", "primal", "tab:blue"), ("bias_dual", "dual", "tab:orange")]:
        m, s = series(metric)
        ax_bias.plot(ks, m, marker="o", label=label, color=color)
        ax_bias.fill_between(ks, m - s, m + s, alpha=0.2, color=color)
    l_indiv_m, _ = series("L_indiv")
    ax_bias.plot(ks, l_indiv_m, marker="x", linestyle="--", label="avg individual loss", color="gray")
    ax_bias.set_xlabel("Ensemble size K"); ax_bias.set_ylabel("Bias (log-loss units)")
    ax_bias.set_title("Bias vs. ensemble size"); ax_bias.legend(); ax_bias.set_xscale("log", base=2)

    for metric, label, color in [("var_primal", "primal", "tab:blue"), ("var_dual", "dual", "tab:orange")]:
        m, s = series(metric)
        ax_var.plot(ks, m, marker="o", label=label, color=color)
        ax_var.fill_between(ks, m - s, m + s, alpha=0.2, color=color)
    ax_var.set_xlabel("Ensemble size K"); ax_var.set_ylabel("Variance (log-loss units)")
    ax_var.set_title("Variance vs. ensemble size"); ax_var.legend(); ax_var.set_xscale("log", base=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote figure to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bregman bias-variance decomposition over TOMPEI-CMMD classifier ensembles.")
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

    out_dir = os.path.join(BASE_MODELS, "results", args.dataset, "bias_variance")
    summary = write_outputs(pool_size, rows, out_dir)
    plot_bias_variance_vs_k(summary, os.path.join(out_dir, "figures", "bias_variance_vs_k.png"))
