"""
Consolidates every TOMPEI-CMMD classification model's roster config + test
metrics into one summary table - the PI's Step 1 deliverable ("Architecture
1, Model 1: hyperparameter setting A, test AUC/ACC/etc.") across the full
120-config grid, so models can actually be compared and ranked instead of
only inspected one results/ directory at a time.

Usage:
    python aggregate_results_cls.py [--dataset TOMPEI-CMMD] [--sort-by test_auc] [--top 15]

Writes results/{dataset}/summary.csv and results/{dataset}/summary.json
(overwritten each run - these are derived/reproducible from the per-model
metrics files, not source data) and prints a leaderboard to stdout.
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

from fusionLearning.models.roster_cls import MODEL_CONFIGS
from fusionLearning.models.distributed_cls import BASE_MODELS, load_skip_set_cls

# Columns written to summary.csv, in order. test_cm is deliberately left out of
# the CSV (nested list doesn't flatten cleanly) but is kept in summary.json.
CSV_FIELDS = [
    "variant_id", "family", "timm_name", "depth_tier", "resolution_tier", "resolution_px",
    "params_m", "lr", "batch_size", "status",
    "test_loss", "test_acc", "test_auc", "test_precision", "test_sensitivity",
    "test_specificity", "test_f1", "test_mcc", "tested_at",
]


def collect_results(dataset: str = "TOMPEI-CMMD") -> list[dict]:
    skip_set = load_skip_set_cls()
    rows = []

    for cfg in MODEL_CONFIGS:
        model_name = f"{cfg['variant_id']}_{cfg['timm_name']}"
        model_dir = os.path.join(BASE_MODELS, "results", dataset, model_name)
        test_path = os.path.join(model_dir, "metrics", "test_metrics.json")

        row = dict(cfg)  # variant_id/family/timm_name/depth_tier/resolution_tier/resolution_px/params_m/lr/batch_size

        if cfg["variant_id"] in skip_set:
            row["status"] = "skipped"
        elif os.path.exists(test_path):
            row["status"] = "done"
            with open(test_path) as f:
                row.update(json.load(f))
        else:
            row["status"] = "pending"

        rows.append(row)

    return rows


def write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)


def print_leaderboard(rows: list[dict], sort_by: str, top: int) -> None:
    done = [r for r in rows if r["status"] == "done" and sort_by in r]
    done.sort(key=lambda r: r[sort_by], reverse=True)

    print(f"\nTop {min(top, len(done))} by {sort_by} (of {len(done)} completed / {len(rows)} total configs):")
    print(f"{'variant_id':<16} {'timm_name':<24} {'params_m':>9} {'test_acc':>9} "
          f"{'test_auc':>9} {'test_f1':>8} {'sens':>7} {'spec':>7}")
    for r in done[:top]:
        print(f"{r['variant_id']:<16} {r['timm_name']:<24} {r['params_m']:>9.2f} "
              f"{r.get('test_acc', float('nan')):>9.4f} {r.get('test_auc', float('nan')):>9.4f} "
              f"{r.get('test_f1', float('nan')):>8.4f} {r.get('test_sensitivity', float('nan')):>7.4f} "
              f"{r.get('test_specificity', float('nan')):>7.4f}")

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nStatus: {counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate TOMPEI-CMMD classification results across the full roster.")
    parser.add_argument("--dataset", type=str, default="TOMPEI-CMMD")
    parser.add_argument("--sort-by", type=str, default="test_auc",
                         choices=["test_auc", "test_acc", "test_f1", "test_sensitivity",
                                  "test_specificity", "test_precision", "test_mcc"])
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    rows = collect_results(args.dataset)

    out_dir = os.path.join(BASE_MODELS, "results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    write_csv(rows, os.path.join(out_dir, "summary.csv"))
    write_json(rows, os.path.join(out_dir, "summary.json"))

    print(f"Wrote {len(rows)} rows to {out_dir}/summary.{{csv,json}}")
    print_leaderboard(rows, args.sort_by, args.top)
