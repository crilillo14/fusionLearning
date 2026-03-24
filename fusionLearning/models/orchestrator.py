"""
Mass train, validate, test, and infer models across all arch-encoder pairs.

Usage:
    python orchestrator.py <dataset> [<dataset> ...] [--all] [--arch ARCH] [--encoder ENCODER]

Examples:
    python orchestrator.py CUB --all
    python orchestrator.py CUB Cityscapes ADE20K --all
    python orchestrator.py CUB --arch Unet --encoder resnet50
"""

import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

import argparse
import traceback

from fusionLearning.models.distributed import (
    launch_training,
    available_encoder_types,
    arch_dict,
    flat_encoders,
    load_skip_set,
    record_failure,
)

SUPPORTED_DATASETS = ["CUB", "Cityscapes", "ADE20K"]


def run_pair(arch_name: str, encoder: str, dataset: str) -> bool:
    """Returns True on success, False on failure."""
    key = f"{arch_name}__{encoder}"
    skip = load_skip_set()
    if key in skip:
        print(f"  [SKIP] {arch_name} + {encoder} — in skip list")
        return False

    print(f"\n{'='*60}")
    print(f"  dataset={dataset}  arch={arch_name}  encoder={encoder}")
    print(f"{'='*60}")
    try:
        launch_training(arch_name, encoder, dataset)
        return True
    except Exception as e:
        print(f"  [FAIL] {arch_name} + {encoder}: {e}")
        traceback.print_exc()
        record_failure(arch_name, encoder, str(e))
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mass training across arch-encoder pairs and datasets"
    )
    parser.add_argument(
        "datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        metavar="DATASET",
        help=f"One or more datasets to train on. Choices: {SUPPORTED_DATASETS}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all arch-encoder pairings for each dataset.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        choices=list(arch_dict.keys()),
        help="Single architecture to train (use with --encoder).",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        choices=flat_encoders,
        help="Single encoder to train (use with --arch).",
    )
    args = parser.parse_args()

    if not args.all and not (args.arch and args.encoder):
        parser.error("Specify --all or both --arch and --encoder.")

    results = {"success": [], "skip": [], "fail": []}

    for dataset in args.datasets:
        if args.all:
            pairs = [(arch, enc) for arch in arch_dict for enc in flat_encoders]
        else:
            pairs = [(args.arch, args.encoder)]

        for arch_name, encoder in pairs:
            key = f"{arch_name}__{encoder}"
            skip = load_skip_set()
            if key in skip:
                print(f"  [SKIP] {dataset}/{arch_name}+{encoder}")
                results["skip"].append(f"{dataset}/{key}")
                continue

            ok = run_pair(arch_name, encoder, dataset)
            if ok:
                results["success"].append(f"{dataset}/{key}")
            else:
                results["fail"].append(f"{dataset}/{key}")

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  success : {len(results['success'])}")
    print(f"  skipped : {len(results['skip'])}")
    print(f"  failed  : {len(results['fail'])}")
    if results["fail"]:
        print("\n  Failed pairs:")
        for f in results["fail"]:
            print(f"    {f}")
    print(f"{'='*60}\n")
