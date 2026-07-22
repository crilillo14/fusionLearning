"""
Mass train, validate, test, and infer classification models across the
TOMPEI-CMMD 100-config roster (10 timm architecture families x 10 variants
each - see roster_cls.py).

Usage:
    python orchestrator_cls.py [--all] [--family FAMILY] [--model VARIANT_ID]

Examples:
    python orchestrator_cls.py --all
    python orchestrator_cls.py --family resnet
    python orchestrator_cls.py --model resnet_00
"""

import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

import argparse
import traceback

from fusionLearning.models.distributed_cls import (
    launch_training_cls,
    load_skip_set_cls,
    record_failure_cls,
)
from fusionLearning.models.roster_cls import MODEL_CONFIGS, FAMILIES

DATASET = "TOMPEI-CMMD"


def run_variant(variant_id: str, dataset: str = DATASET) -> bool:
    """Returns True on success, False on failure."""
    skip = load_skip_set_cls()
    if variant_id in skip:
        print(f"  [SKIP] {variant_id} — in skip list")
        return False

    print(f"\n{'='*60}")
    print(f"  dataset={dataset}  variant={variant_id}")
    print(f"{'='*60}")
    try:
        launch_training_cls(variant_id, dataset)
        return True
    except Exception as e:
        print(f"  [FAIL] {variant_id}: {e}")
        traceback.print_exc()
        record_failure_cls(variant_id, str(e))
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mass training across the TOMPEI-CMMD 100-config classification roster"
    )
    parser.add_argument("--all", action="store_true", help="Train all 100 configs.")
    parser.add_argument("--family", type=str, choices=FAMILIES,
                         help="Train all 10 variants within one architecture family.")
    parser.add_argument("--model", type=str,
                         choices=[c["variant_id"] for c in MODEL_CONFIGS],
                         help="Train a single variant_id.")
    args = parser.parse_args()

    if not args.all and not args.family and not args.model:
        parser.error("Specify --all, --family, or --model.")

    if args.all:
        variant_ids = [c["variant_id"] for c in MODEL_CONFIGS]
    elif args.family:
        variant_ids = [c["variant_id"] for c in MODEL_CONFIGS if c["family"] == args.family]
    else:
        variant_ids = [args.model]

    results = {"success": [], "skip": [], "fail": []}

    for variant_id in variant_ids:
        skip = load_skip_set_cls()
        if variant_id in skip:
            print(f"  [SKIP] {variant_id}")
            results["skip"].append(variant_id)
            continue

        ok = run_variant(variant_id)
        if ok:
            results["success"].append(variant_id)
        else:
            results["fail"].append(variant_id)

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"  success : {len(results['success'])}")
    print(f"  skipped : {len(results['skip'])}")
    print(f"  failed  : {len(results['fail'])}")
    if results["fail"]:
        print("\n  Failed variants:")
        for f in results["fail"]:
            print(f"    {f}")
    print(f"{'='*60}\n")
