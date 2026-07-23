"""
Base-model roster for TOMPEI-CMMD binary classification benchmarking.

10 architecture families x 4 depth tiers x 3 resolution tiers = 120 configs.

This replaces the previous 100-config roster (10 families x 10 capacity/LR
variants, all at a single fixed 512x512 resolution). The new grid varies two
axes deliberately instead of one:

  - depth tier (tiny/small/medium/large): a real timm checkpoint per family
    per tier, chosen to stay in a *realistic in-house* capacity range (train
    set is only 2892 images) rather than reaching for each family's largest
    available pretrained checkpoint - e.g. ViT stops at vit_base (85.8M), not
    vit_large (303M); ConvNeXt stops at convnext_small (49.5M), not
    convnext_large (196M). Transformer/hybrid families (Swin, MaxViT, CoAtNet)
    don't have meaningfully smaller-than-~15-30M pretrained checkpoints, so
    their "tiny" tier sits higher in absolute params than CNN families' tiny
    tier - an inherent property of those architectures, not a modeling choice.

  - resolution tier (lo/mid/hi = 384/512/768px, see consts.RESOLUTION_TIERS_CLS):
    lesion regions are small relative to the ~2000x2000 native mammogram, so
    downsampling too aggressively risks losing the signal entirely. Sized
    against confirmed remote hardware (4x NVIDIA L40S, 48GB each) - real
    memory profiling on that hardware is still needed before trusting the
    large-depth x hi-res combos not to OOM (batch sizes below are best-guess
    estimates, same caveat the previous roster carried for its batch sizes).

Naming: variant_id = f"{family}_{depth_char}{res_digit}", e.g. "resnet_S2"
(small depth, mid res), "maxvit_L3" (large depth, hi res). depth_char in
{T,S,M,L}, res_digit in {1,2,3} for {lo,mid,hi} - kept as a digit (not a
letter) specifically to avoid collision with the depth-tier letters.

Architecture note: classic Inception-v3 (and Inception-ResNet-v2, InceptionNeXt)
were all considered for the 10th slot and rejected - Inception-v3's timm
checkpoints are all the *same* architecture with different training recipes
(no size variants), and every alternative that does provide 4 real capacity
points turned out to require mixing genuinely different architectures per
depth tier, which breaks the controlled-variable premise of the depth axis
(every other family holds architecture fixed and varies only scale - this one
must too). Xception is used instead: a single, consistent, pure-CNN
architecture (no attention, no token mixing) built entirely on depthwise
separable convolutions - a spatial (depthwise) conv followed by a pointwise
1x1 "cross-channel" conv - with a genuine single-architecture depth ladder in
timm (legacy_xception -> xception41 -> xception65 -> xception71: 20.8/24.9/
37.9/40.3M), scaling purely via middle-flow block count, same as every other
family's depth tiers.

family -> grad-cam strategy (see inference_cls.py, unchanged by this pivot):
  "cnn_staged"        - spatially-meaningful per-stage feature maps via
                         features_only=True (works for any architecture that
                         keeps a spatial grid through its stages - includes
                         the hybrid conv+windowed-attention families MaxViT/
                         CoAtNet, not just pure CNNs; unverified against a
                         real trained checkpoint, flag if gradcam_from_paths_cls
                         fails for these two families and fall back to
                         "transformer_final" if so)
  "transformer_final"  - only the final block is spatially interpretable
                          (pure patch/window transformers - ViT, Swin)
"""

from __future__ import annotations

from fusionLearning.models.consts import RESOLUTION_TIERS_CLS

GRADCAM_FAMILY_TYPE: dict[str, str] = {
    "resnet": "cnn_staged",
    "densenet": "cnn_staged",
    "xception": "cnn_staged",
    "efficientnet": "cnn_staged",
    "convnext": "cnn_staged",
    "regnet": "cnn_staged",
    "vit": "transformer_final",
    "swin": "transformer_final",
    "maxvit": "cnn_staged",
    "coatnet": "cnn_staged",
}

# depth tier -> real timm checkpoint name, one per family, in tiny->large order.
FAMILY_TIMM_NAMES: dict[str, list[str]] = {
    "resnet":         ["resnet18", "resnet34", "resnet50", "resnet101"],
    "densenet":       ["densenet121", "densenet169", "densenet201", "densenet161"],
    "xception":       ["legacy_xception", "xception41", "xception65", "xception71"],
    "efficientnet":   ["efficientnet_b0", "efficientnet_b2", "efficientnet_b4", "efficientnet_b5"],
    "convnext":       ["convnext_atto", "convnext_nano", "convnext_tiny", "convnext_small"],
    "regnet":         ["regnety_004", "regnety_016", "regnety_032", "regnety_080"],
    "vit":            ["vit_tiny_patch16_224", "vit_small_patch16_224", "vit_medium_patch16_gap_240", "vit_base_patch16_224"],
    "swin":           ["swin_tiny_patch4_window7_224", "swin_small_patch4_window7_224", "swin_s3_base_224", "swin_base_patch4_window7_224"],
    "maxvit":         ["maxvit_nano_rw_256", "maxvit_tiny_tf_224", "maxvit_small_tf_224", "maxvit_base_tf_224"],
    "coatnet":        ["coatnet_nano_rw_224", "coatnet_0_rw_224", "coatnet_1_rw_224", "coatnet_2_rw_224"],
}

# Approximate params_m per family per depth tier (verified via
# `timm.create_model(name, num_classes=1)` param counts at authoring time -
# for reference/reporting only, not consumed by training code).
FAMILY_PARAMS_M: dict[str, list[float]] = {
    "resnet":         [11.18, 21.29, 23.51, 42.50],
    "densenet":       [6.95, 12.49, 18.09, 26.47],
    "xception":       [20.81, 24.92, 37.87, 40.29],
    "efficientnet":   [4.01, 7.70, 17.55, 28.34],
    "convnext":       [3.37, 14.95, 27.82, 49.46],
    "regnet":         [3.90, 10.31, 17.92, 37.17],
    "vit":            [5.52, 21.67, 38.33, 85.80],
    "swin":           [27.52, 48.84, 70.36, 86.74],
    "maxvit":         [14.94, 30.40, 68.16, 118.70],
    "coatnet":        [14.63, 26.67, 40.95, 72.84],
}

DEPTH_TIERS: list[str] = ["tiny", "small", "medium", "large"]
DEPTH_CHAR: dict[str, str] = {"tiny": "T", "small": "S", "medium": "M", "large": "L"}

RESOLUTION_TIERS: list[str] = ["lo", "mid", "hi"]
RESOLUTION_DIGIT: dict[str, str] = {"lo": "1", "mid": "2", "hi": "3"}

# Base per-GPU batch size at mid (512px) resolution, by depth tier - reuses the
# previous roster's capacity-tiering precedent (24/16/10/6). Scaled by a
# resolution multiplier below (activation memory scales roughly with pixel
# count, i.e. ~(mid/res)^2) and rounded to a clean number. UNMEASURED - tune
# against the actual remote L40S boxes before a full 120-model unattended run.
BASE_BATCH_SIZE_BY_DEPTH: dict[str, int] = {"tiny": 24, "small": 16, "medium": 10, "large": 6}

BATCH_SIZE_TABLE: dict[str, dict[str, int]] = {
    "tiny":   {"lo": 32, "mid": 24, "hi": 10},
    "small":  {"lo": 24, "mid": 16, "hi": 8},
    "medium": {"lo": 16, "mid": 10, "hi": 6},
    "large":  {"lo": 10, "mid": 6, "hi": 4},
}

# Peak LR by depth tier only (not resolution) - larger pretrained backbones get
# a gentler peak LR for fine-tuning, same rationale as the previous roster.
LR_BY_DEPTH: dict[str, float] = {"tiny": 0.01, "small": 0.008, "medium": 0.006, "large": 0.004}

FAMILIES: list[str] = sorted(FAMILY_TIMM_NAMES.keys())


def _build_model_configs() -> list[dict]:
    configs = []
    for family in FAMILIES:
        timm_names = FAMILY_TIMM_NAMES[family]
        params = FAMILY_PARAMS_M[family]
        for depth_idx, depth_tier in enumerate(DEPTH_TIERS):
            for res_tier in RESOLUTION_TIERS:
                variant_id = f"{family}_{DEPTH_CHAR[depth_tier]}{RESOLUTION_DIGIT[res_tier]}"
                configs.append({
                    "variant_id": variant_id,
                    "family": family,
                    "timm_name": timm_names[depth_idx],
                    "depth_tier": depth_tier,
                    "resolution_tier": res_tier,
                    "resolution_px": RESOLUTION_TIERS_CLS[res_tier],
                    "params_m": params[depth_idx],
                    "lr": LR_BY_DEPTH[depth_tier],
                    "batch_size": BATCH_SIZE_TABLE[depth_tier][res_tier],
                })
    return configs


MODEL_CONFIGS: list[dict] = _build_model_configs()
MODEL_CONFIGS_BY_ID: dict[str, dict] = {c["variant_id"]: c for c in MODEL_CONFIGS}


def get_config(variant_id: str) -> dict:
    if variant_id not in MODEL_CONFIGS_BY_ID:
        raise ValueError(f"Unknown variant_id: {variant_id!r}. See roster_cls.MODEL_CONFIGS.")
    return MODEL_CONFIGS_BY_ID[variant_id]


def gradcam_strategy(family: str) -> str:
    return GRADCAM_FAMILY_TYPE[family]


if __name__ == "__main__":
    assert len(MODEL_CONFIGS) == 120, f"expected 120 configs, got {len(MODEL_CONFIGS)}"
    assert len(FAMILIES) == 10
    for fam in FAMILIES:
        n = sum(1 for c in MODEL_CONFIGS if c["family"] == fam)
        assert n == 12, f"{fam} has {n} variants, expected 12"
    assert len(MODEL_CONFIGS_BY_ID) == 120, "variant_id collision detected"
    print(f"{len(MODEL_CONFIGS)} configs across {len(FAMILIES)} families, all OK")
