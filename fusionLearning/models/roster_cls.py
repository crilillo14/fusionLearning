"""
Base-model roster for TOMPEI-CMMD binary classification benchmarking.

10 timm architecture families x 10 variants each = 100 configs, for later use as
diverse base-model predictions feeding into fusion methods (fusionLearning/fusion/).

Variant axes per family:
  - encoder capacity: distinct timm model names spanning small -> large within the family
    (verified importable + carrying default ImageNet-pretrained weights via
    `timm.list_models(pretrained=True)` as of timm==1.0.25)
  - hyperparameters: peak LR decreases ~3%/variant as capacity grows (standard fine-tuning
    practice - larger pretrained backbones get a gentler peak LR), batch size is tiered by
    parameter count so a fixed 500x500x3-input batch roughly fits typical GPU memory
    (unmeasured guess - tune against the actual remote GPU, see notes/tompei_cmmd_classification_spec.md)
  - densenet only has 5 distinct architectures with default pretrained weights in timm's
    registry, so its 10 slots are 5 archs x 2 explicit LR settings each - a genuine
    LR-sensitivity comparison rather than padding.

family -> grad-cam strategy (see inference_cls.py):
  "cnn_staged"        - CNN backbones with spatially-meaningful per-stage feature maps
  "transformer_final"  - patch/window transformers; only the final block is spatially
                          interpretable via reshape_transform
"""

from __future__ import annotations

GRADCAM_FAMILY_TYPE: dict[str, str] = {
    "resnet": "cnn_staged",
    "resnext": "cnn_staged",
    "seresnet": "cnn_staged",
    "densenet": "cnn_staged",
    "efficientnet": "cnn_staged",
    "mobilenet": "cnn_staged",
    "regnet": "cnn_staged",
    "convnext": "cnn_staged",
    "vit": "transformer_final",
    "swin": "transformer_final",
}

MODEL_CONFIGS: list[dict] = [
    {"variant_id": "resnet_00", "family": "resnet", "timm_name": "resnet18", "params_m": 11.18, "lr": 0.01, "batch_size": 16},
    {"variant_id": "resnet_01", "family": "resnet", "timm_name": "resnet26", "params_m": 13.95, "lr": 0.0097, "batch_size": 16},
    {"variant_id": "resnet_02", "family": "resnet", "timm_name": "resnet34", "params_m": 21.29, "lr": 0.0094, "batch_size": 16},
    {"variant_id": "resnet_03", "family": "resnet", "timm_name": "resnet50", "params_m": 23.51, "lr": 0.0091, "batch_size": 16},
    {"variant_id": "resnet_04", "family": "resnet", "timm_name": "resnet50d", "params_m": 23.53, "lr": 0.0088, "batch_size": 16},
    {"variant_id": "resnet_05", "family": "resnet", "timm_name": "resnet101", "params_m": 42.5, "lr": 0.0085, "batch_size": 10},
    {"variant_id": "resnet_06", "family": "resnet", "timm_name": "resnet101d", "params_m": 42.52, "lr": 0.0082, "batch_size": 10},
    {"variant_id": "resnet_07", "family": "resnet", "timm_name": "resnet152", "params_m": 58.15, "lr": 0.0079, "batch_size": 10},
    {"variant_id": "resnet_08", "family": "resnet", "timm_name": "resnet152d", "params_m": 58.17, "lr": 0.0076, "batch_size": 10},
    {"variant_id": "resnet_09", "family": "resnet", "timm_name": "resnet200d", "params_m": 62.65, "lr": 0.0073, "batch_size": 6},
    {"variant_id": "resnext_00", "family": "resnext", "timm_name": "resnext26ts", "params_m": 8.25, "lr": 0.01, "batch_size": 24},
    {"variant_id": "resnext_01", "family": "resnext", "timm_name": "resnext50_32x4d", "params_m": 22.98, "lr": 0.0097, "batch_size": 16},
    {"variant_id": "resnext_02", "family": "resnext", "timm_name": "resnext50d_32x4d", "params_m": 23.0, "lr": 0.0094, "batch_size": 16},
    {"variant_id": "resnext_03", "family": "resnext", "timm_name": "cspresnext50", "params_m": 18.52, "lr": 0.0091, "batch_size": 16},
    {"variant_id": "resnext_04", "family": "resnext", "timm_name": "eca_resnext26ts", "params_m": 8.25, "lr": 0.0088, "batch_size": 24},
    {"variant_id": "resnext_05", "family": "resnext", "timm_name": "gcresnext50ts", "params_m": 13.62, "lr": 0.0085, "batch_size": 16},
    {"variant_id": "resnext_06", "family": "resnext", "timm_name": "skresnext50_32x4d", "params_m": 25.43, "lr": 0.0082, "batch_size": 16},
    {"variant_id": "resnext_07", "family": "resnext", "timm_name": "resnext101_32x4d", "params_m": 42.13, "lr": 0.0079, "batch_size": 10},
    {"variant_id": "resnext_08", "family": "resnext", "timm_name": "resnext101_32x8d", "params_m": 86.74, "lr": 0.0076, "batch_size": 6},
    {"variant_id": "resnext_09", "family": "resnext", "timm_name": "resnext101_64x4d", "params_m": 81.41, "lr": 0.0073, "batch_size": 6},
    {"variant_id": "seresnet_00", "family": "seresnet", "timm_name": "legacy_seresnet18", "params_m": 11.27, "lr": 0.01, "batch_size": 16},
    {"variant_id": "seresnet_01", "family": "seresnet", "timm_name": "legacy_seresnet34", "params_m": 21.45, "lr": 0.0097, "batch_size": 16},
    {"variant_id": "seresnet_02", "family": "seresnet", "timm_name": "legacy_seresnet50", "params_m": 26.04, "lr": 0.0094, "batch_size": 16},
    {"variant_id": "seresnet_03", "family": "seresnet", "timm_name": "legacy_seresnet101", "params_m": 47.28, "lr": 0.0091, "batch_size": 10},
    {"variant_id": "seresnet_04", "family": "seresnet", "timm_name": "legacy_seresnet152", "params_m": 64.77, "lr": 0.0088, "batch_size": 6},
    {"variant_id": "seresnet_05", "family": "seresnet", "timm_name": "legacy_senet154", "params_m": 113.04, "lr": 0.0085, "batch_size": 6},
    {"variant_id": "seresnet_06", "family": "seresnet", "timm_name": "ecaresnet26t", "params_m": 13.96, "lr": 0.0082, "batch_size": 16},
    {"variant_id": "seresnet_07", "family": "seresnet", "timm_name": "ecaresnet50d", "params_m": 23.53, "lr": 0.0079, "batch_size": 16},
    {"variant_id": "seresnet_08", "family": "seresnet", "timm_name": "ecaresnet50t", "params_m": 23.53, "lr": 0.0076, "batch_size": 16},
    {"variant_id": "seresnet_09", "family": "seresnet", "timm_name": "ecaresnet101d", "params_m": 42.52, "lr": 0.0073, "batch_size": 10},
    {"variant_id": "densenet_00", "family": "densenet", "timm_name": "densenet121", "params_m": 6.95, "lr": 0.01, "batch_size": 24},
    {"variant_id": "densenet_01", "family": "densenet", "timm_name": "densenet121", "params_m": 6.95, "lr": 0.00485, "batch_size": 24},
    {"variant_id": "densenet_02", "family": "densenet", "timm_name": "densenet161", "params_m": 26.47, "lr": 0.0094, "batch_size": 16},
    {"variant_id": "densenet_03", "family": "densenet", "timm_name": "densenet161", "params_m": 26.47, "lr": 0.00455, "batch_size": 16},
    {"variant_id": "densenet_04", "family": "densenet", "timm_name": "densenet169", "params_m": 12.49, "lr": 0.0088, "batch_size": 16},
    {"variant_id": "densenet_05", "family": "densenet", "timm_name": "densenet169", "params_m": 12.49, "lr": 0.00425, "batch_size": 16},
    {"variant_id": "densenet_06", "family": "densenet", "timm_name": "densenet201", "params_m": 18.09, "lr": 0.0082, "batch_size": 16},
    {"variant_id": "densenet_07", "family": "densenet", "timm_name": "densenet201", "params_m": 18.09, "lr": 0.00395, "batch_size": 16},
    {"variant_id": "densenet_08", "family": "densenet", "timm_name": "densenetblur121d", "params_m": 6.97, "lr": 0.0076, "batch_size": 24},
    {"variant_id": "densenet_09", "family": "densenet", "timm_name": "densenetblur121d", "params_m": 6.97, "lr": 0.00365, "batch_size": 24},
    {"variant_id": "efficientnet_00", "family": "efficientnet", "timm_name": "efficientnet_b0", "params_m": 4.01, "lr": 0.01, "batch_size": 24},
    {"variant_id": "efficientnet_01", "family": "efficientnet", "timm_name": "efficientnet_b1", "params_m": 6.51, "lr": 0.0097, "batch_size": 24},
    {"variant_id": "efficientnet_02", "family": "efficientnet", "timm_name": "efficientnet_b2", "params_m": 7.7, "lr": 0.0094, "batch_size": 24},
    {"variant_id": "efficientnet_03", "family": "efficientnet", "timm_name": "efficientnet_b3", "params_m": 10.7, "lr": 0.0091, "batch_size": 16},
    {"variant_id": "efficientnet_04", "family": "efficientnet", "timm_name": "efficientnet_b4", "params_m": 17.55, "lr": 0.0088, "batch_size": 16},
    {"variant_id": "efficientnet_05", "family": "efficientnet", "timm_name": "efficientnet_b5", "params_m": 28.34, "lr": 0.0085, "batch_size": 16},
    {"variant_id": "efficientnet_06", "family": "efficientnet", "timm_name": "tf_efficientnet_b6", "params_m": 40.74, "lr": 0.0082, "batch_size": 10},
    {"variant_id": "efficientnet_07", "family": "efficientnet", "timm_name": "tf_efficientnet_b7", "params_m": 63.79, "lr": 0.0079, "batch_size": 6},
    {"variant_id": "efficientnet_08", "family": "efficientnet", "timm_name": "efficientnetv2_rw_s", "params_m": 22.15, "lr": 0.0076, "batch_size": 16},
    {"variant_id": "efficientnet_09", "family": "efficientnet", "timm_name": "efficientnetv2_rw_m", "params_m": 51.09, "lr": 0.0073, "batch_size": 10},
    {"variant_id": "mobilenet_00", "family": "mobilenet", "timm_name": "mobilenetv2_050", "params_m": 0.69, "lr": 0.01, "batch_size": 24},
    {"variant_id": "mobilenet_01", "family": "mobilenet", "timm_name": "mobilenetv2_100", "params_m": 2.23, "lr": 0.0097, "batch_size": 24},
    {"variant_id": "mobilenet_02", "family": "mobilenet", "timm_name": "mobilenetv2_140", "params_m": 4.32, "lr": 0.0094, "batch_size": 24},
    {"variant_id": "mobilenet_03", "family": "mobilenet", "timm_name": "mobilenetv3_small_050", "params_m": 0.57, "lr": 0.0091, "batch_size": 24},
    {"variant_id": "mobilenet_04", "family": "mobilenet", "timm_name": "mobilenetv3_small_100", "params_m": 1.52, "lr": 0.0088, "batch_size": 24},
    {"variant_id": "mobilenet_05", "family": "mobilenet", "timm_name": "mobilenetv3_large_100", "params_m": 4.2, "lr": 0.0085, "batch_size": 24},
    {"variant_id": "mobilenet_06", "family": "mobilenet", "timm_name": "mobilenetv3_large_150d", "params_m": 13.34, "lr": 0.0082, "batch_size": 16},
    {"variant_id": "mobilenet_07", "family": "mobilenet", "timm_name": "mobilenetv4_conv_small", "params_m": 2.49, "lr": 0.0079, "batch_size": 24},
    {"variant_id": "mobilenet_08", "family": "mobilenet", "timm_name": "mobilenetv4_conv_medium", "params_m": 8.44, "lr": 0.0076, "batch_size": 24},
    {"variant_id": "mobilenet_09", "family": "mobilenet", "timm_name": "mobilenetv4_conv_large", "params_m": 31.31, "lr": 0.0073, "batch_size": 10},
    {"variant_id": "regnet_00", "family": "regnet", "timm_name": "regnety_002", "params_m": 2.79, "lr": 0.01, "batch_size": 24},
    {"variant_id": "regnet_01", "family": "regnet", "timm_name": "regnety_004", "params_m": 3.9, "lr": 0.0097, "batch_size": 24},
    {"variant_id": "regnet_02", "family": "regnet", "timm_name": "regnety_008", "params_m": 5.49, "lr": 0.0094, "batch_size": 24},
    {"variant_id": "regnet_03", "family": "regnet", "timm_name": "regnety_016", "params_m": 10.31, "lr": 0.0091, "batch_size": 16},
    {"variant_id": "regnet_04", "family": "regnet", "timm_name": "regnety_032", "params_m": 17.92, "lr": 0.0088, "batch_size": 16},
    {"variant_id": "regnet_05", "family": "regnet", "timm_name": "regnety_040", "params_m": 19.56, "lr": 0.0085, "batch_size": 16},
    {"variant_id": "regnet_06", "family": "regnet", "timm_name": "regnety_064", "params_m": 29.29, "lr": 0.0082, "batch_size": 16},
    {"variant_id": "regnet_07", "family": "regnet", "timm_name": "regnety_080", "params_m": 37.17, "lr": 0.0079, "batch_size": 10},
    {"variant_id": "regnet_08", "family": "regnet", "timm_name": "regnety_120", "params_m": 49.58, "lr": 0.0076, "batch_size": 10},
    {"variant_id": "regnet_09", "family": "regnet", "timm_name": "regnety_160", "params_m": 80.57, "lr": 0.0073, "batch_size": 6},
    {"variant_id": "convnext_00", "family": "convnext", "timm_name": "convnext_atto", "params_m": 3.37, "lr": 0.01, "batch_size": 24},
    {"variant_id": "convnext_01", "family": "convnext", "timm_name": "convnext_femto", "params_m": 4.83, "lr": 0.0097, "batch_size": 24},
    {"variant_id": "convnext_02", "family": "convnext", "timm_name": "convnext_pico", "params_m": 8.53, "lr": 0.0094, "batch_size": 24},
    {"variant_id": "convnext_03", "family": "convnext", "timm_name": "convnext_nano", "params_m": 14.95, "lr": 0.0091, "batch_size": 16},
    {"variant_id": "convnext_04", "family": "convnext", "timm_name": "convnext_tiny", "params_m": 27.82, "lr": 0.0088, "batch_size": 16},
    {"variant_id": "convnext_05", "family": "convnext", "timm_name": "convnext_small", "params_m": 49.46, "lr": 0.0085, "batch_size": 10},
    {"variant_id": "convnext_06", "family": "convnext", "timm_name": "convnext_base", "params_m": 87.57, "lr": 0.0082, "batch_size": 6},
    {"variant_id": "convnext_07", "family": "convnext", "timm_name": "convnext_large", "params_m": 196.23, "lr": 0.0079, "batch_size": 4},
    {"variant_id": "convnext_08", "family": "convnext", "timm_name": "convnextv2_tiny", "params_m": 27.87, "lr": 0.0076, "batch_size": 16},
    {"variant_id": "convnext_09", "family": "convnext", "timm_name": "convnextv2_base", "params_m": 87.69, "lr": 0.0073, "batch_size": 6},
    {"variant_id": "vit_00", "family": "vit", "timm_name": "vit_tiny_patch16_224", "params_m": 5.52, "lr": 0.01, "batch_size": 24},
    {"variant_id": "vit_01", "family": "vit", "timm_name": "vit_small_patch16_224", "params_m": 21.67, "lr": 0.0097, "batch_size": 16},
    {"variant_id": "vit_02", "family": "vit", "timm_name": "vit_small_patch32_224", "params_m": 22.49, "lr": 0.0094, "batch_size": 16},
    {"variant_id": "vit_03", "family": "vit", "timm_name": "vit_base_patch16_224", "params_m": 85.8, "lr": 0.0091, "batch_size": 6},
    {"variant_id": "vit_04", "family": "vit", "timm_name": "vit_base_patch32_224", "params_m": 87.46, "lr": 0.0088, "batch_size": 6},
    {"variant_id": "vit_05", "family": "vit", "timm_name": "vit_base_patch16_224_miil", "params_m": 85.77, "lr": 0.0085, "batch_size": 6},
    {"variant_id": "vit_06", "family": "vit", "timm_name": "vit_large_patch16_224", "params_m": 303.3, "lr": 0.0082, "batch_size": 4},
    {"variant_id": "vit_07", "family": "vit", "timm_name": "vit_medium_patch16_gap_240", "params_m": 38.33, "lr": 0.0079, "batch_size": 10},
    {"variant_id": "vit_08", "family": "vit", "timm_name": "vit_relpos_small_patch16_224", "params_m": 21.6, "lr": 0.0076, "batch_size": 16},
    {"variant_id": "vit_09", "family": "vit", "timm_name": "vit_relpos_medium_patch16_224", "params_m": 38.23, "lr": 0.0073, "batch_size": 10},
    {"variant_id": "swin_00", "family": "swin", "timm_name": "swin_tiny_patch4_window7_224", "params_m": 27.52, "lr": 0.01, "batch_size": 16},
    {"variant_id": "swin_01", "family": "swin", "timm_name": "swin_small_patch4_window7_224", "params_m": 48.84, "lr": 0.0097, "batch_size": 10},
    {"variant_id": "swin_02", "family": "swin", "timm_name": "swin_base_patch4_window7_224", "params_m": 86.74, "lr": 0.0094, "batch_size": 6},
    {"variant_id": "swin_03", "family": "swin", "timm_name": "swin_base_patch4_window12_384", "params_m": 86.88, "lr": 0.0091, "batch_size": 6},
    {"variant_id": "swin_04", "family": "swin", "timm_name": "swin_large_patch4_window7_224", "params_m": 195.0, "lr": 0.0088, "batch_size": 4},
    {"variant_id": "swin_05", "family": "swin", "timm_name": "swin_large_patch4_window12_384", "params_m": 195.2, "lr": 0.0085, "batch_size": 4},
    {"variant_id": "swin_06", "family": "swin", "timm_name": "swin_s3_tiny_224", "params_m": 27.56, "lr": 0.0082, "batch_size": 16},
    {"variant_id": "swin_07", "family": "swin", "timm_name": "swin_s3_small_224", "params_m": 48.97, "lr": 0.0079, "batch_size": 10},
    {"variant_id": "swin_08", "family": "swin", "timm_name": "swin_s3_base_224", "params_m": 70.36, "lr": 0.0076, "batch_size": 6},
    {"variant_id": "swin_09", "family": "swin", "timm_name": "swin_tiny_patch4_window7_224", "params_m": 27.52, "lr": 0.00365, "batch_size": 16},
]

MODEL_CONFIGS_BY_ID: dict[str, dict] = {c["variant_id"]: c for c in MODEL_CONFIGS}
FAMILIES: list[str] = sorted({c["family"] for c in MODEL_CONFIGS})


def get_config(variant_id: str) -> dict:
    if variant_id not in MODEL_CONFIGS_BY_ID:
        raise ValueError(f"Unknown variant_id: {variant_id!r}. See roster_cls.MODEL_CONFIGS.")
    return MODEL_CONFIGS_BY_ID[variant_id]


def gradcam_strategy(family: str) -> str:
    return GRADCAM_FAMILY_TYPE[family]


if __name__ == "__main__":
    assert len(MODEL_CONFIGS) == 100
    assert len(FAMILIES) == 10
    for fam in FAMILIES:
        n = sum(1 for c in MODEL_CONFIGS if c["family"] == fam)
        assert n == 10, f"{fam} has {n} variants, expected 10"
    print(f"{len(MODEL_CONFIGS)} configs across {len(FAMILIES)} families, all OK")
