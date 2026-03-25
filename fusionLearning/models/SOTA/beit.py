"""
BeiT for semantic segmentation via HuggingFace transformers.

Uses BeitForSemanticSegmentation (UperNet decoder on top of BeiT encoder).
All weights are randomly initialized — no pretrained checkpoint loaded.

Interface matches the SMP arch_dict convention:
    model = BeiT3Segmentation(classes=21)
    logits = model(x)  # [B, num_classes, H, W]

The HuggingFace model outputs at 1/4 resolution; we bilinearly upsample
back to the input resolution in forward().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BeitConfig, BeitForSemanticSegmentation


class BeiT3Segmentation(nn.Module):
    """
    BeiT encoder + UperNet segmentation head.

    Args:
        classes:    Number of output segmentation classes.
        img_size:   Expected input spatial size (BeiT uses fixed patch grid).
        **kwargs:   Absorbs SMP-style params (encoder_name, encoder_weights,
                    in_channels) so this can sit in arch_dict unchanged.
    """

    def __init__(self, classes: int, img_size: int = 512, **kwargs):
        super().__init__()
        config = BeitConfig(
            # Encoder — BeiT-base scale
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            image_size=img_size,
            patch_size=16,
            # Segmentation head
            num_labels=classes,
            semantic_loss_ignore_index=255,
            # Disable auxiliary head to keep forward output clean
            use_auxiliary_head=False,
            out_indices=[3, 5, 7, 11],  # UperNet multi-scale features
        )
        # Instantiated from config → random init, no pretrained weights
        self.model = BeitForSemanticSegmentation(config)
        self.num_classes = classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        out = self.model(pixel_values=x)
        # logits: [B, num_classes, H/4, W/4]  (UperNet output before final upsample)
        logits = out.logits
        # Upsample to input resolution
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = BeiT3Segmentation(classes=21)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")   # (2, 21, 512, 512)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
