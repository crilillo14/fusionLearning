"""
Mask2Former for semantic segmentation via HuggingFace transformers.

Mask2Former uses a Swin Transformer backbone + pixel decoder + masked-attention
transformer decoder. Its native output is a set of (mask, class) query pairs
rather than a dense [B, C, H, W] logit map.

We convert the query output to a standard dense logit map for compatibility
with CrossEntropyLoss:

    seg_logits[b, c, h, w] = Σ_q  class_logits[b, q, c]  *  mask_logits[b, q, h, w]

This is a linear combination over queries, keeping everything in raw logit
space so CrossEntropyLoss can apply log_softmax as usual.

Interface matches the SMP arch_dict convention:
    model = Mask2FormerSegmentation(classes=21)
    logits = model(x)  # [B, num_classes, H, W]

All weights are randomly initialized — Mask2FormerConfig() with no pretrained
checkpoint. If you want to load a pretrained backbone later, swap the config
for a from_pretrained call and re-randomize the head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation


class Mask2FormerSegmentation(nn.Module):
    """
    Mask2Former for semantic segmentation.

    Args:
        classes:  Number of output segmentation classes.
        **kwargs: Absorbs SMP-style params (encoder_name, encoder_weights,
                  in_channels) so this can sit in arch_dict unchanged.
    """

    def __init__(self, classes: int, **kwargs):
        super().__init__()
        config = Mask2FormerConfig(
            num_labels=classes,
            # Swin-Tiny backbone (smallest available) for reasonable memory use
            backbone_config={
                "_target_": "transformers.SwinConfig",
                "image_size": 224,
                "in_channels": 3,
                "patch_size": 4,
                "embed_dim": 96,
                "depths": [2, 2, 6, 2],
                "num_heads": [3, 6, 12, 24],
                "window_size": 7,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            },
            # Pixel decoder
            feature_size=256,
            mask_feature_size=256,
            # Transformer decoder
            hidden_dim=256,
            num_queries=100,
            encoder_layers=6,
            decoder_layers=10,
            # Disable auxiliary losses — only need final output for benchmarking
            use_auxiliary_loss=False,
        )
        # From-config init → all weights random, no pretrained checkpoint
        self.model = Mask2FormerForUniversalSegmentation(config)
        self.num_classes = classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        out = self.model(pixel_values=x)

        # masks_queries_logits : [B, num_queries, H/4, W/4]
        # class_queries_logits : [B, num_queries, num_classes + 1]
        #   (last dim is the "no-object" class — drop it for semantic seg)
        mask_logits  = out.masks_queries_logits                    # [B, Q, h, w]
        class_logits = out.class_queries_logits[..., :-1]          # [B, Q, C]

        # Upsample masks to input resolution
        mask_logits = F.interpolate(
            mask_logits, size=(H, W), mode="bilinear", align_corners=False
        )  # [B, Q, H, W]

        # Dense logit map: linear combination of query masks weighted by class scores
        # Stays in raw logit space → compatible with CrossEntropyLoss
        seg_logits = torch.einsum("bqc,bqhw->bchw", class_logits, mask_logits)
        # [B, num_classes, H, W]

        return seg_logits


if __name__ == "__main__":
    model = Mask2FormerSegmentation(classes=21)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape: {out.shape}")   # (2, 21, 512, 512)
    print(f"Params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
