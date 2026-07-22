import os
import random

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from fusionLearning.models.roster_cls import gradcam_strategy

LABEL_NAMES = {0: "normal", 1: "lesion"}


def _sample_indices(dataset, n):
    total = len(dataset)
    n = min(n, total, 50)
    return random.sample(range(total), n)


def inference_from_paths_cls(model, modelDir, test_dataloader, n=20):
    """
    Samples n random test images, runs the trained model, and saves a grid figure
    with each sample's predicted label/probability and true label - green border
    on correct predictions, red on incorrect. Direct classification analog of
    inference.py's inference_from_paths (no mask to show, so 1 image + caption
    instead of a 3-column image/mask/prediction layout).
    """
    dataset = test_dataloader.dataset
    device = next(model.parameters()).device
    model.eval()

    indices = _sample_indices(dataset, n)
    n = len(indices)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4), squeeze=False)
    axes = axes.reshape(-1)

    for ax_idx, idx in enumerate(indices):
        image, label, filename = dataset[idx]
        with torch.no_grad():
            logit = model(image.unsqueeze(0).to(device))
            prob = torch.sigmoid(logit).item()
        pred = 1 if prob >= 0.5 else 0
        true = int(label.item())
        correct = pred == true

        ax = axes[ax_idx]
        img_np = image.cpu().permute(1, 2, 0).numpy()[..., 0]
        ax.imshow(img_np, cmap="gray")
        color = "seagreen" if correct else "firebrick"
        ax.set_title(f"pred: {LABEL_NAMES[pred]} ({prob:.2f})\ntrue: {LABEL_NAMES[true]}",
                      fontsize=9, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(modelDir, "figures", "inference.png"), dpi=130)
    plt.close(fig)


def _resolve_stage_layers(model, timm_name):
    """
    CNN families: real per-stage feature layers, found by asking timm's
    features_only mode for this architecture's stage module names (reduction>=4
    filters out the stem, keeping the ~4 genuine stages) and resolving those
    dotted names against the actual trained classifier's named_modules(). This
    works generically across resnet/densenet/efficientnet/convnext/etc without
    hardcoding per-family attribute paths, since features_only just hooks
    existing submodules by name rather than restructuring the backbone.
    """
    fm = timm.create_model(timm_name, features_only=True, pretrained=False)
    stage_names = [fi["module"] for fi in fm.feature_info.info if fi["reduction"] >= 4]
    named = dict(model.named_modules())
    return [(name, named[name]) for name in stage_names if name in named]


def _vit_reshape_transform(tensor, num_prefix_tokens=1):
    tensor = tensor[:, num_prefix_tokens:, :]
    side = int(tensor.shape[1] ** 0.5)
    result = tensor.reshape(tensor.shape[0], side, side, tensor.shape[2])
    return result.permute(0, 3, 1, 2)


def _swin_reshape_transform(tensor):
    # timm's SwinTransformerBlock keeps activations as [B, H, W, C] internally
    # (not flattened to [B, N, C] the way ViT does), so no reshape is needed there.
    if tensor.ndim == 4:
        return tensor.permute(0, 3, 1, 2)
    side = int(tensor.shape[1] ** 0.5)
    result = tensor.reshape(tensor.shape[0], side, side, tensor.shape[2])
    return result.permute(0, 3, 1, 2)


def _resolve_final_layer(model, family):
    """
    Transformer families ("transformer_final" in roster_cls.GRADCAM_FAMILY_TYPE):
    only the final block is spatially interpretable, since intermediate blocks
    operate on flattened token sequences rather than spatial grids. Needs a
    reshape_transform to unflatten tokens back into (C, H, W) for Grad-CAM.
    """
    if family == "vit":
        num_prefix = getattr(model, "num_prefix_tokens", 1)
        layer = model.blocks[-1].norm1
        return [("final_block", layer)], (lambda t: _vit_reshape_transform(t, num_prefix))
    if family == "swin":
        layer = model.layers[-1].blocks[-1].norm1
        return [("final_block", layer)], _swin_reshape_transform
    raise ValueError(f"No transformer_final Grad-CAM handling defined for family={family!r}")


def gradcam_from_paths_cls(model, modelDir, test_dataloader, timm_name, family, n=8):
    """
    Generates Grad-CAM explanations for n random test images, saved to
    modelDir/figures/gradcam_{timm_name}.png.

    CNN families ("cnn_staged"): one heatmap column per major stage, giving a
    genuine per-layer progression of what the network attends to with depth.
    Transformer families ("vit"/"swin", "transformer_final"): a single column at
    the final block only - see roster_cls.GRADCAM_FAMILY_TYPE and
    notes/tompei_cmmd_classification_spec.md section 5 for the "when applicable"
    rationale.
    """
    strategy = gradcam_strategy(family)
    device = next(model.parameters()).device
    model.eval()

    reshape_transform = None
    if strategy == "cnn_staged":
        layers = _resolve_stage_layers(model, timm_name)
    else:
        layers, reshape_transform = _resolve_final_layer(model, family)

    if not layers:
        print(f"gradcam_from_paths_cls: no target layers resolved for {timm_name}, skipping.")
        return

    dataset = test_dataloader.dataset
    indices = _sample_indices(dataset, n)
    n = len(indices)
    n_cols = len(layers)

    samples = []
    for idx in indices:
        image, label, filename = dataset[idx]
        input_tensor = image.unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(input_tensor)).item()
        pred = 1 if prob >= 0.5 else 0
        samples.append({
            "input_tensor": input_tensor,
            "rgb_img": image.permute(1, 2, 0).cpu().numpy(),
            "pred": pred,
            "true": int(label.item()),
        })

    fig, axes = plt.subplots(n, n_cols, figsize=(n_cols * 3, n * 3), squeeze=False)

    # Grad-CAM needs gradients w.r.t. activations, so this must run outside
    # torch.no_grad() even though the model is in .eval() mode.
    for col, (stage_name, layer) in enumerate(layers):
        with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape_transform) as cam:
            for row, s in enumerate(samples):
                target = [BinaryClassifierOutputTarget(s["pred"])]
                grayscale_cam = cam(input_tensor=s["input_tensor"], targets=target)[0]
                overlay = show_cam_on_image(s["rgb_img"], grayscale_cam, use_rgb=True)

                ax = axes[row, col]
                ax.imshow(overlay)
                if row == 0:
                    ax.set_title(stage_name, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"true={LABEL_NAMES[s['true']]}\npred={LABEL_NAMES[s['pred']]}",
                                    fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(modelDir, "figures", f"gradcam_{timm_name}.png"), dpi=110)
    plt.close(fig)
