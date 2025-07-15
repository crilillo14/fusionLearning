"""

Geometric and photometric transform pipelines using torchvision v2.

Tweak as needed. Don't want to overdo it and take out birds out the image.

~ Only apply g transform to mask, image geo + photometric.

"""




from torchvision.transforms import InterpolationMode, v2
from PIL import Image
import torch
import random


# only needed for compiled models.
# decision discontinued for now. Will be reimplemented for H100 training.

def crop_to_multiple(img: Image.Image, multiple : int = 32) -> Image.Image:
    """
    Crop image to make its dimensions divisible by multiple (32 for compile models)
    Args:
        img: torchvision datapoint Image
        multiple: The number to make dimensions divisible by
    Returns:
        Cropped torchvision datapoint Image
    """
    w, h = img.size
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    right = left + new_w
    bottom = top + new_h
    
    return img.crop((left, top, right, bottom))


# edited to be even more subtle...


geoTransforms = v2.Compose([
    crop_to_multiple,
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
    v2.RandomPerspective(distortion_scale=0.05, p=0.05,
                         interpolation=InterpolationMode.BILINEAR),
    v2.RandomPosterize(bits=2, p=0.05)
])

# Mask-specific geometric transforms (use nearest-neighbour interpolation to preserve class labels)
maskGeoTransforms = v2.Compose([
    crop_to_multiple,
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(5, interpolation=InterpolationMode.NEAREST),
    v2.RandomPerspective(distortion_scale=0.05, p=0.05,
                         interpolation=InterpolationMode.NEAREST),
])

# ----------------------------------------------------------------------------------
# Functional pair transform that guarantees identical geometry with different modes
from torchvision.transforms.functional import rotate, hflip, vflip, perspective

def _get_perspective_params(width: int, height: int, distortion_scale: float = 0.05):
    """Mimic torchvision RandomPerspective param generator."""
    half_w = distortion_scale * width / 2
    half_h = distortion_scale * height / 2
    tl = (random.uniform(0, half_w), random.uniform(0, half_h))
    tr = (random.uniform(width - half_w, width), random.uniform(0, half_h))
    bl = (random.uniform(0, half_w), random.uniform(height - half_h, height))
    br = (random.uniform(width - half_w, width), random.uniform(height - half_h, height))
    startpoints = [tl, tr, br, bl]
    endpoints   = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]
    return startpoints, endpoints

def geom_transform_pair(img, mask, degrees: float = 5, perspective_scale: float = 0.05, p_persp: float = 0.05):
    """Apply the same random geometric ops to img & mask with appropriate interpolation mode."""
    # horizontal flip
    if torch.rand(()) < 0.5:
        img = hflip(img)
        mask = hflip(mask)

    # vertical flip
    if torch.rand(()) < 0.5:
        img = vflip(img)
        mask = vflip(mask)

    # rotation
    angle = torch.empty(1).uniform_(-degrees, degrees).item()
    img = rotate(img, angle, interpolation=InterpolationMode.BILINEAR, expand=False)
    mask = rotate(mask, angle, interpolation=InterpolationMode.NEAREST, expand=False)

    # perspective
    if torch.rand(()) < p_persp:
        startpts, endpts = _get_perspective_params(img.size[0], img.size[1], distortion_scale=perspective_scale)
        img = perspective(img, startpts, endpts, InterpolationMode.BILINEAR)
        mask = perspective(mask, startpts, endpts, InterpolationMode.NEAREST)

    # crop to multiple of 32
    img = crop_to_multiple(img)
    mask = crop_to_multiple(mask)
    return img, mask

photometricTransforms = v2.Compose([
    v2.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.001),
    v2.RandomGrayscale(p=0.01),
    v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.01),
    v2.RandomAutocontrast(p=0.01),
    v2.RandomEqualize(p=0.01)
])