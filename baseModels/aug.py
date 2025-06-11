"""

Geometric and photometric transform pipelines using torchvision v2.

Tweak as needed. Don't want to overdo it and take out birds out the image.

~ Only apply g transform to mask, image geo + photometric.

"""




from torchvision.transforms import functional as F
from torchvision.transforms import v2
from PIL import Image
import torch

def pad_to_multiple(img, multiple=32, fill=0):
    """
    Pad image to make its dimensions divisible by multiple
    Args:
        img: PIL Image or Tensor
        multiple: The number to make dimensions divisible by
        fill: Fill value for padding
    Returns:
        Padded image
    """
    if isinstance(img, Image.Image):
        w, h = img.size
    else:  # tensor
        h, w = img.shape[-2:]
    
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    
    padding = (0, 0, new_w - w, new_h - h)
    return F.pad(img, padding, fill=fill)

geoTransforms = v2.Compose([
    pad_to_multiple,
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(10),
    v2.RandomPerspective(distortion_scale=0.1, p=0.1),
    v2.RandomPosterize(bits=1, p=0.1)
])

photometricTransforms = v2.Compose([
    pad_to_multiple,
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomGrayscale(p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.1),
    v2.RandomAutocontrast(p=0.1),
    v2.RandomEqualize(p=0.1),

])