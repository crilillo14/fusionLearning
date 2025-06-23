"""

Geometric and photometric transform pipelines using torchvision v2.

Tweak as needed. Don't want to overdo it and take out birds out the image.

~ Only apply g transform to mask, image geo + photometric.

"""




from torchvision.transforms import InterpolationMode, v2
from PIL import Image


# only needed for compiled models.
# decision discontinued for now. Will be reimplemented for H100 training.

def pad_to_multiple(img: Image.Image, multiple : int = 32, fill : int = 0) -> Image.Image:
    """
    Pad image to make its dimensions divisible by multiple (32 for compile models)
    Args:
        img: torchvision datapoint Image
        multiple: The number to make dimensions divisible by
        fill: Fill value for padding
    Returns:
        Padded torchvision datapoint Image
    """
    w, h = img.size
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    
    padding = (0, 0, new_w - w, new_h - h)
    return v2.Pad(padding, fill=fill)(img)


# edited to be even more subtle...


geoTransforms = v2.Compose([
    pad_to_multiple,
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(5, interpolation=InterpolationMode.BILINEAR),
    v2.RandomPerspective(distortion_scale=0.05, p=0.05,
                         interpolation=InterpolationMode.BILINEAR),
    v2.RandomPosterize(bits=2, p=0.05)
])


photometricTransforms = v2.Compose([
    v2.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.001),
    v2.RandomGrayscale(p=0.01),
    v2.RandomAdjustSharpness(sharpness_factor=1.2, p=0.01),
    v2.RandomAutocontrast(p=0.01),
    v2.RandomEqualize(p=0.01)
])