"""

Geometric and photometric transform pipelines using torchvision v2.

Tweak as needed. Don't want to overdo it and take out birds out the image.

~ Only apply g transform to mask, image geo + photometric.

"""




from torchvision.transforms import functional as F
from torchvision.transforms import v2
from PIL import Image

geoTransforms = v2.Compose([
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomRotation(10),
    v2.RandomPerspective(distortion_scale=0.1, p=0.1),
    v2.RandomPosterize(bits=1, p=0.1),

])

photometricTransforms = v2.Compose([
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomGrayscale(p=0.1),
    v2.RandomAdjustSharpness(sharpness_factor=2.0, p=0.1),
    v2.RandomAutocontrast(p=0.1),
    v2.RandomEqualize(p=0.1),

])