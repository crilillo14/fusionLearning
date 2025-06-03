
from torchvision.transforms import functional as F
from torchvision.transforms import v2
from PIL import Image

transforms = v2.Compose([
    v2.ToImage(),
    v2.ToTensor(),
])

