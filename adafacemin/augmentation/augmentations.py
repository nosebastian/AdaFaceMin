from typing import Tuple
import torch
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

__all__ = ["RandomResolution", "RandomKeep", "RandomPhotometric"]

class RandomResolution(torch.nn.Module):
    def __init__(self, scale:tuple[float, float]=(0.2, 1.0)):
        super().__init__()
        self.scale = scale
        
    @staticmethod
    def get_params(img, scale):
        aug_set = list(set(F.InterpolationMode)-{'hamming'})
        width, height = F.get_image_size(img)
        side_ratio = np.random.uniform(scale[0], scale[1])
        h = int(height * side_ratio)
        w = int(width * side_ratio)
        di = np.random.choice(aug_set)
        ui = np.random.choice(aug_set)
        aa = np.random.choice([True, False])
        return h, w, di, ui, aa
        
    def forward(self, img: torch.Tensor | Image.Image) -> torch.Tensor:
        width, height = F.get_image_size(img) 
        h, w, di, ui, aa = self.get_params(img, self.scale)
        downsampled = F.resize(img, (h, w), interpolation=di, antialias=aa)
        upsampled = F.resize(downsampled, (height, width), interpolation=ui, antialias=aa)
        return upsampled
    
class RandomKeep(transforms.RandomResizedCrop):
    def __init__(self, scale=(0.2, 1.0), ratio=(0.75, 1.3333333333333333), fill=0, padding_mode="constant"):
        super().__init__(100, scale, ratio, InterpolationMode.BILINEAR, False)
        self.fill = fill
        self.padding_mode = padding_mode
        
    @staticmethod
    def get_params(img, scale, ratio):
        return transforms.RandomResizedCrop.get_params(img, scale, ratio)
        
    def forward(self, img: torch.Tensor | Image.Image) -> torch.Tensor:
        width, height = F.get_image_size(img) 
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        croped = F.crop(img, i, j, h, w)
        padded = F.pad(croped, (j, i, width - j - w, height - i - h), self.fill, self.padding_mode)
        return padded

class RandomPhotometric(transforms.ColorJitter):
    def __init__(self, brightness: float | Tuple[float, float] = 0.5, contrast: float | Tuple[float, float] = 0.5, saturation: float | Tuple[float, float] = 0.5) -> None:
        super().__init__(brightness, contrast, saturation, 0)
        
    def forward(self, img: torch.Tensor | Image.Image) -> torch.Tensor:
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast,
                            self.saturation, self.hue)
        for fn_id in fn_idx.cpu().numpy():
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)

        return img