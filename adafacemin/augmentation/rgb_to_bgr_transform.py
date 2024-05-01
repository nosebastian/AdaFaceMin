import torch

__all__ = ["RGBToBGRTransform"]

class RGBToBGRTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image[[2,1,0], :, :]
