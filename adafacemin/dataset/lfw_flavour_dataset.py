from enum import Enum
from torchvision.datasets import VisionDataset
from pathlib import Path
from typing import Callable, Literal
from PIL import Image

__all__: list[str] = ['LFWFlavourDataset']


class LFWFlavourDataset(VisionDataset):
    def __init__(
        self, 
        root: str | Path, 
        dataset: Literal['cfp_ff', 'vgg2_fp', 'lfw', 'agedb_30', 'cfp_fp', 'calfw', 'cplfw'] | str,
        transform:Callable=None, 
        target_transform:Callable=None
    ) -> None:
        super(LFWFlavourDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.root: Path = root if isinstance(root, Path) else Path(root)
        self.dataset = dataset
        self.items = list((self.root / self.dataset / 'images').rglob(f'*.jpg'))
        self.items: list[Path] = sorted(self.items, key=lambda x: int(x.stem))
        self.key: list[str] = (self.root / self.dataset / 'key.csv').read_text().strip().split('\n')[1:]
        self.key = [line.split(',')[-1].strip() for line in self.key]
        self.key = [int(line == 'True' or line == '1')  for line in self.key]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, index):
        image_path = self.items[index]
        image = Image.open(image_path)
        label = {
            'index' : index,
            'pair_label' : self.key[index//2],
            'dataset' : self.dataset,
        }
        if self.transforms is not None:
            image, label = self.transforms(image, label)
        return image, label
