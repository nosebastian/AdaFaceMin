from pathlib import Path
from lightning import LightningDataModule
from torchvision.datasets import ImageFolder, VisionDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from ..augmentation import RandomResolution, RandomKeep, RandomPhotometric, RGBToBGRTransform
from ..dataset import LFWFlavourDataset

__all__: list[str] = ["SampleDataModule"]

class SampleDataModule(LightningDataModule):
    def __init__(
        self, 
        train_root: str | Path,
        val_root: str | Path,
        test_root: str | Path,
        test_sets: list[str] = [
            'cfp_ff', 'vgg2_fp', 
            'lfw', 'agedb_30', 
            'cfp_fp', 'calfw'
        ],
        num_workers: int = 4,
        batch_size: int = 32,
    ):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.test_root = test_root
        self.test_sets = test_sets
        self.num_workers = num_workers
        self.batch_size = batch_size

    def prepare_data(self):
        # download, split, etc...
        pass

    def setup(self, stage=None):
        # build the dataset
        default_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            RGBToBGRTransform()
        ])
            
        train_transform = transforms.Compose([
            transforms.RandomApply([RandomResolution()], p=0.5),
            transforms.RandomApply([RandomKeep()], p=0.5),
            transforms.RandomApply([RandomPhotometric()], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            RGBToBGRTransform()
        ])
        
        self.train_dataset: VisionDataset = ImageFolder(self.train_root, allow_empty=True, transform=train_transform)
        self.val_dataset: VisionDataset = ImageFolder(self.val_root, allow_empty=True, transform=default_transform)
        self.test_datasets: list[VisionDataset] = [LFWFlavourDataset(self.test_root, set, transform=default_transform) for set in self.test_sets]

    def train_dataloader(self) -> DataLoader:
        # return a DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        # return a DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self)  -> list[DataLoader]:
        # return a DataLoader
        return [DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers) for ds in self.test_datasets]
