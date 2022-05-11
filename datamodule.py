from typing import List, Optional, Tuple

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import CheXpertDataSet


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_path: str, 
                 batch_size: int, 
                 img_size: Tuple[int, int], 
                 img_crop: int, 
                 class_names: List[str],
                 num_workers: int = 1
                ):
        super(CheXpertDataModule, self).__init__()
        self.train_path = f"{data_path}/train.csv"
        self.val_path = f"{data_path}/valid.csv"

        self.batch_size = batch_size
        self.img_size = img_size
        self.img_crop = img_crop
        self.class_names = class_names
        self.num_workers = num_workers
    
    def setup_transforms(self):
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
        transform_list = []
        transform_list.append(transforms.RandomResizedCrop(self.img_size))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize)      
        transform_sequence=transforms.Compose(transform_list)

        self.transforms = transform_sequence
    
    def setup(self, stage: Optional[str] = None):
        self.setup_transforms()
        self.train_dataset = CheXpertDataSet(
            self.train_path,
            transform=self.transforms,
            prefix="../"
        )

        self.val_dataset = CheXpertDataSet(
            self.val_path,
            transform=self.transforms,
            prefix="../",
            num_workers=self.num_workers
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
