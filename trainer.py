import pytorch_lightning as pl
from model import CheXpertModel
from datamodule import CheXpertDataModule
from typing import List, Optional
from torch import nn
from torch import optim

class CheXpertLightningModule(pl.LightningModule):
    def __init__(self, 
                 model_alias: str, 
                 feature_size: int,
                 output_size: int,
                 learning_rate: float = 1e-3,
                 hidden_layers: List[int] = [256, 32],
                 pretrained=True
                ):
        super().__init__()
        self.model = CheXpertModel(
            model_alias=model_alias,
            feature_size=feature_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            pretrained=pretrained
        )

        self.lr = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        return loss

if __name__ == "__main__":
    class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    
    model = CheXpertLightningModule(
        model_alias="cspresnet50",
        feature_size=1024,
        output_size=14
    )

    datamodule = CheXpertDataModule(
        "../CheXpert-v1.0-small/",
        batch_size=32,
        img_size=(320, 320),
        img_crop=224,
        class_names=class_names
    )

    trainer = pl.Trainer(accelerator="gpu", gpus=1)
    trainer.fit(model, datamodule)