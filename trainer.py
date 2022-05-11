from typing import List, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from sklearn.metrics import accuracy_score, f1_score
from torch import nn, optim

from src.datamodule import CheXpertDataModule
from src.model import CheXpertModel


class CheXpertLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_alias: str,
        feature_size: int,
        output_size: int,
        learning_rate: float = 1e-3,
        hidden_layers: List[int] = [256, 32],
        pretrained=True,
    ):
        super().__init__()
        self.model = CheXpertModel(
            model_alias=model_alias,
            feature_size=feature_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            pretrained=pretrained,
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
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.log("val/loss", loss.mean().cpu())

        return y_hat

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        predictions = (outputs > 0.5).cpu().long().numpy()
        _, y = batch
        y = y.cpu().numpy()

        self.log(
            "val/accuracy", accuracy_score(y, predictions), on_step=False, on_epoch=True
        )

        self.log(
            "val/f1_score",
            f1_score(y, predictions, average="macro"),
            on_step=False,
            on_epoch=True,
        )


if __name__ == "__main__":
    class_names = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    ]

    model = CheXpertLightningModule(
        model_alias="cspresnet50", feature_size=1024, output_size=14, pretrained=False
    )

    datamodule = CheXpertDataModule(
        "../CheXpert-v1.0-small/",
        batch_size=32,
        img_size=(320, 320),
        img_crop=224,
        class_names=class_names,
    )
    log_callback = ModelCheckpoint()
    logger = WandbLogger(log_model="all")
    trainer = pl.Trainer(
        logger,
        callbacks=[log_callback],
        accelerator="gpu",
        gpus=1,
        profiler="simple",
        log_every_n_steps=1,
        max_epochs=50,
    )
    trainer.fit(model, datamodule)
