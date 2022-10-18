import os
from galaxy_mnist import GalaxyMNIST

import torch
from torch import optim, nn, utils, Tensor
import torchvision.models as models
from torchmetrics.functional import accuracy
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl
from pl_bolts.models.autoencoders.base_vae import VAE
from pytorch_lightning.loggers import WandbLogger
import wandb

class DataModule(pl.LightningDataModule):
    def __init__(self, transform, config):
        super().__init__()
        self.transform = transform
        self.config = config
        self.data_dir = self.config['data_dir']
        self.batch_size = self.config['batch_size']
        self.data_size = self.config['data_size']
        self.no_transform = A.ToFloat()

    def prepare_data(self):
        # download
        GalaxyMNIST(self.data_dir, train=True, download=True)
        GalaxyMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage = None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            gmnist = GalaxyMNIST(self.data_dir, train=True, transform=self.no_transform)
            train_size = int(7000*self.data_size) # 8000 total in original train set
            self.gmnist_train, self.gmnist_val = random_split(gmnist, [train_size, 8000-train_size]) # Random validation split?
            #self.gmnist_train = gmnist
            #self.gmnist_val   = GalaxyMNIST(self.data_dir, train=False, transform=None)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            #self.mnist_test = GalaxyMNIST(self.data_dir, train=False, transform=self.transform) # test on augmented images
            self.gmnist_test = GalaxyMNIST(self.data_dir, train=False, transform=self.no_transform) # test on unaugmented images

        if stage == "predict" or stage is None:
            #self.mnist_predict = GalaxyMNIST(self.data_dir, train=False, transform=self.transform) # predict on augmented images
            self.gmnist_predict = GalaxyMNIST(self.data_dir, train=False, transform=self.no_transform) # predict on unaugmented images

    def train_dataloader(self):
        return DataLoader(self.gmnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.config['num_workers'])

    def val_dataloader(self):
        return DataLoader(self.gmnist_val, batch_size=self.batch_size, num_workers=self.config['num_workers'])

    def test_dataloader(self):
        return DataLoader(self.gmnist_test, batch_size=self.batch_size, num_workers=self.config['num_workers'])

    def predict_dataloader(self):
        return DataLoader(self.gmnist_predict, batch_size=self.batch_size, num_workers=self.config['num_workers'])

# define the LightningModule
# Logging is key for this one, but not included in the pl_bolts.models.autoencoders.base_vae
class Classifier(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        #self.save_hyperparameters(ignore=['model'])
        self.config = config

    def forward(self, x):
        x_tmp = x.permute(0,3,1,2).double()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        # validation metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        x_tmp = x.permute(0,3,1,2).double()
        y_hat = self.model(x_tmp)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        acc = accuracy(y_hat, y)
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])


def main():
    wandb.init()

    ### Eventually Grid search over latent_dim
    latent_dim = 20

    # Model architecture
    if wandb.config['model']=='resnet50': ### This will need to be changed.
        model = models.resnet50()
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=wandb.config["num_classes"],
            bias=True
        )
    else:
        raise NotImplementedError

    # Configure architecture.
    # https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
    vae = VAE(
        input_height=64,
        enc_type: str = "resnet50",
        enc_out_dim: int = 2048,
        kl_coeff: float = 0.1, # coefficient for kl term of the loss
        latent_dim: int = 256,
        lr: float = 1e-4,
        #config=wandb.config
    )
    # Initialise logger
    wandb_logger = WandbLogger(project=wandb.config['project'], job_type='train')
    # Setup trainer
    trainer = pl.Trainer(
        limit_train_batches=None, max_epochs=wandb.config['epochs'],
        logger=wandb_logger,
        accelerator="gpu", gpus=1, num_nodes=1, # since im using slurm
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.EarlyStopping("val_loss", min_delta=0.0001, patience=50),
        ],
        # I think I am double logging as wandb is logging the model weights as well?
        enable_checkpointing = pl.callbacks.ModelCheckpoint(
            monitor='val/loss',
            dirpath='./checkpoints',
            filename='vae_galaxyMNIST-epoch{epoch:02d}-val_loss{val/loss:.2f}',
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=True
        )
    )
    # Fit
    trainer.fit(
        model=vae.double(),
        train_dataloaders=DataModule(transform=transform, config=wandb.config)
    )
    # Test after fitting
    trainer.test(
        dataloaders=DataModule(transform=transform, config=wandb.config),
        ckpt_path="best"
    )


if __name__=="__main__":
    main()
