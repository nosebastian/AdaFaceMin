from typing import Literal
from lightning.pytorch import LightningModule
import torch
from torch import Tensor
import torch.optim  as optim
import torch.optim.lr_scheduler as scheduler
import torch.nn as nn
import pandas as pd
import numpy as np
from ..backbone import build_model
from ..loss import build_head
from ..metric import LFWFlavourAccuracy

__all__: list[str] = ['FaceRecognitionModel']

class FaceRecognitionModel(LightningModule):
    def __init__(
        self,
        model_name:Literal['ir_101', 'ir_50', 'ir_se_50', 'ir_34', 'ir_18'],
        head_type:Literal['arcface', 'cosface', 'adafacenet'],
        class_num:int=70722,
        m:float=0.4,
        t_alpha:float=0.333,
        h:float=64.,
        s:float=1.0,
        optimizer:str = 'SGD',
        optimizer_kwargs:dict[str, float] = {'lr':0.1, 'momentum':0.9, 'weight_decay':5e-4},
        ls_scheduler:str|None = None,
        ls_scheduler_kwargs:dict[str, float] = {},
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(model_name)
        self.head = build_head(head_type, 512, class_num, m, t_alpha, h, s)
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        self.ls_scheduler = ls_scheduler
        self.ls_scheduler_kwargs = ls_scheduler_kwargs
        self.criterion = nn.CrossEntropyLoss()
        
        self.test_accuracy = LFWFlavourAccuracy()
        
    def forward(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        embeddings, norms = self.model(images)
        cos_theta_m = self.head(embeddings, norms, labels)
        if isinstance(cos_theta_m, tuple):
            cos_theta_m, bad_grad = cos_theta_m
            labels[bad_grad.squeeze(-1)] = -100 # ignore_index
        return cos_theta_m, norms, embeddings, labels

    def training_step(self, batch, batch_idx):
        images, labels = batch
        cos_thetas, norms, embeddings, labels = self.forward(images, labels)
        loss_train = self.criterion(cos_thetas, labels)
        self.log('train_loss', loss_train, on_step=True, on_epoch=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        cos_thetas, norms, embeddings, labels = self.forward(images, labels)
        loss_val = self.criterion(cos_thetas, labels)
        self.log('val_loss', loss_val, on_step=True, on_epoch=True)
        return loss_val

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        embeddings, norms = self.model(images)
        self.test_accuracy.update(embeddings, labels['index'], labels['pair_label'])
        self.log('test_accuracy', self.test_accuracy, on_step=False, on_epoch=True, rank_zero_only=True)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer)(self.parameters(), **self.optimizer_kwargs)
        if self.ls_scheduler is not None:
            scheduler = getattr(scheduler, self.ls_scheduler)(optimizer, **self.ls_scheduler_kwargs)
            return [optimizer], [scheduler]
        return optimizer
