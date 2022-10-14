import pytorch_lightning as pl
import torch

from data import nuScenesDataset, CollateFn
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes
from typing import Any, Literal
from model import model_registry


def make_data_loaders(cfg, mode):
    assert mode in ['train', 'val', 'test']
    dataset_kwargs = {}
    dataset_kwargs.update(cfg.model.params)
    dataset_kwargs.update(cfg.data.params)
    dataset_cfg = cfg.data.get(mode, {}).dataset
    nusc = NuScenes(dataset_cfg.version, dataset_cfg.root)
    dataloader_params = cfg.data.get(mode, {}).dataloader
    dataloader_params.shuffle = True
    dataloader_params.pin_memory = True
    return DataLoader(nuScenesDataset(nusc, mode, dataset_kwargs),
                      collate_fn=CollateFn, **dataloader_params)


class LiteModel(pl.LightningModule):
    def __init__(self, cfg, **kwargs):
        super(LiteModel, self).__init__()
        self.cfg = cfg
        model_class = model_registry.get(self.cfg.model.name)
        self.model = model_class(**self.cfg.model.params)

    def training_step(self, batch, batch_idx):
        results = self.model(batch, "train")
        loss = results["loss"].mean()
        for key, val in results.items():
            self.log(key, val, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def train_dataloader(self):
        return make_data_loaders(self.cfg, "train")

    def val_dataloader(self):
        return make_data_loaders(self.cfg, "val")

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int):
        datum = {}
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                datum[key] = val.to(device)
            else:
                datum[key] = val
        return datum

    def configure_optimizers(self, use_pl_optimizer: Literal[True] = True):
        if self.cfg.optim.optimizer != "Adam":
            raise NotImplementedError(f"Unknown type optimizer {self.cfg.optim.name}")
        optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optim.lr)
        return {
            "optimizer": optim,
            "lr_scheduler": torch.optim.lr_scheduler.StepLR(optim, step_size=self.cfg.optim.lr_epoch,
                                                            gamma=self.cfg.optim.lr_decay)
        }
