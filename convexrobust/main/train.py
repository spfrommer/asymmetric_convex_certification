import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import LightningDataModule

from convexrobust.model.base_certifiable import BaseCertifiable
from convexrobust.utils import dirs, file_utils, pretty
from convexrobust.utils import torch_utils as TU

from typing import Dict
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """Loaded from the CLI arguments."""
    train: bool = False
    balance: bool = True


@dataclass
class ModelBlueprint:
    model: BaseCertifiable
    epochs: int
    load_model: bool = False
    load_eval_results: bool = False


BlueprintDict = Dict[str, ModelBlueprint]
ModelDict = Dict[str, BaseCertifiable]


def train_models(blueprints: BlueprintDict, *args) -> ModelDict:
    return {
        name: _train_model(name, blueprint, *args) \
        for (name, blueprint) in blueprints.items()
    }


def _train_model(
        name: str, blueprint: ModelBlueprint,
        experiment_directory: str, datamodule: LightningDataModule, config: TrainConfig
    ) -> BaseCertifiable:

    model_root = dirs.out_path(experiment_directory, 'model')
    model_path = dirs.path(model_root, name)
    checkpoint_dir = dirs.path(model_path, 'checkpoints')
    checkpoint_path = dirs.path(checkpoint_dir, 'model.ckpt')

    if blueprint.load_model or not config.train:
        pretty.subsection_print(f'Loading classifier {name}')
        return TU.load_model_from_checkpoint(checkpoint_path, blueprint).to(TU.device())

    pretty.subsection_print(f'Training classifier {name}')
    file_utils.create_empty_directory(model_path)

    model = blueprint.model.to(TU.device())
    model.epochs_n = blueprint.epochs  # Some LR schedulers require number of epochs

    logger = TensorBoardLogger(
        model_root, name, version='tensorboard', default_hp_metric=False
    )

    file_utils.create_empty_directory(checkpoint_dir)
    checkpoint = ModelCheckpoint(checkpoint_dir, filename='model', monitor='val_loss')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=blueprint.epochs, logger=logger, gpus=TU.gpu_n(),
        num_sanity_val_steps=0, callbacks=[checkpoint, lr_monitor]
    )
    trainer.fit(model, datamodule)

    if blueprint.epochs == 0:
        trainer.save_checkpoint(checkpoint_path)  # Useful if loading pretrained model

    # Load model with best validation performance, not just most recent
    model = TU.load_model_from_checkpoint(checkpoint_path, blueprint)

    if config.balance:
        pretty.subsection_print(f'Balancing classifier {name}')
        _balance_classifier(model, datamodule)
        _save_class_balance(model.class_balance.item(), checkpoint_path)

    return model

def _balance_classifier(model: BaseCertifiable, datamodule: LightningDataModule) -> None:
    model.eval().to(TU.device())
    model.class_balance.fill_(0.0)
    TU.check_dataloader_deterministic(datamodule.test_dataloader())
    model.balance(datamodule)

def _save_class_balance(class_balance: float, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path)
    checkpoint['state_dict']['class_balance'].fill_(class_balance)
    torch.save(checkpoint, checkpoint_path)
