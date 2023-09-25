from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as dists
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
import torchmetrics
import torchvision

from convexrobust.utils import torch_utils as TU
from convexrobust.model.certificate import Certificate

import lib.smoothingSplittingNoise.src.attacks as rs_attacks

from abc import abstractmethod

from typing import Optional, Union


class BaseCertifiable(pl.LightningModule):
    """Lightning base class for all certificate-generating models.

    All child classes must implement forward, certify, balance, and configure_optimizers.

    The forward and certify methods MUST incorporate the class_balance parameter.

    The configure_optimizers function is a PyTorch Lightning requirement.
    """
    def __init__(
            self, loss, adv_norm: Optional[Union[int, str]]=None,
            adv_eps: Optional[Union[float, list[float]]]=None, stability=False,
        ):
        """
        Args:
            loss: A loss function. Takes in two arguments, predicted logits and target classes.
            adv_norm: 1, 2, 'inf', None. The norm to attack (None if no adversarial training).
            adv_eps: The adversarial perturbation bound. Can either be a constant float or a
                list of floats, one for each epoch (for epochs past the length of the list,
                the last epilon is used)
            stability (bool): Whether to apply stability training (see RS4A).
        """
        super().__init__()

        assert [stability, (adv_norm is not None)].count(True) <= 1

        self.loss = loss
        self.adv_norm = adv_norm
        # If adv_eps is a single float, make a list of length one
        self.adv_eps = [adv_eps] if isinstance(adv_eps, float) else adv_eps

        # Child classes must incorporate class_balance in their forward and certify methods
        self.class_balance = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        # Some methods need to be certified externally via command line (abCROWN)
        self.external_certification = False

        self.stability = stability


    def class_balance_prediction_shift(self) -> Tensor:
        """ A version of class_balance that can be added directly to multiclass output logits. """
        return TU.from_single_logit(self.class_balance)

    def predict(self, signal: Tensor) -> Tensor:
        """Computes the "hard" prediction associated to the "soft" forward call predictions.

        Args:
            signal (Tensor): [batch_n x ...].

        Returns:
            Tensor: [batch_n] long. Predicted classes for each signal in the batch.
        """
        return self.forward(signal).argmax(dim=1)

    def predict_fast_approximate(self, signal: Tensor) -> Tensor:
        """Compute a fast approximation of the hard prediction. Useful for methods that need to be
        iteratively balanced and typically have long prediction times (e.g. randomized smoothing).

        Args:
            signal (Tensor): [batch_n x ...].

        Returns:
            Tensor: [batch_n] long. Predicted classes for each signal in the batch.
        """
        return self.predict(signal)  # By default, just do regular prediction

    def training_signal_modify(self, signal: Tensor) -> Tensor:
        """Potentially modifies the signal that is used for training. Useful for stuff like training
        on noisy data.

        Args:
            signal (Tensor): [batch_n x ...].

        Returns:
            Tensor: [batch_n x ...]. Has the same shape as signal.
        """
        return signal

    def extra_loss(self, signal: Tensor, target: Tensor) -> Tensor:
        """Can add a regularization term to the loss of each batch.

        Args:
            signal (Tensor): [batch_n x ...].
            target (Tensor): [batch_n].

        Returns:
            Tensor: scalar. Added to the loss for the batch.
        """
        return torch.tensor(0.0)


    @abstractmethod
    def forward(self, signal: Tensor) -> Tensor:
        """Computes the prediction output logits over a batch. MUST be appropriately shifted by the
        class balance.

        Args:
            signal (Tensor): [batch_n x ...].

        Returns:
            Tensor: [batch_n x class_n]. Logits for each class.
        """
        pass

    @abstractmethod
    def certify(self, signal: Tensor, target: Tensor) -> tuple[Tensor, Certificate]:
        """Certifies a SINGLE input signal. MUST take into account the class balance.

        Args:
            signal (Tensor): [...]. A single input signal (not batched).
            target (Tensor): [...]. A single target class (not batched). Ignored for most schemes.

        Returns:
            tuple[Tensor, Certificate]: The provable certificate of robustness for the model's
                predictions. on the input. Returned prediction Tensor should be same as that
                returned by predict().
        """
        pass

    @abstractmethod
    def configure_optimizers(self):
        """See PyTorch Lightning documentation."""
        pass

    @abstractmethod
    def balance(self, datamodule: LightningDataModule) -> None:
        """Set class_balance parameter such that the class 0 and class 1 accuracies are the same
        over datamodule.eval_iterator. For preset implementations, see balance.py.

        Args:
            datamodule (LightningDataModule): The datamodule whose eval_iterator should be balanced.
        """
        pass

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict:
        assert self.training
        assert self.class_balance == 0.0 and not self.class_balance.requires_grad

        if self.current_epoch == 0 and batch_idx == 0 and len(batch[0].shape) > 2:
            self.log_images(batch, 'train')
            self.log_attributes()

        return self.compute_step_variables(batch)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> dict:
        assert not self.training

        if self.current_epoch == 0 and batch_idx == 0 and len(batch[0].shape) > 2:
            self.log_images(batch, 'val')

        with torch.no_grad():
            vars = self.compute_step_variables(batch)
            self.log('val_loss', vars['loss']) # For model checkpointing
            return vars

    def compute_step_variables(self, batch: tuple[Tensor, Tensor]) -> dict:
        """Computes a dictionary of output variables for each step. The 'loss' variable is used for
        backpropagation for the training_step. Other returned variables are aggregated in
        log_outputs for debugging at the end of each epoch.
        """
        signal_orig, target = batch[0], batch[1]

        signal = self.training_signal_modify(signal_orig)
        if self.adv_norm is not None:
            t = self.current_epoch
            epsilon = self.adv_eps[t] if t < len(self.adv_eps) else self.adv_eps[-1]
            with torch.enable_grad():
                signal_clone = signal.detach().clone()
                signal, _ = rs_attacks.pgd_attack(
                    TU.LossWrapper(self, self.loss), signal_clone, target,
                    epsilon, adv=self.adv_norm, steps=50
                )
            signal = signal.detach()

        pred = self.forward(signal)

        stability_loss = torch.tensor(0.0)
        if self.stability:
            signal_tilde = self.training_signal_modify(signal_orig)
            pred_tilde = self.forward(signal_tilde)
            pred_forecast = dists.Categorical(logits=pred)
            pred_tilde_forecast = dists.Categorical(logits=pred_tilde)
            stability_loss = 6.0 * dists.kl_divergence(pred_forecast, pred_tilde_forecast).mean()

        vars = {}
        vars['target'] = target
        vars['pred'] = pred.detach()

        classification_loss = self.loss(pred, target)
        extra_loss = self.extra_loss(signal, target)

        vars['class_loss'] = classification_loss.detach()
        vars['stability_loss'] = stability_loss.detach()
        vars['extra_loss'] = extra_loss.detach()
        vars['loss'] = classification_loss + extra_loss + stability_loss

        return vars


    def training_epoch_end(self, outputs: list[dict]) -> None:
        self.log_outputs(outputs, 'train')

    def validation_epoch_end(self, outputs: list[dict]) -> None:
        self.log_outputs(outputs, 'valid')

    def log_outputs(self, outputs: list[dict], stage: str) -> None:
        """Logs statistics from outputs at the end of each epoch."""
        losses = self._calc_means_containing_key('loss', outputs)
        self._add_scalars_single(f'loss_{stage}', losses)

        pred = self._collect_outputs('pred', outputs)
        target = self._collect_outputs('target', outputs)

        kwargs = {'task': 'multiclass', 'num_classes': 2}
        accs = {
            'acc_composite': torchmetrics.functional.accuracy(pred, target, **kwargs),
            'acc_0': torchmetrics.functional.accuracy(pred, target, ignore_index=1, **kwargs),
            'acc_1': torchmetrics.functional.accuracy(pred, target, ignore_index=0, **kwargs)
        }
        self._add_scalars_single(f'acc_{stage}', accs)
        self.logger.experiment.flush()

    def _calc_means_containing_key(self, in_key: str, outputs: list[dict]) -> dict[str, Tensor]:
        """Collects the outputs for each key containing in_key and averages."""
        return {
            key: self._collect_outputs(key, outputs, True).mean()
            for key in outputs[0].keys() if in_key in key
        }

    def _collect_outputs(self, key: str, outputs: list[dict], stack=False) -> Tensor:
        """Aggregates a specific key across all outputs from an epoch into one Tensor."""
        cat_func = torch.stack if stack else torch.cat
        return cat_func([r[key] for r in outputs])

    def _add_scalars_single(self, prefix: str, values: dict) -> None:
        """Per-epoch logging of all values in a dict with a desired prefix."""
        for (name, val) in values.items():
            self.logger.experiment.add_scalar(f'{prefix}_{name}', val, self.current_epoch)


    def on_train_start(self) -> None:
        """Specify organization of custom scalars in the tensorboard interface."""
        layout = {
            'Loss Simple': {
                'Train': ['Multiline', ['loss_train_loss']],
                'Valid': ['Multiline', ['loss_valid_loss']],
            },
            'Accuracy Simple': {
                'Train': ['Multiline', ['acc_train_acc_composite']],
                'Valid': ['Multiline', ['acc_valid_acc_composite']],
            },
            'Loss': {
                'Train': ['Multiline', [
                    'loss_train_loss', 'loss_train_class_loss', 'loss_train_extra_loss'
                ]],
                'Valid': ['Multiline', [
                    'loss_valid_loss', 'loss_valid_class_loss', 'loss_valid_extra_loss'
                ]],
            },
            'Accuracy': {
                'Train': ['Multiline', [
                    'acc_train_acc_composite', 'acc_train_acc_0', 'acc_train_acc_1'
                ]],
                'Valid': ['Multiline', [
                    'acc_valid_acc_composite', 'acc_valid_acc_0', 'acc_valid_acc_1'
                ]],
            },
        }
        self.logger.experiment.add_custom_scalars(layout)

    def log_images(self, batch: tuple[Tensor, Tensor], stage: str) -> None:
        """Log images from the batch to tensorboard."""
        experiment = self.logger.experiment

        signal, target = batch[0].clone(), batch[1].clone()
        signal_modify = self.training_signal_modify(signal).clamp(0, 1)

        channel_n, tag_size = signal.shape[1], 4

        for i in range(signal.shape[0]):
            # Tag the upper left hand corner of the image with a square representing the class.
            tag = torch.ones(channel_n, tag_size, tag_size).to(TU.device()) * target[i]
            signal[i][:, 0:tag_size, 0:tag_size] = tag
            signal_modify[i][:, 0:tag_size, 0:tag_size] = tag

        grid = torchvision.utils.make_grid(signal)
        grid_modify = torchvision.utils.make_grid(signal_modify)

        experiment.add_image(f'{stage}/Raw', grid)
        experiment.add_image(f'{stage}/Modify', grid_modify)

    def log_attributes(self) -> None:
        """Logs attributes of the model to tensorboard."""
        experiment = self.logger.experiment

        attributes = self.__dict__
        attributes = {k:v for k,v in attributes.items() if not k.startswith('_')}
        experiment.add_text('attributes', str(attributes), 0)
        if 'noise' in attributes.keys():
            noise_attributes = self.noise.__dict__
            experiment.add_text('noise_attributes', str(noise_attributes), 0)
