from __future__ import annotations

import torch
from torch import Tensor
from pytorch_lightning import LightningDataModule
import numpy as np

from torchmetrics.functional.classification import binary_accuracy
from sklearn import metrics

from convexrobust.model.base_certifiable import BaseCertifiable
from convexrobust.utils import pretty
from convexrobust.utils import torch_utils as TU


def binary_search_balance(model: BaseCertifiable, datamodule: LightningDataModule) -> None:
    """Balance a binary classifier using binary search. Used for models with no direct class
    balance to prediction relationship."""

    def prediction_sweep(balance, do_tqdm=False):
        model.class_balance.fill_(balance)
        pred, target = _compute_pred_target(model, datamodule, do_tqdm=do_tqdm, hard_pred=True)
        class_0_acc, class_1_acc = _compute_class_accuracies(pred, target)
        return class_0_acc, class_1_acc, class_1_acc - class_0_acc >= 0

    tolerance, initial_bound = 0.01, 0.2

    print('Evaluating with zero balance -- subsequent bounds update will take equal time')
    class_0_acc, class_1_acc, _ = prediction_sweep(0.0, True)
    print(f'(Original) class 0 acc: {class_0_acc}, class 1 acc: {class_1_acc}')

    if abs(class_1_acc - class_0_acc) <= tolerance:
        print(f'Initial error within tolerance, returning')
        return

    print(f'Balance bounds: [-inf, inf] -- evaluating...', end='\r')

    bounds, it = [-1000, 1000], 0
    def update_bounds(new_bounds):
        nonlocal bounds, it
        bounds = new_bounds
        it += 1
        print(f'Balance bounds: [{bounds[0]: 6.4f}, {bounds[1]: 6.4f}]; class accs: ' + \
              f'[{class_0_acc: 6.3f}; {class_1_acc: 6.3f}] (iter {it:>3}) -- evaluating...',
              end='\r')

    if class_1_acc - class_0_acc < 0.0:
        update_bounds([-initial_bound, 0.0])
        class_0_acc, class_1_acc, error_positive = prediction_sweep(bounds[0], False)
        while not error_positive:
            update_bounds([bounds[0] * 2, bounds[1]])
            class_0_acc, class_1_acc, error_positive = prediction_sweep(bounds[0], False)
    else:
        update_bounds([0.0, initial_bound])
        class_0_acc, class_1_acc, error_positive = prediction_sweep(bounds[1], False)
        while error_positive:
            update_bounds([bounds[0], bounds[1] * 2])
            class_0_acc, class_1_acc, error_positive = prediction_sweep(bounds[1], False)

    while abs(class_1_acc - class_0_acc) > tolerance:
        m = (bounds[0] + bounds[1]) / 2
        class_0_acc, class_1_acc, error_positive = prediction_sweep(m, False)
        update_bounds([m, bounds[1]] if error_positive else [bounds[0], m])

    print(f'Balance bounds: [{bounds[0]: 6.4f}, {bounds[1]: 6.4f}]; class accs: ' + \
          f'[{class_0_acc: 6.3f}; {class_1_acc: 6.3f}] (iter {it:>3}) ' + \
          f'-- final balance: {model.class_balance.item(): 6.3f}')


def direct_balance(model: BaseCertifiable, datamodule: LightningDataModule) -> None:
    """Computes the optimal balance in closed form, assuming that the class_balance parameter is
    simply added on to the prediction logits."""
    pred, target = _compute_pred_target(model, datamodule, hard_pred=False)
    class_0_acc, class_1_acc = _compute_class_accuracies(TU.make_single_logit_hard(pred), target)
    print(f'(Original) class 0 acc: {class_0_acc}, class 1 acc: {class_1_acc}')

    fpr, tpr, thresholds = metrics.roc_curve(TU.numpy(target), TU.logit_sign() * TU.numpy(pred))
    # Optimize such that class accuracies are balanced
    threshold = thresholds[np.argmin(abs(tpr - (1 - fpr)))]
    class_0_acc, class_1_acc = _compute_class_accuracies(
        TU.make_single_logit_hard(pred + threshold), target
    )

    print(f'(Final) Class 0 acc: {class_0_acc}, class 1 acc: {class_1_acc}')
    pretty.subsection_print(f'Got optimal balance: {threshold.item()}')
    model.class_balance.fill_(threshold)


@torch.no_grad()
def _compute_pred_target(
        model: BaseCertifiable, datamodule: LightningDataModule, do_tqdm=True, hard_pred=False,
    ) -> tuple[Tensor, Tensor]:

    all_pred, all_target = [], []

    for (signal, target) in datamodule.eval_iterator(do_tqdm):
        if hard_pred:
            pred = model.predict_fast_approximate(signal)
        else:
            pred = TU.to_single_logit(model.forward(signal))

        all_pred.append(pred)
        all_target.append(target)

    return torch.cat(all_pred), torch.cat(all_target)


def _compute_class_accuracies(pred: Tensor, target: Tensor) -> tuple[float, float]:
    class_0_acc = binary_accuracy(pred, target, ignore_index=1)
    class_1_acc = binary_accuracy(pred, target, ignore_index=0)
    return class_0_acc, class_1_acc
