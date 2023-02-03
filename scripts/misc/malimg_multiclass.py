from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Surpress tensorflow cuda errors

import random
import warnings
import click
import collections
import itertools
import dacite
import os.path as op
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.patches as patches
from collections import Counter

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule

from convexrobust.data import datamodules
from convexrobust.data import malimg
from convexrobust.model.insts.convex import ConvexMalimg

from convexrobust.utils import dirs, pretty, file_utils
from convexrobust.utils import torch_utils as TU

from convexrobust.model.certificate import Certificate, Norm
from convexrobust.main.train import TrainConfig
from convexrobust.main.evaluate import EvaluateConfig
import convexrobust.main.train as main_train
import convexrobust.main.evaluate as main_evaluate
from convexrobust.main.train import ModelBlueprint, BlueprintDict, ModelDict

from typing import Optional


matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'savefig.transparent': True,
})

def figs_path(file_name: str) -> str:
    return op.join(f'./figs/malimg-multiclass', file_name)


class MalwareClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=24)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.normalize = datamodules.get_normalize_layer('malimg')

    def forward(self, x):
        x = self.normalize(x)
        return self.model(x)

    def training_step(self, batch, _):
        return self.compute_loss(batch)

    def validation_step(self, batch, _):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss)
        return loss

    def compute_loss(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


@click.command(context_settings={'show_default': True})

@click.option('--clear/--no_clear', default=False, help="""
Whether to clear all old models and results and start fresh.
""")
@click.option('--tensorboard/--no_tensorboard', default=True, help="""
Whether to launch tensorboard showing the results of training. Works even with no_train.
""")
@click.option('--seed', default=1, help="""
The random seed.
""")

@click.option('--train/--no_train', default=False, help="""
Whether to train models or load them from a previous training. You can retrain only
certain models by specifying --train and fixing the other models with the load_model
flag in ModelBlueprint.
""")
@click.option('--balance/--no_balance', default=True, help="""
Whether to balance the test set performance of methods after training such that
the accuracies are the same across both classes.
""")

@click.option('--eval_n', default=5000, help="""
How many test data points to evaluate / balance over.
""")
@click.option('--verify_cert/--no_verify_cert', default=False, help="""
Whether to verify any certificates generated during evaluation with a PGD attack.
Does not verify nondeterministic randomized smoothing certificates.
""")
@click.option('--empirical_cert/--no_empirical_cert', default=False, help="""
Whether to compute empirical robustness certificates with a PGD attack.
""")
def run(
        clear, tensorboard, seed,
        train, balance,
        eval_n, verify_cert, empirical_cert
    ) -> None:

    ##### SETUP

    assert not (clear and (not train)) # If clear old models, must train!
    init(seed)

    pretty.section_print('Loading data and assembling parameters')
    params = locals()

    train_config = dacite.from_dict(data_class=TrainConfig, data=params)
    evaluate_config = dacite.from_dict(data_class=EvaluateConfig, data=params)

    experiment_root_directory = f'malimg-multiclass'
    convex_experiment_directory = f'{experiment_root_directory}/convex_binary'
    multiclass_experiment_directory = f'{experiment_root_directory}/multiclass'

    if clear:
        file_utils.create_empty_directory(dirs.out_path(experiment_root_directory))
    if tensorboard:
        TU.launch_tensorboard(dirs.out_path(experiment_root_directory), 6006)

    # Has Allaple.A as class 1, all other class 0 (certified)
    datamodule_binarized = datamodules.get_datamodule('malimg', eval_n=eval_n)
    # Has malware classes 0-23 inclusive and allaple.A as 24
    datamodule_malware = datamodules.get_datamodule(
        'malimg', eval_n=eval_n, datamodule_args={'binarize': False}
    )

    # Make sure using same datasets for train / test / val split
    # Train and val only malware, test everything
    datamodule_malware.dataset_train.file_list = \
        datamodule_binarized.dataset_train.file_list[:datamodule_binarized.dataset_train.class_cutoff]
    datamodule_malware.dataset_val.file_list = \
        datamodule_binarized.dataset_val.file_list[:datamodule_binarized.dataset_val.class_cutoff]
    datamodule_malware.dataset_test.file_list = datamodule_binarized.dataset_test.file_list


    ##### TRAINING

    pretty.section_print('Creating convex model...')
    convex_blueprints = { 'convex': ModelBlueprint(ConvexMalimg(reg=0.075), 150, True) }
    convex_model: ConvexMalimg = main_train._train_model(
        'convex', convex_blueprints['convex'],
        convex_experiment_directory, datamodule_binarized, train_config
    )

    pretty.section_print('Creating malware model...')
    multiclass_train_config = TrainConfig(train=train, balance=False)
    multiclass_model: MalwareClassifier = main_train._train_model(
        'multiclass', ModelBlueprint(MalwareClassifier(), 150, True),
        multiclass_experiment_directory, datamodule_malware, multiclass_train_config
    )

    convex_model.eval()
    multiclass_model.eval()


    ##### EVALUATING

    pretty.section_print('Evaluating...')

    y_pred, y_true, certificate = [], [], []

    for batch in datamodule_malware.eval_iterator():
        signal, target = batch
        y_true.append((target.item() + 1) % 25)

        is_malware = (convex_model.predict(signal) == 0).item()

        if not is_malware:
            y_pred.append(0) # The ''clean''
            certificate.append(Certificate.zero())
            continue

        predicted_malware = multiclass_model(signal).argmax(dim=1).item() + 1
        y_pred.append(predicted_malware)
        certificate.append(convex_model.certify(signal, target)[1])


    ##### PLOTTING

    pretty.section_print('Plotting...')

    cnf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cnf_matrix, ['Clean'] + [str(i+1) for i in range(24)], normalize=True)
    plt.savefig(figs_path(f'confusion.pdf'), bbox_inches='tight')


    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    malware_classes, malware_counts = np.unique(y_true, return_counts=True)
    count_sort_indices = np.argsort(-malware_counts)
    plot_malware_classes = malware_classes[count_sort_indices][1:6] # Ignore Allaple.A

    # Remove Yuner.A
    plot_malware_classes = np.delete(plot_malware_classes, 1)

    fig, axs = plt.subplots(1, 3)
    style = {'showmedians': True}
    parts_all = []

    malware_names = [malimg.malware_names[malware_class - 1] for malware_class in plot_malware_classes]

    malware_names = [n[:7] + '...' if len(n) > 10 else n for n in malware_names]

    for i, norm in enumerate([Norm.L1, Norm.L2, Norm.LInf]):
        certificates_lists = []

        for malware_class in plot_malware_classes:
            indices = np.where(y_true == malware_class)[0]
            certificates_lists.append([certificate[i].radius[norm] for i in indices])

        parts_all.append(axs[i].violinplot(certificates_lists, **style))

    for parts in parts_all:
        for i, pc in enumerate(parts['bodies']):
            if i == 1:
                pc.set_facecolor('#D43F3A')
            if i == 2:
                pc.set_facecolor('#448c3b')
            if i == 3:
                pc.set_facecolor('#aa52b3')
            pc.set_edgecolor('black')

        for partname in ('cbars','cmins','cmaxes','cmedians'):
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1)

    axs[0].set_ylabel(r'$\ell_1$ certificates')
    axs[1].set_ylabel(r'$\ell_2$ certificates')
    axs[2].set_ylabel(r'$\ell_{\infty}$ certificates')

    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    for ax in axs:
        set_axis_style(ax, malware_names)

    plt.tight_layout()
    plt.savefig(figs_path(f'violin.pdf'), bbox_inches='tight')

    pretty.section_print('Done executing (ctrl+c to exit)')

def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels, rotation=45)
    ax.set_xlim(0.25, len(labels) + 0.75)

def init(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    pretty.init()
    warnings.filterwarnings('ignore')

def plot_confusion_matrix(
        cm, classes, normalize=False,
    ):
    """
    Adapted from https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = plt.gca()

    im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    # tick_marks = tick_marks[::2]
    # classes = classes[::2]
    classes[1::2] = [''] * len(classes[1::2])
    # plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # plt.text(j, i, cm[i, j],
                 # horizontalalignment="center",
                 # color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    rect = patches.Rectangle((0.5, 0.5), 24, 24, linewidth=2, edgecolor='k', facecolor='none')
    ax.add_patch(rect)

if __name__ == "__main__":
    run()

