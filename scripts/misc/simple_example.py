from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Surpress tensorflow cuda errors

import random
import warnings
import click
import collections
import itertools
import dacite
import numpy as np

import torch

from convexrobust.data import datamodules
from convexrobust.model.insts.convex import ConvexMnist

from convexrobust.utils import dirs, pretty, file_utils
from convexrobust.utils import torch_utils as TU

from convexrobust.main.train import TrainConfig
from convexrobust.main.evaluate import EvaluateConfig
import convexrobust.main.train as main_train
import convexrobust.main.evaluate as main_evaluate
from convexrobust.main.train import ModelBlueprint, BlueprintDict, ModelDict



@click.command(context_settings={'show_default': True})

@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38', help="""
The dataset to use. All should be downloaded automatically.
""")
@click.option('--clear/--no_clear', default=False, help="""
Whether to clear all old models and results and start fresh.
""")
@click.option('--tensorboard/--no_tensorboard', default=True, help="""
Whether to launch tensorboard showing the results of training. Works even with no_train.
""")
@click.option('--seed', default=1, help="""
The random seed.
""")


@click.option('--train/--no_train', default=True, help="""
Whether to train models or load them from a previous training. You can retrain only
certain models by specifying --train and fixing the other models with the load_model
flag in ModelBlueprint.
""")
@click.option('--balance/--no_balance', default=True, help="""
Whether to balance the test set performance of methods after training such that
the accuracies are the same across both classes.
""")

@click.option('--eval_n', default=1000, help="""
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
        data, clear, tensorboard, seed,
        train, balance,
        eval_n, verify_cert, empirical_cert
    ) -> None:

    assert not (clear and (not train)) # If clear old models, must train!
    init(seed)

    pretty.section_print('Loading data and assembling parameters')
    params = locals() # Combine args + datamodule and experiment_directory attributes

    train_config = dacite.from_dict(data_class=TrainConfig, data=params)
    evaluate_config = dacite.from_dict(data_class=EvaluateConfig, data=params)

    experiment_directory = f'simple_example'

    if clear:
        file_utils.create_empty_directory(dirs.out_path(experiment_directory))
    if tensorboard:
        TU.launch_tensorboard(dirs.out_path(experiment_directory), 6006)

    pretty.section_print(f'Training')

    datamodule = datamodules.get_datamodule(data, eval_n=eval_n)
    blueprints: BlueprintDict = {
        'convex': ModelBlueprint(ConvexMnist(), 10)
        # Add your model here, or in convexrobust/main/main.py
    }

    pretty.section_print('Creating models')
    models: ModelDict = main_train.train_models(
        blueprints, experiment_directory, datamodule, train_config
    )

    pretty.section_print('Evaluating models')
    results = main_evaluate.evaluate_models(
        models, blueprints, experiment_directory, datamodule, evaluate_config
    )

    pretty.section_print('Done executing, written to /out/simple_example (ctrl+c to exit)')

def init(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    pretty.init()
    warnings.filterwarnings('ignore')


if __name__ == "__main__":
    run()

