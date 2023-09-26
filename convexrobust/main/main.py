from __future__ import annotations

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Surpress tensorflow cuda errors

import warnings
# Surpress lightning-bolt warnings https://github.com/Lightning-Universe/lightning-bolts/issues/563
warnings.simplefilter('ignore')
original_filterwarnings = warnings.filterwarnings
def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, 'append':True})
warnings.filterwarnings = _filterwarnings

import random
import collections
import click
import numpy as np
from dataclasses import dataclass
import dacite

import torch
from pytorch_lightning import LightningDataModule

from convexrobust.data import datamodules
from convexrobust.model.linf_certifiable import LInfCertifiable
from convexrobust.model.randsmooth_certifiable import RandsmoothCertifiable
from convexrobust.model.insts.convex import (
    ConvexCifar, ConvexMnist, ConvexFashionMnist, ConvexMalimg, ConvexSimple
)
from convexrobust.model.insts.randsmooth import (
    RandsmoothCifar, RandsmoothMnist, RandsmoothFashionMnist,RandsmoothMalimg, RandsmoothSimple
)
from convexrobust.model.insts.cayley import (
    CayleyCifar, CayleyMnist, CayleyFashionMnist
)
from convexrobust.model.insts.abcrown import (
    ABCROWNCifar, ABCROWNMnist, ABCROWNFashionMnist, ABCROWNMalimg
)

from convexrobust.utils import dirs, pretty, file_utils
from convexrobust.utils import torch_utils as TU

from convexrobust.main.train import TrainConfig
from convexrobust.main.evaluate import EvaluateConfig
import convexrobust.main.train as main_train
import convexrobust.main.evaluate as main_evaluate
from convexrobust.main.train import ModelBlueprint, BlueprintDict, ModelDict

from typing import Optional


experiments = ['standard', 'ablation', 'ablation_noaugment']


def randsmooth_blueprints(
        randsmooth_class: type[RandsmoothCertifiable], epochs: int, sigma_scale: float,
        data_in_n: int, nb=100, large_splitderandomized_sigma=None, load=False
    ) -> BlueprintDict:

    constructor_params = {'n': 10000, 'cert_n_scale': 10, 'nb': nb, 'data_in_n': data_in_n}


    blueprint_params = {'load_model': True, 'load_eval_results': True} if load else {}

    blueprints = {}

    for factor in [1, 2, 3, 4]:
        params = {'sigma': sigma_scale * factor, **constructor_params}
        blueprints.update({
            f'randsmooth_splitderandomized_{factor}': ModelBlueprint(
                randsmooth_class(noise='split_derandomized', **params),
                epochs * factor, **blueprint_params
            ),
            f'randsmooth_laplace_{factor}': ModelBlueprint(
                randsmooth_class(noise='laplace', **params), epochs * factor, **blueprint_params
            ),
            f'randsmooth_gauss_{factor}': ModelBlueprint(
                randsmooth_class(noise='gauss', **params), epochs * factor, **blueprint_params
            ),
            f'randsmooth_uniform_{factor}': ModelBlueprint(
                randsmooth_class(noise='uniform', **params), epochs * factor, **blueprint_params
            ),
        })

    if large_splitderandomized_sigma is not None:
        params = {'sigma': large_splitderandomized_sigma, **constructor_params}
        blueprints[f'randsmooth_splitderandomized_large'] = ModelBlueprint(
            randsmooth_class(noise='split_derandomized', **params), epochs * 4, **blueprint_params
        )

    return blueprints


def get_blueprints(datamodule: LightningDataModule, experiment: str) -> BlueprintDict:
    if experiment in ['ablation', 'ablation_noaugment']:
        assert datamodule.name in ['cifar10_catsdogs', 'cifar10_dogscats']

    data_args = {'data_in_n': datamodule.in_n}

    if datamodule.name == 'mnist_38':
        return {
            'convex_noreg': ModelBlueprint(ConvexMnist(), 60, False),
            'convex_reg': ModelBlueprint(ConvexMnist(reg=0.01), 60, False),
            'cayley': ModelBlueprint(CayleyMnist(**data_args), 60, False),
            'abcrown': ModelBlueprint(ABCROWNMnist(), 60, False),
            **randsmooth_blueprints(RandsmoothMnist, 60, 0.75, datamodule.in_n),
            # Commented out by default since install is tricky -- see lib/linf_dist
            # for install instructions and uncomment to run this baseline
            'linf': ModelBlueprint(LInfCertifiable(
                'MLPModel(depth=5,width=5120,identity_val=10.0,scalar=True)',
                dirs.pretrain_path('mnist_38', 'model.pth'),
                [1, 28, 28],
                **data_args
            ), 0, False),
        }
    elif datamodule.name == 'fashion_mnist_shirts':
        return {
            'convex_noreg': ModelBlueprint(ConvexFashionMnist(), 60, True),
            'convex_reg': ModelBlueprint(ConvexFashionMnist(reg=0.01), 60, True),
            'cayley': ModelBlueprint(CayleyFashionMnist(**data_args), 60, True),
            'abcrown': ModelBlueprint(ABCROWNFashionMnist(), 60, True),
            **randsmooth_blueprints(RandsmoothFashionMnist, 60, 0.75, datamodule.in_n, True),
            # 'linf': ModelBlueprint(LInfCertifiable(
                # 'MLPModel(depth=5,width=5120,identity_val=10.0,scalar=True)',
                # dirs.pretrain_path('fashion_mnist_shirts', 'model.pth'),
                # [1, 28, 28],
                # **data_args
            # ), 0, False),
        }
    elif datamodule.name == 'malimg':
        return {
            'convex_noreg': ModelBlueprint(ConvexMalimg(reg=0.0), 150, True, True),
            'convex_reg': ModelBlueprint(ConvexMalimg(reg=0.075), 150),
            'abcrown': ModelBlueprint(ABCROWNMalimg(), 150, True, True),
            **randsmooth_blueprints(
                RandsmoothMalimg, 150, 3.5, datamodule.in_n, nb=32,
                large_splitderandomized_sigma=100, load=True
                # large_splitderandomized_sigma=None, load=True
            ),
        }
    elif datamodule.name in ['cifar10_catsdogs', 'cifar10_dogscats']:
        if experiment == 'standard':
            return {
                'convex_noreg': ModelBlueprint(ConvexCifar(), 150, True, True),
                'convex_reg': ModelBlueprint(ConvexCifar(reg=0.0075), 150, True, False),
                'cayley': ModelBlueprint(CayleyCifar(**data_args), 150, True, True),
                'abcrown': ModelBlueprint(ABCROWNCifar(), 150, True, True),
                **randsmooth_blueprints(RandsmoothCifar, 600, 0.75, datamodule.in_n, load=True),
            }
        elif experiment == 'ablation':
            # Experiments for appendix G.2
            blueprints = {
                'convex_nofeaturemap': ModelBlueprint(
                    ConvexCifar(apply_feature_map=False, reg=0.0), 150
                )
            }

            for i, reg in enumerate([0.0, 0.0025, 0.005, 0.0075, 0.01]):
                blueprints[f'convex_reg_{i}'] = ModelBlueprint(
                    ConvexCifar(reg=reg), 150
                )

            return blueprints
        elif experiment == 'ablation_noaugment':
            # Experiments for appendix G.3
            return {
                'convex_nofeaturemap': ModelBlueprint(
                    ConvexCifar(apply_feature_map=False), 500, False
                )
            }
        else:
            raise RuntimeError('Bad experiment type')
    elif datamodule.name == 'circles':
        return {
            'convex_noreg': ModelBlueprint(ConvexSimple, 30, False, {})
        }
    else:
        raise RuntimeError('Bad dataset')


@click.command(context_settings={'show_default': True})

@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38', help="""
The dataset to use. All should be downloaded automatically.
""")
@click.option('--experiment', type=click.Choice(experiments), default='standard', help="""
Which experiment to run (e.g. standard, ablation, etc).
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


@click.option('--train/--no_train', default=False, help="""
Whether to train models or load them from a previous training. You can retrain only
certain models by specifying --train and fixing the other models with the load_model
flag in ModelBlueprint.
""")
@click.option('--augment_data/--no_augment_data', default=True, help="""
Whether to apply data augmentation in the datamodule (e.g. random cropping).
Does NOT affect noise augmentation for randomized smoothing methods.
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
        data, experiment, clear, tensorboard, seed,
        train, augment_data, balance,
        eval_n, verify_cert, empirical_cert
    ) -> None:

    assert not (clear and (not train)) # If clear old models, must train!
    init(seed)

    pretty.section_print('Loading data and assembling parameters')
    params = locals() # Combine args + datamodule and experiment_directory attributes

    train_config = dacite.from_dict(data_class=TrainConfig, data=params)
    evaluate_config = dacite.from_dict(data_class=EvaluateConfig, data=params)

    datamodule = datamodules.get_datamodule(data, eval_n=eval_n, no_transforms=not augment_data)
    experiment_directory = f'{data}-{experiment}'

    if clear:
        file_utils.create_empty_directory(dirs.out_path(experiment_directory))
    if tensorboard:
        TU.launch_tensorboard(dirs.out_path(experiment_directory), 6006)

    blueprints: BlueprintDict = get_blueprints(datamodule, experiment)

    pretty.section_print('Creating models')
    models: ModelDict = main_train.train_models(
        blueprints, experiment_directory, datamodule, train_config
    )

    pretty.section_print('Evaluating models')
    _ = main_evaluate.evaluate_models(
        models, blueprints, experiment_directory, datamodule, evaluate_config
    )

    pretty.section_print('Done executing (ctrl+c to exit)')

def init(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    pretty.init()


if __name__ == "__main__":
    run()
