from __future__ import annotations

import warnings
# Surpress lightning-bolt warnings https://github.com/Lightning-Universe/lightning-bolts/issues/563
warnings.simplefilter('ignore')
original_filterwarnings = warnings.filterwarnings
def _filterwarnings(*args, **kwargs):
    return original_filterwarnings(*args, **{**kwargs, 'append':True})
warnings.filterwarnings = _filterwarnings

import torch

import numpy as np
import collections
from collections import defaultdict
import click
import tabulate

import os.path as op
import matplotlib.pyplot as plt

from convexrobust.data import datamodules
from convexrobust.model.certificate import Norm, Certificate
from convexrobust.utils import dirs, file_utils, pretty
from convexrobust.utils import torch_utils as TU
from convexrobust.main import main

from convexrobust.main.plot import labels_dict

from convexrobust.main.evaluate import Result, ResultDict

from sklearn.metrics import ConfusionMatrixDisplay

from collections import OrderedDict

from typing import Optional


# Used to reproduce the plots for the paper. From the hyperparam table in Section E of appendix
labels_dict['mnist_38_paper'] = labels_dict['standard_1_splitting_4']
labels_dict['fashion_mnist_shirts_paper'] = labels_dict['standard_1']
labels_dict['cifar10_catsdogs_paper'] = labels_dict['standard_2']
labels_dict['malimg_paper'] = labels_dict['standard_4_splitting_large']


@click.group()
def cli():
    pass


@cli.command(context_settings={'show_default': True})
@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38')
def single(data):
    cert_times = get_cert_times(data)

    for (label, avg_time) in cert_times.items():
        print(f'{label}: {avg_time}')

def get_cert_times(data):
    pretty.init()

    experiment_directory = f'{data}-standard'
    file_utils.ensure_created_directory(f'./timing')
    results: ResultDict = file_utils.read_pickle(dirs.out_path(experiment_directory, 'results.pkl'))

    labels = labels_dict[f'{data}_paper']

    cert_times_dict = {}

    for method_name, label in labels.items():
        if method_name not in results.keys():
            cert_times_dict[label] = -1
            continue

        if method_name == 'abcrown':
            l1_time = float(file_utils.read_file(dirs.out_path(experiment_directory, 'abcrown_times', 'time_l1.txt')))
            l2_time = float(file_utils.read_file(dirs.out_path(experiment_directory, 'abcrown_times', 'time_l2.txt')))
            linf_time = float(file_utils.read_file(dirs.out_path(experiment_directory, 'abcrown_times', 'time_linf.txt')))

            eval_n = 1000

            # Based on how many properties are being evaluated in scripts/abcrown
            prop_n = {
                'mnist_38': 15 * eval_n,
                'fashion_mnist_shirts': 14 * eval_n,
                'cifar10_catsdogs': 14 * eval_n,
                'malimg': 13 * eval_n,
            }[data]

            avg_prop_time = (l1_time + l2_time + linf_time) / prop_n
            cert_times_dict[label] = avg_prop_time

            continue

        cert_times = [r.debug_vars['cert_time'] for r in results[method_name] if 'cert_time' in r.debug_vars]
        avg_cert_time = np.array(cert_times).mean()
        if 'randsmooth' in method_name:
            avg_cert_time *= 10 # Take into account cert_n_scale

        cert_times_dict[label] = avg_cert_time

    return cert_times_dict


@cli.command()
@click.pass_context
def all(ctx):
    datas = ['mnist_38', 'malimg', 'fashion_mnist_shirts', 'cifar10_catsdogs']
    cert_times_data = {data: get_cert_times(data) for data in datas}
    cert_times_data = transpose(cert_times_data)

    headers = datas

    rows = [[k] + [f'{v[data]:0.3g}' for data in datas] for (k, v) in cert_times_data.items()]

    print(tabulate.tabulate(rows, headers=headers))

    table = tabulate.tabulate(rows, headers=headers, tablefmt='latex')
    print(table)


# Adapted from:
# https://stackoverflow.com/questions/21976875/transpose-dict-of-dicts-without-nans
def transpose(dct):
    d = defaultdict(dict)
    for key1, inner in dct.items():
        for key2, value in inner.items():
            d[key2][key1] = value
    return d

if __name__ == "__main__":
    cli()
