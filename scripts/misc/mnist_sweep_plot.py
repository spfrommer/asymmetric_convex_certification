from __future__ import annotations

import torch

import numpy as np
import collections
import itertools
import click
import tqdm

import os
import os.path as op
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from convexrobust.data import datamodules
from convexrobust.model.certificate import Norm, Certificate
from convexrobust.utils import dirs, file_utils, pretty
from convexrobust.utils import torch_utils as TU
from convexrobust.main import main

from convexrobust.main.evaluate import Result, ResultDict

from sklearn.metrics import ConfusionMatrixDisplay

from collections import OrderedDict

from typing import Optional


matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 16,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'savefig.transparent': True,
})


def figs_path(file_name: str, global_params) -> str:
    return op.join(f'./figs/{global_params.all_experiments_directory}', file_name)

def get_clean_accuracy(results: ResultDict, name: str) -> float:
    return np.mean([(r.target == r.pred).float().item() for r in results[name]])

def get_cert_accuracies(
        results: ResultDict, plot_radii: list[float], norm: Norm, empirical=False
    ) -> dict[str, list[float]]:

    cert_accuracies = {}
    for (name, result_list) in results.items():
        result_list = filter_target_class(result_list)
        cert_radii = np.array(get_cert_radii(result_list, norm, empirical))
        cert_accuracies[name] = [np.mean(cert_radii >= thresh) for thresh in plot_radii]
    return cert_accuracies


def get_max_radius(results: ResultDict, norm: Norm) -> float:
    result_list = filter_target_class(sum(results.values(), []))
    cert_radii = get_cert_radii(result_list, norm, False)
    emp_radii = get_cert_radii(result_list, norm, True)
    return max(cert_radii + emp_radii)


def get_cert_radii(result_list: list[Result], norm: Norm, empirical: bool) -> list[float]:
    def get_cert(result: Result) -> Optional[Certificate]:
        return result.empirical_certificate if empirical else result.certificate

    def has_radius(result: Result) -> bool:
        return (result.target == result.pred).item() and (get_cert(result) is not None)

    return [get_cert(r).radius[norm] if has_radius(r) else -1 for r in result_list]

def filter_target_class(result_list: list[Result]) -> list[Result]:
    return [r for r in result_list if r.target == TU.CERT_CLASS]

def plot_cert_heatmap(cert_results: np.ndarray, norm: Norm, global_params) -> None:
    fig = plt.figure()
    ax = plt.gca()
    im = plt.imshow(cert_results, cmap='plasma')
    plt.xlabel('Sensitive class')
    plt.ylabel('Non-sensitive class')

    plt.xticks([0, 2, 4, 6, 8])
    plt.yticks([0, 2, 4, 6, 8])

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size = '5%', pad = 0.5)
    fig.add_axes(cax)
    fig.colorbar(im, cax = cax, orientation = 'horizontal')

    plt.tight_layout()
    # plt.savefig(figs_path(f'sweep_{norm}.png', global_params))
    plt.savefig(figs_path(f'sweep_{norm}.pdf', global_params),bbox_inches='tight')

@click.command(context_settings={'show_default': True})
@click.option('--data', type=click.Choice(datamodules.names), default='mnist_38')
@click.option('--classes_n', type=int, default=10)
@click.option('--experiment', type=click.Choice(main.experiments), default='standard')
@click.option('--clear_figs/--no_clear_figs', default=True)
def run(data, classes_n, experiment, clear_figs):
    pretty.init()

    pretty.section_print('Assembling parameters')
    all_experiments_directory = f'{data}-sweep'
    local_vars = locals()
    global_params = collections.namedtuple('Params', local_vars.keys())(*local_vars.values())

    file_utils.ensure_created_directory(f'./figs/{all_experiments_directory}', clear=clear_figs)


    for norm in [Norm.L1, Norm.L2, Norm.LInf]:
        pretty.section_print('Loading results')
        cert_results = np.zeros((classes_n, classes_n))
        for l0, l1 in tqdm.tqdm(list(itertools.product(range(classes_n), range(classes_n)))):
            if l0 == l1:
                cert_results[l0, l1] = np.nan
                continue
            experiment_directory = f'{all_experiments_directory}/{l0}-{l1}'
            results = file_utils.read_pickle(dirs.out_path(experiment_directory, 'results.pkl'))
            results_filtered = filter_target_class(results['convex'])
            cert_radii = np.array(get_cert_radii(results_filtered, norm, empirical=False))
            cert_radii[cert_radii < 0] = 0
            cert_results[l0, l1] = np.median(cert_radii)

        pretty.section_print(f'Plotting results {norm}')
        plot_cert_heatmap(cert_results, norm, global_params)


if __name__ == "__main__":
    run()
