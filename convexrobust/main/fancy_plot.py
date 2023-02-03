import torch
from torch.utils.data import DataLoader
from torch import Tensor

import numpy as np
import collections
import click

import os.path as op
import matplotlib
from matplotlib import transforms
from mpl_toolkits.axes_grid1 import Divider, Size
import matplotlib.pyplot as plt

from convexrobust.data import datamodules
from convexrobust.model.certificate import Norm
from convexrobust.utils import dirs, file_utils, pretty, vis_utils
from convexrobust.utils import torch_utils as TU
from convexrobust.main import main

from convexrobust.main.evaluate import Result, ResultDict

from sklearn.metrics import ConfusionMatrixDisplay

from collections import OrderedDict
from typing import Type, Dict, List, Any
from dataclasses import dataclass

matplotlib.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'font.size': 10,
    'text.usetex': True,
    'pgf.rcfonts': False,
    'savefig.transparent': True,
})


def figs_path(file_name, global_params):
    return op.join(f'./figs/{global_params.experiment_directory}', file_name)


def clean_confusion_plot(results: ResultDict, global_params):
    for (name, res) in results.items():
        fig = plt.figure(figsize=(8, 6), dpi=200)
        targets = TU.numpy(torch.cat([r.target for r in res]))
        preds = TU.numpy(torch.cat([r.pred for r in res]))

        ConfusionMatrixDisplay.from_predictions(targets, preds, cmap='plasma', normalize='true')

        fig.tight_layout()
        plt.savefig(figs_path(f'{name}_confusion.png', global_params))
        plt.close()

labels_dict = {
    'ablation_reg': OrderedDict([
        ('convex_reg_0', r'$\lambda=0.0$'),
        ('convex_reg_1', r'$\lambda=0.0025$'),
        ('convex_reg_2', r'$\lambda=0.005$'),
        ('convex_reg_3', r'$\lambda=0.0075$'),
        ('convex_reg_4', r'$\lambda=0.01$'),
    ]),
    'ablation_feature_map': OrderedDict([
        ('convex_nofeaturemap', r'$\varphi=\textrm{Id}$'),
        ('convex_reg_0', r'Standard $\varphi$'),
    ]),
}

for i in range(1, 5):
    labels_dict[f'standard_{i}'] = OrderedDict([
        # ('convex_noreg', 'Convex*'),
        ('convex_reg', 'Convex*'),
        (f'randsmooth_gauss_{i}', 'RS Gaussian'),
        (f'randsmooth_laplace_{i}', 'RS Laplacian'),
        (f'randsmooth_uniform_{i}', 'RS Uniform'),
        (f'randsmooth_splitderandomized_{i}', 'Splitting'),
        ('abcrown', r'$\alpha,\beta$-CROWN'),
        ('cayley', 'Cayley'),
        ('linf', r'$\ell_{\infty}$ Nets'),
    ])
    labels_dict[f'standard_sdr_max_{i}'] = OrderedDict([
        # ('convex_noreg', 'Convex*'),
        ('convex_reg', 'Convex*'),
        (f'randsmooth_gauss_{i}', 'RS Gaussian'),
        (f'randsmooth_laplace_{i}', 'RS Laplacian'),
        (f'randsmooth_uniform_{i}', 'RS Uniform'),
        ('randsmooth_splitderandomized_4', 'Splitting'),
        ('abcrown', r'$\alpha,\beta$-CROWN'),
        ('cayley', 'Cayley'),
        ('linf', r'$\ell_{\infty}$ Nets'),
    ])
    labels_dict[f'standard_sdr_large_{i}'] = OrderedDict([
        # ('convex_noreg', 'Convex*'),
        ('convex_reg', 'Convex*'),
        (f'randsmooth_gauss_{i}', 'RS Gaussian'),
        (f'randsmooth_laplace_{i}', 'RS Laplacian'),
        (f'randsmooth_uniform_{i}', 'RS Uniform'),
        ('randsmooth_splitderandomized_large', 'Splitting'),
        ('abcrown', r'$\alpha,\beta$-CROWN'),
        ('cayley', 'Cayley'),
        ('linf', r'$\ell_{\infty}$ Nets'),
    ])

figsize_dict = {'large': (4, 2.3), 'small': (2.3, 2.0)}

norm_str_dict = {Norm.L1: r'$\ell_1$', Norm.L2: r'$\ell_2$', Norm.LInf: r'$\ell_{\infty}$'}

line_colors = ['#88CCEE', '#CC6677', '#DDCC77', '#117733',
               '#332288', '#AA4499', '#44AA99', '#999933',
               '#882255', '#661100', '#6699CC', '#888888']

def certified_radius_plot(results: ResultDict, global_params, norm=Norm.L2):
    fig = plt.figure(dpi=72)
    figsize = figsize_dict[global_params.figsize]
    ax = setup_axes(fig, global_params.x_label, global_params.y_label, figsize)
    labels = labels_dict[global_params.labels]

    results = {k: v for (k, v) in results.items() if k in labels.keys()}

    max_radius = get_max_radius(results, norm) * 1.1
    if global_params.x_log:
        ax.set_xscale('log')
        min_x = {Norm.LInf: 0.000001, Norm.L2: 0.00001, Norm.L1: 0.01}[norm]
        plot_radii = np.logspace(np.log10(min_x), np.log10(max_radius), num=1000)
    else:
        plot_radii = np.linspace(0, max_radius, num=1000)

    accuracies = get_cert_accuracies(results, plot_radii, norm)

    filtered_accuracies = OrderedDict(
        [(k,accuracies[k]) for k in labels.keys() if k in accuracies.keys()]
    )
    for i, (name, accs) in enumerate(filtered_accuracies.items()):
        if name == 'abcrown':
            # Scatter plot places where acc changes
            changes = np.where(np.array(accs[:-1]) != np.array(accs[1:]))[0]
            plt.scatter(
                plot_radii[changes], np.array(accs)[changes], label=labels[name],
                color=line_colors[i], alpha=0.8, marker='x'
            )
        else:
            style = 'dashed' if 'convex' in name else 'solid'
            plt.plot(plot_radii, accs, label=labels[name], color=line_colors[i], alpha=0.8, linestyle=style)

    if global_params.title is not None:
        plt.title(global_params.title)

    if global_params.x_label:
        plt.xlabel(f'{norm_str_dict[norm]} radius')

    if global_params.y_label:
        plt.ylabel('Certified accuracy')
    else:
        ax.axes.yaxis.set_ticklabels([])

    if global_params.label_acc:
        legend_text_size = 8

        log_plots_bottom_left = True

        legend_width = 0.4 if global_params.figsize == 'large' else 0.7
        if global_params.x_log and log_plots_bottom_left:
            legend = plt.legend(
                bbox_to_anchor=(0, 0.0, legend_width, 0.55), mode='expand',
                handlelength=1, handletextpad=0.3, prop={'size': legend_text_size}
            )
        elif (not global_params.y_label):
            legend = plt.legend(
                bbox_to_anchor=(1 - legend_width, 0.55, legend_width, 0.45), mode='expand',
                handlelength=1, handletextpad=0.3, prop={'size': legend_text_size}
            )
        else:
            legend = plt.legend(
                # bbox_to_anchor=(1 - legend_width, 0.35, legend_width, 0.65), mode='expand',
                bbox_to_anchor=(1 - legend_width, 0.31, legend_width, 0.69), mode='expand',
                handlelength=1, handletextpad=0.3, prop={'size': legend_text_size}
            )

        for i, (name, _) in enumerate(filtered_accuracies.items()):
            acc = get_cert_accuracies(results, [0], norm)[name][0]
            if acc > 0.99999:
                text = f'[100 % clean]'
            else:
                text = f'[{acc * 100:.1f}% clean]'

            transform = legend.get_texts()[i].get_transform()
            trans = 0.66 if global_params.figsize == 'large' else 0.67
            offset = transforms.ScaledTranslation(trans, -0.02, fig.dpi_scale_trans)
            transform = transform + offset

            plt.text(0, 0.0, text, ha='left', va='bottom', transform=transform,
                     zorder=100, size=legend_text_size)
    else:
        legend = plt.legend(loc='upper right', handlelength=1, handletextpad=0.3)

    plt.xlim([min(plot_radii), max(plot_radii)])
    plt.ylim([0, 1])

    postprocess_axes(ax, global_params.x_label, global_params.y_label, figsize)

    # plt.savefig(figs_path(f'cert_{norm.name}_{global_params.labels}.pdf', global_params))
    plt.savefig(figs_path(f'cert_{norm.name}_{global_params.labels}.pgf', global_params))


def randsmooth_sweep_plot(results: ResultDict, global_params, norm=Norm.L2):
    fig = plt.figure(dpi=72)
    figsize = figsize_dict[global_params.figsize]
    ax = setup_axes(fig, global_params.x_label, global_params.y_label, figsize)

    results = {k: v for (k, v) in results.items() if 'randsmooth' in k}

    max_radius = get_max_radius(results, norm) * 1.1
    if global_params.x_log:
        ax.set_xscale('log')
        min_x = {Norm.LInf: 0.000001, Norm.L2: 0.00001, Norm.L1: 0.01}[norm]
        plot_radii = np.logspace(np.log10(min_x), np.log10(max_radius), num=1000)
    else:
        plot_radii = np.linspace(0, max_radius, num=1000)

    accuracies = get_cert_accuracies(results, plot_radii, norm)

    labels_map = {
        'gauss': 'RS Gaussian', 'laplace': 'RS Laplacian', 'uniform': 'RS Uniform',
        'splitderandomized': 'Splitting'
    }
    for i, smoothing in enumerate(labels_map.keys()):
        for multiplier in range(1, 5):
            name = f'randsmooth_{smoothing}_{multiplier}'
            label = labels_map[smoothing] + r' $(n \cdot \sigma)$' if multiplier == 1 else None
            alpha = 1 - (multiplier - 1) * 0.225
            plt.plot(
                plot_radii, accuracies[name], label=label, color=line_colors[i], alpha=alpha
            )

    if global_params.title is not None:
        plt.title(global_params.title)

    if global_params.x_label:
        plt.xlabel(f'{norm_str_dict[norm]} radius')

    if global_params.y_label:
        plt.ylabel('Certified accuracy')
    else:
        ax.axes.yaxis.set_ticklabels([])

    legend = plt.legend(loc='upper right')

    plt.xlim([min(plot_radii), max(plot_radii)])
    plt.ylim([0, 1])

    postprocess_axes(ax, global_params.x_label, global_params.y_label, figsize)

    # plt.savefig(figs_path(f'randsmooth_sweep_cert_{norm.name}.pdf', global_params))
    plt.savefig(figs_path(f'randsmooth_sweep_cert_{norm.name}.pgf', global_params))



def setup_axes(fig, x_label: bool, y_label: bool, figsize):
    if y_label:
        h = [Size.Fixed(0.45), Size.Fixed(figsize[0]), Size.Fixed(0.01)]
    else:
        h = [Size.Fixed(0.15), Size.Fixed(figsize[0]), Size.Fixed(0.01)]
    if x_label:
        v = [Size.Fixed(0.4), Size.Fixed(figsize[1]), Size.Fixed(0.01)]
    else:
        v = [Size.Fixed(0.3), Size.Fixed(figsize[1]), Size.Fixed(0.2)]

    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax = fig.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1)
    )
    return ax

def postprocess_axes(ax, x_label: bool, y_label: bool, figsize):
    ax.figure.set_size_inches(
        figsize[0] + (0.6 if y_label else 0.3),
        figsize[1] + (0.7 if x_label else 0.4)
    )

def get_max_radius(results: ResultDict, norm: Norm):
    all_results = sum(results.values(), [])
    all_radii = [result.certificate.radius[norm] for result in all_results
                 if result.certificate is not None]
    all_emp_radii = [result.empirical_certificate.radius[norm] for result in all_results
                 if result.empirical_certificate is not None]
    return max(all_radii + all_emp_radii)

def get_cert_accuracies(results: ResultDict, plot_radii: List[float],
                        norm: Norm, empirical=False) -> Dict[str, List[float]]:
    cert_accuracies = {}
    for (name, result) in results.items():
        def get_cert(r):
            return r.empirical_certificate if empirical else r.certificate

        def has_radius(r):
            return (r.target == r.pred).item() and (get_cert(r) is not None)

        result = [r for r in result if r.target == TU.CERT_CLASS]
        cert_radii = np.array([get_cert(r).radius[norm] if has_radius(r) else -1 for r in result])

        cert_accuracies[name] = [np.mean(cert_radii >= thresh) for thresh in plot_radii]
    return cert_accuracies

def set_axes_size(w,h, ax=None):
    """w, h: width, height in inches"""
    # From: https://stackoverflow.com/questions/44970010/axes-class-set-explicitly-size-width-height-of-axes-in-given-units
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh + 0.2)

def make_decision_plot(model, model_name, global_params, threshold=False, plot_cert=False):
    # Combos: soft 2-classifier, soft 1-classifier, hard classifier, maybe want to plot one class
    scale = 1
    plot_params = vis_utils.ClassifierPlotParams(
        xlim=[-scale, scale], ylim=[-scale, scale],
        cmap=plt.cm.get_cmap('seismic', 255))

    signals, targets = TU.fetch_dataset(global_params.datamodule.test_data, global_params.eval_n)

    fig = plt.figure(dpi=72, figsize=(3.2,2.8))
    ax = plt.gca()
    plt.xlim(plot_params.xlim)
    plt.ylim(plot_params.ylim)
    ax.set_aspect('equal', adjustable='box')

    with torch.no_grad():
        vis_utils.plot_scalar_prediction(model, plot_params, threshold=threshold)

        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(rotation=90)

        vis_utils.plot_data(signals, targets, plot_params)

        if plot_cert:
            radii = []
            for i, signal in enumerate(signals):
                pred, certificate = model.certify(signal.unsqueeze(0), targets[i])
                radii.append(certificate.radius[Norm.L2])
            vis_utils.plot_radii(signals, radii)

    plt.tight_layout()

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if threshold:
        plt.savefig(figs_path(f'decision_{model_name}_threshold.pdf', global_params), bbox_inches='tight')
    else:
        plt.savefig(figs_path(f'decision_{model_name}.pdf', global_params), bbox_inches='tight')


@click.command()
@click.option('--data', type=click.Choice(datamodules.names), default='cifar10_catsdogs')
@click.option('--experiment', type=click.Choice(main.experiments), default='standard')

@click.option('--clear_figs/--no_clear_figs', default=False)
@click.option('--figsize', type=click.Choice(figsize_dict.keys()), default='large')
@click.option('--labels', type=click.Choice(labels_dict.keys()), default='standard_1')
@click.option('--x_label/--no_x_label', default=True)
@click.option('--y_label/--no_y_label', default=True)
@click.option('--x_log/--no_x_log', default=False)
@click.option('--label_acc/--no_label_acc', default=True)
@click.option('--title', type=str, default=None)
@click.option('--randsmooth_sweep/--no_randsmooth_sweep', default=False)
def run(data, experiment, clear_figs, figsize, labels, x_label, y_label, x_log, label_acc, title, randsmooth_sweep):
    pretty.init()

    pretty.section_print('Assembling parameters')
    experiment_directory = f'{data}-{experiment}'
    local_vars = locals()
    global_params = collections.namedtuple('Params', local_vars.keys())(*local_vars.values())

    file_utils.ensure_created_directory(f'./figs/{experiment_directory}', clear=clear_figs)

    pretty.section_print('Loading results')
    results: ResultsDict = file_utils.read_pickle(dirs.out_path(experiment_directory, 'results.pkl'))

    pretty.section_print('Plotting results')
    # clean_confusion_plot(results, global_params)
    if randsmooth_sweep:
        randsmooth_sweep_plot(results, global_params, norm=Norm.L1)
        randsmooth_sweep_plot(results, global_params, norm=Norm.L2)
        randsmooth_sweep_plot(results, global_params, norm=Norm.LInf)
    else:
        certified_radius_plot(results, global_params, norm=Norm.L1)
        certified_radius_plot(results, global_params, norm=Norm.L2)
        certified_radius_plot(results, global_params, norm=Norm.LInf)


if __name__ == "__main__":
    run()
