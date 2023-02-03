import torch

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from dataclasses import dataclass
from typing import Tuple

from convexrobust.utils import torch_utils


@dataclass
class ClassifierPlotParams:
    xlim: Tuple[int, int]
    ylim: Tuple[int, int]
    cmap: matplotlib.colors.Colormap


def plot_data(signals, targets, plot_params):
    """Plots 2d data colored by target values. class 0 = red, class 1 = blue"""
    plt.scatter(torch_utils.numpy(signals[:, 0]),
                torch_utils.numpy(signals[:, 1]),
                edgecolors='k', c=1 - torch_utils.numpy(targets), s=40, cmap=plot_params.cmap)

def plot_radii(signals, radii):
    fig = plt.gcf()
    ax = fig.gca()
    for signal, radius in zip(signals, radii):
        if radius > 0:
            ax.add_patch(plt.Circle(signal.tolist(), radius, edgecolor='k', facecolor=(0,0,0,0)))


def plot_prediction(model, plot_params, single_logit=False, take_argmax=False, class_sel=None):
    """Plots prediction of a model that accepts elements in R2."""
    # Adapted from https://github.com/Formulator/Spiral/blob/master/PyTorch_Spiral_DataLoader.ipynb
    x_min, x_max = plot_params.xlim
    y_min, y_max = plot_params.ylim

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 30
    # spacing = min(x_max - x_min, y_max - y_min) / 100

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1),
                      YY.ravel().reshape(-1,1)))
    data = torch.tensor(data).float().to(model.device)

    Z = model.predict(data) if class_sel is None else model.forward(data)
    hard_classifier = (len(Z) == XX.size) and class_sel is None

    #Convert PyTorch tensor to NumPy for plotting.
    Z = Z.detach().cpu().numpy()
    alpha = np.ones(Z.shape[0]) * 0.6
    if hard_classifier:
        invalid_indices = np.where(Z < 0)[0]
        alpha[invalid_indices] = 0
    else:
        if take_argmax:
            Z = np.argmax(Z, axis=1)
        elif class_sel is not None:
            Z = Z[:, class_sel]
        else:
            assert Z.shape[1] == 2
            Z = np.exp(Z)
            Z = (Z[:, 1] - Z[:, 0]) / 2 + 0.5

    alpha = alpha.reshape(XX.shape)
    Z = Z.reshape(XX.shape)

    plt.imshow(Z, extent=plot_params.xlim + plot_params.ylim, origin='lower',
               cmap=plot_params.cmap, alpha=alpha, vmin=0, vmax=1)
    # plt.imshow(Z, extent=plot_params.xlim + plot_params.ylim, origin='lower',
               # cmap=plot_params.cmap, alpha=alpha, vmin=np.min(Z), vmax=np.max(Z))

def plot_scalar_prediction(model, plot_params, threshold=False):
    """Plots prediction of a model that accepts elements in R2."""
    # Adapted from https://github.com/Formulator/Spiral/blob/master/PyTorch_Spiral_DataLoader.ipynb
    x_min, x_max = plot_params.xlim
    y_min, y_max = plot_params.ylim

    # Set grid spacing parameter
    spacing = min(x_max - x_min, y_max - y_min) / 200

    # Create grid
    XX, YY = np.meshgrid(np.arange(x_min, x_max, spacing),
                         np.arange(y_min, y_max, spacing))

    # Concatenate data to match input
    data = np.hstack((XX.ravel().reshape(-1,1),
                      YY.ravel().reshape(-1,1)))

    if threshold:
        Z = 1 - model.predict(torch.tensor(data).float().to(model.device))
    else:
        Z = model.forward(torch.tensor(data).float().to(model.device))

    #Convert PyTorch tensor to NumPy for plotting.
    Z = Z.detach().cpu().numpy()
    alpha = np.ones(Z.shape[0]) * 0.6

    Z = Z.reshape(XX.shape)

    v_bound = [0, 1] if threshold else [-max(Z.min(), Z.max()), max(Z.min(), Z.max())]

    # plt.imshow(Z, extent=plot_params.xlim + plot_params.ylim, origin='lower',
               # cmap=plot_params.cmap, vmin=Z.min(), vmax=Z.max())
    plt.imshow(Z, extent=plot_params.xlim + plot_params.ylim, origin='lower',
               cmap=plot_params.cmap, vmin=v_bound[0], vmax=v_bound[1])
               # cmap=plot_params.cmap, vmin=Z.min(), vmax=Z.max())
               # cmap=plot_params.cmap, vmin=-10, vmax=10)
