import os
# Surpress tensorflow cuda errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import warnings
import click
import tqdm

from convexrobust.data import datamodules
from convexrobust.utils import torch_utils as TU


@click.command()
@click.option('--data', type=click.Choice(datamodules.names), default='cifar10_catsdogs')
def run(data):
    warnings.filterwarnings('ignore')

    # Adapted from https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html#:~:text=mean%3A%20simply%20divide%20the%20sum,%2F%20count%20%2D%20total_mean%20**%202)
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    data_n = 0
    image_size = None

    datamodule = datamodules.get_datamodule(data, no_transforms=True)
    for (signals, _) in tqdm.tqdm(datamodule.train_dataloader()):
        psum += signals.sum(axis=[0,2,3])
        psum_sq += (signals ** 2).sum(axis=[0,2,3])
        data_n += signals.shape[0]
        if image_size is None:
            image_size = signals.shape[-1]

    count = data_n * image_size * image_size

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))


if __name__ == "__main__":
    run()
