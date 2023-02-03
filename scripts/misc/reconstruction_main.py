import os
# Surpress tensorflow cuda errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import click
import tqdm
import csv

import torch

from mosek.fusion import *

from convexrobust.data import datamodules
from convexrobust.utils import torch_utils as TU

def flatten(signals):
    return signals.reshape(-1, 3 * 32 * 32)

def expand(signals):
    return signals.reshape(-1, 3, 32, 32)

def get_signals():
    datamodule = datamodules.get_datamodule('cifar10_catsdogs', no_transforms=True)
    signals, targets = TU.fetch_dataset(datamodule.dataset_train, 10000)
    signals_val, targets_val = TU.fetch_dataset(datamodule.dataset_val, 10000)

    signals_c0 = torch.cat((signals[targets==0], signals_val[targets_val==0]), dim=0).to('cpu')
    signals_c1 = torch.cat((signals[targets==1], signals_val[targets_val==1]), dim=0).to('cpu')

    signals_c0, signals_c1 = flatten(signals_c0), flatten(signals_c1)

    return signals_c0, signals_c1

def construct_mosek_problem(search_images):
    search_n = search_images.shape[0]

    M = Model()
    test_image_param = M.parameter(3072)

    alpha = M.variable(search_n, Domain.greaterThan(0.0))
    M.constraint(Expr.sum(alpha), Domain.equalsTo(1.0))

    t = M.variable()
    error = Expr.sub(test_image_param, Expr.mul(alpha, search_images.astype('double')))

    M.constraint(Expr.vstack(t, error), Domain.inQCone())
    M.objective(ObjectiveSense.Minimize, t)

    return M, test_image_param, alpha

@click.command()
def run():
    print('Loading signals')
    warnings.filterwarnings('ignore')
    signals_c1, signals_c0 = get_signals()

    search_images = TU.numpy(signals_c0)

    print('Constructing problem')
    M, test_image_param, alpha = construct_mosek_problem(search_images)
    # M.setLogHandler(sys.stdout)

    print('Solving...')
    with open('res/reconstruct.csv', 'w+') as csvfile:
        writer = csv.writer(csvfile)
        for test_image in tqdm.tqdm(TU.numpy(signals_c1)):
            test_image_param.setValue(test_image.astype('double'))
            M.solve()
            assert M.getProblemStatus() == ProblemStatus.PrimalAndDualFeasible
            error = torch.tensor(test_image - alpha.level() @ search_images)

            writer.writerow([error.norm(1).item(), error.norm(2).item(),
                             error.norm(float('inf')).item()])
            csvfile.flush()


            # test_image = torch.tensor(test_image.reshape(3, 32, 32))
            # reconstruction = torch.tensor((alpha.level() @ search_images).reshape(3, 32, 32))
            # print(f'{error.norm(1)}, {error.norm(2)}, {error.norm(float("inf"))}')
            # TU.imshow(torch.cat([test_image, reconstruction], dim=2))
            # import pdb; pdb.set_trace()

    # closest_image = get_closest_cvxpy(search_images, test_image)
    # TU.imshow(torch.cat([expand(torch.tensor(test_image)), expand(closest_image)], dim=3))


if __name__ == "__main__":
    run()
