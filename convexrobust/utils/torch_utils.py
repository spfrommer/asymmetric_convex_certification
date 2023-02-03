import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import re
import threading
import socket
import os
import tqdm
from contextlib import contextmanager

import matplotlib
# matplotlib.use('QtAgg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from convexrobust.utils import file_utils


def device():
    return torch.device('cuda')
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gpu_n():
    return 1 if torch.cuda.is_available() else 0


def numpy(tensor):
    return tensor.detach().cpu().numpy()


def norm_ball_conversion_factor(to_norm: float, from_norm: float, dim: int):
    if to_norm <= from_norm:
        return 1
    return 1 / (dim ** (1 / from_norm - 1 / to_norm))


CERT_CLASS = 0 # The "sensitive class"
NON_CERT_CLASS = 1 - CERT_CLASS

def cert_class_tensor():
    return torch.tensor([CERT_CLASS]).long().to(device())

def non_cert_class_tensor():
    return torch.tensor([NON_CERT_CLASS]).long().to(device())

def logit_sign():
    return -1 if CERT_CLASS == 0 else 1

def from_single_logit(pred):
    # Takes tensor of length batch_n
    # Scalars converted to length 1 tensor
    if type(pred) == float:
        pred = torch.tensor(pred).to(device())
    if pred.dim() == 0:
        pred = pred.reshape(1)
    return 0.5 * logit_sign() * torch.stack([-pred, pred], dim=1)

def to_single_logit(pred):
    return logit_sign() * (pred[:, 1] - pred[:, 0])

def make_single_logit_hard(pred):
    return from_single_logit(pred).argmax(dim=1)


def CustomBCEWithLogitsLoss(pred, target):
    # TODO: inefficient (a lot of conversions)
    pred = to_single_logit(pred)
    return F.binary_cross_entropy_with_logits(
        logit_sign() * pred, target.float()
    )

class LossWrapper():
    def __init__(self, model, loss):
        self.model = model
        self.loss_func = loss

    def loss(self, x, y):
        pred = self.model(x)
        return self.loss_func(pred, y)

@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.
    From https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
    '''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def imshow(X):
    if X.shape[0] == 1:
        X = X[0]
    if torch.is_tensor(X):
        X = numpy(X)
    X = np.moveaxis(X, 0, -1)

    plt.figure()
    plt.imshow(X)
    plt.axis('off')
    plt.show()

def histshow(X):
    X = X.flatten().tolist()
    plt.figure()
    plt.hist(X, 50)
    plt.show()


def load_model_from_checkpoint(checkpoint_path, blueprint):
    model = blueprint.model.load_from_checkpoint(checkpoint_path, strict=False)

    # Handle pytorch None weight bug... https://github.com/pytorch/pytorch/issues/16675
    # Relevant for cayley transform 'alpha' parameter
    state_dict = torch.load(checkpoint_path)['state_dict']
    for (name, data) in state_dict.items():
        if 'alpha' in name:
            # Fix sequential list indexing
            name_eval = 'model.' + re.sub(r'\.(\d+)\.', r'[\1].', name)
            exec(name_eval + ' = nn.Parameter(data, requires_grad=True).to(device())')
    model.to(device())

    return model


def launch_tensorboard(tensorboard_dir, port):
    # Use threading so tensorboard is automatically closed on process end
    command = f'tensorboard --bind_all --port {port} '\
              f'--logdir {tensorboard_dir} > /dev/null '\
              f'--window_title {socket.gethostname()} 2>&1'
    t = threading.Thread(target=os.system, args=(command,))
    t.start()

    print(f'Launching tensorboard on http://localhost:{port}')



def fetch_dataset(dataset, fetch_n):
    signals, targets = next(iter(DataLoader(dataset, batch_size=fetch_n)))
    signals, targets = signals.to(device()), targets.to(device())
    return signals, targets


def fetch_dataloader(dataloader, fetch_n, do_tqdm=False):
    if do_tqdm:
        pbar = tqdm.tqdm(total=fetch_n)

    i = 0
    for (signals, targets) in dataloader:
        for (signal, target) in zip(signals, targets):
            yield signal.to(device()).unsqueeze(0), target.to(device()).unsqueeze(0)
            if do_tqdm:
                pbar.update(1)
            i += 1

            if i >= fetch_n:
                if do_tqdm:
                    pbar.close()
                return

    if do_tqdm:
        pbar.close()


def fetch_dataloader_batch(dataloader, fetch_n):
    signals, targets = [], []
    for (signal, target) in fetch_dataloader(dataloader, fetch_n):
        signals.append(signal)
        targets.append(target)

    return torch.cat(signals, dim=0), torch.cat(targets, dim=0)


def check_dataloader_deterministic(dataloader):
    s1, t1 = next(iter(fetch_dataloader(dataloader, 1)))
    s2, t2 = next(iter(fetch_dataloader(dataloader, 1)))
    assert (s1 == s2).all() and (t1 == t2)
