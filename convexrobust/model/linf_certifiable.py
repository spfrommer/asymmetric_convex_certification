import torch
import numpy as np

from convexrobust.data import datamodules

from convexrobust.utils import torch_utils as TU

from convexrobust.model.certificate import Certificate
from convexrobust.model.base_certifiable import BaseCertifiable
from convexrobust.model import balance

from collections import OrderedDict

from lib.linf_dist.main import parse_function_call
from lib.linf_dist.ell_inf_models import *


class LInfCertifiable(BaseCertifiable):
    def __init__(self, model, load_path, input_shape, data_in_n: int, **kwargs):
        super().__init__(loss=None, **kwargs)
        self.save_hyperparameters()

        self.normalize = datamodules.get_normalize_layer('mnist_38', average_stddev=True)

        model_name, params = parse_function_call(model)
        model = globals()[model_name](input_dim=np.array(input_shape), num_classes=2, **params)

        state_dict = torch.load(load_path)['state_dict']
        if next(iter(state_dict)).startswith('module.'):
            new_state_dict = OrderedDict([(k[7:], v) for k, v in state_dict.items()])
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        self.model = model

        with torch.no_grad():
            self.model.scalar.fill_(1.0)

        mean = self.normalize.means
        std = self.normalize.sds

        self.up = ((1 - mean) / std).view(-1, 1, 1).to(TU.device())
        self.down = ((0 - mean) / std).view(-1, 1, 1).to(TU.device())

        self.data_in_n = data_in_n

    def forward(self, x):
        x = self.normalize(x)
        return self.model.forward(x) + self.class_balance_prediction_shift()

    def certify(self, x, target):
        x = self.normalize(x)
        balance = (-TU.logit_sign() * self.class_balance).item()
        pred = self.model.forward(x, class_balance=balance)

        if pred.argmax() != target:
            return pred.argmax(dim=1), Certificate.zero()

        def is_certified(eps):
            outputs, worst = self.model.forward(
                x, target, eps=eps, up=self.up, down=self.down,
                class_balance=balance
            )
            return target == worst.argmax()

        tolerance = 0.01
        L, R = 0.0, 0.1
        while is_certified(R):
            R *= 2
        assert is_certified(L)
        while (R - L) > tolerance:
            m = (L + R) / 2
            if is_certified(m):
                L = m
            else:
                R = m
        eps = L

        certificate = Certificate.from_linf(eps * self.normalize.sds.item(), self.data_in_n)

        return pred.argmax(dim=1), certificate

    def balance(self, datamodule):
        balance.direct_balance(self, datamodule)

    def configure_optimizers(self):
        # Not actually used, load pretrained nets
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return [optimizer], []
