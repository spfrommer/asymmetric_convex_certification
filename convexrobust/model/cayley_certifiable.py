import torch
import math

from convexrobust.data.datamodules import NormalizeLayer
from convexrobust.model.certificate import Certificate
from convexrobust.model.base_certifiable import BaseCertifiable

from convexrobust.model import balance

from lib.orthconv.utils import margin_loss

custom_loss = lambda yhat, y: margin_loss(yhat, y, 0.5, 1.0, 1.0)

class CayleyCertifiable(BaseCertifiable):
    def __init__(self, model, data_in_n: int, normalize: NormalizeLayer=None, **kwargs):
        super().__init__(loss=custom_loss, **kwargs)

        self.model = model
        self.data_in_n = data_in_n
        self.normalize = normalize

    def forward(self, x):
        if self.normalize is not None:
            x = self.normalize(x)
        return self.model.forward(x) + self.class_balance_prediction_shift()

    def certify(self, x, _):
        assert x.shape[0] == 1
        pred = self.forward(x)
        margins = torch.sort(pred, 1)[0]

        certified_margin = (margins[:,-1] - margins[:, -2])

        certificate = Certificate.from_l2(
            certified_margin.item() / math.sqrt(2), self.data_in_n
        )

        return pred.argmax(dim=1), certificate

    def balance(self, datamodule):
        balance.direct_balance(self, datamodule)