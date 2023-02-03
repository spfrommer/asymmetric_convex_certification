from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
import torch.autograd.functional as AF
from jacobian import JacobianReg
from pytorch_lightning import LightningDataModule

from convexrobust.utils import torch_utils as TU

from convexrobust.model.certificate import Norm, Certificate
from convexrobust.model.base_certifiable import BaseCertifiable
from convexrobust.model.modules import ConvexModule
from convexrobust.model import balance

class ABCROWNCertifiable(BaseCertifiable):
    """The base class for all alpha-beta-CROWN certifiable models.

    Unfortunately, the alpha beta CROWN solver doesn't have an API and must be run from
    the command line. So the certification procedures are left empty.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(loss=nn.CrossEntropyLoss(), **kwargs)

        self.external_certification = True

    def certify(self, x: Tensor, _) -> tuple[Tensor, Certificate]:
        raise NotImplementedError()

    def balance(self, datamodule: LightningDataModule) -> None:
        balance.direct_balance(self, datamodule)
