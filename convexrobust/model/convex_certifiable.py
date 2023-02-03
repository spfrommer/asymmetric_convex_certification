from __future__ import annotations

import torch
from torch import Tensor
import torch.autograd.functional as AF
from jacobian import JacobianReg
from pytorch_lightning import LightningDataModule

from convexrobust.utils import torch_utils as TU

from convexrobust.model.certificate import Norm, Certificate
from convexrobust.model.base_certifiable import BaseCertifiable
from convexrobust.model.modules import ConvexModule
from convexrobust.model import balance

class ConvexCertifiable(BaseCertifiable):
    """The base class for all convexly certified subclasses.

    Child classes must supply the convex prediction model and override configure_optimizers.
    """
    def __init__(self, convex_model: ConvexModule, reg=0.0, approx_reg=False, **kwargs) -> None:
        """
        Args:
            convex_model (ConvexModule): The learned convex function used for prediction. Refers to
                'g' or '\hat{g}' in the paper.
            reg (float, optional): The Jacobian regularization parameter. Defaults to 0.0.
            approx_reg (bool, optional): Whether to use an approximation of the jacobian
                regularization. Faster, but not as good regularization.
        """
        super().__init__(loss=TU.CustomBCEWithLogitsLoss, **kwargs)

        self.convex_model = convex_model
        self.convex_model.init_project()

        self.reg = reg
        self.approx_reg = approx_reg
        if self.reg > 0.0 and approx_reg:
            self.jac_reg = JacobianReg()

    def optimizer_step(self, *args, **kwargs) -> None:
        """Called after every gradient step -- projects weight matrices to ensure convexity."""
        super().optimizer_step(*args, **kwargs)
        self.convex_model.project()

    def lipschitz_forward(self, x: Tensor) -> tuple[Tensor, dict[Norm, float]]:
        """The optional "feature map". Refers to 'phi' in the paper.

        Args:
            x (Tensor): [batch_n x ...]. The input signal.

        Returns:
            tuple[Tensor, dict[Norm, float]]: The modified signal and an associated dictionary of
                Lipschitz constants for each norm. Defaults to the identity function.
        """
        return x, {Norm.L1: 1, Norm.L2: 1, Norm.LInf: 1}

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.lipschitz_forward(x) # Apply 'phi' from the paper
        x = self.convex_model(x) # Apply 'g' / '\hat{g}' from the paper
        # Shift output logits directly by class balance
        return TU.from_single_logit(x) + self.class_balance_prediction_shift()

    def certify(self, x: Tensor, _) -> tuple[Tensor, Certificate]:
        # The core certification procedure from the paper
        assert x.shape[0] == 1

        # Compute output margin with class balance
        x_feat, lips = self.lipschitz_forward(x)
        pred = self.convex_model.forward(x_feat) + self.class_balance
        pred *= TU.logit_sign()

        if pred >= 0:
            # Don't certify the non-sensitive class
            return TU.non_cert_class_tensor(), Certificate.zero()

        margin = -pred.squeeze(0)
        # The jacobian of the convex network with respect to x's feature-space representation
        # This is the subgradient v(phi(x)) from the paper
        jac = AF.jacobian(
            self.convex_model.forward, x_feat, strict=True, create_graph=False
        ).squeeze()

        # The certified radii for each norm, jacobian norms computed according to the dual norm
        certificate = Certificate({
            Norm.L1: (margin / jac.norm(float('inf'))).item() / lips[Norm.L1],
            Norm.L2: (margin / jac.norm(2)).item() / lips[Norm.L2],
            Norm.LInf: (margin / jac.norm(1)).item() / lips[Norm.LInf]
        })

        return TU.cert_class_tensor(), certificate

    def balance(self, datamodule: LightningDataModule) -> None:
        balance.direct_balance(self, datamodule)

    def extra_loss(self, x: Tensor, _) -> Tensor:
        # Jacobian regularization loss
        if self.reg == 0.0:
            return torch.tensor(0.0)

        with torch.enable_grad():
            x, _ = self.lipschitz_forward(x)
            if not self.approx_reg:
                jac = AF.jacobian(
                    self.convex_model.forward, x, strict=True, create_graph=True
                ).squeeze()
                extra_loss = self.reg * jac.norm(2)
            else:
                x.requires_grad = True
                output = self.convex_model(x).unsqueeze(1)
                extra_loss = self.reg * self.jac_reg(x, output)

        return extra_loss
