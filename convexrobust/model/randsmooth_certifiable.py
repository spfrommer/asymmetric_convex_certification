from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from pytorch_lightning import LightningDataModule

import scipy.stats as stats

from convexrobust.data.datamodules import NormalizeLayer

from convexrobust.utils import torch_utils as TU

from convexrobust.model.certificate import Certificate, Norm
from convexrobust.model.base_certifiable import BaseCertifiable, Certificate
from convexrobust.model import balance

import lib.smoothingSplittingNoise.src.smooth as rs_smooth
import lib.smoothingSplittingNoise.src.noises as rs_noises

from typing import Optional


def custom_loss(preds, targets):
    return -torch.distributions.Categorical(logits=preds).log_prob(targets).mean()


class RandsmoothCertifiable(BaseCertifiable):
    """Generates certificates for a smoothed classifier using randomized smoothing. Implementation
    is a modification of https://github.com/alevine0/smoothingSplittingNoise, which in turn modifies
    https://github.com/tonyduan/rs4a.
    """
    def __init__(
            self, model: Module, data_in_n: int, normalize: NormalizeLayer=None, sigma=0.25,
            n0=100, n=100000, nb=100, alpha=0.001, cert_n_scale=1, noise='gauss',
            init_batchnorm_channels: Optional[int]=None, **kwargs
        ):
        """
        Args:
            model (Module): The module used to make predictions.
            data_in_n (int): The dataset input dimensionality. Used to convert between norm balls.
            normalize (NormalizeLayer, optional): An optional normalization layer. Defaults to None.
            sigma (float, optional): The noise standard deviation. Defaults to 0.25.
            n0 (int, optional): Number of samples to guess the certificate class. Defaults to 100.
            n (int, optional): Number of samples for certification. Defaults to 100000.
            nb (int, optional): Number of samples per batch for certification. Defaults to 100.
            alpha (float, optional): The probabilistic correctness tolerance. Defaults to 0.001.
            cert_n_scale (int, optional): Optionally increase the confidence of the certification
                procedure by virtually projecting the number of samples used for certification.
                Works as described in the original Cohen et. al. paper. Helps increase the
                performance of the baselines. Defaults to 1.
            noise (str, optional): ['gauss', 'uniform', 'laplace', 'split', 'split_derandomized'].
                The noise distribution for training and certification. Defaults to 'gauss'.
            init_batchnorm_channels (int, optional): If not None, applies an initial batchnorm layer
                with the specified number of channels. Defaults to None.
        """
        super().__init__(loss=custom_loss, **kwargs)

        self.model = model
        self.normalize = normalize

        self.data_in_n = data_in_n

        noise_args = {'dim': self.data_in_n, 'sigma': sigma, 'device': TU.device()}
        noises = {
            'gauss': rs_noises.Gaussian, 'uniform': rs_noises.Uniform, 'laplace': rs_noises.Laplace,
            'split': rs_noises.SplitMethod, 'split_derandomized': rs_noises.SplitMethodDerandomized
        }
        self.noise = noises[noise](**noise_args)

        self.n0 = n0
        self.n = n
        self.nb = nb
        self.alpha = alpha
        self.cert_n_scale = cert_n_scale

        self.bn = None
        if init_batchnorm_channels is not None:
            self.bn = nn.BatchNorm2d(init_batchnorm_channels)

    def training_signal_modify(self, x: Tensor) -> Tensor:
        # Perturb the training data with noisy augmentations.
        return self.noise.sample(x.view(len(x), -1)).view(x.shape)

    def forward(self, x: Tensor) -> Tensor:
        if self.normalize is not None:
            x = self.normalize(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.model.forward(x) + self.class_balance_prediction_shift()

    def predict(self, x: Tensor) -> Tensor:
        """Provides hard predictions for randomized smoothing, overriding default."""
        with TU.evaluating(self):
            preds = [self._predict_single(x_single) for x_single in x]
            return torch.cat(preds, dim=0)

    def predict_fast_approximate(self, x: Tensor) -> Tensor:
        """Approximates the randomized smoothing prediction with fewer samples."""
        with TU.evaluating(self):
            n_override = 100  # Number of samples to use for fast approximate prediction
            # Project to the number of samples we want to be certifying for
            n_scale = int((self.n * self.cert_n_scale) // n_override)
            preds = [self._predict_single(x_single, n_override, n_scale) for x_single in x]
            return torch.cat(preds, dim=0)

    def _predict_single(
            self, x_single: Tensor, n_override: Optional[int]=None, n_scale=1
        ) -> Tensor:

        """Predicts for a batch_size 1 tensor, produces a singleton Tensor list class output.

        Args:
            x_single (Tensor): [...]. A signal tensor with only one batch element.
            n_override (int, optional): An optional n to use for prediction instead of the class
                member. Defaults to None.
            n_scale (int, optional): An optional n_scale to use for prediction. Works similarly to
                the class member cert_n_scale, but for prediction. n and n_scale are used for
                balancing. Defaults to 1.

        Returns:
            Tensor: [1] long. A singleton list tensor with the prediction.
        """
        x_single = x_single.unsqueeze(0) # Add a singleton batch dimension

        if isinstance(self.noise, rs_noises.SplitMethodDerandomized):
            # The splitting method derandomized has a special prediction API
            if n_override is None:
                counts = rs_smooth.smooth_predict_hard_derandomized(
                    self.forward, x_single, self.noise, noise_batch_size=self.nb
                )
            else:
                counts = rs_smooth.smooth_predict_hard_derandomized_subset(
                    self.forward, x_single, self.noise, n_override, noise_batch_size=self.nb
                )
            return self.noise.classify_and_certify_l1_exact_from_counts(counts)[0]

        # Standard prediction procedure for all non-splitting noises
        counts = rs_smooth.smooth_predict_hard(
            self.forward, x_single, self.noise, self.n if n_override is None else n_override,
            raw_count=True, noise_batch_size=self.nb
        )
        (na, nb), (ca, cb) = torch.topk(counts.int(), 2)

        if stats.binomtest(na * n_scale, (na + nb) * n_scale, 0.5).pvalue <= self.alpha:
            return ca.unsqueeze(0)

        return torch.tensor([-1]).type_as(x_single)

    def certify(self, x: Tensor, _) -> tuple[Tensor, Certificate]:
        assert x.shape[0] == 1

        with TU.evaluating(self):
            if isinstance(self.noise, rs_noises.SplitMethodDerandomized):
                # The splitting method derandomized has a special certification API
                counts = rs_smooth.smooth_predict_hard_derandomized(
                    self.forward, x, self.noise, noise_batch_size=self.nb
                )
                cats, certs = self.noise.classify_and_certify_l1_exact_from_counts(counts)
                certificate = Certificate.from_l1(certs.item(), self.data_in_n)
                return cats, certificate

            # Standard certification procedure for all non-splitting noises
            preds = rs_smooth.smooth_predict_hard(self.forward, x, self.noise, self.n0)
            top_cats = preds.probs.argmax(dim=1)
            prob_lb = rs_smooth.certify_prob_lb(
                self.forward, x, top_cats, 2 * self.alpha, self.noise,
                self.n, noise_batch_size=self.nb, sample_size_scale=self.cert_n_scale
            )

            if prob_lb > 0.5:
                certificate = Certificate({
                    Norm.L1: self.noise.certify_l1(prob_lb).item(),
                    Norm.L2: self.noise.certify_l2(prob_lb).item(),
                    Norm.LInf: self.noise.certify_linf(prob_lb).item(),
                })
                return top_cats, certificate

            return torch.tensor([-1]).type_as(x), Certificate.zero()

    def balance(self, datamodule: LightningDataModule) -> None:
        balance.binary_search_balance(self, datamodule)
