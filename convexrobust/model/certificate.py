from collections import defaultdict
from enum import Enum
from typing import Dict, Type, TypeVar

from convexrobust.utils import torch_utils as TU

class Norm(Enum):
    L1 = 1
    L2 = 2
    LInf = float('inf')

# See https://github.com/python/typing/issues/58#issuecomment-326240794
CertificateT = TypeVar('CertificateT', bound='Certificate')

class Certificate:
    radius: Dict[Norm, float]

    def __init__(self, radius: Dict[Norm, float]):
        # A default dict that returns 0.0 if no radius is specified for a norm
        self.radius = defaultdict(float, radius)

    def __str__(self) -> str:
        return self.radius.__str__()

    @classmethod
    def zero(cls: Type[CertificateT]) -> CertificateT:
        return cls({})

    @classmethod
    def from_l1(cls: Type[CertificateT], radius: float, dim: int) -> CertificateT:
        return cls(cls._convert_radii(1, radius, dim))

    @classmethod
    def from_l2(cls: Type[CertificateT], radius: float, dim: int) -> CertificateT:
        return cls(cls._convert_radii(2, radius, dim))

    @classmethod
    def from_linf(cls: Type[CertificateT], radius: float, dim: int) -> CertificateT:
        return cls(cls._convert_radii(float('inf'), radius, dim))

    @classmethod
    def _convert_radii(_, from_norm, radius, dim):
        return {
            Norm.L1: radius * TU.norm_ball_conversion_factor(1, from_norm, dim),
            Norm.L2: radius * TU.norm_ball_conversion_factor(2, from_norm, dim),
            Norm.LInf: radius * TU.norm_ball_conversion_factor(float('inf'), from_norm, dim),
        }
