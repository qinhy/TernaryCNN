"""Public API for TernaryCNN's reusable ternary layers."""

from importlib.metadata import version

from .acts import ActModels
from .bit import Bit, convert_to_ternary
from .convs import Conv2dModels, Conv2dModules
from .linear import LinearModels, LinearModules
from .norms import NormModels
from .pool import PoolModels, PoolModules
from .resnet import ResNetModels, ResNetModules

__version__ = version("ternarycnn")

__all__ = [
    "ActModels",
    "Bit",
    "Conv2dModels",
    "Conv2dModules",
    "LinearModels",
    "LinearModules",
    "NormModels",
    "PoolModels",
    "PoolModules",
    "ResNetModels",
    "ResNetModules",
    "convert_to_ternary",
]

