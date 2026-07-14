from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from ternarylayers.bit import Bit

class Linear(Bit.Linear):
    pass

class Conv2d(Bit.Conv2d):
    pass
