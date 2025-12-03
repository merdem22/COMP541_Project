"""
Placeholder for future graph reasoning module.

Intended to sit after fusion and before the detection head.
"""

import torch
from torch import nn


class GraphModuleStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No-op pass-through for now.
        return x
