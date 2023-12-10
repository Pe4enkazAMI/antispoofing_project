from typing import Optional
import torch
from torch import Tensor 
import torch.nn as nn


class ASLoss(nn.CrossEntropyLoss):
    def __init__(self, 
                 weight: Tensor | None = torch.tensor([1.0, 1.0]),
                 size_average=None, 
                 ignore_index: int = -100, 
                 reduce=None, 
                 reduction: str = 'mean', 
                 label_smoothing: float = 0) -> None:
        super().__init__(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, logits: Tensor, targets: Tensor, *args, **kwargs) -> Tensor:
        return {"ASLoss" : super().forward(logits, targets)}
