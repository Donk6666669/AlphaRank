import logging

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.metric import jit_distributed_available
from torchmetrics.utilities.data import dim_zero_cat


logger = logging.getLogger(__name__)


def cat_states(func):
    def wrapper(self, *args, **kwargs):
        if not jit_distributed_available() and not self._is_synced:
            output_dict = {
                attr: getattr(self, attr) for attr in self._reductions
            }

            for attr, reduction_fn in self._reductions.items():
                # pre-concatenate metric states that are lists to reduce number of all_gather operations
                if (
                    reduction_fn == dim_zero_cat
                    and isinstance(output_dict[attr], list)
                    and len(output_dict[attr]) >= 1
                ):
                    setattr(self, attr, dim_zero_cat(output_dict[attr]))
        return func(self, *args, **kwargs)

    return wrapper


class Accuracy(Metric):
    def __init__(self, threshold: float = 0.5, hard_set: bool = None):
        super().__init__()
        self.threshold = threshold
        self.hard_set = hard_set
        self.add_state("pred", default=[], dist_reduce_fx="cat")
        self.add_state("label", default=[], dist_reduce_fx="cat")

    def update(self, pred: torch.Tensor, label: torch.Tensor, **kwargs):
        if self.hard_set is None:
            self.pred.append(pred.flatten())
            self.label.append(label.flatten())
        else:
            if self.hard_set:
                self.pred.append(pred[kwargs["hard"]].flatten())
                self.label.append(label[kwargs["hard"]].flatten())
            else:
                self.pred.append(pred[~kwargs["hard"]].flatten())
                self.label.append(label[~kwargs["hard"]].flatten())

    @cat_states
    def compute(self):
        if len(self.pred) == 0:
            return None
        self.pred = (self.pred > self.threshold).float()
        return (self.pred == self.label).float().mean()
