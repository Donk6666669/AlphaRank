import logging
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.metrics.rmsd import weighted_rigid_align
from protenix.model.modules.frames import (
    expressCoordinatesInFrame,
    gather_frame_atom_by_indices,
)
from protenix.model.utils import expand_at_dim
from protenix.openfold_local.utils.checkpointing import get_checkpoint_fn
from protenix.utils.torch_utils import cdist


class RankNetLoss(nn.Module):
    def __init__(
        self,
        weight_by_diff: bool = False,
        normalize_logits: bool = False,
        weight_penalty: float = 0.01,
    ):
        super().__init__()
        self.weight_by_diff = weight_by_diff
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(
        self,
        pair1_logit: torch.Tensor,
        pair2_logit: torch.Tensor,
        pair_label: torch.Tensor,
        **kwargs,
    ):
        # pair_label = pair2_label - pair1_label
        target = (pair_label > 0).float()
        pred_logit = pair2_logit - pair1_logit

        weight = None
        if self.weight_by_diff:
            weight = torch.abs(pair_label)
        loss = F.binary_cross_entropy_with_logits(
            pred_logit, target, weight=weight
        )

        if self.normalize_logits:
            loss += self.weight_penalty * (
                torch.mean(pair1_logit**2) + torch.mean(pair2_logit**2)
            )

        result = {
            "loss": loss,
        }
        return result