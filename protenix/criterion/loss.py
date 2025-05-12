import torch
import torch.nn as nn
import torch.nn.functional as F


class RankNetLoss(nn.Module):
    def __init__(
        self,
        normalize_logits: bool = False,
        weight_penalty: float = 0.01,
    ):
        super().__init__()
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(
        self,
        pm1_pred: torch.Tensor,
        pm2_pred: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        # label = pm2_label > pm1_label
        target = label.float()
        pred_logit = pm2_pred - pm1_pred

        loss = F.binary_cross_entropy_with_logits(pred_logit, target)

        if self.normalize_logits:
            loss += self.weight_penalty * (
                torch.mean(pm1_pred**2) + torch.mean(pm2_pred**2)
            )

        result = {
            "loss": loss,
        }
        return result


class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        **kwargs,
    ):
        # label = pm2_label > pm1_label
        target = label.float()
        loss = F.binary_cross_entropy(pred, target)

        result = {
            "loss": loss,
        }
        return result
