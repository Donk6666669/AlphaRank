import logging

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch_scatter import scatter_mean
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


class TaskAssayMetric(Metric):
    def __init__(self, task: str = None, by_assay: bool = False, **kwargs):
        super().__init__()
        self.task = task
        self.by_assay = by_assay
        if self.task is not None:
            self.task = SIUDataset.TASK[self.task.upper()]
        self.add_state("pred", default=[], dist_reduce_fx="cat")
        self.add_state("label", default=[], dist_reduce_fx="cat")
        if self.by_assay:
            self.add_state("assay", default=[], dist_reduce_fx="cat")

    def update(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        task: int = None,
        assay: torch.Tensor = None,
        **kwargs
    ):
        if self.task is None or task == self.task.value:
            self.pred.append(pred.flatten())
            self.label.append(label.flatten())
            if self.by_assay:
                self.assay.append(assay.flatten())


class SpearmanR(TaskAssayMetric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    @cat_states
    def compute(self):
        if len(self.pred) == 0:
            return None
        if not self.by_assay:
            corr, _ = spearmanr(
                self.pred.cpu().numpy(), self.label.cpu().numpy()
            )
        else:
            assay = torch.unique(self.assay)
            corr = []
            for a in assay:
                mask = self.assay == a
                pred = self.pred[mask].cpu().numpy()
                label = self.label[mask].cpu().numpy()

                # skip assay with less than 5 ligands
                if len(pred) < 5:
                    continue
                # if pred or label is constant array, skip
                if np.all(pred == pred[0]) or np.all(label == label[0]):
                    continue
                corr.append(spearmanr(pred, label)[0])
            corr = torch.tensor(corr).mean().item()
        return torch.tensor(corr).to(self.label.device)


class PearsonR(TaskAssayMetric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    @cat_states
    def compute(self):
        if len(self.pred) == 0:
            return None
        if not self.by_assay:
            if self.pred.cpu().numpy().all() == 0:
                corr = 0
            elif np.any(np.isnan(self.pred.cpu().numpy())) or np.any(
                np.isinf(self.pred.cpu().numpy())
            ):
                corr = 0
            else:
                corr, _ = pearsonr(
                    self.pred.cpu().numpy(), self.label.cpu().numpy()
                )

        else:
            assay = torch.unique(self.assay)
            corr = [0.0]
            for a in assay:
                mask = self.assay == a
                pred = self.pred[mask].cpu().numpy()
                label = self.label[mask].cpu().numpy()

                # skip assay with less than 5 ligands
                if len(pred) < 5:
                    continue
                if np.all(pred == pred[0]) or np.all(label == label[0]):
                    continue
                elif np.any(np.isnan(pred)) or np.any(np.isnan(label)):
                    continue
                else:
                    corr.append(pearsonr(pred, label)[0])
            corr = torch.tensor(corr).mean().item()
        return torch.tensor(corr).to(self.pred.device)


class RMSE(TaskAssayMetric):

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    @cat_states
    def compute(self):
        if len(self.pred) == 0:
            return None

        if not self.by_assay:
            mse = F.mse_loss(self.pred, self.label)
            rmse = torch.sqrt(mse)
        else:
            mse = F.mse_loss(self.pred, self.label, reduction="none")
            rmse = torch.sqrt(mse)
            assay = self.assay.unique(sorted=True, return_inverse=True)[1]
            rmse = scatter_mean(rmse, assay).mean()
        return rmse
