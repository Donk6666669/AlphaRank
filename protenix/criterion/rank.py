import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from protenix.utils.timetool import with_time
from protenix.utils.torchtool import is_rank_zero

from .components.metrics import Accuracy

logger = logging.getLogger(__name__)


class RankCriterion(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

        # name, metric, step_compute
        self.train_metrics = [
            ("acc", Accuracy(), True),
            ("acc_hard", Accuracy(hard_set=True), True),
            ("acc_easy", Accuracy(hard_set=False), True),
        ]
        self.val_metrics = [
            ("acc", Accuracy(), False),
            ("acc_hard", Accuracy(hard_set=True), False),
            ("acc_easy", Accuracy(hard_set=False), False),
        ]

        for name, metric, _ in self.train_metrics + self.val_metrics:
            self.add_module(name, metric)

        self.samples = []

    def reset(self):
        for _, metric, _ in self.train_metrics + self.val_metrics:
            metric.reset()
        self.samples = []

    def forward(self, outputs: Dict[str, torch.Tensor]):
        result = {}

        loss_dict = self.loss(**outputs)
        result.update(loss_dict)

        # prepare metric inputs & update metrics
        metric_inputs = {
            "pred": outputs["pred"],
            "label": outputs["label"],
            "hard": outputs["hard"],
        }
        metrics = (
            self.train_metrics
            if self.training
            else self.train_metrics + self.val_metrics
        )
        for name, metric, is_compute in metrics:
            metric.update(**metric_inputs)
            if is_compute:
                value = metric.compute()
                if value is not None:
                    result[name] = value

        self.save_history(
            outputs["pred"],
            outputs["label"],
            outputs["hard"],
            outputs["idx"],
        )
        return result

    def save_history(self, pred, label, hard, idx):
        for item in zip(pred, label, hard, idx):
            self.samples.append(
                {
                    "pred": item[0].item(),
                    "label": item[1].item(),
                    "hard": item[2].item(),
                    "idx": item[3].item(),
                }
            )

    def compute(self, verbose=False, save_name=None):
        results = {}
        metrics = (
            self.train_metrics
            if self.training
            else self.train_metrics + self.val_metrics
        )
        for name, metric, _ in metrics:
            if metric._update_called:
                if verbose:
                    value, time_cost = with_time(
                        metric.compute, pretty_time=True
                    )()
                    if is_rank_zero() and value is not None:
                        logger.info(f"- {name}: {value} ({time_cost})")
                else:
                    value = metric.compute()
                if value is not None:
                    results[name] = value

        if save_name is not None and len(results) > 0:
            root = Path("results")
            root.mkdir(parents=True, exist_ok=True)
            file_path = root / f"{save_name}.json"
            logger.info(f"Saving results to {file_path}...")
            with open(file_path, "w") as f:
                json.dump(
                    {
                        "metrics": {
                            name: (
                                value.item()
                                if hasattr(value, "item")
                                else value
                            )
                            for name, value in results.items()
                        },
                        "detail": self.samples,
                    },
                    f,
                )
        return results
