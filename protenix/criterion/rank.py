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
            ("acc_rerank", Accuracy(hard_set=True), True),
            ("acc_screen", Accuracy(hard_set=False), True),
        ]
        self.val_metrics = [
            ("acc", Accuracy(), False),
            ("acc_rerank", Accuracy(hard_set=True), False),
            ("acc_screen", Accuracy(hard_set=False), False),
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



class ListRankCriterion(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

        # 这里保留你的指标配置，没变
        self.train_metrics = [
            ("acc", Accuracy(), True),
            ("acc_rerank", Accuracy(hard_set=True), True),
            ("acc_screen", Accuracy(hard_set=False), True),
        ]
        self.val_metrics = [
            ("acc", Accuracy(), False),
            ("acc_rerank", Accuracy(hard_set=True), False),
            ("acc_screen", Accuracy(hard_set=False), False),
        ]

        for name, metric, _ in self.train_metrics + self.val_metrics:
            self.add_module(name, metric)

        self.samples = []

    def reset(self):
        for _, metric, _ in self.train_metrics + self.val_metrics:
            metric.reset()
        self.samples = []

    def forward(self, outputs: dict):
        """
        outputs字典要求包含：
        - pm1_pred, pm2_pred, pm3_pred  用于loss计算
        - pred, label, hard, idx      用于指标计算和保存
        """

        result = {}

        # 调用 PlackettLuceLoss，只用3个预测
        loss_dict = self.loss(
            pm1_pred=outputs["pm1_pred"],
            pm2_pred=outputs["pm2_pred"],
            pm3_pred=outputs["pm3_pred"],
        )
        result.update(loss_dict)

        # 准备指标输入，保持原来格式
        metric_inputs = {
            "pred": outputs["pred"],
            "label": outputs["label"],
            "hard": outputs["hard"],
        }
        metrics = (
            self.train_metrics if self.training else self.train_metrics + self.val_metrics
        )
        for name, metric, is_compute in metrics:
            metric.update(**metric_inputs)
            if is_compute:
                value = metric.compute()
                if value is not None:
                    result[name] = value

        # 保存预测历史，保存的是指标相关的pred等
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
            self.train_metrics if self.training else self.train_metrics + self.val_metrics
        )
        for name, metric, _ in metrics:
            if metric._update_called:
                if verbose:
                    # 你的with_time和is_rank_zero实现这里需要
                    value, time_cost = with_time(metric.compute, pretty_time=True)()
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
                                value.item() if hasattr(value, "item") else value
                            )
                            for name, value in results.items()
                        },
                        "detail": self.samples,
                    },
                    f,
                )
        return results
    
class ListListRankCriterion(nn.Module):
    def __init__(self, loss: nn.Module):
        super().__init__()
        self.loss = loss

        # 这里取消所有metric，PlackettLuceLoss不依赖label等
        # self.train_metrics = []
        # self.val_metrics = []

        self.samples = []

    def reset(self):
        # 如果以后有metric，可以恢复这里
        # for _, metric, _ in self.train_metrics + self.val_metrics:
        #     metric.reset()
        self.samples = []

    def forward(self, outputs: dict):
        """
        outputs 期望包含:
            - pm1_pred: tensor or batch of tensor
            - pm2_pred: tensor or batch of tensor
            - pm3_pred: tensor or batch of tensor
            - idx: tensor or list of sample idx
        """
        result = {}

        # 调用 loss，传入对应参数
        loss_dict = self.loss(
            pm1_pred=outputs["pm1_pred"],
            pm2_pred=outputs["pm2_pred"],
            pm3_pred=outputs["pm3_pred"],
        )
        result.update(loss_dict)

        # 因为没有 metric，暂时不更新metric和计算

        # 保存预测历史
        self.save_history(
            outputs["pm1_pred"],
            outputs["pm2_pred"],
            outputs["pm3_pred"],
            outputs["idx"],
        )
        return result

    def save_history(self, pm1_pred, pm2_pred, pm3_pred, idx):
        """
        pm1_pred, pm2_pred, pm3_pred: tensor, 可能含 batch 维度
        idx: tensor 或 list，batch size
        """
        # 确保都是tensor
        if not isinstance(pm1_pred, torch.Tensor):
            pm1_pred = torch.tensor(pm1_pred)
        if not isinstance(pm2_pred, torch.Tensor):
            pm2_pred = torch.tensor(pm2_pred)
        if not isinstance(pm3_pred, torch.Tensor):
            pm3_pred = torch.tensor(pm3_pred)
        if not isinstance(idx, torch.Tensor):
            idx = torch.tensor(idx)

        # 支持batch，遍历保存
        for i in range(pm1_pred.shape[0]):
            self.samples.append(
                {
                    "pm1_pred": pm1_pred[i].item(),
                    "pm2_pred": pm2_pred[i].item(),
                    "pm3_pred": pm3_pred[i].item(),
                    "idx": idx[i].item(),
                }
            )

    def compute(self, verbose=False, save_name=None):
        """
        目前无metric计算，直接返回空dict
        """
        results = {}

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
