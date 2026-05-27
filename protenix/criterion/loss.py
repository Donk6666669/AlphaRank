import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    
# class PlackettLuceLoss(nn.Module):
#     def __init__(self, tau=1.0):
#         super().__init__()
#         self.tau = tau

#     def forward(self, pred, label, **kwargs):
#         indices = torch.argsort(label, descending=True)
#         sorted_scores = pred[indices]
#         thetas = torch.exp(sorted_scores / self.tau)

#         loss = 0.0
#         n = thetas.size(0)
#         for k in range(n):
#             denom = thetas[k:].sum()
#             numer = thetas[k]
#             log_prob = torch.log(numer / denom + 1e-8)
#             mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
#             loss -= mu_k * log_prob

#         return {"loss": loss}

# class PlackettLuceLoss(nn.Module):
#     def __init__(self, tau=1.0):
#         super().__init__()
#         self.tau = tau

#     def forward(self, pm1_pred, pm2_pred, pm3_pred):
#         # 把三个预测值合成一个Tensor，形状 (3,)
#         preds = torch.stack([pm1_pred, pm2_pred, pm3_pred])  # shape: (3,)

#         # 因为真实亲和力已排好序，按顺序计算Plackett-Luce概率
#         thetas = torch.exp(preds / self.tau)  # shape: (3,)

#         loss = 0.0
#         n = thetas.size(0)  # 3
#         for k in range(n):
#             denom = thetas[k:].sum()
#             numer = thetas[k]
#             log_prob = torch.log(numer / denom + 1e-8)
#             mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
#             loss -= mu_k * log_prob

#         return {"loss": loss}
class PlackettLuceLoss(nn.Module):
    def __init__(self, tau=0.7, normalize_logits=False, weight_penalty=0.01):
        super().__init__()
        self.tau = tau
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred])  # shape (3,)

        thetas = torch.exp(preds / self.tau)
        loss = 0.0
        n = thetas.size(0)
        for k in range(n):
            denom = thetas[k:].sum()
            numer = thetas[k]
            log_prob = torch.log(numer / denom + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            loss -= mu_k * log_prob

        if self.normalize_logits:
            # L2 penalty on logits
            loss += self.weight_penalty * torch.mean(preds ** 2)

        return {"loss": loss}
    
class RefinedPlackettLuceLoss(nn.Module):
    def __init__(self, tau=0.5, normalize_logits=False, weight_penalty=0.01):
        super().__init__()
        self.tau = tau
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred])  # shape (3,)

        # 数值稳定化：减去最大值
        preds_stable = preds / self.tau
        preds_stable = preds_stable - preds_stable.max()  

        # 计算 theta
        thetas = torch.exp(preds_stable)

        loss = 0.0
        n = thetas.size(0)
        for k in range(n):
            denom = thetas[k:].sum()
            numer = thetas[k]
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)  # 双重1e-8防NaN
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            loss -= mu_k * log_prob

        if self.normalize_logits:
            # L2 penalty on logits (用原始 preds，不是稳定化后的)
            loss += self.weight_penalty * torch.mean(preds ** 2)

        return {"loss": loss}

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class StablePlackettLuceLoss(nn.Module):
    def __init__(self, tau=0.5, normalize_logits=False, weight_penalty=0.0):
        super().__init__()
        self.tau = tau
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred])  # shape (3,)

        # 数值稳定化 + tau 缩放
        preds_scaled = preds / self.tau

        loss = 0.0
        n = preds_scaled.size(0)
        eps = 1e-8

        for k in range(n):
            # 直接用 logsumexp 计算 denom
            log_denom = torch.logsumexp(preds_scaled[k:], dim=0)
            log_numer = preds_scaled[k]
            log_prob = log_numer - log_denom
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            loss -= mu_k * log_prob

        if self.normalize_logits:
            loss += self.weight_penalty * torch.mean(preds ** 2)

        return {"loss": loss}

import torch
import torch.nn as nn
import math

class TestPlackettLuceLoss(nn.Module):
    def __init__(self, 
                 tau_start=1.0, 
                 tau_end=0.1, 
                 anneal_steps=1000, 
                 normalize_logits=False, 
                 weight_penalty=0.01):
        super().__init__()
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty
        # 保存当前 step
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def get_tau(self):
        # 线性退火
        progress = min(self.step.item() / self.anneal_steps, 1.0)
        return self.tau_start + (self.tau_end - self.tau_start) * progress

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        tau = self.get_tau()
        self.step += 1  # 每次调用 forward 递增

        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred])  # shape (3,)

        # 数值稳定化
        preds_stable = preds / tau
        preds_stable = preds_stable - preds_stable.max()

        thetas = torch.exp(preds_stable)

        loss = 0.0
        n = thetas.size(0)
        for k in range(n):
            denom = thetas[k:].sum()
            numer = thetas[k]
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            loss -= mu_k * log_prob

        if self.normalize_logits:
            loss += self.weight_penalty * torch.mean(preds ** 2)

        # 保持原有接口，只返回 loss
        return {"loss": loss}
    
import torch
import torch.nn as nn
import math

class CosinePlackettLuceLoss(nn.Module):
    def __init__(self, 
                 tau_start=1.0, 
                 tau_end=0.6, 
                 anneal_steps=1000, 
                 normalize_logits=False, 
                 weight_penalty=0.01):
        super().__init__()
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.anneal_steps = anneal_steps
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty
        self.register_buffer("step", torch.tensor(0, dtype=torch.long))

    def get_tau(self):
        # 余弦退火
        progress = min(self.step.item() / self.anneal_steps, 1.0)
        cos_factor = (1 + math.cos(math.pi * progress)) / 2  # 从1->0
        tau = self.tau_end + (self.tau_start - self.tau_end) * cos_factor
        return tau

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        tau = self.get_tau()
        self.step += 1  # 每次调用 forward 递增

        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred])  # shape (3,)

        # 数值稳定化
        preds_stable = preds / tau
        preds_stable = preds_stable - preds_stable.max()

        thetas = torch.exp(preds_stable)

        loss = 0.0
        n = thetas.size(0)
        for k in range(n):
            denom = thetas[k:].sum()
            numer = thetas[k]
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            loss -= mu_k * log_prob

        if self.normalize_logits:
            loss += self.weight_penalty * torch.mean(preds ** 2)

        return {"loss": loss}

# class CombinedRankLoss(nn.Module):
#     def __init__(self, lambda_rank=1.0, tau=1.0):
#         super().__init__()
#         self.ranknet_loss = RankNetLoss()
#         self.pl_loss = PlackettLuceLoss(tau=tau)
#         self.lambda_rank = lambda_rank

#     def forward(self, pred=None, label=None,
#                 pm1_pred=None, pm2_pred=None, **kwargs):
#         # RankNet 部分（pairwise）
#         ranknet_out = self.ranknet_loss(pm1_pred, pm2_pred, label)
#         ranknet_loss = ranknet_out["loss"]

#         # Plackett-Luce 部分（listwise）
#         pl_out = self.pl_loss(pred, label)
#         pl_loss = pl_out["loss"]

#         # 加权组合
#         total_loss = ranknet_loss + self.lambda_rank * pl_loss

#         return {
#             "loss": total_loss,
#             "ranknet_loss": ranknet_loss,
#             "plackett_luce_loss": pl_loss,
#         }

class CombinedRankLoss(nn.Module):
    def __init__(self, lambda_rank=1.0, tau=1.0, normalize_logits=False, weight_penalty=0.01):
        super().__init__()
        self.ranknet_loss = RankNetLoss(normalize_logits=normalize_logits, weight_penalty=weight_penalty)
        self.pl_loss = PlackettLuceLoss(tau=tau)
        self.lambda_rank = lambda_rank

    def forward(self, pred=None, label=None,
                pm1_pred=None, pm2_pred=None, **kwargs):
        # RankNet 部分（pairwise）
        ranknet_out = self.ranknet_loss(pm1_pred, pm2_pred, label)
        ranknet_loss = ranknet_out["loss"]

        # Plackett-Luce 部分（listwise）
        pl_out = self.pl_loss(pred, label)
        pl_loss = pl_out["loss"]

        # 加权组合
        total_loss = ranknet_loss + self.lambda_rank * pl_loss

        return {
            "loss": total_loss,
            "ranknet_loss": ranknet_loss,
            "plackett_luce_loss": pl_loss,
        }


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CombinedListwiseLoss(nn.Module):
    def __init__(self, tau=0.4, lambda_listwise=0.00046, normalize_logits=False, weight_penalty=0.01):
        """
        lambda_listwise: listwise loss 的权重
        tau: Plackett-Luce softmax 温度
        """
        super().__init__()
        self.tau = tau
        self.lambda_listwise = lambda_listwise
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        """
        默认 pm1 > pm2 > pm3
        支持 batch 训练
        """
        batch_size = pm1_pred.size(0)

        # -----------------------
        # 1. 计算 pairwise loss
        # -----------------------
        target12 = torch.ones_like(pm1_pred)  # pm2 > pm1 -> target=1
        target23 = torch.ones_like(pm2_pred)  # pm3 > pm2 -> target=1

        pred12 = pm2_pred - pm1_pred
        pred23 = pm3_pred - pm2_pred

        loss12 = F.binary_cross_entropy_with_logits(pred12, target12)
        loss23 = F.binary_cross_entropy_with_logits(pred23, target23)

        pairwise_loss = loss12 + loss23

        # -----------------------
        # 2. 计算 listwise loss (Plackett-Luce)
        # -----------------------
        # shape (batch, 3)
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred], dim=1)
        preds_stable = preds / self.tau
        preds_stable = preds_stable - preds_stable.max(dim=1, keepdim=True)[0]  # 数值稳定
        thetas = torch.exp(preds_stable)

        n = thetas.size(1)  # 3
        listwise_loss = 0.0
        for k in range(n):
            numer = thetas[:, k]
            denom = thetas[:, k:].sum(dim=1)
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            listwise_loss -= (mu_k * log_prob).mean()

        # -----------------------
        # 3. L2 正则 (可选)
        # -----------------------
        if self.normalize_logits:
            l2_loss = torch.mean(pm1_pred**2 + pm2_pred**2 + pm3_pred**2)
            pairwise_loss += self.weight_penalty * l2_loss
            listwise_loss += self.weight_penalty * torch.mean(preds**2)

        # -----------------------
        # 4. 合并 loss
        # -----------------------
        loss = pairwise_loss + self.lambda_listwise * listwise_loss

        return {"loss": loss}


class CombinedLossWithUncertainty(nn.Module):
    def __init__(self, tau=0.5, normalize_logits=False, weight_penalty=0.01):
        super().__init__()
        self.tau = tau
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

        # 可学习的不确定性参数（初始化 log_sigma=0 => sigma=1）
        self.log_sigma_pair = nn.Parameter(torch.zeros(1))
        self.log_sigma_list = nn.Parameter(torch.zeros(1))

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        # ---------------- Pairwise loss ----------------
        target12 = torch.ones_like(pm1_pred)
        target23 = torch.ones_like(pm2_pred)

        pred12 = pm2_pred - pm1_pred
        pred23 = pm3_pred - pm2_pred

        loss12 = F.binary_cross_entropy_with_logits(pred12, target12)
        loss23 = F.binary_cross_entropy_with_logits(pred23, target23)
        pairwise_loss = loss12 + loss23

        # ---------------- Listwise loss ----------------
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred], dim=1)
        preds_stable = preds / self.tau
        preds_stable = preds_stable - preds_stable.max(dim=1, keepdim=True)[0]
        thetas = torch.exp(preds_stable)

        n = thetas.size(1)
        listwise_loss = 0.0
        for k in range(n):
            numer = thetas[:, k]
            denom = thetas[:, k:].sum(dim=1)
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            listwise_loss -= (mu_k * log_prob).mean()

        # ---------------- Uncertainty weighting ----------------
        sigma_pair = torch.exp(self.log_sigma_pair)
        sigma_list = torch.exp(self.log_sigma_list)

        loss = (pairwise_loss / (2 * sigma_pair**2) +
                listwise_loss / (2 * sigma_list**2) +
                self.log_sigma_pair + self.log_sigma_list)

        return {"loss": loss, 
                "pairwise_loss": pairwise_loss.detach(),
                "listwise_loss": listwise_loss.detach(),
                "sigma_pair": sigma_pair.item(),
                "sigma_list": sigma_list.item()}

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CombinedListwiseLossLearnableLambda(nn.Module):
    def __init__(self, tau=0.5, init_lambda=0.0005, normalize_logits=False, weight_penalty=0.01):
        """
        init_lambda: λ 的初始值（经验设定，比如 0.0005）
        tau: Plackett-Luce softmax 温度
        """
        super().__init__()
        self.tau = tau
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

        # 将 init_lambda 反算成 raw_lambda 作为初始化值
        # softplus(x) ≈ ln(1+e^x)，我们用 inverse_softplus 来初始化
        raw_init = math.log(math.exp(init_lambda) - 1.0)
        self.raw_lambda = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        """
        默认 pm1 > pm2 > pm3
        支持 batch 训练
        """
        batch_size = pm1_pred.size(0)

        # -----------------------
        # 1. pairwise loss
        # -----------------------
        target12 = torch.ones_like(pm1_pred)
        target23 = torch.ones_like(pm2_pred)

        pred12 = pm2_pred - pm1_pred
        pred23 = pm3_pred - pm2_pred

        loss12 = F.binary_cross_entropy_with_logits(pred12, target12)
        loss23 = F.binary_cross_entropy_with_logits(pred23, target23)

        pairwise_loss = loss12 + loss23

        # -----------------------
        # 2. listwise loss (Plackett-Luce)
        # -----------------------
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred], dim=1)
        preds_stable = preds / self.tau
        preds_stable = preds_stable - preds_stable.max(dim=1, keepdim=True)[0]
        thetas = torch.exp(preds_stable)

        n = thetas.size(1)  # 3
        listwise_loss = 0.0
        for k in range(n):
            numer = thetas[:, k]
            denom = thetas[:, k:].sum(dim=1)
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            listwise_loss -= (mu_k * log_prob).mean()

        # -----------------------
        # 3. L2 正则 (可选)
        # -----------------------
        if self.normalize_logits:
            l2_loss = torch.mean(pm1_pred**2 + pm2_pred**2 + pm3_pred**2)
            pairwise_loss += self.weight_penalty * l2_loss
            listwise_loss += self.weight_penalty * torch.mean(preds**2)

        # -----------------------
        # 4. 合并 loss (λ 可学习)
        # -----------------------
        lambda_listwise = F.softplus(self.raw_lambda)  # 保证 λ>0
        loss = pairwise_loss + lambda_listwise * listwise_loss

        return {
            "loss": loss,
            "lambda_listwise": lambda_listwise.detach().cpu().item()
        }

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedListwiseLoss_LambdaBounded(nn.Module):
    def __init__(self, tau=0.5, init_lambda=0.0005, low=4e-4, high=6e-4,
                 normalize_logits=False, weight_penalty=0.01, lambda_reg_weight=0.0):
        """
        init_lambda: 期望的初始 lambda（比如 0.0005）
        low, high: lambda 学习的上下界
        lambda_reg_weight: 可选的 lambda 正则系数 (small, e.g. 1e-6)
        """
        super().__init__()
        assert low < high
        self.tau = tau
        self.low = float(low)
        self.high = float(high)
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty
        self.lambda_reg_weight = lambda_reg_weight

        # 计算 raw 初始化（logit）
        p = (init_lambda - self.low) / (self.high - self.low)
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        raw_init = math.log(p / (1.0 - p))

        self.raw_lambda = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

    def forward(self, pm1_pred, pm2_pred, pm3_pred, return_components=False):
        # pairwise
        target12 = torch.ones_like(pm1_pred)
        target23 = torch.ones_like(pm2_pred)
        pred12 = pm2_pred - pm1_pred
        pred23 = pm3_pred - pm2_pred
        loss12 = F.binary_cross_entropy_with_logits(pred12, target12)
        loss23 = F.binary_cross_entropy_with_logits(pred23, target23)
        pairwise_loss = loss12 + loss23

        # listwise (Plackett-Luce) stable implementation
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred], dim=1)  # B x 3
        preds_stable = preds / self.tau
        log_probs = 0.0
        log_soft = F.log_softmax(preds_stable, dim=1)
        log_probs = log_probs + log_soft[:, 0]
        rem = preds_stable[:, 1:]
        log_soft_rem = F.log_softmax(rem, dim=1)
        log_probs = log_probs + log_soft_rem[:, 0]
        listwise_loss = -log_probs.mean()

        if self.normalize_logits:
            l2_loss = torch.mean(pm1_pred**2 + pm2_pred**2 + pm3_pred**2)
            pairwise_loss = pairwise_loss + self.weight_penalty * l2_loss
            listwise_loss = listwise_loss + self.weight_penalty * torch.mean(preds**2)

        # map raw -> [low, high]
        lam = torch.sigmoid(self.raw_lambda) * (self.high - self.low) + self.low

        reg = 0.0
        if self.lambda_reg_weight > 0:
            init_val = (self.low + self.high) / 2.0
            reg = self.lambda_reg_weight * (lam - init_val).pow(2)

        loss = pairwise_loss + lam * listwise_loss + reg

        if return_components:
            return {
                "loss": loss,
                "pairwise_loss": pairwise_loss.detach(),
                "listwise_loss": listwise_loss.detach(),
                "lambda": float(lam.detach().cpu().numpy()),  # ✅ 确保是 Python float
                "raw_lambda": float(self.raw_lambda.detach().cpu().numpy()),
                "lambda_reg": float(reg.detach().cpu().numpy()) if isinstance(reg, torch.Tensor) else float(reg)
            }
        return {"loss": loss}


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NewCombinedListwiseLoss(nn.Module):
    def __init__(self, tau=0.5, alpha_triplet=-0.1, beta_listwise=0.0, 
                 normalize_logits=False, weight_penalty=0.01):
        """
        alpha_triplet: triplet loss (pm1 > pm3) 的权重
        beta_listwise: listwise loss 的权重
        tau: Plackett-Luce softmax 温度
        """
        super().__init__()
        self.tau = tau
        self.alpha_triplet = alpha_triplet
        self.beta_listwise = beta_listwise
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        """
        默认 pm1 > pm2 > pm3
        支持 batch 训练
        """
        batch_size = pm1_pred.size(0)

        # -----------------------
        # 1. Pairwise loss (L1>2 + L2>3)
        # -----------------------
        target12 = torch.ones_like(pm1_pred)  # pm1 > pm2 -> target=1
        target23 = torch.ones_like(pm2_pred)  # pm2 > pm3 -> target=1

        pred12 = pm2_pred - pm1_pred  # 大的减小的
        pred23 = pm3_pred - pm2_pred

        loss12 = F.binary_cross_entropy_with_logits(pred12, target12)
        loss23 = F.binary_cross_entropy_with_logits(pred23, target23)

        pairwise_loss = loss12 + loss23

        # -----------------------
        # 2. Triplet loss (L1>3)
        # -----------------------
        target13 = torch.ones_like(pm1_pred)  # pm1 > pm3 -> target=1
        pred13 = pm3_pred - pm1_pred
        triplet_loss = F.binary_cross_entropy_with_logits(pred13, target13)

        # -----------------------
        # 3. Listwise loss (Plackett-Luce)
        # -----------------------
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred], dim=1)
        preds_stable = preds / self.tau
        preds_stable = preds_stable - preds_stable.max(dim=1, keepdim=True)[0]  # 数值稳定
        thetas = torch.exp(preds_stable)

        n = thetas.size(1)  # 3
        listwise_loss = 0.0
        for k in range(n):
            numer = thetas[:, k]
            denom = thetas[:, k:].sum(dim=1)
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            listwise_loss -= (mu_k * log_prob).mean()

        # -----------------------
        # 4. L2 正则 (可选)
        # -----------------------
        if self.normalize_logits:
            l2_loss = torch.mean(pm1_pred**2 + pm2_pred**2 + pm3_pred**2)
            pairwise_loss += self.weight_penalty * l2_loss
            listwise_loss += self.weight_penalty * torch.mean(preds**2)

        # -----------------------
        # 5. 合并 loss
        # -----------------------
        # L = L_pairwise + alpha * L_triplet + beta * L_listwise
        loss = pairwise_loss + self.alpha_triplet * triplet_loss + self.beta_listwise * listwise_loss

        return {"loss": loss}

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CombinedListwiseLoss1(nn.Module):
    def __init__(self, 
                 tau=0.5, 
                 lambda_listwise=0.0, 
                 lambda_consistency=-0.1,
                 normalize_logits=False, 
                 weight_penalty=0.01):
        """
        lambda_listwise: listwise loss 的权重
        lambda_consistency: 概率一致性约束的权重
        tau: Plackett-Luce softmax 温度
        """
        super().__init__()
        self.tau = tau
        self.lambda_listwise = lambda_listwise
        self.lambda_consistency = lambda_consistency
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        """
        默认 pm1 > pm2 > pm3
        支持 batch 训练
        """
        batch_size = pm1_pred.size(0)

        # -----------------------
        # 1. 计算 pairwise loss
        # -----------------------
        # 按你的要求：pred12 = pm2 - pm1, pred23 = pm3 - pm2
        # target=1 对应 “右边比分数更大”（pm2>pm1、pm3>pm2）
        target12 = torch.ones_like(pm1_pred)  # pm2 > pm1 -> target=1
        target23 = torch.ones_like(pm2_pred)  # pm3 > pm2 -> target=1

        pred12 = pm2_pred - pm1_pred
        pred23 = pm3_pred - pm2_pred

        loss12 = F.binary_cross_entropy_with_logits(pred12, target12)
        loss23 = F.binary_cross_entropy_with_logits(pred23, target23)

        pairwise_loss = loss12 + loss23

        # -----------------------
        # 2. 概率一致性正则 (Transitivity)
        # -----------------------
        # 由于当前定义的是 P(2>1) 和 P(3>2)，其传递性对应 P(3>1) ≈ P(3>2)*P(2>1)
        p12 = torch.sigmoid(pred12)                    # P(2>1)
        p23 = torch.sigmoid(pred23)                    # P(3>2)
        p31 = torch.sigmoid(pm3_pred - pm1_pred)       # P(3>1)
        consistency_loss = F.mse_loss(p31, p23 * p12)

        # -----------------------
        # 3. 计算 listwise loss (Plackett-Luce)
        # -----------------------
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred], dim=1)
        preds_stable = preds / self.tau
        preds_stable = preds_stable - preds_stable.max(dim=1, keepdim=True)[0]  # 数值稳定
        thetas = torch.exp(preds_stable)

        n = thetas.size(1)  # 3
        listwise_loss = 0.0
        for k in range(n):
            numer = thetas[:, k]
            denom = thetas[:, k:].sum(dim=1)
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            listwise_loss -= (mu_k * log_prob).mean()

        # -----------------------
        # 4. L2 正则 (可选)
        # -----------------------
        if self.normalize_logits:
            l2_loss = torch.mean(pm1_pred**2 + pm2_pred**2 + pm3_pred**2)
            pairwise_loss += self.weight_penalty * l2_loss
            listwise_loss += self.weight_penalty * torch.mean(preds**2)

        # -----------------------
        # 5. 合并 loss
        # -----------------------
        loss = (pairwise_loss 
                + self.lambda_listwise * listwise_loss
                + self.lambda_consistency * consistency_loss)

        return {"loss": loss}

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RefinedCombinedListwiseLoss(nn.Module):
    def __init__(self, 
                 tau=0.5, 
                 lambda_listwise=0.0, 
                 w12=1.0, 
                 w23=1.0,
                 normalize_logits=False, 
                 weight_penalty=0.01):
        """
        lambda_listwise: listwise loss 的权重
        tau: Plackett-Luce softmax 温度
        w12: loss12 的权重
        w23: loss23 的权重
        """
        super().__init__()
        self.tau = tau
        self.lambda_listwise = lambda_listwise
        self.w12 = w12
        self.w23 = w23
        self.normalize_logits = normalize_logits
        self.weight_penalty = weight_penalty

    def forward(self, pm1_pred, pm2_pred, pm3_pred):
        """
        默认 pm1 > pm2 > pm3
        支持 batch 训练
        """
        batch_size = pm1_pred.size(0)

        # -----------------------
        # 1. 计算 pairwise loss
        # -----------------------
        target12 = torch.ones_like(pm1_pred)  # pm2 > pm1 -> target=1
        target23 = torch.ones_like(pm2_pred)  # pm3 > pm2 -> target=1

        pred12 = pm2_pred - pm1_pred
        pred23 = pm3_pred - pm2_pred

        loss12 = F.binary_cross_entropy_with_logits(pred12, target12)
        loss23 = F.binary_cross_entropy_with_logits(pred23, target23)

        # -----------------------
        # 2. 计算 listwise loss (Plackett-Luce)
        # -----------------------
        # shape (batch, 3)
        preds = torch.stack([pm1_pred, pm2_pred, pm3_pred], dim=1)
        preds_stable = preds / self.tau
        preds_stable = preds_stable - preds_stable.max(dim=1, keepdim=True)[0]  # 数值稳定
        thetas = torch.exp(preds_stable)

        n = thetas.size(1)  # 3
        listwise_loss = 0.0
        for k in range(n):
            numer = thetas[:, k]
            denom = thetas[:, k:].sum(dim=1)
            log_prob = torch.log(numer / (denom + 1e-8) + 1e-8)
            mu_k = 1.0 / (math.sqrt(n) * math.log(k + 2))
            #listwise_loss += (mu_k * log_prob).mean()
            listwise_loss += -(mu_k * log_prob).mean()

        # -----------------------
        # 3. L2 正则 (可选)
        # -----------------------
        if self.normalize_logits:
            l2_loss = torch.mean(pm1_pred**2 + pm2_pred**2 + pm3_pred**2)
            loss12 += self.weight_penalty * l2_loss
            loss23 += self.weight_penalty * l2_loss
            listwise_loss += self.weight_penalty * torch.mean(preds**2)

        # -----------------------
        # 4. 合并 loss
        # -----------------------
        loss = self.w12 * loss12 + self.w23 * loss23 + self.lambda_listwise * listwise_loss

        return {"loss": loss}
