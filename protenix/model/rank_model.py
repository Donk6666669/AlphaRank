import torch
import torch.nn.functional as F
from torch import nn
from lorentz import exp_map0, pairwise_dist, oxy_angle

class PairRanker(nn.Module):
    def __init__(self, s_input_dim=449, s_dim=384, z_dim=128):
        super().__init__()
        self.s_input_dim = s_input_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.encoder: nn.Module = None

    def single_forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m: torch.Tensor = None,
        z_pm: torch.Tensor = None,
        z_mp: torch.Tensor = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "single_forward should be implemented in the subclass"
        )

    # def split_forward(
    #     self,
    #     pm1: dict[str, torch.Tensor] = None,
    #     pm2: dict[str, torch.Tensor] = None,
    #     **kwargs,
    # ):
    #     pm1_pred = self.single_forward(**pm1)["pred"]
    #     pm2_pred = self.single_forward(**pm2)["pred"]
    #     return {
    #         "pm1_pred": pm1_pred,
    #         "pm2_pred": pm2_pred,
    #         "pred": F.sigmoid(pm2_pred - pm1_pred),
    #     }
    def split_forward(
        self,
        pm1: dict[str, torch.Tensor] = None,
        pm2: dict[str, torch.Tensor] = None,
        pm3: dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        pm1_pred = self.single_forward(**pm1)["pred"]
        pm2_pred = self.single_forward(**pm2)["pred"]
        pm3_pred = self.single_forward(**pm3)["pred"] if pm3 is not None else None
        pred = F.sigmoid(pm2_pred - pm1_pred)
        ret = {
            "pm1_pred": pm1_pred,
            "pm2_pred": pm2_pred,
            "pred": pred,
        }
        if pm3_pred is not None:
            ret["pm3_pred"] = pm3_pred
        return ret

    def forward(self, **kwargs):
        if "pm1" in kwargs and "pm2" in kwargs:
            return self.split_forward(**kwargs)
        else:
            return self.single_forward(**kwargs)


class LinearPairRanker(PairRanker):
    def __init__(
        self,
        s_input_dim=449,
        s_dim=384,
        z_dim=128,
        strategy="cat_sz",
    ):
        super().__init__(s_input_dim, s_dim, z_dim)

        self.strategy = strategy

        # single feature
        if self.strategy == "s_input":
            self.input_dim = self.s_input_dim * 2
        elif self.strategy == "s_input_p_only":
            self.input_dim = self.s_input_dim
        elif self.strategy == "s_input_m_only":
            self.input_dim = self.s_input_dim
        elif self.strategy == "s":
            self.input_dim = self.s_dim * 2
        elif self.strategy == "s_p_only":
            self.input_dim = self.s_dim
        elif self.strategy == "s_m_only":
            self.input_dim = self.s_dim
        elif self.strategy == "z":
            self.input_dim = self.z_dim
        elif self.strategy == "zdouble":
            self.input_dim = self.z_dim * 2

        # cat two features
        elif self.strategy == "cat_sz":
            self.input_dim = self.s_dim * 2 + self.z_dim
        elif self.strategy == "cat_s_zdouble":
            self.input_dim = self.s_dim * 2 + self.z_dim * 2
        elif self.strategy == "cat_s_m_only_z":
            self.input_dim = self.s_dim + self.z_dim
        elif self.strategy == "cat_s_input_z":
            self.input_dim = self.s_input_dim * 2 + self.z_dim
        elif self.strategy == "cat_s_input_zdouble":
            self.input_dim = self.s_input_dim * 2 + self.z_dim * 2
        elif self.strategy == "cat_s_input_s":
            self.input_dim = self.s_input_dim * 2 + self.s_dim * 2

        # cat three features
        elif self.strategy == "cat_s_input_s_z":
            self.input_dim = self.s_input_dim * 2 + self.s_dim * 2 + self.z_dim
        elif self.strategy == "cat_s_input_s_zdouble":
            self.input_dim = (
                self.s_input_dim * 2 + self.s_dim * 2 + self.z_dim * 2
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1),
        )

    def single_forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m: torch.Tensor = None,
        z_pm: torch.Tensor = None,
        z_mp: torch.Tensor = None,
    ):
        # single feature
        if self.strategy == "s_input":
            x = torch.cat([s_inputs_p, s_inputs_m], dim=1)
        elif self.strategy == "s_input_p_only":
            x = s_inputs_p
        elif self.strategy == "s_input_m_only":
            x = s_inputs_m
        elif self.strategy == "s":
            x = torch.cat([s_p, s_m], dim=1)
        elif self.strategy == "s_p_only":
            x = s_p
        elif self.strategy == "s_m_only":
            x = s_m
        elif self.strategy == "z":
            x = torch.cat([z_pm], dim=1)
        elif self.strategy == "zdouble":
            x = torch.cat([z_pm, z_mp], dim=1)

        # cat two features
        elif self.strategy == "cat_sz":
            x = torch.cat([s_p, s_m, z_pm], dim=1)
        elif self.strategy == "cat_s_zdouble":
            x = torch.cat([s_p, s_m, z_pm, z_mp], dim=1)
        elif self.strategy == "cat_s_m_only_z":
            x = torch.cat([s_m, z_pm], dim=1)
        elif self.strategy == "cat_s_input_z":
            x = torch.cat([s_inputs_p, s_inputs_m, z_pm], dim=1)
        elif self.strategy == "cat_s_input_zdouble":
            x = torch.cat([s_inputs_p, s_inputs_m, z_pm, z_mp], dim=1)
        elif self.strategy == "cat_s_input_s":
            x = torch.cat([s_inputs_p, s_inputs_m, s_p, s_m], dim=1)

        # cat three features
        elif self.strategy == "cat_s_input_s_z":
            x = torch.cat([s_inputs_p, s_inputs_m, s_p, s_m, z_pm], dim=1)
        elif self.strategy == "cat_s_input_s_zdouble":
            x = torch.cat([s_inputs_p, s_inputs_m, s_p, s_m, z_pm, z_mp], dim=1)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        pred = self.encoder(x).squeeze(1)
        return {"pred": pred}


class MLPPairRanker(PairRanker):
    def __init__(
        self,
        s_input_dim=449,
        s_dim=384,
        z_dim=128,
        mid_dim=128,
        dropout=0.5,
        strategy="cat_sz",
    ):
        super().__init__(s_input_dim, s_dim, z_dim)

        self.mid_dim = mid_dim
        self.dropout = dropout
        self.strategy = strategy

        # single feature
        if self.strategy == "s_input":
            self.input_dim = self.s_input_dim * 2
        elif self.strategy == "s_input_p_only":
            self.input_dim = self.s_input_dim
        elif self.strategy == "s_input_m_only":
            self.input_dim = self.s_input_dim
        elif self.strategy == "s":
            self.input_dim = self.s_dim * 2
        elif self.strategy == "s_p_only":
            self.input_dim = self.s_dim
        elif self.strategy == "s_m_only":
            self.input_dim = self.s_dim
        elif self.strategy == "z":
            self.input_dim = self.z_dim
        elif self.strategy == "zdouble":
            self.input_dim = self.z_dim * 2

        # cat two features
        elif self.strategy == "cat_sz":
            self.input_dim = self.s_dim * 2 + self.z_dim
        elif self.strategy == "cat_s_zdouble":
            self.input_dim = self.s_dim * 2 + self.z_dim * 2
        elif self.strategy == "cat_s_m_only_z":
            self.input_dim = self.s_dim + self.z_dim
        elif self.strategy == "cat_s_input_z":
            self.input_dim = self.s_input_dim * 2 + self.z_dim
        elif self.strategy == "cat_s_input_zdouble":
            self.input_dim = self.s_input_dim * 2 + self.z_dim * 2
        elif self.strategy == "cat_s_input_s":
            self.input_dim = self.s_input_dim * 2 + self.s_dim * 2

        # cat three features
        elif self.strategy == "cat_s_input_s_z":
            self.input_dim = self.s_input_dim * 2 + self.s_dim * 2 + self.z_dim
        elif self.strategy == "cat_s_input_s_zdouble":
            self.input_dim = (
                self.s_input_dim * 2 + self.s_dim * 2 + self.z_dim * 2
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.mid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mid_dim, int(self.mid_dim / 2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.mid_dim / 2), 1),
        )

    def single_forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m: torch.Tensor = None,
        z_pm: torch.Tensor = None,
        z_mp: torch.Tensor = None,
    ):
        # single feature
        if self.strategy == "s_input":
            x = torch.cat([s_inputs_p, s_inputs_m], dim=1)
        elif self.strategy == "s_input_p_only":
            x = s_inputs_p
        elif self.strategy == "s_input_m_only":
            x = s_inputs_m
        elif self.strategy == "s":
            x = torch.cat([s_p, s_m], dim=1)
        elif self.strategy == "s_p_only":
            x = s_p
        elif self.strategy == "s_m_only":
            x = s_m
        elif self.strategy == "z":
            x = torch.cat([z_pm], dim=1)
        elif self.strategy == "zdouble":
            x = torch.cat([z_pm, z_mp], dim=1)

        # cat two features
        elif self.strategy == "cat_sz":
            x = torch.cat([s_p, s_m, z_pm], dim=1)
        elif self.strategy == "cat_s_zdouble":
            x = torch.cat([s_p, s_m, z_pm, z_mp], dim=1)
        elif self.strategy == "cat_s_m_only_z":
            x = torch.cat([s_m, z_pm], dim=1)
        elif self.strategy == "cat_s_input_z":
            x = torch.cat([s_inputs_p, s_inputs_m, z_pm], dim=1)
        elif self.strategy == "cat_s_input_zdouble":
            x = torch.cat([s_inputs_p, s_inputs_m, z_pm, z_mp], dim=1)
        elif self.strategy == "cat_s_input_s":
            x = torch.cat([s_inputs_p, s_inputs_m, s_p, s_m], dim=1)

        # cat three features
        elif self.strategy == "cat_s_input_s_z":
            x = torch.cat([s_inputs_p, s_inputs_m, s_p, s_m, z_pm], dim=1)
        elif self.strategy == "cat_s_input_s_zdouble":
            x = torch.cat([s_inputs_p, s_inputs_m, s_p, s_m, z_pm, z_mp], dim=1)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        pred = self.encoder(x).squeeze(1)
        return {"pred": pred}

import torch
import torch.nn as nn

class HYPMLPPairRanker(PairRanker):
    def __init__(
        self,
        s_input_dim=449,
        s_dim=384,
        z_dim=128,
        mid_dim=128,
        dropout=0.5,
        strategy="cat_sz",
        alpha=0.5,
        
    ):
        super().__init__(s_input_dim, s_dim, z_dim)

        self.mid_dim = mid_dim
        self.dropout = dropout
        self.strategy = strategy
        self.alpha = alpha
        #self.beta = beta

        # 定义 input_dim（和你之前的一样）
        if self.strategy == "cat_sz":
            self.input_dim = self.s_dim * 2 + self.z_dim
        elif self.strategy == "cat_s_zdouble":
            self.input_dim = self.s_dim * 2 + self.z_dim * 2
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # pair_emb → 128
        self.mlp_pair = nn.Sequential(
            nn.Linear(self.input_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 128),
        )

        # prot_s → 128
        self.mlp_prot = nn.Sequential(
            nn.Linear(self.s_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 128),
        )

    def single_forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m: torch.Tensor = None,
        z_pm: torch.Tensor = None,
        z_mp: torch.Tensor = None,
        prot_s: torch.Tensor = None,   # 新增
    ):
        # pair embedding
        if self.strategy == "cat_sz":
            pair_emb = torch.cat([s_p, s_m, z_pm], dim=1)
        elif self.strategy == "cat_s_zdouble":
            pair_emb = torch.cat([s_p, s_m, z_pm, z_mp], dim=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # --- 1. MLP 投影 ---
        pair_u = self.mlp_pair(pair_emb)  # (B, 128)
        prot_u = self.mlp_prot(prot_s)    # (B, 128)

        # --- 2. exp map ---
        pair_h = exp_map0(pair_u)  # (B, 128)
        prot_h = exp_map0(prot_u)  # (B, 128)

        # --- 3. 距离 + 角度 ---
        dist = torch.diag(pairwise_dist(pair_h, prot_h))  # (B,)
        angle = oxy_angle(pair_h, prot_h)                 # (B,)

        # --- 4. 线性组合 ---
        pred = self.alpha * dist + (1-self.alpha) * angle      # (B,)

        return {"pred": pred}


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1, multiply=4):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, multiply * d_model),  # Expand to 4×
            nn.GELU(),  # Activation
            nn.Linear(multiply * d_model, d_model),  # Project back
            nn.Dropout(dropout),  # Dropout after final projection
        )

    def forward(self, x):
        # Pre-LayerNorm → MLP → Residual
        x = x + self.mlp(self.norm(x))
        return x


class CombinePairRanker(PairRanker):
    def __init__(
        self,
        s_input_dim=449,
        s_dim=384,
        z_dim=128,
        n_residue=1,
        dropout=0.1,
        strategy="from_z",
        agg_strategy="sum",
        agg_topk=64,
    ):
        super().__init__(s_input_dim, s_dim, z_dim)
        self.strategy = strategy
        self.agg_strategy = agg_strategy
        self.agg_topk = agg_topk
        if self.strategy == "from_z":
            self.encoder = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.z_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.z_dim, 1),
            )
        elif self.strategy == "from_s":
            self.encoder_s_p = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_s_m = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_s_pm = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_agg = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.s_dim, 1),
            )
        elif self.strategy == "from_sz":
            self.encoder_s_p = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_s_m = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_z = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.z_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.z_dim, self.s_dim),
            )
            self.encoder_agg = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.s_dim, 1),
            )

    def single_forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m: torch.Tensor = None,
        z_pm: torch.Tensor = None,
        z_mp: torch.Tensor = None,
        mask_p: torch.Tensor = None,
        mask_m: torch.Tensor = None,
        mask_pm: torch.Tensor = None,
        len_p: torch.Tensor = None,
        len_m: torch.Tensor = None,
        **kwargs,
    ):
        if self.strategy == "from_z":
            x = self.encoder(z_pm)
        elif self.strategy == "from_s":
            x = self.encoder_agg(
                torch.einsum(
                    "bpik,bimk->bpmk",
                    self.encoder_s_p(s_p).unsqueeze(2),  # bz, len_p, 1, s_dim
                    self.encoder_s_m(s_m).unsqueeze(1),  # bz, 1, len_m, s_dim
                )
                * self.encoder_s_pm(
                    torch.einsum(
                        "bpik,bimk->bpmk",
                        s_p.unsqueeze(2),  # bz, len_p, 1, s_dim
                        s_m.unsqueeze(1),  # bz, 1, len_m, s_dim
                    )
                )  # bz, len_p, len_m, s_dim
            )  # bz, len_p, len_m, 1
        elif self.strategy == "from_sz":
            x = self.encoder_agg(
                torch.einsum(
                    "bpik,bimk->bpmk",
                    self.encoder_s_p(s_p).unsqueeze(2),  # bz, len_p, 1, s_dim
                    self.encoder_s_m(s_m).unsqueeze(1),  # bz, 1, len_m, s_dim
                )
                * self.encoder_z(z_pm)  # bz, len_p, len_m, s_dim
            )  # bz, len_p, len_m, 1

        x = x.squeeze()  # bz, len_p, len_m

        mask_pm = mask_pm.float()
        total_pair = mask_pm.sum(dim=(1, 2))
        if self.agg_strategy == "sum":
            x = x * mask_pm
            x = x.sum(dim=(1, 2))
        elif self.agg_strategy == "mean":
            x = x * mask_pm
            x = x.sum(dim=(1, 2)) / total_pair
        elif self.agg_strategy == "sum_sigmoid":
            x = (torch.sigmoid(x) * mask_pm).sum(dim=(1, 2))
        elif self.agg_strategy == "mean_sigmoid":
            x = (torch.sigmoid(x) * mask_pm).sum(dim=(1, 2)) / total_pair
        elif self.agg_strategy == "topk_sum":
            x = x * mask_pm + -torch.inf * (1 - mask_pm)
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            x[x == -torch.inf] = 0
            x = x.sum(dim=-1)
        elif self.agg_strategy == "topk_mean":
            x = x * mask_pm + -torch.inf * (1 - mask_pm)
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            mask_top = x != -torch.inf
            x = x * mask_top
            x = x.sum(dim=-1) / torch.min(
                total_pair,
                torch.tensor([self.agg_topk] * x.shape[0]).to(x),
            )
        elif self.agg_strategy == "topk_sum_sigmoid":
            x = torch.sigmoid(x * mask_pm + -torch.inf * (1 - mask_pm))
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            x = x.sum(dim=-1)
        elif self.agg_strategy == "topk_mean_sigmoid":
            x = torch.sigmoid(x * mask_pm + -torch.inf * (1 - mask_pm))
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            x = x.sum(dim=-1) / torch.min(
                total_pair,
                torch.tensor([self.agg_topk] * x.shape[0]).to(x),
            )
        else:
            raise ValueError(f"Unknown agg_strategy: {self.agg_strategy}")
        x = x.squeeze()
        return {"pred": x}


class TripletRanker(nn.Module):
    def __init__(self, s_input_dim=449, s_dim=384, z_dim=128):
        super().__init__()
        self.s_input_dim = s_input_dim
        self.s_dim = s_dim
        self.z_dim = z_dim
        self.encoder: nn.Module = None

    def forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m1: torch.Tensor = None,
        s_inputs_m2: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m1: torch.Tensor = None,
        s_m2: torch.Tensor = None,
        z_pm1: torch.Tensor = None,
        z_pm2: torch.Tensor = None,
        z_m1m2: torch.Tensor = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "single_forward should be implemented in the subclass"
        )


class LinearTripletRanker(TripletRanker):
    def __init__(
        self,
        s_input_dim=449,
        s_dim=384,
        z_dim=128,
        strategy="cat_sz",
    ):
        super().__init__(s_input_dim, s_dim, z_dim)

        self.strategy = strategy

        # single feature
        if self.strategy == "s_input":
            self.input_dim = self.s_input_dim * 3
        elif self.strategy == "s_input_p_only":
            self.input_dim = self.s_input_dim
        elif self.strategy == "s_input_m1_only":
            self.input_dim = self.s_input_dim
        elif self.strategy == "s_input_m2_only":
            self.input_dim = self.s_input_dim
        elif self.strategy == "s_input_m1_m2":
            self.input_dim = self.s_input_dim * 2
        elif self.strategy == "s":
            self.input_dim = self.s_dim * 3
        elif self.strategy == "s_p_only":
            self.input_dim = self.s_dim
        elif self.strategy == "s_m1_only":
            self.input_dim = self.s_dim
        elif self.strategy == "s_m2_only":
            self.input_dim = self.s_dim
        elif self.strategy == "s_m1_m2":
            self.input_dim = self.s_dim * 2
        elif self.strategy == "z":
            self.input_dim = self.z_dim * 3
        elif self.strategy == "z_pm1_only":
            self.input_dim = self.z_dim
        elif self.strategy == "z_pm2_only":
            self.input_dim = self.z_dim
        elif self.strategy == "z_m1m2":
            self.input_dim = self.z_dim

        # cat two features
        elif self.strategy == "cat_sz":
            self.input_dim = self.s_dim * 3 + self.z_dim * 3
        elif self.strategy == "cat_s_z_pm1_pm2":
            self.input_dim = self.s_dim * 3 + self.z_dim * 2
        elif self.strategy == "cat_s_z_m1m2":
            self.input_dim = self.s_dim * 3 + self.z_dim
        elif self.strategy == "cat_s_m1_m2_z_pm1_pm2":
            self.input_dim = self.s_dim * 2 + self.z_dim * 2
        elif self.strategy == "cat_s_m1_m2_z_m1m2":
            self.input_dim = self.s_dim * 2 + self.z_dim

        # cat three features
        elif self.strategy == "cat_s_input_s_z":
            self.input_dim = (
                self.s_input_dim * 3 + self.s_dim * 3 + self.z_dim * 3
            )

        # diff
        elif self.strategy == "diff_s_input":
            self.input_dim = self.s_input_dim
        elif self.strategy == "diff_s":
            self.input_dim = self.s_dim
        elif self.strategy == "diff_z":
            self.input_dim = self.z_dim
        elif self.strategy == "diff_s_z":
            self.input_dim = self.s_dim + self.z_dim
        elif self.strategy == "diff_cat_s_input":
            self.input_dim = self.s_input_dim * 3
        elif self.strategy == "diff_cat_s":
            self.input_dim = self.s_dim * 3
        elif self.strategy == "diff_cat_z":
            self.input_dim = self.z_dim * 3
        elif self.strategy == "diff_cat_s_z":
            self.input_dim = self.s_dim * 3 + self.z_dim * 3

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # init encoder
        self.init_encoder()

    def init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 1),
        )

    def forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m1: torch.Tensor = None,
        s_inputs_m2: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m1: torch.Tensor = None,
        s_m2: torch.Tensor = None,
        z_pm1: torch.Tensor = None,
        z_pm2: torch.Tensor = None,
        z_m1m2: torch.Tensor = None,
        **kwargs,
    ):
        # single feature
        if self.strategy == "s_input":
            x = torch.cat([s_inputs_p, s_inputs_m1, s_inputs_m2], dim=1)
        elif self.strategy == "s_input_p_only":
            x = s_inputs_p
        elif self.strategy == "s_input_m1_only":
            x = s_inputs_m1
        elif self.strategy == "s_input_m2_only":
            x = s_inputs_m2
        elif self.strategy == "s_input_m1_m2":
            x = torch.cat([s_inputs_m1, s_inputs_m2], dim=1)
        elif self.strategy == "s":
            x = torch.cat([s_p, s_m1, s_m2], dim=1)
        elif self.strategy == "s_p_only":
            x = s_p
        elif self.strategy == "s_m1_only":
            x = s_m1
        elif self.strategy == "s_m2_only":
            x = s_m2
        elif self.strategy == "s_m1_m2":
            x = torch.cat([s_m1, s_m2], dim=1)
        # elif self.strategy == "z":
        #     x = torch.cat([z_pm1, z_pm2, z_m1m2], dim=1)
        elif self.strategy == "z_pm1_pm2":
            x = torch.cat([z_pm1, z_pm2], dim=1)
        elif self.strategy == "z_pm1_only":
            x = z_pm1
        elif self.strategy == "z_pm2_only":
            x = z_pm2
        # elif self.strategy == "z_m1m2":
        #     x = z_m1m2

        # cat two features
        # elif self.strategy == "cat_sz":
        #     x = torch.cat([s_p, s_m1, s_m2, z_pm1, z_pm2, z_m1m2], dim=1)
        elif self.strategy == "cat_s_z_pm1_pm2":
            x = torch.cat([s_p, s_m1, s_m2, z_pm1, z_pm2], dim=1)
        # elif self.strategy == "cat_s_z_m1m2":
        #     x = torch.cat([s_p, s_m1, s_m2, z_m1m2], dim=1)
        elif self.strategy == "cat_s_m1_m2_z_pm1_pm2":
            x = torch.cat([s_m1, s_m2, z_pm1, z_pm2], dim=1)
        # elif self.strategy == "cat_s_m1_m2_z_m1m2":
        #     x = torch.cat([s_m1, s_m2, z_m1m2], dim=1)

        # cat three features
        elif self.strategy == "cat_s_input_m1_m2_s_m1_m2_z_pm1_pm2":
            x = torch.cat(
                [
                    s_inputs_m1,
                    s_inputs_m2,
                    s_p,
                    s_m1,
                    s_m2,
                    z_pm1,
                    z_pm2,
                ],
                dim=1,
            )

        # diff
        elif self.strategy == "diff_s_input":
            x = s_inputs_m1 - s_inputs_m2
        elif self.strategy == "diff_s":
            x = s_m1 - s_m2
        elif self.strategy == "diff_z":
            x = z_pm1 - z_pm2
        elif self.strategy == "diff_s_z":
            x = torch.cat([s_m1 - s_m2, z_pm1 - z_pm2], dim=1)
        elif self.strategy == "diff_cat_s_input":
            x = torch.cat(
                [s_inputs_m1 - s_inputs_m2, s_inputs_m1, s_inputs_m2], dim=1
            )
        elif self.strategy == "diff_cat_s":
            x = torch.cat([s_m1 - s_m2, s_m1, s_m2], dim=1)
        elif self.strategy == "diff_cat_z":
            x = torch.cat([z_pm1 - z_pm2, z_pm1, z_pm2], dim=1)
        elif self.strategy == "diff_cat_s_z":
            x = torch.cat(
                [s_m1 - s_m2, z_pm1 - z_pm2, s_m1, s_m2, z_pm1, z_pm2], dim=1
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        pred = F.sigmoid(self.encoder(x).squeeze(1))
        return {"pred": pred}


class MLPTripletRanker(LinearTripletRanker):
    def __init__(
        self,
        s_input_dim=449,
        s_dim=384,
        z_dim=128,
        mid_dim=128,
        dropout=0.5,
        strategy="cat_sz",
    ):
        self.mid_dim = mid_dim
        self.dropout = dropout
        super().__init__(s_input_dim, s_dim, z_dim, strategy)

    def init_encoder(self):
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.mid_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mid_dim, int(self.mid_dim / 2)),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(int(self.mid_dim / 2), 1),
        )


class CombineTripletRanker(TripletRanker):
    def __init__(
        self,
        s_input_dim=449,
        s_dim=384,
        z_dim=128,
        n_residue=1,
        dropout=0.1,
        strategy="from_z",
        agg_strategy="sum",
        agg_topk=64,
    ):
        super().__init__(s_input_dim, s_dim, z_dim)
        self.strategy = strategy
        self.agg_strategy = agg_strategy
        self.agg_topk = agg_topk
        if self.strategy == "from_z":
            self.encoder = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.z_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.z_dim, 1),
            )
        elif self.strategy == "from_s":
            self.encoder_s_p = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_s_m = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_s_pm = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_agg = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.s_dim, 1),
            )
        elif self.strategy == "from_sz":
            self.encoder_s_p = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_s_m = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
            )
            self.encoder_z = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.z_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.z_dim, self.s_dim),
            )
            self.encoder_agg = nn.Sequential(
                *[ResidualMLPBlock(d_model=self.s_dim, dropout=dropout)]
                * n_residue,
                nn.Linear(self.s_dim, 1),
            )

    def single_forward(
        self,
        s_inputs_p: torch.Tensor = None,
        s_inputs_m: torch.Tensor = None,
        s_p: torch.Tensor = None,
        s_m: torch.Tensor = None,
        z_pm: torch.Tensor = None,
        z_mp: torch.Tensor = None,
        mask_p: torch.Tensor = None,
        mask_m: torch.Tensor = None,
        mask_pm: torch.Tensor = None,
        len_p: torch.Tensor = None,
        len_m: torch.Tensor = None,
        **kwargs,
    ):
        if self.strategy == "from_z":
            x = self.encoder(z_pm)
        elif self.strategy == "from_s":
            x = self.encoder_agg(
                torch.einsum(
                    "bpik,bimk->bpmk",
                    self.encoder_s_p(s_p).unsqueeze(2),  # bz, len_p, 1, s_dim
                    self.encoder_s_m(s_m).unsqueeze(1),  # bz, 1, len_m, s_dim
                )
                * self.encoder_s_pm(
                    torch.einsum(
                        "bpik,bimk->bpmk",
                        s_p.unsqueeze(2),  # bz, len_p, 1, s_dim
                        s_m.unsqueeze(1),  # bz, 1, len_m, s_dim
                    )
                )  # bz, len_p, len_m, s_dim
            )  # bz, len_p, len_m, 1
        elif self.strategy == "from_sz":
            x = self.encoder_agg(
                torch.einsum(
                    "bpik,bimk->bpmk",
                    self.encoder_s_p(s_p).unsqueeze(2),  # bz, len_p, 1, s_dim
                    self.encoder_s_m(s_m).unsqueeze(1),  # bz, 1, len_m, s_dim
                )
                * self.encoder_z(z_pm)  # bz, len_p, len_m, s_dim
            )  # bz, len_p, len_m, 1

        x = x.squeeze()  # bz, len_p, len_m

        mask_pm = mask_pm.float()
        total_pair = mask_pm.sum(dim=(1, 2))
        if self.agg_strategy == "sum":
            x = x * mask_pm
            x = x.sum(dim=(1, 2))
        elif self.agg_strategy == "mean":
            x = x * mask_pm
            x = x.sum(dim=(1, 2)) / total_pair
        elif self.agg_strategy == "sum_sigmoid":
            x = (torch.sigmoid(x) * mask_pm).sum(dim=(1, 2))
        elif self.agg_strategy == "mean_sigmoid":
            x = (torch.sigmoid(x) * mask_pm).sum(dim=(1, 2)) / total_pair
        elif self.agg_strategy == "topk_sum":
            x = x * mask_pm + -torch.inf * (1 - mask_pm)
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            x[x == -torch.inf] = 0
            x = x.sum(dim=-1)
        elif self.agg_strategy == "topk_mean":
            x = x * mask_pm + -torch.inf * (1 - mask_pm)
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            mask_top = x != -torch.inf
            x = x * mask_top
            x = x.sum(dim=-1) / torch.min(
                total_pair,
                torch.tensor([self.agg_topk] * x.shape[0]).to(x),
            )
        elif self.agg_strategy == "topk_sum_sigmoid":
            x = torch.sigmoid(x * mask_pm + -torch.inf * (1 - mask_pm))
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            x = x.sum(dim=-1)
        elif self.agg_strategy == "topk_mean_sigmoid":
            x = torch.sigmoid(x * mask_pm + -torch.inf * (1 - mask_pm))
            x = x.view(x.shape[0], -1).topk(self.agg_topk, dim=-1).values
            x = x.sum(dim=-1) / torch.min(
                total_pair,
                torch.tensor([self.agg_topk] * x.shape[0]).to(x),
            )
        else:
            raise ValueError(f"Unknown agg_strategy: {self.agg_strategy}")
        x = x.squeeze()
        return x

    def forward(
        self,
        s_inputs_p=None,
        s_inputs_m1=None,
        s_inputs_m2=None,
        s_p=None,
        s_m1=None,
        s_m2=None,
        z_pm1=None,
        z_pm2=None,
        z_m1m2=None,
        mask_pm1=None,
        mask_pm2=None,
        **kwargs,
    ):
        pred_pm1 = self.single_forward(
            s_inputs_p=s_inputs_p,
            s_inputs_m=s_inputs_m1,
            s_p=s_p,
            s_m=s_m1,
            z_pm=z_pm1,
            mask_pm=mask_pm1,
            **kwargs,
        )

        pred_pm2 = self.single_forward(
            s_inputs_p=s_inputs_p,
            s_inputs_m=s_inputs_m2,
            s_p=s_p,
            s_m=s_m2,
            z_pm=z_pm2,
            mask_pm=mask_pm2,
            **kwargs,
        )

        pred = F.sigmoid(pred_pm2 - pred_pm1)
        return {
            "pred": pred,
            "pm1_pred": pred_pm1,
            "pm2_pred": pred_pm2,
        }
