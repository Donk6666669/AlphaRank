import torch
import torch.nn.functional as F
from torch import nn


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

    def split_forward(
        self,
        pm1: dict[str, torch.Tensor] = None,
        pm2: dict[str, torch.Tensor] = None,
        **kwargs,
    ):
        pm1_pred = self.single_forward(**pm1)["pred"]
        pm2_pred = self.single_forward(**pm2)["pred"]
        return {
            "pm1_pred": pm1_pred,
            "pm2_pred": pm2_pred,
            "pred": F.sigmoid(pm2_pred - pm1_pred),
        }

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


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # Expand to 4×
            nn.GELU(),  # Activation
            nn.Linear(4 * d_model, d_model),  # Project back
            nn.Dropout(dropout),  # Dropout after final projection
        )

    def forward(self, x):
        # Pre-LayerNorm → MLP → Residual
        x = x + self.mlp(self.norm(x))
        return x


class CombinePairRanker(PairRanker):
    def __init__(
        self, s_input_dim=449, s_dim=384, z_dim=128, n_residue=1, dropout=0.1
    ):
        super().__init__(s_input_dim, s_dim, z_dim)
        self.encoder = nn.Sequential(
            *[ResidualMLPBlock(d_model=self.z_dim, dropout=dropout)]
            * n_residue,
            nn.Linear(self.z_dim, 1),
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
        x = self.encoder(z_pm)
        x = x * mask_pm[..., None]
        x = x.sum(dim=(1, 2)).squeeze()
        return {
            "pred": x,
        }


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
