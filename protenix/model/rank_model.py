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
            self.input_dim = self.s_input_dim * 2 + self.s_dim * 2 + self.z_dim * 2

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
        return {
            "pred": pred
        }


class TripletRanker(nn.Module):
    pass

