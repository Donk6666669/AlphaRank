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

        if strategy == "cat_sz":
            self.input_dim = self.s_dim * 2 + z_dim
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

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
        if self.strategy == "cat_sz":
            x = torch.cat([s_p, s_m, z_pm], dim=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        pred = self.encoder(x).squeeze(1)
        return {
            "pred": pred
        }


class TripletRanker(nn.Module):
    pass

