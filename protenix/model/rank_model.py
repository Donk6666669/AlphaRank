import torch
from torch import nn


class CrossIndependentRanker(nn.Module):
    def __init__(self, s_input_dim=449, s_dim=384, z_dim=128):
        super().__init__()
        self.s_input_dim = s_input_dim
        self.s_dim = s_dim
        self.z_dim = z_dim

    def forward(
        self,
        s_inputs: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor = None,
        pocket_mask: torch.Tensor = None,
        mol_mask: torch.Tensor = None,
    ):
        raise NotImplementedError(
            "IndependentRanker is an abstract class. Please use a subclass."
        )


class IRSimpleClassifier(CrossIndependentRanker):
    def __init__(
        self, s_input_dim=449, s_dim=384, z_dim=128, mid_dim=128, dropout=0.5
    ):
        super().__init__(s_input_dim, s_dim, z_dim)

        self.input_dim = s_dim + z_dim
        self.mid_dim = mid_dim
        self.classifier = nn.Sequential(
            nn.Linear(s_dim * 2 + z_dim, self.mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.mid_dim, 1),
        )

    def forward(
        self,
        s_inputs: torch.Tensor = None,
        s: torch.Tensor = None,
        z: torch.Tensor = None,
        pocket_mask: torch.Tensor = None,
        mol_mask: torch.Tensor = None,
    ):
        """
        s_inputs: (N_token, s_input_dim)
        s: (N_token, s_dim)
        z: (N_token, N_token, z_dim)
        pocket_mask: (N_token, 1)
        mol_mask: (N_token, 1)
        """
        # Concatenate s and z
        s_pocket = s[pocket_mask].mean(dim=0)
        s_mol = s[mol_mask].mean(dim=0)
        z_pocket_mol = z[pocket_mask][:, mol_mask].mean(
            dim=(0, 1)
        )

        x = torch.cat([s_pocket, s_mol, z_pocket_mol], dim=0)
        x = self.classifier(x)
        return x


class AffinityClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)  # output shape: (batch,)


class AffinityClassifier_mlp(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(1)
