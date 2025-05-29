import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import MLPPairRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import numpy as np
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem

# ---------------------------
# Config
# ---------------------------
test_set = "jacs_set"
#test_set = "merck"
data_paths = [f"/data/{test_set}_lmdb/"]

# List of checkpoints for ensembling
ckpt_paths = [
     "/data/checkpoints/pair_seed_1/checkpoints/epoch=049-step=800.ckpt",
    "/data/checkpoints/pair_seed_2/checkpoints/epoch=049-step=800.ckpt",
    "/data/checkpoints/pair_seed_3/checkpoints/epoch=048-step=784.ckpt",
    "/data/checkpoints/pair_seed_2025/checkpoints/epoch=048-step=784.ckpt",
    "/data/checkpoints/pair_seed_2026/checkpoints/epoch=048-step=784.ckpt"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.__version__ >= "1.12":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ---------------------------
# Load Data
# ---------------------------
all_data = []
for path in data_paths:
    test_data = LMDBDataset(path)
    with test_data.env.begin(db=test_data.db[test_data.DATA_DB], write=False) as txn:
        keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        for key in tqdm(keys):
            item = test_data[key]
            name = item["name"]
            if "flip" in name:
                continue
            prot_inputs, lig_inputs, prot_s, lig_s, p = item["emb"]
            target, ligand = name.split(",")[:2]

            all_data.append({
                "target": target,
                "ligand": ligand,
                "s_p": torch.tensor(prot_s, dtype=torch.float32),
                "s_m": torch.tensor(lig_s, dtype=torch.float32),
                "z_pm": torch.tensor(p, dtype=torch.float32)
            })

print(f"Loaded {len(all_data)} entries.")

# ---------------------------
# Dataset and Loader
# ---------------------------
class RankerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    return {
        "target": [item["target"] for item in batch],
        "ligand": [item["ligand"] for item in batch],
        "s_p": torch.stack([item["s_p"] for item in batch]),
        "s_m": torch.stack([item["s_m"] for item in batch]),
        "z_pm": torch.stack([item["z_pm"] for item in batch]),
    }

dataset = RankerDataset(all_data)
loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Load Ensemble Models
# ---------------------------
models = []
for path in ckpt_paths:
    print(f"Loading checkpoint from {path}")
    model = MLPPairRanker().to(device)
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)

# ---------------------------
# Inference (with ensemble)
# ---------------------------
targets, ligands, preds = [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc="Forwarding"):
        s_p = batch["s_p"].to(device)
        s_m = batch["s_m"].to(device)
        z_pm = batch["z_pm"].to(device)

        model_preds = []
        for model in models:
            output = model.single_forward(s_p=s_p, s_m=s_m, z_pm=z_pm)["pred"]
            model_preds.append(output.unsqueeze(0))  # shape: [1, B]

        # Average across models → shape [B]

        avg_output = torch.cat(model_preds, dim=0).mean(dim=0)
        #max_output = torch.cat(model_preds, dim=0).max(dim=0)[0]
        #avg_output = max_output

        targets.extend(batch["target"])
        ligands.extend(batch["ligand"])
        preds.extend(avg_output.cpu().numpy())

# ---------------------------
# Evaluation
# ---------------------------
df = pd.DataFrame({
    "targets": targets,
    "ligands": ligands,
    "score": preds
})

results = {}
for i in range(len(df)):
    target = df["targets"][i].lower()
    ligand = str(df["ligands"][i]).replace("&", "/")
    score = df["score"][i]
    if target not in results:
        results[target] = {}
    results[target][ligand] = score

target_acc_res = {}
pearson_corrs = []
spearman_corrs = []

for target in df["targets"].unique():
    target = target.lower()
    path = f"./benchmark/dataset/fep/public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/{test_set}/{target}_edges.csv"
    if not os.path.exists(path):
        continue

    df_target = pd.read_csv(path)
    labels, pair_preds = [], []
    for i in range(len(df_target)):
        ligand1 = str(df_target["Ligand 1"][i])
        ligand2 = str(df_target["Ligand 2"][i])
        ddg = df_target["ddG (kcal/mol)"][i]
        if np.isnan(ddg) or "flip" in ligand1 or "flip" in ligand2:
            continue
        score1 = results[target].get(ligand1)
        score2 = results[target].get(ligand2)
        if score1 is None or score2 is None:
            continue
        labels.append(ddg)
        pair_preds.append(score1 - score2)

    if not labels:
        continue

    acc = np.mean(np.sign(labels) == np.sign(pair_preds))
    auc = roc_auc_score([0 if l < 0 else 1 for l in labels], pair_preds)

    pearson_corr = pearsonr(labels, pair_preds)[0]
    spearman_corr = spearmanr(labels, pair_preds)[0]
    pearson_corrs.append(pearson_corr)
    spearman_corrs.append(spearman_corr)

    target_acc_res[target] = {"acc": acc, "auc": auc, "n": len(labels)}

mean_acc = np.mean([x["acc"] for x in target_acc_res.values()])
mean_auc = np.mean([x["auc"] for x in target_acc_res.values()])
print(f"Mean acc = {mean_acc:.4f}")
print(f"Mean auc = {mean_auc:.4f}")

# ---------------------------
# Per-Target Affinity Correlation
# ---------------------------
per_target_results = {}

for target in df["targets"].unique():
    target_lower = target.lower()
    sdf_path = f"./benchmark/dataset/fep/public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/{test_set}/{target_lower}_ligands.sdf"
    if not os.path.exists(sdf_path):
        print(f"SDF file not found for {target_lower}")
        continue

    suppl = Chem.SDMolSupplier(sdf_path)
    ligand_to_affinity = {}
    for mol in suppl:
        if mol is None:
            continue
        ligand_name = mol.GetProp("_Name")
        if mol.HasProp("r_exp_dg"):
            try:
                affinity = -float(mol.GetProp("r_exp_dg"))
                ligand_to_affinity[ligand_name] = affinity
            except ValueError:
                continue

    actual_affinities, predicted_scores = [], []
    for ligand, actual_affinity in ligand_to_affinity.items():
        pred = results.get(target_lower, {}).get(ligand)
        if pred is not None:
            actual_affinities.append(actual_affinity)
            predicted_scores.append(pred)

    if len(actual_affinities) >= 2:
        pearson_corr, _ = pearsonr(predicted_scores, actual_affinities)
        spearman_corr, _ = spearmanr(predicted_scores, actual_affinities)
        per_target_results[target_lower] = {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "n": len(actual_affinities)
        }
        print(f"{target_lower}: Pearson = {pearson_corr:.4f}, Spearman = {spearman_corr:.4f}, N = {len(actual_affinities)}")
    else:
        print(f"{target_lower}: Not enough data for correlation")

summary_df = pd.DataFrame.from_dict(per_target_results, orient='index').reset_index()
summary_df.columns = ["target", "pearson", "spearman", "n"]

mean_pearson = summary_df["pearson"].mean()
mean_spearman = summary_df["spearman"].mean()

print(f"\nAverage Pearson Correlation: {mean_pearson:.4f}")
print(f"Average Spearman Correlation: {mean_spearman:.4f}")