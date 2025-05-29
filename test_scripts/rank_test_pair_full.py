import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import CombinePairRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import os
import numpy as np
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr

# ---------------------------
# Step 1: Load and preprocess LMDB data
# ---------------------------
test_set = "jacs_set"
test_set = "merck"
data_paths = [f"/data/{test_set}_no_reduce/"]
data_paths = [f"/data/{test_set}_pair_constraint_new/"]
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_09-44-57/checkpoints/last.ckpt"
ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-14_07-22-47/checkpoints/epoch=020-step=19698.ckpt"
ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-14_17-04-08/checkpoints/epoch=020-step=10080.ckpt"
ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-14_17-04-08/checkpoints/last.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-16_18-33-35/checkpoints/epoch=017-step=8640.ckpt"
# ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-19_22-54-28/checkpoints/epoch=005-step=5754.ckpt"
# ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-19_22-53-58/checkpoints/epoch=023-step=23016.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-19_22-53-58/checkpoints/last.ckpt"

#ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-21_12-12-18/checkpoints/epoch=005-step=2898.ckpt"

ckpt_path = "/data/checkpoints/combine_pair_2025-05-21_18-03-07/checkpoints/epoch=015-step=7712.ckpt"

ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-22_10-24-31/checkpoints/epoch=021-step=10604.ckpt"
ckpt_path = "/data/checkpoints/DataModule.CombinePairRanker.RankCriterion.2025-05-22_10-24-51/checkpoints/epoch=024-step=12050.ckpt"

all_data = []

for path in data_paths:
    print(f"Reading from {path}")
    test_data = LMDBDataset(path)
    with test_data.env.begin(db=test_data.db[test_data.DATA_DB], write=False) as txn:
        keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        for key in tqdm(keys):
            item = test_data[key]
            name = item["name"]
            if "flip" in name:
                continue
            prot_inputs, lig_inputs, prot_s, lig_s, p = item["emb"]

            target, ligand = name.split(",")

            all_data.append({
                "target": target,
                "ligand": ligand.replace("&", "/"),
                "s_p": torch.tensor(prot_s, dtype=torch.float32),
                "s_m": torch.tensor(lig_s, dtype=torch.float32),
                "z_pm": torch.tensor(p, dtype=torch.float32)
            })

print(f"Loaded {len(all_data)} entries.")

# ---------------------------
# Step 2: Dataset and Dataloader
# ---------------------------
class RankerDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    targets = [item["target"] for item in batch]
    ligands = [item["ligand"] for item in batch]
    z_pm_list = [item["z_pm"] for item in batch]
    s_p_list = [item["s_p"] for item in batch]
    s_m_list = [item["s_m"] for item in batch]

    len_p_list = [z.shape[0] for z in z_pm_list]
    len_m_list = [z.shape[1] for z in z_pm_list]

    max_len_p = max(len_p_list)
    max_len_m = max(len_m_list)
    dim_z = z_pm_list[0].shape[2]
    dim_s = s_p_list[0].shape[1]

    # Pad z_pm
    z_pm_padded = []
    mask_pm = []
    for z_pm in z_pm_list:
        len_p, len_m, _ = z_pm.shape
        padded = torch.zeros((max_len_p, max_len_m, dim_z), dtype=z_pm.dtype)
        padded[:len_p, :len_m, :] = z_pm
        z_pm_padded.append(padded)

        mask = torch.zeros((max_len_p, max_len_m), dtype=torch.bool)
        mask[:len_p, :len_m] = 1
        mask_pm.append(mask)

    # Pad s_p
    s_p_padded = []
    for s_p in s_p_list:
        len_p = s_p.shape[0]
        padded = torch.zeros((max_len_p, dim_s), dtype=s_p.dtype)
        padded[:len_p, :] = s_p
        s_p_padded.append(padded)

    # Pad s_m
    s_m_padded = []
    for s_m in s_m_list:
        len_m = s_m.shape[0]
        padded = torch.zeros((max_len_m, dim_s), dtype=s_m.dtype)
        padded[:len_m, :] = s_m
        s_m_padded.append(padded)

    return {
        "target": targets,
        "ligand": ligands,
        "z_pm": torch.stack(z_pm_padded),
        "mask_pm": torch.stack(mask_pm),
        "s_p": torch.stack(s_p_padded),
        "s_m": torch.stack(s_m_padded),
        "len_p": torch.tensor(len_p_list),
        "len_m": torch.tensor(len_m_list),
    }

dataset = RankerDataset(all_data)
loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Step 3: Load model and checkpoint
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = CombinePairRanker().to(device)
model = CombinePairRanker(n_residue=1, strategy="from_z", agg_strategy="mean_sigmoid").to(device)

print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

if "state_dict" in checkpoint:
    state_dict = {
        k.replace("model.", ""): v
        for k, v in checkpoint["state_dict"].items()
        if k.startswith("model.")
    }
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.eval()

# ---------------------------
# Step 4: Forward pass
# ---------------------------
targets, ligands, preds = [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc="Forwarding"):
        z_pm = batch["z_pm"].to(device)
        mask_pm = batch["mask_pm"].to(device)
        len_p = batch["len_p"].to(device)
        len_m = batch["len_m"].to(device)
        s_p = batch["s_p"].to(device)
        s_m = batch["s_m"].to(device)

        #output = model.single_forward(z_pm=z_pm, mask_pm=mask_pm, len_p=len_p, len_m=len_m)["pred"]
        output = model.single_forward(
            s_p=s_p,
            s_m=s_m,
            z_pm=z_pm,
            mask_pm=mask_pm,
            len_p=len_p,
            len_m=len_m
        )["pred"]

        targets.extend(batch["target"])
        ligands.extend(batch["ligand"])
        preds.extend(output.cpu().numpy())

df = pd.DataFrame({
    "targets": targets,
    "ligands": ligands,
    "score": preds
})

# ---------------------------
# Step 5: Build results dict
# ---------------------------
results = {}
for i in range(len(df)):
    target = df["targets"][i].lower()
    ligand = df["ligands"][i]
    score = df["score"][i]
    if target not in results:
        results[target] = {}
    results[target][ligand] = score

# ---------------------------
# Step 6: Direction Accuracy + AUC
# ---------------------------
target_acc_res = {}

pair_pearsons = []
pair_spearmans = []

for target in df["targets"].unique():
    target = target.lower()
    path = f"./benchmark/dataset/fep/public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/{test_set}/{target}_edges.csv"
    if not os.path.exists(path):
        continue

    df_target = pd.read_csv(path)
    labels, preds_deltas = [], []

    for i in range(len(df_target)):
        ligand1 = str(df_target["Ligand 1"][i])
        ligand2 = str(df_target["Ligand 2"][i])
        ddg = df_target["ddG (kcal/mol)"][i]
        if np.isnan(ddg) or ligand1 not in results[target] or ligand2 not in results[target]:
            continue

        score1 = results[target][ligand1]
        score2 = results[target][ligand2]
        pred_delta = score1 - score2

        labels.append(ddg)
        preds_deltas.append(pred_delta)
        

    if len(labels) == 0:
        continue
    pearson_corr, _ = pearsonr(preds_deltas, labels)
    spearman_corr, _ = spearmanr(preds_deltas, labels)
    pair_pearsons.append(pearson_corr)
    pair_spearmans.append(spearman_corr)
    acc = np.mean(np.sign(labels) == np.sign(preds_deltas))
    auc = roc_auc_score([0 if l < 0 else 1 for l in labels], preds_deltas)
    target_acc_res[target] = {"acc": acc, "auc": auc, "n": len(labels)}
    print(f"{target:20s}: acc = {acc:.4f}, auc = {auc:.4f}, n = {len(labels)}")

mean_acc = np.mean([v["acc"] for v in target_acc_res.values()])
mean_auc = np.mean([v["auc"] for v in target_acc_res.values()])
print(f"\nMean Accuracy: {mean_acc:.4f}")
print(f"Mean AUC: {mean_auc:.4f}")
print(f"Mean Pair Pearson Correlation: {np.mean(pair_pearsons):.4f}")
print(f"Mean Pair Spearman Correlation: {np.mean(pair_spearmans):.4f}")

# ---------------------------
# Step 7: Affinity Correlation
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

    actual_affinities = []
    predicted_scores = []

    for ligand, actual_affinity in ligand_to_affinity.items():
        if target_lower in results and ligand in results[target_lower]:
            predicted_score = results[target_lower][ligand]
            actual_affinities.append(actual_affinity)
            predicted_scores.append(predicted_score)

    if len(actual_affinities) >= 2:
        pearson_corr, _ = pearsonr(predicted_scores, actual_affinities)
        spearman_corr, _ = spearmanr(predicted_scores, actual_affinities)
        per_target_results[target_lower] = {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
            "n": len(actual_affinities)
        }
        print(f"{target_lower:20s}: Pearson = {pearson_corr:.4f}, Spearman = {spearman_corr:.4f}, N = {len(actual_affinities)}")
    else:
        print(f"{target_lower:20s}: Not enough data for correlation")

summary_df = pd.DataFrame.from_dict(per_target_results, orient='index').reset_index()
summary_df.columns = ["target", "pearson", "spearman", "n"]

mean_pearson = summary_df["pearson"].mean()
mean_spearman = summary_df["spearman"].mean()
print(f"\nAverage Pearson Correlation: {mean_pearson:.4f}")
print(f"Average Spearman Correlation: {mean_spearman:.4f}")