import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import MLPPairRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import os
import numpy as np
# ---------------------------
# Step 1: Load and preprocess LMDB data
# ---------------------------



if torch.__version__ >= "1.12":
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

test_set = "jacs_set"
#test_set = "merck"
data_paths = [
    f"/data/{test_set}_lmdb/"
]
#data_paths = [f"/data/{test_set}_pair_constraint_new/"]
#data_paths = [f"/data/{test_set}_no_reduce/"]

ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_10-38-27/checkpoints/epoch=024-step=750.ckpt"
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_09-44-57/checkpoints/epoch=020-step=6153.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_09-44-57/checkpoints/epoch=017-step=5274.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_09-44-57/checkpoints/last.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_10-38-27/checkpoints/last.ckpt"
ckpt_path = '/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-54-21/checkpoints/epoch=022-step=3450.ckpt'
#ckpt_path = '/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-54-21/checkpoints/last.ckpt'

ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-55-22/checkpoints/epoch=018-step=2717.ckpt"
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-55-34/checkpoints/epoch=018-step=3287.ckpt"
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-11_23-29-46/checkpoints/epoch=025-step=3900.ckpt"
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-11_23-29-46/checkpoints/last.ckpt"

ckpt_path = '/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-54-21/checkpoints/epoch=022-step=3450.ckpt'

#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_09-44-57/checkpoints/last.ckpt"

ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_10-38-27/checkpoints/epoch=024-step=750.ckpt"
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_10-38-27/checkpoints/last.ckpt"


ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-20_11-48-48/checkpoints/epoch=024-step=400.ckpt"

ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-20_11-48-17/checkpoints/epoch=022-step=690.ckpt"

ckpt_path = "/log/train/af3rank/DataModule.MLPPairRanker.RankCriterion.2025-05-20_18-22-56/checkpoints/epoch=049-step=800.ckpt"
#ckpt_path = "/log/train/af3rank/DataModule.MLPPairRanker.RankCriterion.2025-05-20_19-17-26/checkpoints/epoch=049-step=800.ckpt"
#ckpt_path = "/log/train/af3rank/DataModule.MLPPairRanker.RankCriterion.2025-05-20_20-42-26/checkpoints/epoch=098-step=1584.ckpt"



#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-21_11-27-05/checkpoints/epoch=023-step=720.ckpt"

#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-21_11-27-14/checkpoints/epoch=024-step=750.ckpt"

#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-54-21/checkpoints/epoch=022-step=3450.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-54-21/checkpoints/epoch=020-step=3150.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-11_23-29-46/checkpoints/last.ckpt"

#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-25_14-41-03/checkpoints/epoch=022-step=920.ckpt"

# ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-25_14-21-14/checkpoints/epoch=024-step=875.ckpt"
# ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-25_16-36-39/checkpoints/epoch=045-step=1610.ckpt"
# ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-25_16-35-42/checkpoints/epoch=048-step=1715.ckpt"
# ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-25_16-52-22/checkpoints/epoch=042-step=1505.ckpt"








all_data = []

for path in data_paths:
    #print(f"Reading from {path}")
    test_data = LMDBDataset(path)
    with test_data.env.begin(db=test_data.db[test_data.DATA_DB], write=False) as txn:
        keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        for key in tqdm(keys):
            item = test_data[key]
            name = item["name"]
            if "flip" in name:
                continue
            prot_inputs, lig_inputs, prot_s, lig_s, p = item["emb"]

            target = name.split(",")[0]
            ligand = name.split(",")[1]

            #print(target,ligand)

            if len(prot_inputs.shape)>1:
                prot_inputs = np.mean(prot_inputs, axis=0)
                lig_inputs = np.mean(lig_inputs, axis=0)
                prot_s = np.mean(prot_s, axis=0)
                lig_s = np.mean(lig_s, axis=0)
                p = np.mean(p, axis=(0,1))

            all_data.append({
                "target": target,
                "ligand": ligand,
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
# Step 3: Load model and checkpoint
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPPairRanker().to(device)


print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

# Handle Lightning-style checkpoint
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
# Step 4: Forward pass and collect predictions
# ---------------------------
targets, ligands, preds = [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc="Forwarding"):
        s_p = batch["s_p"].to(device)
        s_m = batch["s_m"].to(device)
        z_pm = batch["z_pm"].to(device)

        output = model.single_forward(s_p=s_p, s_m=s_m, z_pm=z_pm)["pred"]

        targets.extend(batch["target"])
        ligands.extend(batch["ligand"])
        preds.extend(output.cpu().numpy())

# ---------------------------
# Step 5: Save predictions and compute overall AUC + EF@1%
# ---------------------------
df = pd.DataFrame({
    "targets": targets,
    "ligands": ligands,
    "score": preds
})

print(df["targets"])


# build a dic of dic to save the results. dic[target][ligand] = score

results = {}

for i in range(len(df)):
    target = df["targets"][i].lower()
    ligand = str(df["ligands"][i])
    if "&" in ligand:
        # replace to /
        ligand = ligand.replace("&", "/")

    score = df["score"][i]

    if target not in results:
        results[target] = {}
    results[target][ligand] = score


#print(results["syk"].keys())


res = []



accs = []


target_acc_res = {}

pearson_corrs = []
spearman_corrs = []

for target in df["targets"].unique():
    target = target.lower()
    path = f"./benchmark/dataset/fep/public_binding_free_energy_benchmark/fep_benchmark_inputs/structure_inputs/{test_set}/{target}_edges.csv"
    #read the csv file
    df_target = pd.read_csv(path)
    #print(df_target.columns)
    # Index(['Ligand 1', 'Ligand 2', 'ddG (kcal/mol)'], dtype='object')
    labels = []

    preds = []
    #get ddg
    for i in range(len(df_target)):
        ligand1 = str(df_target["Ligand 1"][i])
        ligand2 = str(df_target["Ligand 2"][i])
        ddg = df_target["ddG (kcal/mol)"][i]
        if np.isnan(ddg):
            continue
        #print(ddg)
        
        if abs(ddg)<=0:
            continue
        # get score of ligand1 and ligand2
        #print(target, ligand1, ligand2)
        #print(target, ligand1, ligand2)
        if "flip" in ligand1 or "flip" in ligand2:
            continue
        if ddg==0:
            print(target,ligand1,ligand2)
        score1 = results[target][ligand1]
        score2 = results[target][ligand2]

        pred_delta = score1 - score2

        # check if ddg and pred_delta are the same sign

        labels.append(ddg)
        
        preds.append(pred_delta)
    # Compute direction accuracy: how often the sign of predicted ddG matches the actual ddG
    preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds)) * 2 - 1
    preds = np.array(preds)
    dir_labels = [0 if label < 0 else 1 for label in labels]
    dir_labels = (np.array(labels) > 0).astype(int)
    acc = accuracy_score(dir_labels, (preds > 0.0).astype(int))
    
    dir_preds = preds
    
    #print(dir_labels, preds)
    auc = roc_auc_score(dir_labels, preds)
    target_acc_res[target] = {
        "acc": acc,
        "auc": auc,
        "n": len(labels)
    }

    # also calc pearson and spearman correlation

    from scipy.stats import pearsonr, spearmanr
    # pearson_corr = pearsonr(labels, preds)[0]
    # spearman_corr = spearmanr(labels, preds)[0]
    # pearson_corrs.append(pearson_corr)
    # spearman_corrs.append(spearman_corr)

mean_pearson = np.mean(pearson_corrs)
mean_spearman = np.mean(spearman_corrs)
print(f"Mean Pearson correlation: {mean_pearson:.4f}")
print(f"Mean Spearman correlation: {mean_spearman:.4f}")

    
# check 1 and 0
# get correlation
import numpy as np
from scipy.stats import pearsonr, spearmanr




for target in target_acc_res:
    acc = target_acc_res[target]["acc"]
    auc = target_acc_res[target]["auc"]
    n = target_acc_res[target]["n"]
    print(f"{target:20s}: acc = {acc:.4f}, auc = {auc:.4f}, n = {n}")

# print average acc and auc

mean_acc = np.mean([target_acc_res[target]["acc"] for target in target_acc_res])
mean_auc = np.mean([target_acc_res[target]["auc"] for target in target_acc_res])

print(f"Mean acc = {mean_acc:.4f}")
print(f"Mean auc = {mean_auc:.4f}")


from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
import numpy as np

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
        #print(ligand_name)
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
        print(f"{target_lower}: Pearson = {pearson_corr:.4f}, Spearman = {spearman_corr:.4f}, N = {len(actual_affinities)}")
    else:
        print(f"{target_lower}: Not enough data for correlation")

# # Save per-target results to CSV
summary_df = pd.DataFrame.from_dict(per_target_results, orient='index').reset_index()
summary_df.columns = ["target", "pearson", "spearman", "n"]
# summary_df.to_csv("per_target_affinity_correlation.csv", index=False)
# print("Saved per-target correlations to per_target_affinity_correlation.csv")

# Compute and print mean correlations

# exclude cmet

#summary_df = summary_df[~summary_df["target"].isin(["cmet"])]

mean_pearson = summary_df["pearson"].mean()
mean_spearman = summary_df["spearman"].mean()

print(f"\nAverage Pearson Correlation: {mean_pearson:.4f}")
print(f"Average Spearman Correlation: {mean_spearman:.4f}")










