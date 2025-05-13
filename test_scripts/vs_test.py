import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import MLPPairRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import pandas as pd
import os

# ---------------------------
# Step 1: Load and preprocess LMDB data
# ---------------------------

test = "pcba"


if test == "pcba":
    data_paths = [
    "/data/pcba_full_0/",
    "/data/pcba_full_1/",
    "/data/pcba_full_2/",
    "/data/pcba_full_3/",

    ]
elif test == "dude":
    data_paths = [
        "/data/dude_full_0/",
        "/data/dude_full_1/",
        "/data/dude_full_2/",
        "/data/dude_full_3/",
        "/data/dude_remain_0/",
        "/data/dude_remain_1/",
        "/data/dude_remain_2/",
        "/data/dude_remain_3/",
    ]

ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_10-38-27/checkpoints/epoch=024-step=750.ckpt"
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_09-44-57/checkpoints/epoch=020-step=6153.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-09_09-44-57/checkpoints/last.ckpt"
ckpt_path = '/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-54-21/checkpoints/epoch=022-step=3450.ckpt'
ckpt_path = "/data/checkpoints/DataModule.MLPPairRanker.RankCriterion.2025-05-10_21-55-22/checkpoints/epoch=018-step=2717.ckpt"




all_data = []

for path in data_paths:
    print(f"Reading from {path}")
    test_data = LMDBDataset(path)
    with test_data.env.begin(db=test_data.db[test_data.DATA_DB], write=False) as txn:
        keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        for key in tqdm(keys):
            item = test_data[key]
            name = item["name"]
            prot_inputs, lig_inputs, prot_s, lig_s, p = item["emb"]

            label = 1 if "active" in name else 0
            all_data.append({
                "name": name,
                "label": label,
                "s_p": torch.tensor(prot_s, dtype=torch.float32),
                "s_m": torch.tensor(lig_s, dtype=torch.float32),
                "z_pm": torch.tensor(p, dtype=torch.float32)
            })

print(f"Loaded {len(all_data)} entries.")

# get unique names and only keep the first occurrence

unique_names = set()

new_data_list = []

for data in all_data:
    name = data["name"]
    if name not in unique_names:
        unique_names.add(name)
        new_data_list.append(data)

print(f"Filtered to {len(new_data_list)} unique entries.")

# save unique names to a file

with open(f"unique_{test}_names.txt", "w") as f:
    for name in unique_names:
        f.write(f"{name}\n")




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
        "names": [item["name"] for item in batch],
        "labels": [item["label"] for item in batch],
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
names, labels, preds = [], [], []

with torch.no_grad():
    for batch in tqdm(loader, desc="Forwarding"):
        s_p = batch["s_p"].to(device)
        s_m = batch["s_m"].to(device)
        z_pm = batch["z_pm"].to(device)

        output = model.single_forward(s_p=s_p, s_m=s_m, z_pm=z_pm)["pred"]

        names.extend(batch["names"])
        labels.extend(batch["labels"])
        preds.extend(output.cpu().numpy())

# ---------------------------
# Step 5: Save predictions and compute overall AUC + EF@1%
# ---------------------------
df = pd.DataFrame({
    "name": names,
    "label": labels,
    "score": preds
})
df["target"] = df["name"].apply(lambda x: x.split("_")[0])

print(df["target"])

df.to_csv("predictions.csv", index=False)
print("Saved predictions to predictions.csv")

# Compute overall AUC
overall_auc = roc_auc_score(df["label"], df["score"])
print(f"Overall ROC AUC: {overall_auc:.4f}")

# Compute overall EF@1%
top_percent = 0.01
df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)
top_n = max(int(len(df_sorted) * top_percent), 1)
top_df = df_sorted.iloc[:top_n]

n_total = len(df)
n_actives = df["label"].sum()
n_top_actives = top_df["label"].sum()
expected = n_actives * top_percent
overall_ef1 = n_top_actives / expected if expected > 0 else 0.0

print(f"Overall Enrichment Factor @1%: {overall_ef1:.4f}")

# ---------------------------
# Step 6: Per-target AUC and EF@1%
# ---------------------------
per_target_results = []
for target, group in df.groupby("target"):
    labels = group["label"].values
    scores = group["score"].values
    #print(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        continue  # skip invalid targets

    try:
        auc = roc_auc_score(labels, scores)
    except:
        continue

    sorted_group = group.sort_values("score", ascending=False).reset_index(drop=True)
    top_n = max(int(len(sorted_group) * top_percent), 1)
    top_hits = sorted_group.iloc[:top_n]
    actives_in_top = top_hits["label"].sum()
    expected = labels.sum() * top_percent
    ef1 = actives_in_top / expected if expected > 0 else 0.0
    print(ef1)

    per_target_results.append({
        "target": target,
        "auc": auc,
        "ef1": ef1
    })

results_df = pd.DataFrame(per_target_results)
results_df.to_csv("per_target_results.csv", index=False)

mean_auc = results_df["auc"].mean()
mean_ef1 = results_df["ef1"].mean()

print(f"Mean AUC across targets: {mean_auc:.4f}")
print(f"Mean EF@1% across targets: {mean_ef1:.4f}")
print("Per-target results saved to per_target_results.csv")





