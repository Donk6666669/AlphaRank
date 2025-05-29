import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import MLPPairRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
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












import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import MLPTripletRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import os
from collections import defaultdict

available_targets = {
        "ptp1b": "JACS8|PTP1B",
        "bace1": "JACS8|BACE",
        "tyk2": "JACS8|TYK2",
        "pfkfb3": "Merck8|PFKFB3",
        "hif-2alpha": "Merck8|HIF-2α",
        "cmet": "Merck8|c-Met",
        "shp2": "Merck8|SHP2",
        "cdk2": "JACS8|CDK2",
        "jnk1": "JACS8|JNK1",
        "mcl1": "JACS8|MCL1",
        "p38": "JACS8|P38",
        "thrombin": "JACS8|Thrombin",
        "eg5": "Merck8|EG5",
        "syk": "Merck8|SYK",
        "tnks2": "Merck8|TNKS2",
        "cdk8": "Merck8|CDK8",
    }

# ---------------------------
# Configuration
# ---------------------------

#test_set = "merck"
test_set = "jacs"
data_paths = [f"/data/{test_set}_triplet_no_order/"]
data_paths = [f"/data/{test_set}_triplet_lmdb/"]
#data_paths = [f"/data/{test_set}_triplet_constraint_new"]
ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-12_11-09-46/checkpoints/last.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-12_11-37-09/checkpoints/epoch=024-step=7325.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-12_11-37-09/checkpoints/last.ckpt"
# ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-12_11-37-09/checkpoints/epoch=017-step=5274.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-12_20-10-10/checkpoints/epoch=018-step=5567.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-12_20-10-10/checkpoints/last.ckpt"


#ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-20_10-51-43/checkpoints/epoch=022-step=690.ckpt"
ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-20_10-52-16/checkpoints/epoch=024-step=400.ckpt"
ckpt_path = "/log/train/af3rank/DataModule.MLPTripletRanker.RankCriterion.2025-05-20_17-13-56/checkpoints/epoch=048-step=784.ckpt"
#ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-21_01-16-43/checkpoints/epoch=021-step=352.ckpt"
#ckpt_path = "/data/checkpoints/triplet_seed_3/checkpoints/epoch=049-step=800.ckpt"

#ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-12_20-10-41/checkpoints/epoch=024-step=7325.ckpt"

ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-25_14-41-40/checkpoints/epoch=021-step=550.ckpt"


strategy = "s_input"
strategy = 'cat_s_m1_m2_z_pm1_pm2'
#strategy = "diff_cat_s_input"
#strategy = 'diff_s'
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Enable TF32 for better performance (optional)
if torch.__version__ >= "1.12":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True




# ---------------------------
# Load LMDB and Preprocess
# ---------------------------
all_data = []
count = 0
for path in data_paths:
    print(f"Reading from {path}")
    test_data = LMDBDataset(path)
    with test_data.env.begin(db=test_data.db[test_data.DATA_DB], write=False) as txn:
        keys = [key.decode() for key in txn.cursor().iternext(values=False)]
        for key in tqdm(keys):
            item = test_data[key]
            name = item["name"]  # expected format: target;lig1;lig2;ddg
            if "flip" in name:
                continue
            try:
                target, lig1, lig2, ddg = name.split(";")
                # if "Merck8" not in available_targets[target]:
                #     continue
                ddg = float(ddg)
                
                if  abs(ddg)<=0:
                    count+=1
                    continue
                #print(ddg)
            except:
                print(f"Skipping malformed name: {name}")
                continue

            

            s_inputs_p, s_inputs_m1, s_inputs_m2, s_p, s_m1, s_m2, z_pm1, z_pm2, _ = item["emb"]

            # if multi dim then reduce
            if len(s_inputs_m1.shape) > 1:
                s_inputs_p = s_inputs_p.mean(axis=0)
                s_inputs_m1 = s_inputs_m1.mean(axis=0)
                s_inputs_m2 = s_inputs_m2.mean(axis=0)
                s_p = s_p.mean(axis=0)
                s_m1 = s_m1.mean(axis=0)
                s_m2 = s_m2.mean(axis=0)
                z_pm1 = z_pm1.mean(axis=(0, 1))
                z_pm2 = z_pm2.mean(axis=(0, 1))
            

            # s_inputs_m1, s_inputs_m2 = s_inputs_m2, s_inputs_m1
            # s_m1, s_m2 = s_m2, s_m1
            # z_pm1, z_pm2 = z_pm2, z_pm1

            


            if ddg>0:
                s_inputs_m1, s_inputs_m2 = s_inputs_m2, s_inputs_m1
                s_m1, s_m2 = s_m2, s_m1
                z_pm1, z_pm2 = z_pm2, z_pm1
            
            # if "reverse" in test_set:
            #     s_inputs_m1, s_inputs_m2 = s_inputs_m2, s_inputs_m1
            #     s_m1, s_m2 = s_m2, s_m1
            #     z_pm1, z_pm2 = z_pm2, z_pm1


            all_data.append({
                "s_inputs_p": torch.tensor(s_inputs_p, dtype=torch.float32),
                "s_inputs_m1": torch.tensor(s_inputs_m1, dtype=torch.float32),
                "s_inputs_m2": torch.tensor(s_inputs_m2, dtype=torch.float32),
                "s_p": torch.tensor(s_p, dtype=torch.float32),
                "s_m1": torch.tensor(s_m1, dtype=torch.float32),
                "s_m2": torch.tensor(s_m2, dtype=torch.float32),
                "z_pm1": torch.tensor(z_pm1, dtype=torch.float32),
                "z_pm2": torch.tensor(z_pm2, dtype=torch.float32),
                "target": target,
                "label": ddg,
                "lig1": lig1,
                "lig2": lig2,
            })
            # # swap
            # all_data.append({
            #     "s_inputs_p": torch.tensor(s_inputs_p, dtype=torch.float32),
            #     "s_inputs_m1": torch.tensor(s_inputs_m2, dtype=torch.float32),
            #     "s_inputs_m2": torch.tensor(s_inputs_m1, dtype=torch.float32),
            #     "s_p": torch.tensor(s_p, dtype=torch.float32),
            #     "s_m1": torch.tensor(s_m2, dtype=torch.float32),
            #     "s_m2": torch.tensor(s_m1, dtype=torch.float32),
            #     "z_pm1": torch.tensor(z_pm2, dtype=torch.float32),
            #     "z_pm2": torch.tensor(z_pm1, dtype=torch.float32),
            #     "target": target,
            #     "label": 1

            # })
print(f"Loaded {len(all_data)} entries.")
print(count)
# ---------------------------
# Dataset and Dataloader
# ---------------------------
class TripletDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    return {
        "s_inputs_p": torch.stack([x["s_inputs_p"] for x in batch]),
        "s_inputs_m1": torch.stack([x["s_inputs_m1"] for x in batch]),
        "s_inputs_m2": torch.stack([x["s_inputs_m2"] for x in batch]),
        "s_p": torch.stack([x["s_p"] for x in batch]),
        "s_m1": torch.stack([x["s_m1"] for x in batch]),
        "s_m2": torch.stack([x["s_m2"] for x in batch]),
        "z_pm1": torch.stack([x["z_pm1"] for x in batch]),
        "z_pm2": torch.stack([x["z_pm2"] for x in batch]),
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.float32),
        "target": [x["target"] for x in batch],
        "lig1": [x["lig1"] for x in batch],
        "lig2": [x["lig2"] for x in batch],
    }

loader = DataLoader(TripletDataset(all_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Load Model
# ---------------------------
model = MLPTripletRanker(strategy=strategy).to(device)
print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
model.load_state_dict(state_dict)
model.eval()

# ---------------------------
# Inference and Metric Evaluation
# ---------------------------
all_preds = []
all_labels = []
per_target_preds = defaultdict(list)
per_target_labels = defaultdict(list)

with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        targets = batch["target"]
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "target", "lig1", "lig2"]}
        labels = batch["label"].to(device)

        outputs = model(**inputs)["pred"]
        preds = outputs.cpu().numpy()
        labels_np = labels.cpu().numpy()

        
        lig1_names = batch["lig1"]
        lig2_names = batch["lig2"]
        #print(len(targets))
        pair_score1_lis = [results[target][lig1] for target, lig1 in zip(targets, lig1_names)]
        pair_score2_lis = [results[target][lig2] for target, lig2 in zip(targets, lig2_names)]

        alpha = 0.5

        # final score is all_preds and difference in score2 and score1
        #print("666", pair_score1_lis)

        final_scores = alpha * np.array(preds) + (1 - alpha) * (np.array(pair_score1_lis) - np.array(pair_score2_lis))
        #final_scores = np.array(pair_score1_lis) - np.array(pair_score2_lis)

        preds = final_scores

        all_preds.extend(preds)
        all_labels.extend(labels_np)



        for t, p, l in zip(targets, preds, labels_np):
            per_target_preds[t].append(p)
            per_target_labels[t].append(l)

# ---------------------------
# Per-target Metrics
# ---------------------------
print("\n[Per-Target Evaluation Results]")
per_target_results = []


per_target_pearson = []
per_target_spearman = []

for target in sorted(per_target_preds.keys()):
    preds = np.array(per_target_preds[target])
    labels = np.array(per_target_labels[target])

    if len(np.unique(labels)) < 2:
        print(f"{target:20s} | Skipped (only one class)")
        continue
    binary_labels = (labels > 0).astype(int)

    # normalize the preds to -1 1
    preds = (preds - np.min(preds)) / (np.max(preds) - np.min(preds)) * 2 - 1

    print(preds)
    
    auc = roc_auc_score(binary_labels, preds)
    # if is nan
    if np.isnan(auc):
        auc = 1
    # check different thresholds
   

    acc = accuracy_score(binary_labels, (preds > 0.0).astype(int))
    print(f"{target:20s} | AUC: {auc:.4f} | ACC: {acc:.4f} | N: {len(binary_labels)}")

    per_target_results.append((target, auc, acc, len(binary_labels)))

    # pearson
    from scipy.stats import pearsonr, spearmanr
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    # print(f"Pearson correlation: {pearson_corr:.4f}")
    # print(f"Spearman correlation: {spearman_corr:.4f}")

    per_target_pearson.append(pearson_corr)
    per_target_spearman.append(spearman_corr)

# average over targets

avg_auc = np.mean([result[1] for result in per_target_results])
avg_acc = np.mean([result[2] for result in per_target_results])

print(f"\nAverage AUC: {avg_auc:.4f}")
print(f"Average ACC: {avg_acc:.4f}")

avg_pearson = np.mean(per_target_pearson)
avg_spearman = np.mean(per_target_spearman)
print(f"\nAverage Pearson correlation: {avg_pearson:.4f}")
print(f"Average Spearman correlation: {avg_spearman:.4f}")








