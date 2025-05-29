import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import CombineTripletRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import os
from collections import defaultdict

# ---------------------------
# Configuration
# ---------------------------
test_set = "merck"
data_paths = [f"/data/{test_set}_triplet_no_order/"]
ckpt_path = "/data/checkpoints/DataModule.MLPTripletRanker.RankCriterion.2025-05-20_10-52-16/checkpoints/epoch=024-step=400.ckpt"
ckpt_path = "/data/checkpoints/DataModule.CombineTripletRanker.RankCriterion.2025-05-21_11-26-54/checkpoints/epoch=003-step=1928.ckpt"
ckpt_path = "/data/checkpoints/DataModule.CombineTripletRanker.RankCriterion.2025-05-21_11-26-44/checkpoints/epoch=013-step=6762.ckpt"
strategy = "from_z"  # <-- SET THIS: from_z | from_s | from_sz
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.__version__ >= "1.12":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# ---------------------------
# Load LMDB and Preprocess
# ---------------------------
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
            try:
                target, lig1, lig2, ddg = name.split(";")
                ddg = float(ddg)
            except:
                print(f"Skipping malformed name: {name}")
                continue

            s_inputs_p, s_inputs_m1, s_inputs_m2, s_p, s_m1, s_m2, z_pm1, z_pm2, _ = item["emb"]

            # if ddg > 0:
            #     s_inputs_m1, s_inputs_m2 = s_inputs_m2, s_inputs_m1
            #     s_m1, s_m2 = s_m2, s_m1
            #     z_pm1, z_pm2 = z_pm2, z_pm1
            
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
                "label": int(ddg < 0)
            })


print(f"Loaded {len(all_data)} entries.")

# ---------------------------
# Dataset and Dataloader
# ---------------------------
from torch.nn.utils.rnn import pad_sequence

class TripletDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)



def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence

    # Extract
    s_inputs_p_list = [x["s_inputs_p"] for x in batch]
    s_inputs_m1_list = [x["s_inputs_m1"] for x in batch]
    s_inputs_m2_list = [x["s_inputs_m2"] for x in batch]
    s_p_list = [x["s_p"] for x in batch]
    s_m1_list = [x["s_m1"] for x in batch]
    s_m2_list = [x["s_m2"] for x in batch]
    z_pm1_list = [x["z_pm1"] for x in batch]
    z_pm2_list = [x["z_pm2"] for x in batch]

    # Sequence lengths
    len_p = [x.shape[0] for x in s_inputs_p_list]
    len_m1 = [x.shape[0] for x in s_inputs_m1_list]
    len_m2 = [x.shape[0] for x in s_inputs_m2_list]
    max_len_p = max(len_p)
    max_len_m1 = max(len_m1)
    max_len_m2 = max(len_m2)

    # Feature dimensions
    s_input_dim = s_inputs_p_list[0].shape[1]
    s_dim = s_p_list[0].shape[1]
    z_dim = z_pm1_list[0].shape[2]

    # Pad 2D tensors
    s_inputs_p = pad_sequence(s_inputs_p_list, batch_first=True)
    s_inputs_m1 = pad_sequence(s_inputs_m1_list, batch_first=True)
    s_inputs_m2 = pad_sequence(s_inputs_m2_list, batch_first=True)
    s_p = pad_sequence(s_p_list, batch_first=True)
    s_m1 = pad_sequence(s_m1_list, batch_first=True)
    s_m2 = pad_sequence(s_m2_list, batch_first=True)

    # Pad z_pm1, z_pm2 manually to 4D [B, max_len_p, max_len_m, z_dim]
    z_pm1 = torch.zeros(len(batch), max_len_p, max_len_m1, z_dim)
    z_pm2 = torch.zeros(len(batch), max_len_p, max_len_m2, z_dim)
    mask_pm1 = torch.zeros(len(batch), max_len_p, max_len_m1)
    mask_pm2 = torch.zeros(len(batch), max_len_p, max_len_m2)

    for i in range(len(batch)):
        lp = len_p[i]
        lm1 = len_m1[i]
        lm2 = len_m2[i]
        z_pm1[i, :lp, :lm1, :] = batch[i]["z_pm1"]
        z_pm2[i, :lp, :lm2, :] = batch[i]["z_pm2"]
        mask_pm1[i, :lp, :lm1] = 1.0
        mask_pm2[i, :lp, :lm2] = 1.0

    return {
        "s_inputs_p": s_inputs_p,
        "s_inputs_m1": s_inputs_m1,
        "s_inputs_m2": s_inputs_m2,
        "s_p": s_p,
        "s_m1": s_m1,
        "s_m2": s_m2,
        "z_pm1": z_pm1,
        "z_pm2": z_pm2,
        "mask_pm1": mask_pm1,
        "mask_pm2": mask_pm2,
        "label": torch.tensor([x["label"] for x in batch], dtype=torch.float32),
        "target": [x["target"] for x in batch],
    }

loader = DataLoader(TripletDataset(all_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Load Model
# ---------------------------
model = CombineTripletRanker(strategy=strategy).to(device)
print(f"Loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
model.load_state_dict(state_dict)
model.eval()

# ---------------------------
# Inference and Metrics
# ---------------------------
all_preds = []
all_labels = []
per_target_preds = defaultdict(list)
per_target_labels = defaultdict(list)

with torch.no_grad():
    for batch in tqdm(loader, desc="Predicting"):
        targets = batch["target"]
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ["label", "target"]}
        labels = batch["label"].to(device)

        outputs = model(**inputs)["pred"]
        #preds = outputs.cpu().numpy()
        preds = np.atleast_1d(outputs.cpu().numpy())
        labels_np = labels.cpu().numpy()

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

for target in sorted(per_target_preds.keys()):
    preds = np.array(per_target_preds[target])
    labels = np.array(per_target_labels[target])

    if len(np.unique(labels)) < 2:
        print(f"{target:20s} | Skipped (only one class)")
        continue

    auc = roc_auc_score(labels, preds)
    acc = accuracy_score(labels, (preds > 0.5).astype(int))
    print(f"{target:20s} | AUC: {auc:.4f} | ACC: {acc:.4f} | N: {len(labels)}")

    per_target_results.append((target, auc, acc, len(labels)))

avg_auc = np.mean([result[1] for result in per_target_results])
avg_acc = np.mean([result[2] for result in per_target_results])

print(f"\nAverage AUC: {avg_auc:.4f}")
print(f"Average ACC: {avg_acc:.4f}")

# ---------------------------
# Overall Metrics
# ---------------------------
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

overall_auc = roc_auc_score(all_labels, all_preds)
overall_acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))

print(f"\n[Overall Evaluation]")
print(f"Overall AUC: {overall_auc:.4f}")
print(f"Overall ACC: {overall_acc:.4f}")