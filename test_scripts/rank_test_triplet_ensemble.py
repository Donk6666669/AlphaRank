import torch
from torch.utils.data import Dataset, DataLoader
from protenix.model.rank_model import MLPTripletRanker
from protenix.utils.lmdb import LMDBDataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import os
from collections import defaultdict

# ---------------------------
# Configuration
# ---------------------------

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

test_set = "jacs_triplet_lmdb"
test_set = "merck_triplet_lmdb"
#test_set = "jacs_set"
data_paths = [f"/data/{test_set}/"]
#data_paths = [f"/data/{test_set}_triplet_constraint"]

ckpt_paths = [
    "/data/checkpoints/triplet_seed_1/checkpoints/epoch=047-step=768.ckpt",
    "/data/checkpoints/triplet_seed_2/checkpoints/epoch=041-step=672.ckpt",
    "/data/checkpoints/triplet_seed_3/checkpoints/epoch=049-step=800.ckpt",
    "/data/checkpoints/triplet_seed_2025/checkpoints/epoch=047-step=768.ckpt",
    "/data/checkpoints/triplet_seed_2026/checkpoints/epoch=043-step=704.ckpt",
   
]

strategy = 'cat_s_m1_m2_z_pm1_pm2'
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
                if "Merck8" not in available_targets[target]:
                    continue
                ddg = float(ddg)
            except:
                print(f"Skipping malformed name: {name}")
                continue

            s_inputs_p, s_inputs_m1, s_inputs_m2, s_p, s_m1, s_m2, z_pm1, z_pm2, _ = item["emb"]

            if ddg < 0:
                s_inputs_m1, s_inputs_m2 = s_inputs_m2, s_inputs_m1
                s_m1, s_m2 = s_m2, s_m1
                z_pm1, z_pm2 = z_pm2, z_pm1

            if "reverse" in test_set:
                s_inputs_m1, s_inputs_m2 = s_inputs_m2, s_inputs_m1
                s_m1, s_m2 = s_m2, s_m1
                z_pm1, z_pm2 = z_pm2, z_pm1
            if len(s_inputs_m1.shape) > 1:
                s_inputs_p = s_inputs_p.mean(axis=0)
                s_inputs_m1 = s_inputs_m1.mean(axis=0)
                s_inputs_m2 = s_inputs_m2.mean(axis=0)
                s_p = s_p.mean(axis=0)
                s_m1 = s_m1.mean(axis=0)
                s_m2 = s_m2.mean(axis=0)
                z_pm1 = z_pm1.mean(axis=(0, 1))
                z_pm2 = z_pm2.mean(axis=(0, 1))

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
                "label": -ddg
            })

print(f"Loaded {len(all_data)} entries.")

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
    }

loader = DataLoader(TripletDataset(all_data), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# ---------------------------
# Load Multiple Models
# ---------------------------
models = []
for ckpt_path in ckpt_paths:
    print(f"Loading checkpoint from {ckpt_path}")
    model = MLPTripletRanker(strategy=strategy).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)

# ---------------------------
# Inference with Ensemble
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

        # Average ensemble
        ensemble_preds = None
        for model in models:
            output = model(**inputs)["pred"]
            if ensemble_preds is None:
                ensemble_preds = output
            else:
                ensemble_preds += output
        ensemble_preds /= len(models)

        # max ensemble
        # ensemble_preds = None
        # for model in models:
        #     output = model(**inputs)["pred"]
        #     if ensemble_preds is None:
        #         ensemble_preds = output
        #     else:
        #         ensemble_preds = torch.max(ensemble_preds, output)

        preds = ensemble_preds.cpu().numpy()
        labels_np = labels.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels_np)

        for t, p, l in zip(targets, preds, labels_np):
            per_target_preds[t].append(p)
            per_target_labels[t].append(l)

# ---------------------------
# Per-target Evaluation
# ---------------------------
print("\n[Per-Target Evaluation Results]")
per_target_results = []

for target in sorted(per_target_preds.keys()):
    preds = np.array(per_target_preds[target])
    labels = np.array(per_target_labels[target])

    if len(np.unique(labels)) < 2:
        print(f"{target:20s} | Skipped (only one class)")
        continue
    binary_labels = (labels > 0).astype(int)
    auc = roc_auc_score(binary_labels, preds)
    acc = accuracy_score(binary_labels, (preds > 0.5).astype(int))
    print(f"{target:20s} | AUC: {auc:.4f} | ACC: {acc:.4f} | N: {len(labels)}")
    per_target_results.append((target, auc, acc, len(labels)))

avg_auc = np.mean([r[1] for r in per_target_results])
avg_acc = np.mean([r[2] for r in per_target_results])
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