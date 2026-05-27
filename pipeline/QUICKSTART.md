# 蛋白-小分子亲和力预测 — 快速上手

**你只需要准备3样东西，运行一条命令，就能得到亲和力打分 CSV。**

当前流程已改为**单容器模式**：所有 Protenix predict、LMDB transform、protein LMDB 抽取和 openbind inference 都在 `alpharank_hyper_decoy_wukelin` 中完成。主控脚本仍从 `/home/wukelin/ProtenixAffinity` 启动，运行时会自动同步必要脚本到容器 `/project` 映射目录。

---

## 第一步：准备输入文件

### 1. FASTA 文件（蛋白质序列）

```fasta
>ProteinName
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK...
```

- 第一行 `>` 后面是蛋白名（作为结果中的 target 列名）
- 支持多蛋白（每个蛋白各占两行）

### 2. SMI 文件（小分子 SMILES）

```
ligand_001 CCO
ligand_002 c1ccccc1
ligand_003 CC(C)O
CC(=O)Oc1ccccc1C(=O)O ligand_004
```

- 格式：`名字 SMILES` 或 `SMILES 名字`（自动识别）
- 每行一个小分子

### 3. MSA 目录（预计算的多序列比对）

放在**运行容器可访问**的路径下，例如：
```
/data/rerank/protenix/chembl_bdb/openbind/msa/
```

> 如果这个 MSA 目录只存在于旧挂载 `/data/protein/...`，pipeline 会自动复制到单容器挂载 `/msa/data/...`，使其在 `alpharank_hyper_decoy_wukelin` 的 `/data/...` 下可见。

> 如果还没有 MSA，可以先在 [ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/MSA.ipynb) 生成。

---

## 第二步：确认容器正在运行

```bash
docker ps | grep alpharank_hyper_decoy_wukelin
```

应该看到这个容器：
```
alpharank_hyper_decoy_wukelin
```

---

## 第三步：运行！

```bash
cd /home/wukelin/ProtenixAffinity

python pipeline/run_pipeline.py \
    --fasta  /home/wukelin/ProtenixAffinity/test_data/test_protein.fasta \
    --smi    /home/wukelin/ProtenixAffinity/test_data/test_ligands.smi   \
    --msa_dir /data/rerank/protenix/chembl_bdb/openbind/msa              \
    --project_name  AlphaRank                                             \
    --output_dir /data/rerank/protenix/chembl_bdb/alpharank/output       \
    --lmdb_dir   /data/rerank/protenix/chembl_bdb/alpharank/pair         \
    --ckpt_path /log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt \
    --gpu    0                                                            \
    --seeds  101                                                          \
    --mode   full
```

> **`--output_dir` / `--lmdb_dir`** 是运行容器内的中间文件存放路径，
> 只要磁盘有空间，路径不存在时会自动创建，按项目名区分即可。
> **`--gpu`** 同时控制 Protenix predict 和 openbind inference；如果 GPU 0 已满，可以改成空闲 GPU，例如 `--gpu 3`。

运行时间大约：每个 蛋白-小分子对 约 10~30 秒（取决于序列长度和 GPU 速度）。

---

## 第四步：查看结果

Pipeline 完成后，结果 CSV 保存在：

```
output/AlphaRank/results/<target_name>.csv
```

例如 `output/AlphaRank/results/ev-a71_2a.csv`：

```csv
ligand,score
test_lig_002,-11.511119
test_lig_001,-11.539435
test_lig_003,-11.542738
```

- **score 是负的双曲测地距离，通常数值越大（越接近 0）预测亲和力越强**
- 文件按 target 分组，每个 target 一个 CSV

---

## 常见问题

**Q: 提示 `docker: not found` 或容器连不上**
```bash
docker ps  # 检查Docker是否正常
docker start alpharank_hyper_decoy_wukelin  # 重启容器
```

**Q: MSA 路径找不到**
- MSA 目录必须在**运行容器可访问的路径**（即宿主机 `/msa/data/...` 下）
- 如果原始 MSA 在旧路径 `/data/protein/...` 下，pipeline 会自动复制到 `/msa/data/...`
- 检查：`ls /msa/data/<你的msa路径>/`

**Q: 运行中途报错，想从某步继续**
```bash
# 从已有LMDB开始（跳过protenix predict，直接做transform+inference）
python pipeline/run_pipeline.py ... --mode from_lmdb
```

**Q: 想先只生成JSON和脚本检查配置**
```bash
python pipeline/run_pipeline.py ... --mode generate_only
```

**Q: 如何指定不同的模型权重**
```bash
python pipeline/run_pipeline.py ... \
    --ckpt_path /log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt
```

---

## 参数速查

| 参数 | 必填 | 说明 |
|---|---|---|
| `--fasta` | ✅ | 蛋白质 FASTA 文件（宿主机绝对路径） |
| `--smi` | ✅ | 小分子 SMI 文件 |
| `--msa_dir` | ✅ | MSA 目录（运行容器内路径，形如 `/data/...`） |
| `--project_name` | ✅ | 项目名，用于区分输出 |
| `--output_dir` | ✅ | 运行容器中间输出目录（容器内路径）|
| `--lmdb_dir` | ✅ | 运行容器 LMDB/JSON/Shell 中间文件目录（容器内路径）|
| `--gpu` | | GPU 编号，默认 `0` |
| `--ckpt_path` | | 排序模型 checkpoint，有默认值 |
| `--mode` | | `full`（默认）/ `generate_only` / `from_lmdb` |

---

## 小测试

用内置测试数据验证环境：

```bash
cd /home/wukelin/ProtenixAffinity
bash test_data/test_run.sh
```

成功后会在 `output/TestRun/results/ev-a71_2a.csv` 看到 3 个小分子的打分结果。

