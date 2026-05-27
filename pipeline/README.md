# ProtenixAffinity Pipeline

一个用于蛋白-小分子亲和力预测的全自动 Pipeline 工具。

> **第一次使用？** 请直接看 [QUICKSTART.md](QUICKSTART.md) — 只需要 3 个输入文件，运行一条命令即可。

## 📋 功能概述

这个Pipeline整合了从输入文件（FASTA + SMI）到最终亲和力评分（CSV）的完整流程：

1. **JSON生成**: 从FASTA和SMI文件生成蛋白-小分子对的JSON配置
2. **脚本生成**: 自动生成protenix predict的Shell脚本
3. **LMDB生成**: 运行protenix predict生成LMDB数据库
4. **数据转换**: 给LMDB添加可读的名字标识
5. **亲和力预测**: 在同一个容器中计算亲和力评分并生成CSV

## 🏗️ 系统架构

### 容器配置

现在完整流程统一在 **一个容器** 中执行：

- **运行容器 (alpharank_hyper_decoy_wukelin)**: 用于运行 protenix predict、数据预处理和 inference
  - 宿主机 `/home/wukelin/ProtenixAffinity2` → 容器 `/project`
  - 宿主机 `/msa/data` → 容器 `/data`

主控脚本仍从当前 codebase（`/home/wukelin/ProtenixAffinity`）启动；运行时会自动把必要的辅助脚本同步到容器 `/project` 映射目录，并把 JSON/Shell 脚本放到容器 `/data` 下的 LMDB 目录中。

## 📂 目录结构

```
ProtenixAffinity/
├── pipeline/
│   ├── generate_json.py       # 生成JSON配置文件
│   ├── generate_sh.py         # 生成Shell脚本（调用 run_predict.py）
│   ├── run_pipeline.py        # 主入口脚本（单容器全流程自动化）
│   ├── run_predict.py         # protenix predict 包装器（核心推断）
│   ├── run_transform.py       # Transform包装器
│   └── README.md              # 本文档
├── test_scripts/
│   └── transform.py           # LMDB名字转换（运行时同步到单容器/project）
├── batch_run.py               # GPU任务调度
└── output/                    # 输出目录（自动生成）
    └── <project_name>/
        ├── json/              # JSON配置文件
        └── scripts/           # Shell脚本
```

## 🚀 快速开始

### 前置要求

1. 准备输入文件：
   - **FASTA文件**: 包含蛋白序列
     ```
     >ProteinName
     MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK...
     ```
   - **SMI文件**: 包含小分子SMILES
     ```
     ligand_001 CCO
     ligand_002 c1ccccc1
     ```
   - **MSA文件**: 预计算的MSA数据（放在运行容器可访问的路径）

     > 单容器的 `/data` 对应宿主机 `/msa/data`。如果你的 MSA 仍在旧宿主路径 `/data/protein/...`，主流程会自动复制到 `/msa/data/...`，确保容器内 `/data/...` 可见。

2. 确保Docker容器正在运行：
  ```bash
  docker ps | grep alpharank_hyper_decoy_wukelin
  ```

### 完整运行示例

```bash
python pipeline/run_pipeline.py \
    --fasta /msa/data/wukelin/protein.fasta \
    --smi /msa/data/wukelin/ligands.smi \
    --msa_dir /data/rerank/protenix/chembl_bdb/openbind/msa \
    --project_name AlphaRank \
    --output_dir /data/rerank/protenix/chembl_bdb/alpharank/output \
    --lmdb_dir /data/rerank/protenix/chembl_bdb/alpharank/pair \
    --ckpt_path /log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt \
    --gpu 0 \
    --seeds 101 \
    --mode full
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--fasta` | 蛋白质FASTA文件路径（宿主机） | `/msa/data/protein.fasta` |
| `--smi` | 小分子SMI文件路径（宿主机） | `/msa/data/ligands.smi` |
| `--msa_dir` | MSA目录（运行容器内路径） | `/data/rerank/.../msa` |
| `--project_name` | 项目名称 | `AlphaRank` |
| `--output_dir` | 运行容器输出目录 | `/data/.../output` |
| `--lmdb_dir` | 运行容器 LMDB/JSON/Shell 中间文件目录 | `/data/.../pair` |
| `--gpu` | GPU编号，同时用于 Protenix predict 和 openbind inference | `0`, `1`, `3` |
| `--seeds` | 随机种子 | `101` |
| `--ckpt_path` | 排序模型checkpoint路径（运行容器内） | `/log/train/.../ema_epoch40-49.ckpt` |
| `--mode` | 运行模式 | `full`, `generate_only`, `from_lmdb` |

### 运行模式

1. **full**: 执行完整pipeline（推荐）
2. **generate_only**: 仅生成JSON和Shell脚本（用于验证配置）
3. **from_lmdb**: 从已有LMDB开始（跳过预测步骤）

## 🔧 分步执行

完整流程（`--mode full`）包含 6 个全自动步骤，无需人工干预：

| 步骤 | 说明 | 执行位置 |
|---|---|---|
| 1 | 生成JSON配置（蛋白-小分子对 + 蛋白-only） | 宿主机 |
| 2 | 生成Shell脚本 | 宿主机 |
| 3 | 复制JSON/Shell脚本并同步辅助脚本 | 宿主机 → 单一运行容器共享目录 |
| 4 | 运行 Protenix predict（AlphaFold3推断，生成pair embedding LMDB） | 单一运行容器（GPU）|
| 5 | 运行 transform.py（给pair embedding LMDB添加可读名字，并自动抽取protein LMDB） | 单一运行容器 |
| 6 | 运行 openbind.py（ranking模型推断，生成CSV） | 单一运行容器（GPU）|

> **重要说明**：步骤4使用 `pipeline/run_predict.py` 调用 Protenix 的 `inference_jsons()` 函数，以 `--reduce` 模式生成 5-tuple pair embedding `(s_inputs_p, s_inputs_m, s_p, s_m, z_pm)`。步骤5会从 pair embedding 中自动抽取 protein embedding `(s_inputs_p, s_p)`，因此单容器模式下不再依赖额外的 protein-only predict。

### 单独运行各步骤

```bash
# 步骤1: 生成JSON
python pipeline/generate_json.py \
    --fasta /path/protein.fasta --smi /path/ligands.smi \
    --msa_dir /data/.../msa --output_dir ./output/json \
    --project_name my_project

# 步骤2: 生成Shell脚本
python pipeline/generate_sh.py \
    --project_name my_project \
    --json_dir /data/.../pair --output_dir /data/.../output \
    --lmdb_dir /data/.../pair \
    --script_dir ./output/scripts --gpu 0 --seeds 101

# 步骤4: 在单一运行容器中运行推断（由run_pipeline自动执行）
docker exec -i alpharank_hyper_decoy_wukelin /bin/zsh -c \
  'cd /project && bash /data/.../pair/my_project_pair.sh'

# 步骤5: transform（由run_pipeline自动执行）
docker exec -i alpharank_hyper_decoy_wukelin /bin/zsh -c \
  'cd /project && python test_scripts/transform.py \
    --old_lmdb /data/.../my_project.lmdb \
    --json_path /data/.../my_project.json \
    --new_lmdb /data/.../my_project_named.lmdb \
    --protein_lmdb /data/.../my_project_protein.lmdb'

# 步骤6: inference（由run_pipeline自动执行）
docker exec -i alpharank_hyper_decoy_wukelin /bin/zsh -c \
  'cd /project && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python test_scripts/openbind.py \
    --data_paths /data/.../my_project_named.lmdb/ \
    --protein_path /data/.../my_project_protein.lmdb/ \
    --ckpt_path /log/train/alpharank_hyper_decoy_new/.../ema_epoch40-49.ckpt \
    --output_dir /project/output/my_project/results'
```

## 📊 输出文件

执行完成后，会生成以下文件：

```
output/<project_name>/
├── json/
│   ├── <project_name>.json          # 蛋白-小分子对JSON
│   └── <project_name>_protein.json  # 蛋白-only JSON（保留作兼容/检查）
├── scripts/
│   ├── <project_name>_pair.sh       # 蛋白-小分子对脚本
│   └── <project_name>_protein.sh    # 蛋白-only脚本（单容器模式默认不执行）
└── results/
    └── <target_name>.csv            # 每个target一个CSV
```

CSV格式（实际输出）：
```csv
ligand,score
test_lig_002,-11.511119
test_lig_001,-11.539435
test_lig_003,-11.542738
```
**score** 为亲和力得分（越小越强）。运行日志中会额外打印 `dist/angle/omega` 等几何描述符。

## 🐛 故障排查

### 常见问题

1. **JSON生成失败**
   - 检查FASTA和SMI文件格式是否正确
   - 确保文件路径存在且可访问

2. **protenix predict运行失败**
  - 检查GPU是否可用：`nvidia-smi`
  - 查看容器日志：`docker logs alpharank_hyper_decoy_wukelin`
  - 确认MSA文件路径正确
  - 如果 GPU 0 已满，改用空闲 GPU，例如 `--gpu 3`

3. **LMDB或中间文件生成失败**
  - 检查磁盘空间：`df -h`
  - 确认 `/data` 路径映射到宿主机 `/msa/data`

4. **Inference失败**
  - 检查运行容器中的路径配置
  - 确认 `--lmdb_dir` 下存在 `<project_name>_named.lmdb` 和 `<project_name>_protein.lmdb`
  - 确认使用了空闲 GPU；`--gpu` 会同时限制 Protenix 和 openbind

### 日志查看

```bash
# 查看运行容器日志
docker logs alpharank_hyper_decoy_wukelin -f
```

## 🔄 更新和维护

### 更新代码

```bash
cd /home/wukelin/ProtenixAffinity
git pull
```

### 清理临时文件

```bash
# 清理输出目录
rm -rf output/<project_name>

# 清理LMDB（谨慎操作）
rm -rf /msa/data/.../pair/*
```

## 📝 最佳实践

1. **命名规范**: 使用描述性的项目名称，如 `CRBN_IKZF1_2024`
2. **路径管理**: 使用绝对路径避免混淆
3. **资源监控**: 运行前检查GPU和磁盘空间
4. **数据备份**: 重要结果及时备份
5. **日志保存**: 保存关键步骤的日志输出

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

[MIT License](LICENSE)

## 📧 联系方式

如有问题，请联系项目维护者。
