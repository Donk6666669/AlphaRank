# AlphaRank

AlphaRank 是一个面向 **protein–ligand binding affinity ranking** 的双曲空间排序模型。

它使用 co-folding 模型提取蛋白和小分子的表示，再将表示投影到 Lorentz 双曲空间中，并以**负的双曲测地距离**作为亲和力排序分数。直观来说，AlphaRank 的目标是：给定一个蛋白和一批候选小分子，输出这些小分子的相对亲和力排序。

本仓库已经把完整流程打包成单容器 Pipeline：用户只需要准备 FASTA、SMI、MSA 和模型 checkpoint，即可从输入文件直接得到最终 CSV 排序结果。

---

## 1. 整体流程

AlphaRank 从输入到输出一共做 6 步：

```text
FASTA + SMI + MSA
        │
        ▼
1. 生成 protein–ligand JSON
        │
        ▼
2. 生成 Protenix/co-folding 推理脚本
        │
        ▼
3. 将 JSON 和脚本放到 Docker 容器可访问的数据目录
        │
        ▼
4. 在 Docker 容器中运行 co-folding，生成 pair embedding LMDB
        │
        ▼
5. 转换 LMDB 名称，并从 pair embedding 中抽取 protein LMDB
        │
        ▼
6. 加载 AlphaRank checkpoint，输出 per-target ranking CSV
```

输出文件格式如下：

```csv
ligand,score
test_lig_002,-11.50997543334961
test_lig_001,-11.539664268493652
test_lig_003,-11.541532516479492
```

`score` 是负的双曲测地距离，通常**数值越大（越接近 0）表示预测亲和力越强**。

---

## 2. 系统要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux |
| GPU | NVIDIA GPU，建议显存 ≥ 16 GB |
| Docker | 20.10+ |
| GPU 容器支持 | NVIDIA Container Toolkit |
| Python | 宿主机只需要能运行 `docker.py`，主要依赖都在 Docker 内 |
| 数据 | 预计算 MSA 目录 + AlphaRank checkpoint |

验证 GPU：

```bash
nvidia-smi
```

验证 Docker GPU 支持：

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

---

## 3. Clone 仓库

```bash
git clone https://github.com/Donk6666669/AlphaRank.git
cd AlphaRank
```

仓库中已经包含：

```text
AlphaRank/
├── docker/                 # Dockerfile 与容器依赖
├── docker-compose.yml      # Docker Compose 配置
├── docker.py               # 容器管理脚本
├── pipeline/               # 一键 Pipeline 主逻辑
├── protenix/               # co-folding 与 AlphaRank 模型代码
├── runner/                 # co-folding 推理入口
├── configs/                # co-folding 推理配置
├── conf/                   # AlphaRank 相关配置
├── test_data/              # 最小示例输入
├── test_scripts/           # LMDB 转换与 AlphaRank inference 脚本
├── lorentz.py              # 双曲空间几何工具
└── README.md               # 当前说明文件
```

---

## 4. 准备数据目录

Docker 容器会使用三个主要挂载目录：

| 宿主机目录 | 容器内目录 | 用途 |
|---|---|---|
| 当前仓库目录 | `/project` | 代码 |
| 数据目录 | `/data` | MSA、中间 LMDB、JSON、Shell 脚本 |
| 日志/权重目录 | `/log` | checkpoint |

推荐目录示例：

```text
/home/yourname/AlphaRank          # 代码仓库
/data/alpharank                   # 数据、MSA、中间结果
/data/alpharank_log               # checkpoint 或日志
```

创建目录：

```bash
mkdir -p /data/alpharank
mkdir -p /data/alpharank_log
mkdir -p container_home
```

如果当前机器已经有类似 `/msa/data` 和 `/msa/data/wukelin/log` 这样的共享目录，也可以直接使用已有目录。

---

## 5. 准备 MSA

MSA 是 co-folding 的必要输入。示例 Pipeline 默认使用以下容器内路径：

```text
/data/rerank/protenix/chembl_bdb/openbind/msa
```

由于容器内 `/data` 会映射到宿主机的数据目录，因此如果你的 `DATA_ROOT=/data/alpharank`，那么宿主机上应准备：

```text
/data/alpharank/rerank/protenix/chembl_bdb/openbind/msa
```

目录结构示例：

```text
msa/
└── EV-A71_2A/
    └── EV-A71_2A.a3m
```

注意：FASTA 中 `>` 后面的蛋白名需要和 MSA 子目录/文件名对应。

---

## 6. 准备 AlphaRank checkpoint

示例 Pipeline 默认 checkpoint 容器内路径为：

```text
/log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt
```

如果你的 `LOG_ROOT=/data/alpharank_log`，那么宿主机上对应文件应放在：

```text
/data/alpharank_log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt
```

checkpoint 文件不建议直接放入 Git 仓库。请单独下载或从项目维护者处获取。

---

## 7. 配置 Docker 挂载

首次启动容器前，需要让 Docker 知道代码、数据和 checkpoint 分别挂载到哪里。

推荐直接写 `.env` 文件：

```bash
cat > .env <<EOF
PROJECT=alpharank
CODE_ROOT=$(pwd)
DATA_ROOT=/data/alpharank
LOG_ROOT=/data/alpharank_log
CONTAINER_HOME=$(pwd)/container_home
TARGET_HOME=/home/$(id -un)
EOF
```

如果你的数据在 `/msa/data`，checkpoint 在 `/msa/data/wukelin/log`，可以写成：

```bash
cat > .env <<EOF
PROJECT=alpharank
CODE_ROOT=$(pwd)
DATA_ROOT=/msa/data
LOG_ROOT=/msa/data/wukelin/log
CONTAINER_HOME=$(pwd)/container_home
TARGET_HOME=/home/$(id -un)
EOF
```

容器名会自动变成：

```text
<PROJECT>_<username>
```

例如：

```text
alpharank_wukelin
```

---

## 8. 构建并启动 Docker 容器

安装宿主机侧的轻量依赖：

```bash
pip install docker-compose sh
```

构建并启动容器：

```bash
python docker.py startd --build
```

构建完成后检查容器：

```bash
docker ps
```

进入容器：

```bash
python docker.py
```

退出容器：

```bash
exit
```

以后如果不需要重新 build，只启动容器即可：

```bash
python docker.py startd
```

---

## 9. 用仓库自带示例数据跑完整 Pipeline

仓库内置了最小测试输入：

```text
test_data/test_protein.fasta
test_data/test_ligands.smi
```

先从 `.env` 读取数据目录和日志目录：

```bash
source .env
```

确认以下文件/目录已经存在：

```bash
# MSA：这里检查的是容器内 /data 对应的宿主机 DATA_ROOT 下路径
ls $DATA_ROOT/rerank/protenix/chembl_bdb/openbind/msa

# checkpoint：这里检查的是容器内 /log 对应的宿主机 LOG_ROOT 下路径
ls $LOG_ROOT/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt
```

运行完整示例：

```bash
PROJECT_NAME=CloneTest GPU=0 bash test_data/test_run.sh
```

如果 GPU 0 忙，可以换成其他 GPU：

```bash
PROJECT_NAME=CloneTest GPU=3 bash test_data/test_run.sh
```

成功时会看到：

```text
✅ Pipeline运行完成！
✅ 完整测试运行成功！
```

结果文件：

```text
output/CloneTest/results/ev-a71_2a.csv
```

查看结果：

```bash
cat output/CloneTest/results/ev-a71_2a.csv
```

示例输出：

```csv
ligand,score
test_lig_002,-11.50997543334961
test_lig_001,-11.539664268493652
test_lig_003,-11.541532516479492
```

---

## 10. 用自己的蛋白和小分子运行

### 10.1 准备 FASTA

文件示例：`my_protein.fasta`

```fasta
>MyProtein
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK...
```

注意：`MyProtein` 会作为 target 名称，也会用于寻找对应 MSA。

### 10.2 准备 SMI

文件示例：`my_ligands.smi`

```text
ligand_001 CCO
ligand_002 c1ccccc1
ligand_003 CC(C)O
```

支持以下两种格式：

```text
ligand_name SMILES
SMILES ligand_name
```

### 10.3 准备 MSA

假设你的 `DATA_ROOT=/data/alpharank`，并希望容器内 MSA 路径为：

```text
/data/my_project/msa
```

那么宿主机上应放在：

```text
/data/alpharank/my_project/msa
```

示例：

```text
/data/alpharank/my_project/msa/MyProtein/MyProtein.a3m
```

### 10.4 运行命令

```bash
python pipeline/run_pipeline.py \
  --fasta /path/to/my_protein.fasta \
  --smi /path/to/my_ligands.smi \
  --msa_dir /data/my_project/msa \
  --project_name MyProject \
  --output_dir /data/my_project/output \
  --lmdb_dir /data/my_project/pair \
  --ckpt_path /log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt \
  --gpu 0 \
  --container alpharank_$(id -un) \
  --host_project_dir $(pwd) \
  --host_data_dir /data/alpharank \
  --mode full
```

参数解释：

| 参数 | 含义 |
|---|---|
| `--fasta` | 宿主机上的 FASTA 文件路径 |
| `--smi` | 宿主机上的 SMI 文件路径 |
| `--msa_dir` | 容器内 MSA 路径，必须以 `/data/...` 表示 |
| `--project_name` | 本次任务名称；输出会放入 `output/<project_name>/` |
| `--output_dir` | 容器内 co-folding 输出目录 |
| `--lmdb_dir` | 容器内 LMDB/JSON/Shell 中间目录 |
| `--ckpt_path` | 容器内 AlphaRank checkpoint 路径，通常以 `/log/...` 表示 |
| `--gpu` | 使用的 GPU 编号 |
| `--container` | Docker 容器名 |
| `--host_project_dir` | 宿主机上的仓库绝对路径，对应容器 `/project` |
| `--host_data_dir` | 宿主机上的数据根目录，对应容器 `/data` |
| `--mode` | `full` 表示从头到尾完整运行 |

最终结果：

```text
output/MyProject/results/<target_name>.csv
```

---

## 11. Pipeline 运行产物

一次完整运行后，本地仓库下会出现：

```text
output/<project_name>/
├── json/
│   ├── <project_name>.json
│   └── <project_name>_protein.json
├── scripts/
│   ├── <project_name>_pair.sh
│   └── <project_name>_protein.sh
└── results/
    └── <target_name>.csv
```

数据目录下会出现：

```text
<DATA_ROOT>/<your_project>/pair/
├── <project_name>.lmdb
├── <project_name>_named.lmdb
├── <project_name>_protein.lmdb
├── <project_name>.json
├── <project_name>_protein.json
├── <project_name>_pair.sh
└── <project_name>_protein.sh
```

---

## 12. 常见问题

### 12.1 Docker 构建失败：Conda Terms of Service

当前 Dockerfile 已自动接受必要 channel 的 ToS。如果仍遇到相关错误，请先拉取最新版：

```bash
git pull
python docker.py startd --build
```

### 12.2 `ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`

这是 PyTorch 2.3 与新版 MKL 的兼容问题。当前 Dockerfile 已固定 `mkl<2025` 和 `intel-openmp<2025`。如果遇到该问题，请重新 build：

```bash
python docker.py startd --build
```

### 12.3 `deepspeed` 与 `torch.library.custom_op` 报错

当前依赖已固定 `deepspeed==0.17.2` 以兼容 PyTorch 2.3.1。重新 build 即可：

```bash
python docker.py startd --build
```

### 12.4 找不到 MSA

检查三件事：

1. FASTA 中的蛋白名是否和 MSA 子目录名一致。
2. `--msa_dir` 是否是容器内路径，例如 `/data/my_project/msa`。
3. 宿主机上是否真的存在对应目录，例如 `$DATA_ROOT/my_project/msa`。

### 12.5 找不到 checkpoint

`--ckpt_path` 是容器内路径，通常以 `/log/...` 开头。它对应宿主机 `.env` 中的 `LOG_ROOT`。

例如：

```text
容器内: /log/train/.../ema_epoch40-49.ckpt
宿主机: $LOG_ROOT/train/.../ema_epoch40-49.ckpt
```

### 12.6 GPU OOM

换一块空闲 GPU：

```bash
PROJECT_NAME=CloneTest GPU=3 bash test_data/test_run.sh
```

或在自定义命令中修改：

```bash
--gpu 3
```

---

## 13. 已验证的新用户流程

本仓库已经验证过以下完整流程：

```text
git clone
  → 写入 .env
  → python docker.py startd --build
  → PROJECT_NAME=CloneTest GPU=3 bash test_data/test_run.sh
  → output/CloneTest/results/ev-a71_2a.csv
```

验证输出：

```csv
ligand,score
test_lig_002,-11.50997543334961
test_lig_001,-11.539664268493652
test_lig_003,-11.541532516479492
```

---

## 14. License

请参考仓库中的 LICENSE。
