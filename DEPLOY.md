# ProtenixAffinity — 部署指南

本文档面向**从零开始部署**的用户：克隆仓库后，按此流程安装 Docker 环境并运行 Pipeline。

---

## 1. 系统要求

| 要求 | 最低配置 |
|------|---------|
| OS | Linux (Ubuntu 20.04+ 推荐) |
| GPU | NVIDIA GPU，≥ 16 GB 显存 |
| CUDA | 12.1 |
| Docker | 20.10+ |
| NVIDIA Container Toolkit | 最新版 |
| 磁盘空间 | ≥ 100 GB（数据 + 模型权重） |

---

## 2. 安装 Docker 与 NVIDIA Container Toolkit

```bash
# 安装 Docker（已安装可跳过）
curl -fsSL https://get.docker.com | bash
sudo usermod -aG docker $USER
newgrp docker

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

---

## 3. 克隆仓库

```bash
git clone https://github.com/THU-ATOM/ProtenixAffinity.git
cd ProtenixAffinity
```

---

## 4. 构建 Docker 镜像

镜像基于 `docker/Dockerfile`，包含 PyTorch 2.3、hhsuite、mmseqs2 及所有 Python 依赖。

```bash
# 安装构建工具（仅首次）
pip install docker-compose sh

# 初始化并构建镜像（首次约需 20-40 分钟，视网络速度而定）
python docker.py startd --build
```

交互式提示：

```
Give a project name [mnist]: ProtenixAffinity
Code root to be mounted at /project [.]: /path/to/ProtenixAffinity
Data root to be mounted at /data [data]: /path/to/your/data
Log root to be mounted at /log [log]: /path/to/your/log
```

**路径说明：**
- `Code root` → 映射到容器 `/project`（填写本仓库的绝对路径）
- `Data root` → 映射到容器 `/data`（放置 MSA 文件、checkpoint、中间数据）
- `Log root` → 映射到容器 `/log`（放置训练 checkpoint）

配置会保存到 `.env` 文件，下次 `python docker.py startd` 直接启动，无需重新配置。

---

## 5. 准备权重文件

将模型 checkpoint 放到 `Log root`（即容器 `/log`）下，路径示例：

```
/your/log/train/alpharank_hyper_decoy_new/
  DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/
    checkpoints/
      ema_epoch40-49.ckpt   ← 指定此文件
```

> 权重文件从项目托管处下载，或联系项目维护者获取。

---

## 6. 准备 MSA 文件

MSA（Multiple Sequence Alignment）是蛋白结构预测的必须输入，需预先生成：

```
/your/data/rerank/protenix/chembl_bdb/openbind/msa/
  <ProteinName>/          ← 蛋白名（与 FASTA 中 > 后的名字一致）
    <ProteinName>.a3m     ← MSA 文件
```

生成 MSA 的方法可参考 [ColabFold MSA](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/MSA.ipynb)。

---

## 7. 运行 Pipeline

### 快速测试（3个小分子）

```bash
# 确认容器在运行
docker ps | grep ProtenixAffinity

# 运行测试
python pipeline/run_pipeline.py \
    --fasta  test_data/test_protein.fasta \
    --smi    test_data/test_ligands.smi \
    --msa_dir /data/rerank/protenix/chembl_bdb/openbind/msa \
    --project_name  TestRun \
    --output_dir /data/alpharank/output \
    --lmdb_dir   /data/alpharank/pair \
    --ckpt_path  /log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt \
    --gpu    0 \
    --container  <你的容器名称> \
    --host_project_dir  /path/to/ProtenixAffinity \
    --host_data_dir     /path/to/your/data \
    --mode   full
```

**关键参数：**

| 参数 | 说明 |
|------|------|
| `--container` | Docker 容器名，默认 `ProtenixAffinity_<用户名>` |
| `--host_project_dir` | 宿主机上本仓库的绝对路径（映射到容器 `/project`） |
| `--host_data_dir` | 宿主机上数据目录（映射到容器 `/data`） |
| `--ckpt_path` | 容器内 checkpoint 路径（以 `/log/` 开头） |
| `--msa_dir` | 容器内 MSA 目录路径（以 `/data/` 开头） |
| `--gpu` | 使用哪块 GPU（0-indexed） |

### 查看容器名称

```bash
docker ps --format '{{.Names}}'
```

---

## 8. 查看结果

Pipeline 完成后，结果 CSV 在：

```
output/<project_name>/results/<target_name>.csv
```

```csv
ligand,score
compound_001,-11.51
compound_002,-11.54
```

**score 是负的双曲测地距离，通常数值越大（越接近 0）预测亲和力越强。**

---

## 9. 故障排查

**`Cannot connect to Docker daemon`**
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER && newgrp docker
```

**`nvidia-smi: command not found` 或 GPU 不可用**
```bash
nvidia-smi           # 验证驱动
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

**`OOM (out of memory)` 错误**
- 换用空闲显存更大的 GPU：`--gpu 1`（或其他编号）
- 检查：`nvidia-smi`

**MSA 路径找不到**
- `--msa_dir` 必须是容器内路径（`/data/...`），对应宿主机 `--host_data_dir/...`
- 检查：`docker exec <容器名> ls /data/rerank/protenix/...`

---

## 10. 目录结构参考

```
ProtenixAffinity/
├── docker/
│   ├── Dockerfile          ← 容器构建文件
│   └── requirements.txt    ← Python 依赖
├── docker-compose.yml      ← Docker Compose 配置
├── docker.py               ← 容器管理脚本
├── pipeline/
│   ├── run_pipeline.py     ← 主入口（单容器模式）
│   ├── QUICKSTART.md       ← 快速上手
│   └── README.md           ← 技术参考
├── test_data/
│   ├── test_protein.fasta  ← 测试蛋白序列
│   └── test_ligands.smi    ← 测试小分子
└── DEPLOY.md               ← 本文档
```
