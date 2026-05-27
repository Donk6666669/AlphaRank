# AlphaRank

AlphaRank 是一个面向 **protein–ligand binding affinity ranking** 的双曲空间排序模型。

它使用 co-folding 模型提取蛋白和小分子的表示，再通过可学习投影映射到 Lorentz 双曲空间中，并以**负的双曲测地距离**作为亲和力排序分数。相比普通欧氏表示，双曲空间的切向放大效应更适合区分结构极其相似、但活性差异显著的 hard inactives。

当前仓库提供从 FASTA + SMI + 预计算 MSA 到最终 per-target CSV 排序结果的一键流程。

---

## 核心功能

- 输入蛋白 FASTA 与小分子 SMI 文件
- 自动生成 Protenix/co-folding 推理 JSON 与 Shell 脚本
- 在单个 Docker 容器中生成 protein–ligand pair embedding LMDB
- 自动转换 LMDB 名称并抽取 protein-only LMDB
- 加载 AlphaRank checkpoint 进行 binding affinity ranking
- 输出每个 target 一个 CSV：`ligand,score`

---

## 当前运行架构

AlphaRank Pipeline 当前为**单容器模式**：

- Docker 镜像构建文件：`docker/Dockerfile`
- 容器内代码路径：`/project`
- 容器内数据路径：`/data`
- 容器内日志/权重路径：`/log`

Pipeline 入口：

```bash
python pipeline/run_pipeline.py ...
```

详细部署步骤见 [DEPLOY.md](DEPLOY.md)。

快速上手见 [pipeline/QUICKSTART.md](pipeline/QUICKSTART.md)。

完整参数说明见 [pipeline/README.md](pipeline/README.md)。

---

## 快速测试

1. 构建并启动 Docker 容器：

```bash
pip install docker-compose sh
python docker.py startd --build
```

2. 准备 checkpoint 与 MSA 数据。

3. 运行测试数据：

```bash
bash test_data/test_run.sh
```

输出示例：

```csv
ligand,score
test_lig_002,-11.511119
test_lig_001,-11.539435
test_lig_003,-11.542738
```

`score` 是负的双曲测地距离，通常**数值越大（越接近 0）表示预测亲和力越强**。

---

## 目录结构

```text
AlphaRank/
├── docker/                 # Dockerfile 与容器依赖
├── docker-compose.yml      # 容器编排配置
├── docker.py               # 容器管理脚本
├── pipeline/               # 一键推理 Pipeline
├── protenix/               # co-folding 表示提取与 AlphaRank 模型代码
├── runner/                 # Protenix/co-folding 推理入口
├── configs/                # 推理配置
├── conf/                   # AlphaRank/训练相关配置（保留用于模型兼容）
├── test_data/              # 最小测试输入
├── test_scripts/           # LMDB 转换与 AlphaRank inference 脚本
└── DEPLOY.md               # 从零部署指南
```

---

## 输入与输出

### 输入

- FASTA：蛋白序列
- SMI：小分子名称与 SMILES
- MSA：预计算多序列比对目录
- checkpoint：AlphaRank 模型权重

### 输出

```text
output/<project_name>/results/<target_name>.csv
```

CSV 格式：

```csv
ligand,score
compound_001,-10.42
compound_002,-11.03
```

---

## 文档

- [DEPLOY.md](DEPLOY.md)：Docker 环境、镜像构建、数据与 checkpoint 准备
- [pipeline/QUICKSTART.md](pipeline/QUICKSTART.md)：最快运行方式
- [pipeline/README.md](pipeline/README.md)：Pipeline 参数与分步流程
- [pipeline/TOOLS.md](pipeline/TOOLS.md)：工具脚本说明

---

## License

请参考 [LICENSE](LICENSE)。
