# Pipeline工具快速索引

## 🎯 我想...

### 快速开始
- **第一次使用** → [QUICKSTART.md](QUICKSTART.md)
- **了解所有功能** → [README.md](README.md)
- **查看工具清单** → [TOOLS.md](TOOLS.md)
- **查看总结** → [SUMMARY.md](SUMMARY.md)

## 当前运行架构

- 全流程统一在单个容器 `alpharank_hyper_decoy_wukelin` 中执行
- 主控脚本从宿主仓库 `/home/wukelin/ProtenixAffinity` 启动
- 容器 `/project` 对应宿主机 `/home/wukelin/ProtenixAffinity2`
- 容器 `/data` 对应宿主机 `/msa/data`
- 如果 MSA 仍在旧宿主路径 `/data/protein/...`，pipeline 会自动复制到 `/msa/data/...`
- 输出 CSV 实际格式为 `ligand,score`

### 运行Pipeline
- **使用示例脚本** → `bash example_run.sh` 或 `bash complete_example.sh`
- **命令行运行** → `python run_pipeline.py --help`
- **验证环境** → `python check_env.py --full`

### 单独使用某个工具
- **生成JSON** → `python generate_json.py --help`
- **生成Shell脚本** → `python generate_sh.py --help`
- **转换LMDB** → `python run_transform.py --help`
- **检查环境** → `python check_env.py --help`

### 遇到问题
- **故障排查** → [README.md#故障排查](README.md)
- **查看所有命令** → [TOOLS.md#常用命令组合](TOOLS.md)
- **检查输入格式** → `python check_env.py --fasta xxx --smi xxx`

## 📂 文件导航

### 核心工具（按使用频率）
1. `run_pipeline.py` ⭐ - 主入口，一键运行
2. `check_env.py` - 环境验证
3. `generate_json.py` - JSON生成
4. `generate_sh.py` - Shell脚本生成
5. `run_transform.py` - LMDB转换

### 配置和示例
1. `example_run.sh` - 基础示例（推荐新手）
2. `complete_example.sh` - 完整示例（带环境检查）
3. `config.example.yaml` - 配置模板

### 文档（按阅读顺序）
1. `QUICKSTART.md` ← 从这里开始
2. `README.md` ← 完整文档
3. `TOOLS.md` ← 工具详解
4. `SUMMARY.md` ← 总结
5. `INDEX.md` ← 本文件

## 🔗 快速链接

### 最常用的命令

```bash
# 1. 环境检查
python pipeline/check_env.py --fasta xxx.fasta --smi xxx.smi --full

# 2. 完整运行（推荐）
python pipeline/run_pipeline.py \
    --fasta xxx.fasta \
    --smi xxx.smi \
    --msa_dir /data/msa \
    --project_name MyProject \
    --output_dir /data/output \
    --lmdb_dir /data/lmdb \
    --gpu 0 \
    --mode full

# 如果 GPU 0 已满，可以改用空闲 GPU，例如 --gpu 3

# 3. 仅生成配置（验证）
python pipeline/run_pipeline.py ... --mode generate_only
```

## 📞 需要帮助？

1. 查看相应的文档
2. 运行 `--help` 查看工具帮助
3. 查看 README.md 的故障排查部分
