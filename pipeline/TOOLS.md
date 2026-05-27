# 📋 ProtenixAffinity Pipeline - 工具清单

## 🎯 已创建的工具和文档

### 核心Pipeline工具

1. **generate_json.py** - JSON配置文件生成器
   - 从FASTA和SMI文件生成蛋白-小分子对的JSON
   - 自动生成蛋白-only的JSON
   - 位置: `pipeline/generate_json.py`

2. **generate_sh.py** - Shell脚本生成器
   - 自动生成protenix predict的执行脚本
   - 支持自定义GPU、seeds等参数
   - 位置: `pipeline/generate_sh.py`

3. **run_pipeline.py** - 主入口脚本 ⭐
   - 串联整个pipeline流程
   - 支持三种运行模式（full/generate_only/from_lmdb）
    - 在单个 `alpharank_hyper_decoy_wukelin` 容器内完成 Protenix predict、transform 和 inference
    - 自动同步辅助脚本，并在需要时把旧路径 MSA 复制到单容器 `/data` 挂载目录
   - 位置: `pipeline/run_pipeline.py`

4. **run_transform.py** - Transform包装器
   - 自动化LMDB名字转换
   - 动态修改transform.py配置
   - 位置: `pipeline/run_transform.py`

5. **check_env.py** - 环境验证工具
   - 验证FASTA和SMI文件格式
   - 检查Docker容器状态
   - 检查GPU和磁盘空间
   - 位置: `pipeline/check_env.py`

### 配置和示例

6. **example_run.sh** - 示例运行脚本
   - 完整的配置参数示例
   - 可直接修改使用
   - 位置: `pipeline/example_run.sh`

7. **config.example.yaml** - 配置文件模板
   - YAML格式的配置示例
   - 包含所有可配置参数
   - 位置: `pipeline/config.example.yaml`

### 文档

8. **README.md** - 完整文档
   - 详细的使用说明
   - 架构说明
   - 故障排查指南
   - 位置: `pipeline/README.md`

9. **QUICKSTART.md** - 快速开始指南
   - 5分钟上手指南
   - 步骤清晰简洁
   - 位置: `pipeline/QUICKSTART.md`

10. **TOOLS.md** - 本文档
    - 工具清单和使用示例
    - 位置: `pipeline/TOOLS.md`

## 🚀 快速使用指南

### 方式1: 使用示例脚本（推荐新手）

```bash
# 1. 复制示例脚本
cp pipeline/example_run.sh pipeline/my_run.sh

# 2. 编辑配置
nano pipeline/my_run.sh

# 3. 运行
bash pipeline/my_run.sh
```

### 方式2: 直接命令行（推荐熟练用户）

```bash
python pipeline/run_pipeline.py \
    --fasta /path/to/protein.fasta \
    --smi /path/to/ligands.smi \
    --msa_dir /data/msa_dir \
    --project_name MyProject \
    --output_dir /data/output \
    --lmdb_dir /data/lmdb \
    --gpu 0 \
    --mode full
```

> `--gpu` 会同时控制 Protenix predict 和 openbind inference；如果 GPU 0 已满，请改为空闲 GPU，例如 `--gpu 3`。

### 方式3: 分步执行（推荐调试）

```bash
# 步骤1: 验证环境
python pipeline/check_env.py \
    --fasta protein.fasta \
    --smi ligands.smi \
    --full

# 步骤2: 仅生成文件（检查配置）
python pipeline/run_pipeline.py \
    ... \
    --mode generate_only

# 步骤3: 执行完整流程
python pipeline/run_pipeline.py \
    ... \
    --mode full
```

## 📊 工具功能对比

| 工具 | 功能 | 适用场景 | 难度 |
|------|------|----------|------|
| run_pipeline.py | 完整流程 | 一键运行 | ⭐ |
| example_run.sh | 配置模板 | 快速上手 | ⭐ |
| generate_json.py | 生成JSON | 单独生成配置 | ⭐⭐ |
| generate_sh.py | 生成脚本 | 单独生成脚本 | ⭐⭐ |
| run_transform.py | LMDB转换 | 单独运行转换 | ⭐⭐ |
| check_env.py | 环境检查 | 故障排查 | ⭐ |

## 🔧 常用命令组合

### 检查环境是否就绪

```bash
python pipeline/check_env.py \
    --fasta /path/to/protein.fasta \
    --smi /path/to/ligands.smi \
    --check-docker \
    --check-gpu \
    --check-disk /data /msa/data
```

### 仅生成JSON文件

```bash
python pipeline/generate_json.py \
    --fasta protein.fasta \
    --smi ligands.smi \
    --msa_dir /data/msa \
    --output_dir ./output/json \
    --project_name MyProject
```

### 仅生成Shell脚本

```bash
python pipeline/generate_sh.py \
    --project_name MyProject \
    --json_dir /data/json \
    --output_dir /data/output \
    --lmdb_dir /data/lmdb \
    --script_dir ./output/scripts \
    --gpu 0
```

### 运行完整Pipeline

```bash
python pipeline/run_pipeline.py \
    --fasta protein.fasta \
    --smi ligands.smi \
    --msa_dir /data/msa \
    --project_name MyProject \
    --output_dir /data/output \
    --lmdb_dir /data/lmdb \
    --gpu 0 \
    --mode full
```

## 📁 输出文件结构

运行完成后的输出结构：

```
output/
└── <project_name>/
    ├── json/
    │   ├── <project_name>.json                # 蛋白-小分子对JSON
    │   └── <project_name>_protein.json        # 蛋白-only JSON
    └── scripts/
        ├── <project_name>_pair.sh             # 蛋白-小分子对脚本
        └── <project_name>_protein.sh          # 蛋白-only脚本

单一运行容器LMDB目录/
├── <project_name>.lmdb/                       # 原始LMDB
├── <project_name>_named.lmdb/                 # 添加名字后的LMDB
└── <project_name>_protein.lmdb/               # 从pair embedding自动抽取的蛋白LMDB

output/<project_name>/results/
└── <target_name>.csv                          # 实际CSV格式：ligand,score
```

## 🎓 使用建议

### 新手使用流程

1. 阅读 `QUICKSTART.md`
2. 运行 `check_env.py` 验证环境
3. 复制并修改 `example_run.sh`
4. 使用 `--mode generate_only` 测试配置
5. 使用 `--mode full` 运行完整流程

### 进阶使用流程

1. 直接使用 `run_pipeline.py` 命令行
2. 根据需要单独运行各个工具
3. 自定义修改脚本和配置

### 调试流程

1. 使用 `check_env.py` 检查环境
2. 使用 `--mode generate_only` 生成文件检查
3. 手动运行各个步骤，观察日志
4. 检查中间文件（JSON、LMDB等）

## 💡 提示和技巧

1. **路径管理**: 始终使用绝对路径，避免混淆
2. **日志保存**: 使用 `tee` 保存运行日志
   ```bash
   bash pipeline/my_run.sh 2>&1 | tee logs/$(date +%Y%m%d_%H%M%S).log
   ```
3. **小数据测试**: 先用少量数据测试流程
4. **资源监控**: 运行前检查GPU和磁盘空间
5. **容器路径**: 注意区分宿主机路径和容器内路径
6. **MSA路径**: 单容器 `/data` 对应宿主机 `/msa/data`；旧 `/data/protein/...` 下的 MSA 会被自动复制

## 🆘 获取帮助

每个工具都支持 `--help` 参数：

```bash
python pipeline/run_pipeline.py --help
python pipeline/generate_json.py --help
python pipeline/check_env.py --help
```

## 📚 相关文档

- [完整文档](README.md) - 详细的使用说明和架构
- [快速开始](QUICKSTART.md) - 5分钟上手指南
- [配置示例](config.example.yaml) - 配置文件模板
- [运行示例](example_run.sh) - Shell脚本示例

## 🔄 更新记录

- 2026-05-27: 改为单容器运行模式
    - 全流程统一在 `alpharank_hyper_decoy_wukelin` 中执行
    - `--gpu` 同时控制 Protenix predict 和 openbind inference
    - protein LMDB 从 pair embedding 自动抽取
    - 支持从旧 `/data/protein/...` 自动迁移 MSA 到 `/msa/data/...`
- 2024-05-26: 创建完整的Pipeline工具集
  - 核心工具：5个Python脚本
  - 配置文件：2个示例文件
  - 文档：3个Markdown文档
