╔══════════════════════════════════════════════════════════════════════╗
║          ProtenixAffinity Pipeline 工具集创建完成！                  ║
╚══════════════════════════════════════════════════════════════════════╝

✨ 已创建的文件清单：

📁 pipeline/
├── 🔧 核心工具 (5个Python脚本)
│   ├── generate_json.py      - 从FASTA+SMI生成JSON配置
│   ├── generate_sh.py         - 生成Shell运行脚本
│   ├── run_pipeline.py        - 主入口，串联完整流程 ⭐
│   ├── run_transform.py       - LMDB名字转换包装器
│   └── check_env.py           - 环境验证工具
│
├── 📝 配置和示例 (3个文件)
│   ├── example_run.sh         - 基础运行示例（可执行）
│   ├── complete_example.sh    - 完整运行示例（可执行）
│   └── config.example.yaml    - YAML配置模板
│
└── 📚 文档 (4个Markdown)
    ├── README.md              - 完整使用文档
    ├── QUICKSTART.md          - 5分钟快速开始
    ├── TOOLS.md               - 工具清单和对比
    └── SUMMARY.md             - 本总结文档

═══════════════════════════════════════════════════════════════════════

🚀 快速开始（三种方式）：

┌─ 方式1: 新手推荐 ─────────────────────────────────────────┐
│ 1. 复制示例脚本                                             │
│    cp pipeline/example_run.sh pipeline/my_run.sh           │
│                                                             │
│ 2. 编辑配置                                                 │
│    nano pipeline/my_run.sh                                 │
│                                                             │
│ 3. 运行                                                     │
│    bash pipeline/my_run.sh                                 │
└─────────────────────────────────────────────────────────────┘

┌─ 方式2: 命令行直接运行 ───────────────────────────────────┐
│ python pipeline/run_pipeline.py \                          │
│     --fasta /path/to/protein.fasta \                       │
│     --smi /path/to/ligands.smi \                           │
│     --msa_dir /data/msa_dir \                              │
│     --project_name MyProject \                             │
│     --output_dir /data/output \                            │
│     --lmdb_dir /data/lmdb \                                │
│     --gpu 0 \                                              │
│     --mode full                                            │
└─────────────────────────────────────────────────────────────┘

┌─ 方式3: 完整示例（带环境检查） ───────────────────────────┐
│ bash pipeline/complete_example.sh                          │
└─────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════

🎯 Pipeline流程说明：

输入                             输出
─────────────────────────────────────────────────────
protein.fasta  ┐
               ├─→ [生成JSON] ──→ project.json
ligands.smi    ┘                  project_protein.json
                                          │
                                          ↓
                                  [生成Shell脚本]
                                          │
                                          ↓
                              ┌───[单容器: protenix predict]──┐
                              │   生成pair embedding LMDB      │
                              │   - project.lmdb               │
                              └────────────┬───────────────────┘
                                          ↓
                                  [Transform: 添加名字]
                              project_named.lmdb
                              project_protein.lmdb
                                          │
                                          ↓
                                 ┌───[单容器: Inference]───┐
                                 │   计算亲和力评分        │
                                 │   → results/<target>.csv│
                                 └─────────────────────────┘

═══════════════════════════════════════════════════════════════════════

📋 主要参数说明：

必需参数：
   --fasta          蛋白质FASTA文件路径（宿主机）
   --smi            小分子SMI文件路径（宿主机）
   --msa_dir        MSA目录（单一运行容器内路径）
   --project_name   项目名称
   --output_dir     单一运行容器输出目录
   --lmdb_dir       单一运行容器 LMDB目录

可选参数：
   --gpu            GPU编号（默认：0）
   --seeds          随机种子（默认：101）
   --mode           运行模式：full | generate_only | from_lmdb
   --container      单一运行容器名称（默认：alpharank_hyper_decoy_wukelin）

═══════════════════════════════════════════════════════════════════════

🔍 输入文件格式要求：

FASTA文件示例 (protein.fasta):
  >ProteinName
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEK...

SMI文件示例 (ligands.smi):
  ligand_001 CCO
  ligand_002 c1ccccc1
  或
  CCO ligand_001
  c1ccccc1 ligand_002

═══════════════════════════════════════════════════════════════════════

🛠️ 常用命令：

环境检查：
  python pipeline/check_env.py \
      --fasta protein.fasta \
      --smi ligands.smi \
      --full

仅生成JSON（验证配置）：
  python pipeline/generate_json.py \
      --fasta protein.fasta \
      --smi ligands.smi \
      --msa_dir /data/msa \
      --output_dir ./output/json \
      --project_name MyProject

完整运行：
  python pipeline/run_pipeline.py \
      [参数...] \
      --mode full

从已有LMDB继续：
  python pipeline/run_pipeline.py \
      [参数...] \
      --mode from_lmdb

═══════════════════════════════════════════════════════════════════════

📖 文档导航：

1. 快速上手（5分钟）
   → 阅读: pipeline/QUICKSTART.md

2. 完整文档（详细说明）
   → 阅读: pipeline/README.md

3. 工具清单（工具对比）
   → 阅读: pipeline/TOOLS.md

4. 获取帮助
   → 运行: python pipeline/run_pipeline.py --help

═══════════════════════════════════════════════════════════════════════

✅ 最佳实践：

1. 首次使用：
   ✓ 先阅读 QUICKSTART.md
   ✓ 运行 check_env.py 验证环境
   ✓ 使用 --mode generate_only 测试配置
   ✓ 用小数据集测试完整流程

2. 日常使用：
   ✓ 使用 example_run.sh 作为模板
   ✓ 使用绝对路径避免混淆
   ✓ 运行前检查空闲 GPU，并用 --gpu 指定
   ✓ 保存运行日志：... 2>&1 | tee logs/run.log

3. 故障排查：
   ✓ 检查 check_env.py 输出
   ✓ 查看容器日志：docker logs <container_name>
   ✓ 查看 GPU 状态：nvidia-smi
   ✓ 阅读 README.md 的故障排查章节

═══════════════════════════════════════════════════════════════════════

🎉 恭喜！Pipeline工具集已完全集成到代码库中

现在你和其他用户可以：
  ✓ 一键运行完整的蛋白-小分子亲和力预测流程
  ✓ 使用标准化的配置和参数
   ✓ 在单个 alpharank_hyper_decoy_wukelin 容器内完成全流程
   ✓ 自动同步辅助脚本并迁移旧路径 MSA
  ✓ 获得清晰的文档和示例支持

═══════════════════════════════════════════════════════════════════════

📧 需要帮助？
   → 查看文档：pipeline/README.md
   → 运行帮助：python pipeline/run_pipeline.py --help
   → 环境检查：python pipeline/check_env.py --full

祝使用愉快！🚀
