#!/bin/bash

# ProtenixAffinity Pipeline 使用示例脚本
# 这个脚本展示了如何使用pipeline工具

# ========== 配置参数 ==========

# 项目配置
PROJECT_NAME="AlphaRank_Example"
GPU="0"
SEEDS="101"
MODE="full"  # full, generate_only, from_lmdb

# 输入文件（宿主机路径）
FASTA_FILE="/msa/data/wukelin/openbind/OpenBind_EV-A71_2A/EV-A71_2A.fasta"
SMI_FILE="/msa/data/wukelin/ligands.smi"

# MSA配置（单一运行容器内路径）
MSA_DIR="/data/rerank/protenix/chembl_bdb/openbind/msa"

# 单一运行容器输出路径（容器内路径，/data 对应宿主机 /msa/data）
OUTPUT_DIR="/data/rerank/protenix/chembl_bdb/alpharank/output"
LMDB_DIR="/data/rerank/protenix/chembl_bdb/alpharank/pair"

# 运行容器名称
CONTAINER="alpharank_hyper_decoy_wukelin"

# ========== 运行Pipeline ==========

echo "🚀 开始运行ProtenixAffinity Pipeline"
echo "项目名称: $PROJECT_NAME"
echo "运行模式: $MODE"
echo ""

python pipeline/run_pipeline.py \
    --fasta "$FASTA_FILE" \
    --smi "$SMI_FILE" \
    --msa_dir "$MSA_DIR" \
    --project_name "$PROJECT_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --lmdb_dir "$LMDB_DIR" \
    --gpu "$GPU" \
    --seeds "$SEEDS" \
    --mode "$MODE" \
    --container "$CONTAINER"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Pipeline执行完成！"
else
    echo ""
    echo "❌ Pipeline执行失败，退出码: $EXIT_CODE"
    exit $EXIT_CODE
fi
