#!/bin/bash

# ============================================================
# ProtenixAffinity Pipeline - 完整使用示例
# ============================================================
# 
# 这个脚本展示了如何使用整合后的Pipeline工具
# 从输入文件（FASTA + SMI）到最终CSV结果的完整流程
#
# 作者: Pipeline工具集
# 日期: 2024-05-26
# ============================================================

set -e  # 遇到错误立即退出

# ============================================================
# 步骤0: 配置参数
# ============================================================

echo "========================================"
echo "ProtenixAffinity Pipeline"
echo "========================================"
echo ""

# 项目基本配置
PROJECT_NAME="MyProject"
GPU="0"
SEEDS="101"
MODE="full"  # full | generate_only | from_lmdb

# 输入文件（宿主机路径）
FASTA_FILE="/msa/data/wukelin/openbind/OpenBind_EV-A71_2A/EV-A71_2A.fasta"
SMI_FILE="/msa/data/wukelin/ligands.smi"

# MSA配置（单一运行容器内路径）
MSA_DIR="/data/rerank/protenix/chembl_bdb/openbind/msa"

# 单一运行容器输出路径（容器内路径，/data 对应宿主机 /msa/data）
OUTPUT_DIR="/data/rerank/protenix/chembl_bdb/${PROJECT_NAME}/output"
LMDB_DIR="/data/rerank/protenix/chembl_bdb/${PROJECT_NAME}/pair"

# 运行容器名称
CONTAINER="alpharank_hyper_decoy_wukelin"

# ============================================================
# 步骤1: 环境检查（推荐）
# ============================================================

echo ""
echo "步骤1: 环境检查"
echo "----------------------------------------"

python pipeline/check_env.py \
    --fasta "$FASTA_FILE" \
    --smi "$SMI_FILE" \
    --check-docker \
    --check-gpu

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ 环境检查未通过，请修复后继续"
    exit 1
fi

# ============================================================
# 步骤2: 运行Pipeline
# ============================================================

echo ""
echo "步骤2: 运行Pipeline"
echo "----------------------------------------"

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

# ============================================================
# 步骤3: 结果汇总
# ============================================================

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline执行完成！"
    echo ""
    echo "输出文件位置："
    echo "  - JSON配置: output/${PROJECT_NAME}/json/"
    echo "  - Shell脚本: output/${PROJECT_NAME}/scripts/"
    echo "  - LMDB数据: ${LMDB_DIR}/"
    echo ""
    echo "下一步："
    echo "  1. 检查生成的LMDB文件"
    echo "  2. 查看最终的CSV结果"
else
    echo "❌ Pipeline执行失败"
    echo "退出码: $EXIT_CODE"
    echo ""
    echo "故障排查："
    echo "  1. 检查日志输出中的错误信息"
    echo "  2. 运行环境检查: python pipeline/check_env.py --full"
    echo "  3. 查看文档: pipeline/README.md"
fi
echo "========================================"

exit $EXIT_CODE
