#!/bin/bash

# 测试运行脚本
# 使用最小数据集验证Pipeline

PROJECT_NAME="TestRun"
GPU="3"

# 测试数据（根据实际情况修改）
FASTA_FILE="/home/wukelin/ProtenixAffinity/test_data/test_protein.fasta"
SMI_FILE="/home/wukelin/ProtenixAffinity/test_data/test_ligands.smi"

# MSA目录（单一运行容器 alpharank_hyper_decoy_wukelin 内路径）
MSA_DIR="/data/rerank/protenix/chembl_bdb/openbind/msa"

# 输出目录和LMDB目录（单一运行容器内路径，/data 对应宿主机 /msa/data）
OUTPUT_DIR="/data/rerank/protenix/chembl_bdb/test_run/output"
LMDB_DIR="/data/rerank/protenix/chembl_bdb/test_run/pair"

# checkpoint路径（单一运行容器内路径）
CKPT_PATH="/log/train/alpharank_hyper_decoy_new/DataModule.HYPMLPPairRanker.ListListRankCriterion.2026-01-07_10-59-36/checkpoints/ema_epoch40-49.ckpt"

echo "============================================"
echo "Pipeline 完整测试运行"
echo "============================================"
echo ""
echo "📊 测试配置："
echo "   - 蛋白: EV-A71_2A (已有MSA)"
echo "   - 小分子: 3个测试分子"
echo "   - GPU: $GPU"
echo "   - 项目: $PROJECT_NAME"
echo ""
echo "⏱️  预计运行时间: 约10-30分钟"
echo ""

# 步骤1: 运行完整流程
echo "开始运行完整流程..."
python pipeline/run_pipeline.py \
    --fasta "$FASTA_FILE" \
    --smi "$SMI_FILE" \
    --msa_dir "$MSA_DIR" \
    --project_name "$PROJECT_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --lmdb_dir "$LMDB_DIR" \
    --ckpt_path "$CKPT_PATH" \
    --gpu "$GPU" \
    --mode full

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 完整测试运行成功！"
    echo "   - JSON配置: output/$PROJECT_NAME/json/"
    echo "   - Shell脚本: output/$PROJECT_NAME/scripts/"
    echo "   - LMDB数据: $LMDB_DIR"
    echo ""
    echo "请查看输出日志了解详细进度"
else
    echo ""
    echo "❌ 测试失败，请检查错误信息"
fi
