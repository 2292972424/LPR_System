#!/bin/bash
# ============================================================
# 数据准备脚本：从原始 CCPD 数据生成训练/验证集
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  LPR_System — 数据准备"
echo "============================================"

# ------ 检测数据集 ------
RAW_DIR="$PROJECT_ROOT/data/raw"
PROCESSED_DIR="$PROJECT_ROOT/data/processed"

echo "[INFO] 原始数据目录: $RAW_DIR"
echo "[INFO] 处理后目录: $PROCESSED_DIR"

# 检查是否有原始数据
if [ -d "$RAW_DIR/ccpd" ]; then
    echo "[OK] 检测到 CCPD 数据集"
elif [ -d "$RAW_DIR/CCPD2020" ]; then
    echo "[OK] 检测到 CCPD2020 数据集"
    ln -sf "$RAW_DIR/CCPD2020" "$RAW_DIR/ccpd" 2>/dev/null
else
    echo "============================================"
    echo "  数据集未找到！"
    echo "============================================"
    echo ""
    echo "请按以下步骤准备数据:"
    echo ""
    echo "方式一 (CCPD公开数据集):"
    echo "  1. 下载CCPD: https://github.com/detectRecog/CCPD"
    echo "  2. 解压到: $RAW_DIR/ccpd/"
    echo "  3. 重新运行此脚本"
    echo ""
    echo "方式二 (自定义数据集):"
    echo "  1. 将图片放入: $RAW_DIR/images/"
    echo "  2. 在 $RAW_DIR/ 下创建标注文件 train.txt 和 val.txt"
    echo "     格式: 图片名.jpg 车牌号"
    echo "  3. 重新运行此脚本"
    echo ""
    exit 1
fi

# ------ 运行预处理 ------
echo "[INFO] 生成训练数据..."
python "$PROJECT_ROOT/prepare_lpr_dataset.py" \
    --ccpd_dir "$RAW_DIR/ccpd" \
    --output_dir "$PROCESSED_DIR" \
    --val_split 0.2

echo ""
echo "[OK] 数据准备完成！"
echo "  训练集: $PROCESSED_DIR/train/"
echo "  验证集: $PROCESSED_DIR/val/"
echo "  标注:   $PROCESSED_DIR/train.txt / val.txt"
