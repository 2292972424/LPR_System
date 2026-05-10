#!/bin/bash
# ============================================================
# 一键训练：YOLO检测 + LPRNet识别
# 用法: bash scripts/train_all.sh [--device cuda] [--epochs 100]
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEVICE="cuda"
EPOCHS=100
BATCH_SIZE=128

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --device) DEVICE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch)  BATCH_SIZE="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "  LPR_System — 一键训练"
echo "============================================"
echo "  设备:      $DEVICE"
echo "  轮数:      $EPOCHS"
echo "  批次大小:  $BATCH_SIZE"
echo "============================================"
echo ""

cd "$PROJECT_ROOT"

# ------ 训练 LPRNet 识别模型 ------
echo "[Step 1/2] 训练 LPRNet 车牌识别模型..."
python train_lprnet.py \
    --data_dir "$PROJECT_ROOT/data/processed" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr 0.001 \
    --warmup_epochs 5 \
    --device "$DEVICE" \
    --output_dir "$PROJECT_ROOT/models/weights"

echo ""
echo "[Step 2/2] 训练 YOLO 检测模型..."
if [ -f "$PROJECT_ROOT/data/raw/ccpd/data.yaml" ]; then
    python train_yolo.py \
        --data "$PROJECT_ROOT/data/raw/ccpd/data.yaml" \
        --epochs "$EPOCHS" \
        --batch $((BATCH_SIZE / 4)) \
        --device "${DEVICE#cuda:}"
else
    echo "[WARN] YOLO 配置文件不存在，跳过检测模型训练"
    echo "  请创建 data/raw/ccpd/data.yaml 后手动运行:"
    echo "  python train_yolo.py --data data/raw/ccpd/data.yaml --epochs $EPOCHS"
fi

echo ""
echo "============================================"
echo "  训练完成！"
echo "============================================"
echo ""
echo "模型权重:"
ls -lh "$PROJECT_ROOT/models/weights/"*.pth 2>/dev/null
ls -lh "$PROJECT_ROOT/models/weights/"*.pt 2>/dev/null
echo ""
echo "下一步: bash scripts/run_demo.sh --input test.jpg"
