#!/bin/bash
# ============================================================
# 一键推理：车牌检测+识别
# 用法: bash scripts/run_demo.sh --input test.jpg [--output result.jpg] [--margin 0.05]
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
INPUT=""
OUTPUT="result.jpg"
MARGIN=0.05
CONF=0.25
YOLO_WEIGHT="$PROJECT_ROOT/models/weights/best.pt"
LPRNET_WEIGHT="$PROJECT_ROOT/models/weights/lprnet_best.pth"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)   INPUT="$2"; shift 2 ;;
        --output)  OUTPUT="$2"; shift 2 ;;
        --margin)  MARGIN="$2"; shift 2 ;;
        --conf)    CONF="$2"; shift 2 ;;
        --yolo)    YOLO_WEIGHT="$2"; shift 2 ;;
        --lprnet)  LPRNET_WEIGHT="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT" ]; then
    echo "用法: bash scripts/run_demo.sh --input <图片路径> [--output result.jpg]"
    echo ""
    echo "可选参数:"
    echo "  --margin  0.05   检测框扩边比例"
    echo "  --conf    0.25   检测置信度阈值"
    echo "  --yolo    PATH   YOLO 权重路径"
    echo "  --lprnet  PATH   LPRNet 权重路径"
    exit 1
fi

cd "$PROJECT_ROOT"

echo "============================================"
echo "  LPR_System — 车牌识别推理"
echo "============================================"
echo "  输入:  $INPUT"
echo "  输出:  $OUTPUT"
echo "============================================"

python demo.py image \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --yolo-weight "$YOLO_WEIGHT" \
    --lprnet-weight "$LPRNET_WEIGHT" \
    --margin "$MARGIN" \
    --conf "$CONF" \
    --no-display

echo ""
echo "[OK] 结果已保存: $OUTPUT"
