#!/bin/bash
# ============================================================
# 一键创建 LPR_System 虚拟环境
# 用法: bash scripts/setup_env.sh
# ============================================================
set -e

ENV_NAME="lpr"
PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  LPR_System — 环境安装脚本"
echo "============================================"
echo ""

# ------ 1. 检测 conda ------
if command -v conda &> /dev/null; then
    CONDA_AVAILABLE=true
    echo "[OK] 检测到 conda"
else
    CONDA_AVAILABLE=false
    echo "[WARN] 未检测到 conda，将使用 venv"
fi

# ------ 2. 创建环境 ------
if $CONDA_AVAILABLE; then
    echo "[INFO] 创建 conda 环境: $ENV_NAME (Python $PYTHON_VERSION)"
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo "[OK] conda 环境创建完成"

    # 激活环境（在脚本中）
    eval "$(conda shell.bash hook)"
    conda activate "$ENV_NAME"

    # 安装 PyTorch (CUDA 12.1)
    echo "[INFO] 安装 PyTorch (CUDA 12.1)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

else
    echo "[INFO] 创建 venv 环境..."
    python3 -m venv "$ENV_NAME"
    source "$ENV_NAME/bin/activate"

    # 安装 PyTorch CPU 版
    echo "[INFO] 安装 PyTorch (CPU)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# ------ 3. 安装依赖 ------
echo "[INFO] 安装项目依赖..."
pip install -r "$PROJECT_ROOT/requirements.txt"

# ------ 4. 安装中文字体 ------
echo "[INFO] 安装中文字体..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq && sudo apt-get install -y -qq fonts-noto-cjk 2>/dev/null || \
    sudo apt-get install -y -qq fonts-wqy-zenhei 2>/dev/null || \
    echo "[WARN] 中文字体安装失败，请手动安装: apt install fonts-noto-cjk"
elif command -v yum &> /dev/null; then
    sudo yum install -y wqy-zenhei-fonts 2>/dev/null || echo "[WARN] 中文字体安装失败"
elif command -v brew &> /dev/null; then
    brew install font-noto-sans-cjk 2>/dev/null || echo "[WARN] 中文字体安装失败"
fi

# ------ 5. 验证 ------
echo ""
echo "[INFO] 环境验证..."
python -c "
import torch
import cv2
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA 可用: {torch.cuda.is_available()}')
print(f'  OpenCV: {cv2.__version__}')
" || echo "[WARN] 部分库导入失败"

echo ""
echo "============================================"
echo "  环境安装完成！"
echo "============================================"
echo ""
echo "激活环境:"
if $CONDA_AVAILABLE; then
    echo "  conda activate $ENV_NAME"
else
    echo "  source $ENV_NAME/bin/activate"
fi
echo ""
echo "下一步:"
echo "  1. 准备数据集: bash scripts/prepare_data.sh"
echo "  2. 训练检测模型: python train_yolo.py --data data/raw/ccpd/data.yaml --epochs 100"
echo "  3. 训练识别模型: python train_lprnet.py --data_dir data/processed --epochs 100"
echo "  4. 运行推理:    python demo.py image --input test.jpg --output result.jpg"
echo ""
