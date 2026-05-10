
## 📸 效果展示

```
输入图片 → YOLO检测车牌 → LPRNet识别字符 → 输出: 皖AZX345
```

| 特性 | 说明 |
|------|------|
| 检测模型 | YOLOv8n，支持蓝牌/绿牌双类别 |
| 识别模型 | LPRNet，逐列分类架构 |
| 准确率 | 验证集 Plate Acc **93.5%** |
| 推理速度 | GPU ~30 FPS，CPU ~5 FPS |
| 支持平台 | Linux (推荐) / Windows / macOS |


<img width="360" height="580" alt="251d0fa709382a98c95d218c02603cb0" src="https://github.com/user-attachments/assets/c668f72a-444e-4e2c-a53b-6d08dc5dbd65" />

---

## 🗂️ 项目结构

```
LPR_System/
├── models/                    # 模型定义
│   ├── __init__.py
│   └── lprnet.py             # LPRNet 识别模型
├── utils/                     # 工具模块
│   ├── charset.py            # 字符集配置 (31省+24字母+34序号)
│   ├── dataset.py            # 数据集加载器
│   └── plate_correction.py   # 车牌图像预处理
├── scripts/                   # 一键脚本
│   ├── setup_env.sh          # 一键创建虚拟环境
│   ├── prepare_data.sh       # 数据准备
│   ├── train_all.sh          # 一键训练
│   └── run_demo.sh           # 一键推理
├── train_lprnet.py           # LPRNet 训练入口
├── train_yolo.py             # YOLO 训练入口
├── demo.py                   # 推理演示入口
├── pipeline.py               # 完整识别管线
├── prepare_lpr_dataset.py    # 数据预处理
├── batch_test_ccpd.py        # 批量测试
├── requirements.txt          # 依赖列表
├── data/                     # 数据集目录
│   ├── raw/                  #   原始数据
│   └── processed/            #   预处理后数据
├── models/weights/           # 模型权重 (需自行训练)
├── outputs/                  # 推理输出
└── README.md
```

---

## 🚀 快速开始（5 分钟）

### 前提条件

- Python ≥ 3.10
- CUDA ≥ 12.1（GPU 训练推荐，CPU 推理也可）
- 至少 8 GB 显存（训练），2 GB（推理）

### Step 1：克隆项目

```bash
git clone https://github.com/你的用户名/LPR_System.git
cd LPR_System
```

### Step 2：一键安装环境

```bash
bash scripts/setup_env.sh
```

这会自动：
- 创建 conda/venv 虚拟环境
- 安装 PyTorch + 所有依赖
- 安装中文字体

### Step 3：下载预训练权重

```bash
# 方式一：使用你已有的权重
cp /path/to/best.pt models/weights/
cp /path/to/lprnet_best.pth models/weights/

# 方式二：自己训练（见下方"训练"章节）
```

### Step 4：运行推理

```bash
# 激活环境
conda activate lpr   # 或 source lpr/bin/activate

# 单张图片
bash scripts/run_demo.sh --input path/to/test.jpg --output result.jpg

# 或者直接 Python
python demo.py image --input test.jpg --output result.jpg --no-display
```

---

## 🏋️ 训练自己的模型

### 1. 准备数据集

**方式A：使用 CCPD 公开数据集**

```bash
# 下载 CCPD 数据集（任选一个子集）
# CCPD2019: https://github.com/detectRecog/CCPD
# 解压到 data/raw/ccpd/

# 预处理
bash scripts/prepare_data.sh
```

**方式B：自定义数据集**

```bash
# 目录结构:
data/raw/
├── train/          # 训练图片
│   ├── img001.jpg
│   └── ...
├── val/            # 验证图片
│   └── ...
├── train.txt       # 标注: img001.jpg 皖AZX345
└── val.txt         # 标注: img002.jpg 京AD12345
```

### 2. 训练模型

```bash
# 一键训练（LPRNet + YOLO）
bash scripts/train_all.sh --device cuda --epochs 100

# 或分别训练
python train_lprnet.py --data_dir data/processed --epochs 100 --batch_size 128
python train_yolo.py --data data/raw/data.yaml --epochs 100 --batch 32
```

### 3. 监控训练

```bash
# TensorBoard
tensorboard --logdir runs/
```

---

## 📖 使用手册

### 命令行参数

```bash
# demo.py — 推理模式
python demo.py image \
    --input  test.jpg          \  # 输入图片路径
    --output result.jpg        \  # 输出图片路径
    --yolo-weight models/weights/best.pt     \
    --lprnet-weight models/weights/lprnet_best.pth \
    --margin  0.05             \  # 检测框扩边比例
    --conf    0.25             \  # 检测置信度阈值
    --no-display                  # 不显示窗口（无桌面环境）
```

```bash
# train_lprnet.py — LPRNet 训练
python train_lprnet.py \
    --data_dir    data/processed \  # 数据目录
    --epochs      100           \  # 训练轮数
    --batch_size  128           \  # 批次大小
    --lr          0.001         \  # 学习率
    --warmup_epochs 5           \  # Warmup 轮数
    --device      cuda          \  # 设备
    --output_dir  models/weights   # 权重保存路径
```

### 批量测试

```bash
python batch_test_ccpd.py \
    --data_dir data/processed/val \
    --yolo-weight models/weights/best.pt \
    --lprnet-weight models/weights/lprnet_best.pth \
    --output_dir batch_results
```

---

## 🔧 技术架构

```
                    ┌─────────────────┐
                    │   输入图片       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   YOLOv8n 检测   │  ← 蓝牌/绿牌双分类
                    │   定位车牌区域   │
                    └────────┬────────┘
                             │ 裁剪
                    ┌────────▼────────┐
                    │   图像预处理     │  ← 灰度+Resize 94×24
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   LPRNet 识别    │  ← 逐列分类 (8个位置)
                    │                  │
                    │  P0: 省份 (31类) │
                    │  P1: 字母 (24类) │
                    │  P2~P7: 序号     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   输出车牌文本   │
                    │  皖AZX345 / 绿牌 │
                    └─────────────────┘
```

### LPRNet 架构特点

- **逐列分类**：8 个独立分类器对应车牌 8 个位置
- **Backbone**：5 阶段卷积 + BatchNorm + ReLU
- **宽度自适应**：`AdaptiveAvgPool2d` 将特征图映射到 8 列
- **动态位数**：蓝牌取前 7 位，绿牌取全部 8 位

---

## 📊 性能参考

| 指标 | 数值 | 备注 |
|------|:----:|------|
| 验证集准确率 | **93.5%** | 56320 训练 / 14081 验证 |
| P0 (省份) | 96.0% | 31 类 |
| P1 (字母) | 98.3% | 24 类 |
| P2~P7 (序号) | 97.5%~98.5% | 34 类 |
| 模型参数量 | 4.48M | 轻量级 |
| GPU 推理 | ~30 FPS | NVIDIA RTX 3060+ |
| CPU 推理 | ~5 FPS | Intel i7+ |

---

## ❓ 常见问题

<details>
<summary><b>Q: 中文显示为 ???</b></summary>

安装中文字体：
```bash
# Ubuntu/Debian
sudo apt install fonts-noto-cjk

# CentOS/RHEL
sudo yum install wqy-zenhei-fonts

# macOS
brew install font-noto-sans-cjk
```
</details>

<details>
<summary><b>Q: CUDA 不可用</b></summary>

```bash
# 检查 CUDA 版本
nvidia-smi

# 安装对应版本的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
</details>

<details>
<summary><b>Q: 绿牌识别失败</b></summary>

绿牌需要 8 位分类器。训练数据中必须包含绿牌样本（`train.txt` 中有 8 位车牌号）。
如果没有绿牌数据，模型只在 7 位上训练过，P7 位置是随机初始化。
</details>

<details>
<summary><b>Q: 内存不足 (OOM)</b></summary>

```bash
# 减小 batch_size
python train_lprnet.py --batch_size 32  # 从 128 降到 32

# 或使用梯度累积
# 修改 train_lprnet.py 中的 accumulate 参数
```
</details>

---

## 📝 引用与致谢

- **LPRNet**: [License Plate Recognition via Deep Neural Networks](https://arxiv.org/abs/1806.10447)
- **YOLOv8**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **CCPD**: [Chinese City Parking Dataset](https://github.com/detectRecog/CCPD)

---

## 📄 License

MIT License © 2025

---

## 🤝 贡献指南

欢迎提 Issue 和 Pull Request！

1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/awesome`)
3. 提交更改 (`git commit -m 'Add awesome feature'`)
4. 推送到分支 (`git push origin feature/awesome`)
5. 提交 Pull Request
READMEEOF

# ============================================================
# 10. LICENSE
# ============================================================
cat > LICENSE << 'LICENSEEOF'
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
LICENSEEOF

# ============================================================
# 11. 给脚本加执行权限
# ============================================================
chmod +x scripts/*.sh

# ============================================================
# 12. 创建 .gitkeep 占位
# ============================================================
touch data/raw/.gitkeep data/processed/.gitkeep models/weights/.gitkeep outputs/.gitkeep

echo ""
echo "============================================"
echo "  ✅ 项目结构初始化完成！"
echo "============================================"
echo ""
echo "下一步 — 上传到 GitHub:"
echo ""
echo "  # 1. 初始化 Git"
echo "  git init"
echo "  git add ."
echo "  git commit -m 'Initial commit: LPR_System'"
echo ""
echo "  # 2. 创建 GitHub 仓库"
echo "  #    在 github.com 上新建名为 LPR_System 的空仓库"
echo ""
echo "  # 3. 关联并推送"
echo "  git remote add origin https://github.com/你的用户名/LPR_System.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
echo "============================================"
```

---

## 📋 执行后验证

```bash
# 检查生成的文件
find /LPR_System -maxdepth 2 -type f -name "*.md" -o -name "*.txt" -o -name "*.sh" -o -name ".gitignore" -o -name "LICENSE" | sort
```

预期输出：
```
LICENSE
README.md
requirements-dev.txt
requirements.txt
scripts/prepare_data.sh
scripts/run_demo.sh
scripts/setup_env.sh
scripts/train_all.sh
.gitignore
```

---

## 📖 初学者复刻指南（README 已内嵌）

README.md 已包含完整的：

| 段落 | 内容 |
|------|------|
| 🚀 快速开始 | 5 分钟上手指南 |
| 🏋️ 训练 | 数据集准备 + 训练命令 |
| 📖 使用手册 | 所有命令行参数 |
| ❓ 常见问题 | Q&A 合集 |
| 🔧 技术架构 | 系统流程图 |
| 📊 性能参考 | 准确率/速度表 |

初学者只需 3 步：

```bash
git clone https://github.com/你的用户名/LPR_System.git
cd LPR_System
bash scripts/setup_env.sh       # 一键装环境
bash scripts/run_demo.sh --input test.jpg  # 一键推理
```
