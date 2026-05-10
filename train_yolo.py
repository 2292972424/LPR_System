#!/usr/bin/env python3
"""
训练 YOLOv8n 车牌检测模型
使用 ultralytics 框架
"""

from ultralytics import YOLO
import torch
import argparse


def train_yolov8n(data_yaml, epochs=100, batch=16, imgsz=640, device=0,
                   pretrained="yolov8n.pt", output_dir="runs/detect/plate"):
    """
    训练 YOLOv8n
    Args:
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch: 批次大小
        imgsz: 输入图像尺寸
        device: GPU设备ID (0,1,... 或 'cpu')
        pretrained: 预训练权重
        output_dir: 输出目录
    """
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] PyTorch版本: {torch.__version__}")
    print(f"[INFO] CUDA可用: {torch.cuda.is_available()}")

    # 加载预训练模型
    model = YOLO(pretrained)

    # 开始训练
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        workers=8,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        patience=20,
        save=True,
        save_period=10,
        project=output_dir,
        name='yolov8n_plate',
        exist_ok=True,
        pretrained=True,
        verbose=True,
        # 数据增强
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )

    # 验证
    metrics = model.val()
    print(f"\n[INFO] 验证结果: mAP50={metrics.box.map50:.4f}, "
          f"mAP50-95={metrics.box.map:.4f}")

    # 导出为最佳格式
    model.export(format='onnx', imgsz=imgsz, half=False)

    return model, metrics


def prepare_data_yaml(dataset_path, output_path="plate_dataset.yaml"):
    """
    准备训练数据配置文件
    """
    yaml_content = f"""
# 车牌检测数据集配置
path: {dataset_path}
train: images/train
val: images/val
test: images/test

# 类别
nc: 1
names:
  0: license_plate
"""
    with open(output_path, 'w') as f:
        f.write(yaml_content.strip())
    print(f"[INFO] 数据配置文件已生成: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="数据集根目录")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--pretrained", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    yaml_path = prepare_data_yaml(args.data)
    train_yolov8n(
        data_yaml=yaml_path,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        pretrained=args.pretrained,
    )