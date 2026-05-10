#!/usr/bin/env python3
"""训练 LPRNet — 支持蓝牌7位+绿牌8位"""

import os
import argparse
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.lprnet import create_lprnet
from utils.dataset import CCPDDataset, CustomPlateDataset
from utils.charset import CHARSET_CONFIG

TOTAL_LENGTH = CHARSET_CONFIG["max_plate_length"]
IGNORE_INDEX = CHARSET_CONFIG["ignore_index"]


def train_epoch(model, dataloader, criterion, optimizer, device, grad_clip=5.0):
    model.train()
    total_loss = 0.0
    correct_per_pos = [0] * TOTAL_LENGTH
    valid_per_pos = [0] * TOTAL_LENGTH
    total_samples = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = 0.0
        for pos, logits in enumerate(outputs):
            pos_labels = labels[:, pos]
            # 只计算非 ignored 位置的 loss
            mask = pos_labels != IGNORE_INDEX
            if mask.any():
                loss += criterion(logits[mask], pos_labels[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_samples += images.size(0)

        with torch.no_grad():
            for pos, logits in enumerate(outputs):
                pos_labels = labels[:, pos]
                mask = pos_labels != IGNORE_INDEX
                if mask.any():
                    preds = logits[mask].argmax(dim=1)
                    correct_per_pos[pos] += (preds == pos_labels[mask]).sum().item()
                    valid_per_pos[pos] += mask.sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(dataloader)
    acc_per_pos = [
        correct_per_pos[p] / max(valid_per_pos[p], 1)
        for p in range(TOTAL_LENGTH)
    ]
    overall_acc = sum(correct_per_pos) / max(sum(valid_per_pos), 1)
    return avg_loss, overall_acc, acc_per_pos


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_per_pos = [0] * TOTAL_LENGTH
    valid_per_pos = [0] * TOTAL_LENGTH
    total_samples = 0
    correct_plates = 0
    valid_plates = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            outputs = model(images)

            loss = 0.0
            for pos, logits in enumerate(outputs):
                pos_labels = labels[:, pos]
                mask = pos_labels != IGNORE_INDEX
                if mask.any():
                    loss += criterion(logits[mask], pos_labels[mask])

            total_loss += loss.item()
            total_samples += images.size(0)

            # 逐位正确率
            for pos, logits in enumerate(outputs):
                pos_labels = labels[:, pos]
                mask = pos_labels != IGNORE_INDEX
                if mask.any():
                    preds = logits[mask].argmax(dim=1)
                    correct_per_pos[pos] += (preds == pos_labels[mask]).sum().item()
                    valid_per_pos[pos] += mask.sum().item()

            # 整牌准确率
            plate_correct = torch.ones(images.size(0), dtype=torch.bool, device=device)
            for pos, logits in enumerate(outputs):
                pos_labels = labels[:, pos]
                mask = pos_labels != IGNORE_INDEX
                preds = logits.argmax(dim=1)
                # 只校验有效位置
                plate_correct &= torch.where(mask, preds == pos_labels,
                                             torch.ones_like(plate_correct))
            valid_plates += images.size(0)
            correct_plates += plate_correct.sum().item()

    avg_loss = total_loss / len(dataloader)
    acc_per_pos = [
        correct_per_pos[p] / max(valid_per_pos[p], 1)
        for p in range(TOTAL_LENGTH)
    ]
    overall_acc = sum(correct_per_pos) / max(sum(valid_per_pos), 1)
    plate_acc = correct_plates / max(valid_plates, 1)
    return avg_loss, overall_acc, acc_per_pos, plate_acc


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [g['lr'] for g in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            scale = self.min_lr / self.base_lrs[0] + \
                    0.5 * (1 - self.min_lr / self.base_lrs[0]) * (1 + math.cos(math.pi * progress))
        for i, pg in enumerate(self.optimizer.param_groups):
            pg['lr'] = self.base_lrs[i] * scale


def main():
    parser = argparse.ArgumentParser(description="训练 LPRNet (7位蓝牌+8位绿牌)")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_ann", type=str, default=None)
    parser.add_argument("--val_ann", type=str, default=None)
    parser.add_argument("--train_img_dir", type=str, default=None)
    parser.add_argument("--val_img_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="models/weights")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 设备: {device}")
    print(f"[INFO] PyTorch: {torch.__version__}")
    print(f"[INFO] 输出位数: {TOTAL_LENGTH} (蓝牌7位 + 绿牌8位)")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 数据集 ──
    if args.data_dir and not args.train_ann and not args.val_ann:
        data_dir = args.data_dir
        train_ann_candidate = os.path.join(data_dir, "train.txt")
        val_ann_candidate = os.path.join(data_dir, "val.txt")
        train_img_candidate = os.path.join(data_dir, "train")
        val_img_candidate = os.path.join(data_dir, "val")
        if os.path.isfile(train_ann_candidate) and os.path.isfile(val_ann_candidate):
            args.train_ann = train_ann_candidate
            args.val_ann = val_ann_candidate
            args.train_img_dir = train_img_candidate if os.path.isdir(train_img_candidate) else data_dir
            args.val_img_dir = val_img_candidate if os.path.isdir(val_img_candidate) else data_dir

    if args.train_ann and args.val_ann:
        train_img_dir = args.train_img_dir or os.path.dirname(args.train_ann)
        val_img_dir = args.val_img_dir or os.path.dirname(args.val_ann)
        print(f"[INFO] 训练标注: {args.train_ann}")
        print(f"[INFO] 训练图片目录: {train_img_dir}")
        print(f"[INFO] 验证标注: {args.val_ann}")
        print(f"[INFO] 验证图片目录: {val_img_dir}")
        train_dataset = CustomPlateDataset(train_img_dir, args.train_ann)
        val_dataset = CustomPlateDataset(val_img_dir, args.val_ann)
    elif args.data_dir:
        full_dataset = CCPDDataset(args.data_dir)
        if len(full_dataset) == 0:
            raise ValueError(f"未找到图片: {args.data_dir}")
        val_size = max(1, int(len(full_dataset) * 0.1))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
    else:
        raise ValueError("请指定数据集")

    print(f"[INFO] 训练集: {len(train_dataset)} 张")
    print(f"[INFO] 验证集: {len(val_dataset)} 张")

    # ── DataLoader ──
    actual_batch = min(args.batch_size, max(len(train_dataset) // 10, 8))
    train_loader = DataLoader(train_dataset, batch_size=actual_batch,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ── 模型 ──
    model = create_lprnet(pretrained_path=args.resume, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 参数量: {total_params:,}")

    # ── 损失 & 优化器 ──
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)

    best_plate_acc = 0.0

    # ── 训练 ──
    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}  |  LR: {current_lr:.6f}")
        print(f"{'='*50}")

        train_loss, train_acc, train_acc_pos = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_acc_pos, plate_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(epoch - 1)

        print(f"\n[Train] Loss: {train_loss:.4f}  |  Char Acc: {train_acc:.4f}")
        print(f"[Val]   Loss: {val_loss:.4f}  |  Char Acc: {val_acc:.4f}  |  Plate Acc: {plate_acc:.4f}")
        label_names = ["P" + str(i) for i in range(TOTAL_LENGTH)]
        if TOTAL_LENGTH == 8:
            label_names[7] = "P7*"  # 标记绿牌位
        print(f"[Val]   逐位: " + " ".join(
            [f"{n}:{a:.3f}" for n, a in zip(label_names, val_acc_pos)]
        ))

        if plate_acc >= best_plate_acc:
            best_plate_acc = plate_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "lprnet_best.pth"))
            print(f"[INFO] ✅ 最佳: Plate Acc {plate_acc:.4f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(args.output_dir, f"lprnet_epoch{epoch}.pth"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "lprnet_last.pth"))

        if epoch >= 15 and best_plate_acc < 0.01:
            print("\n[WARN] 15轮后仍无改善，请检查数据与预处理")
            break

    print(f"\n[INFO] 训练完成！最佳车牌准确率: {best_plate_acc:.4f}")


if __name__ == "__main__":
    main()
