#!/usr/bin/env python3
"""训练 LPRNet — 支持蓝牌7位+绿牌8位 (进度条实时显示指标)"""

import os, sys, argparse, math
import torch, torch.nn as nn, torch.optim as optim
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

    pbar = tqdm(dataloader, desc="Training", ncols=100)
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = 0.0
        for pos, logits in enumerate(outputs):
            pos_labels = labels[:, pos]
            mask = pos_labels != IGNORE_INDEX
            if mask.any():
                loss += criterion(logits[mask], pos_labels[mask])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            for pos, logits in enumerate(outputs):
                pos_labels = labels[:, pos]
                mask = pos_labels != IGNORE_INDEX
                if mask.any():
                    preds = logits[mask].argmax(dim=1)
                    correct_per_pos[pos] += (preds == pos_labels[mask]).sum().item()
                    valid_per_pos[pos] += mask.sum().item()

        # 实时显示
        acc = sum(correct_per_pos) / max(sum(valid_per_pos), 1)
        pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})

    avg_loss = total_loss / len(dataloader)
    acc_per_pos = [correct_per_pos[p] / max(valid_per_pos[p], 1) for p in range(TOTAL_LENGTH)]
    overall_acc = sum(correct_per_pos) / max(sum(valid_per_pos), 1)
    return avg_loss, overall_acc, acc_per_pos


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_per_pos = [0] * TOTAL_LENGTH
    valid_per_pos = [0] * TOTAL_LENGTH
    correct_plates = 0

    pbar = tqdm(dataloader, desc="Validating", ncols=100)
    with torch.no_grad():
        for batch in pbar:
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

            plate_correct = torch.ones(images.size(0), dtype=torch.bool, device=device)
            for pos, logits in enumerate(outputs):
                pos_labels = labels[:, pos]
                mask = pos_labels != IGNORE_INDEX
                preds = logits.argmax(dim=1)
                if mask.any():
                    correct_per_pos[pos] += (preds[mask] == pos_labels[mask]).sum().item()
                    valid_per_pos[pos] += mask.sum().item()
                # 整牌：只校验有效位
                plate_correct &= torch.where(mask, preds == pos_labels,
                                             torch.ones_like(plate_correct))
            correct_plates += plate_correct.sum().item()

            # 实时显示
            acc = sum(correct_per_pos) / max(sum(valid_per_pos), 1)
            p_acc = correct_plates / ((pbar.n - 1) * images.size(0) + images.size(0)) if pbar.n > 0 else 0
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})

    n = len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    acc_per_pos = [correct_per_pos[p] / max(valid_per_pos[p], 1) for p in range(TOTAL_LENGTH)]
    overall_acc = sum(correct_per_pos) / max(sum(valid_per_pos), 1)
    plate_acc = correct_plates / max(n, 1)
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
    print(f"[INFO] 设备: {device}", flush=True)
    print(f"[INFO] PyTorch: {torch.__version__}", flush=True)
    print(f"[INFO] 输出位数: {TOTAL_LENGTH} (蓝牌7位 + 绿牌8位)", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 数据集 ──
    if args.data_dir and not args.train_ann and not args.val_ann:
        d = args.data_dir
        if os.path.isfile(os.path.join(d, "train.txt")) and os.path.isfile(os.path.join(d, "val.txt")):
            args.train_ann = os.path.join(d, "train.txt")
            args.val_ann = os.path.join(d, "val.txt")
            args.train_img_dir = os.path.join(d, "train") if os.path.isdir(os.path.join(d, "train")) else d
            args.val_img_dir = os.path.join(d, "val") if os.path.isdir(os.path.join(d, "val")) else d

    if args.train_ann and args.val_ann:
        train_img_dir = args.train_img_dir or os.path.dirname(args.train_ann)
        val_img_dir = args.val_img_dir or os.path.dirname(args.val_ann)
        train_dataset = CustomPlateDataset(train_img_dir, args.train_ann)
        val_dataset = CustomPlateDataset(val_img_dir, args.val_ann)
    elif args.data_dir:
        full = CCPDDataset(args.data_dir)
        vs = max(1, int(len(full) * 0.1))
        train_dataset, val_dataset = random_split(full, [len(full)-vs, vs],
                                                   generator=torch.Generator().manual_seed(42))
    else:
        raise ValueError("请指定数据集")

    print(f"[INFO] 训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}", flush=True)

    bs = min(args.batch_size, max(len(train_dataset) // 10, 8))
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                            num_workers=2, pin_memory=True)

    model = create_lprnet(pretrained_path=args.resume, device=device)
    print(f"[INFO] 参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)
    best_plate_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        msg = f"\n{'='*50}\nEpoch {epoch}/{args.epochs}  LR={optimizer.param_groups[0]['lr']:.6f}\n{'='*50}"
        print(msg, flush=True)

        train_loss, train_acc, train_pos = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_pos, plate_acc = validate(model, val_loader, criterion, device)
        scheduler.step(epoch - 1)

        print(f"[Train] Loss={train_loss:.4f}  CharAcc={train_acc:.4f}", flush=True)
        print(f"[Val]   Loss={val_loss:.4f}  CharAcc={val_acc:.4f}  PlateAcc={plate_acc:.4f}", flush=True)
        print(f"[Val]   逐位: " + " ".join([f"P{i}:{a:.3f}" for i, a in enumerate(val_pos)]), flush=True)

        if plate_acc >= best_plate_acc:
            best_plate_acc = plate_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "lprnet_best.pth"))
            print(f"[INFO] ✅ 最佳模型 PlateAcc={plate_acc:.4f}", flush=True)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"lprnet_epoch{epoch}.pth"))
        torch.save(model.state_dict(), os.path.join(args.output_dir, "lprnet_last.pth"))

    print(f"\n[INFO] 训练完成！最佳车牌准确率: {best_plate_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
