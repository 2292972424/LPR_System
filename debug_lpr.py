#!/usr/bin/env python3
"""车牌识别诊断 — 测试多种预处理策略"""
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from models.lprnet import create_lprnet
from utils.charset import build_char_maps

DEVICE = 'cuda'
TEST_IMG = sys.argv[1] if len(sys.argv) > 1 else 'test1.jpg'

detector = YOLO('models/weights/best.pt')
lprnet = create_lprnet('models/weights/lprnet_best.pth', DEVICE)
lprnet.eval()
_, idx_to_char = build_char_maps()

print(f"字符集: P0({len(idx_to_char[0])}) P1({len(idx_to_char[1])}) P2~6({len(idx_to_char[2])})")

image = cv2.imread(TEST_IMG)
h, w = image.shape[:2]
print(f"\n原图尺寸: {image.shape}")

results = detector(image, conf=0.1, verbose=False)

for r in results:
    if r.boxes is not None:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            bw, bh = x2 - x1, y2 - y1

            print(f"\nYOLO检测: class={cls}({detector.names[cls]}), conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2}), size={bw}x{bh}")

            # ── 策略1: 不同扩边比例，统一用 direct resize ──
            print(f"\n{'margin':<8} {'crop':<12} {'ratio':<8} {'pred':<12}")
            print("-" * 50)

            for margin in [0.00, 0.10, 0.15, 0.20, 0.25]:
                mw = int(bw * margin)
                mh = int(bh * margin)
                nx1 = max(0, x1 - mw)
                ny1 = max(0, y1 - mh)
                nx2 = min(w, x2 + mw)
                ny2 = min(h, y2 + mh)

                crop = image[ny1:ny2, nx1:nx2]
                cw, ch = crop.shape[1], crop.shape[0]
                ratio = cw / max(ch, 1)

                # 直接 resize (绕过 corner detection 的潜在干扰)
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (94, 24), interpolation=cv2.INTER_CUBIC)
                tensor = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = lprnet(tensor)

                chars = []
                for pos, logits in enumerate(outputs):
                    probs = F.softmax(logits, dim=1).squeeze(0)
                    best_idx = probs.argmax().item()
                    chars.append(idx_to_char[pos].get(best_idx, '?'))

                pred_str = ''.join(chars)
                print(f"{margin:<8.2f} {cw}x{ch:<8} {ratio:<8.2f} {pred_str:<12}")

                # 保存 margin=0.15 的预处理图
                if abs(margin - 0.15) < 0.01:
                    cv2.imwrite('debug_direct_crop.jpg', crop)
                    cv2.imwrite('debug_direct_resized.jpg', resized)

            # ── 策略2: 使用完整 correct_plate 流程 ──
            from utils.plate_correction import preprocess_for_lprnet as ppl
            print(f"\n{'margin':<8} {'pred (correct_plate)':<20}")
            print("-" * 40)
            for margin in [0.00, 0.10, 0.15, 0.20, 0.25]:
                mw = int(bw * margin)
                mh = int(bh * margin)
                nx1 = max(0, x1 - mw)
                ny1 = max(0, y1 - mh)
                nx2 = min(w, x2 + mw)
                ny2 = min(h, y2 + mh)
                crop = image[ny1:ny2, nx1:nx2]

                normalized = ppl(crop, target_size=(94, 24))
                tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    outputs = lprnet(tensor)
                chars = []
                for pos, logits in enumerate(outputs):
                    probs = F.softmax(logits, dim=1).squeeze(0)
                    best_idx = probs.argmax().item()
                    chars.append(idx_to_char[pos].get(best_idx, '?'))
                print(f"{margin:<8.2f} {''.join(chars):<20}")

print(f"\n🎯 正确答案: 皖AZX345")
