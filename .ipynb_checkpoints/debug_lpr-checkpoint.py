#!/usr/bin/env python3
"""车牌识别诊断脚本——逐位分析预测偏差"""
import sys
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from models.lprnet import create_lprnet
from utils.plate_correction import preprocess_for_lprnet
from utils.charset import build_char_maps

DEVICE = 'cuda'
TEST_IMG = sys.argv[1] if len(sys.argv) > 1 else 'test1.jpg'

# ── 1. 加载模型 ──
detector = YOLO('models/weights/best.pt')
lprnet = create_lprnet('models/weights/lprnet_best.pth', DEVICE)
lprnet.eval()

_, idx_to_char = build_char_maps()
print(f"字符集: P0({len(idx_to_char[0])}) P1({len(idx_to_char[1])}) P2~6({len(idx_to_char[2])})")

# ── 2. YOLO 检测 ──
image = cv2.imread(TEST_IMG)
print(f"\n原图尺寸: {image.shape}")

results = detector(image, conf=0.1, verbose=False)
for r in results:
    if r.boxes is not None:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
            print(f"  检测到: class={cls} ({detector.names[cls]}), conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")

            crop = image[y1:y2, x1:x2]
            print(f"  裁剪尺寸: {crop.shape}")

            # ── 3. 预处理 ──
            normalized = preprocess_for_lprnet(crop, target_size=(94, 24))
            tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(DEVICE)
            print(f"  输入tensor: shape={tensor.shape}, range=[{tensor.min():.3f}, {tensor.max():.3f}]")

            # 保存预处理后的图像供人工检查
            cv2.imwrite('debug_crop.jpg', crop)
            cv2.imwrite('debug_preprocessed.jpg', (normalized * 255).astype('uint8'))
            print("  已保存: debug_crop.jpg / debug_preprocessed.jpg")

            # ── 4. 逐位预测 ──
            with torch.no_grad():
                outputs = lprnet(tensor)

            print(f"\n{'位':<4} {'预测idx':<10} {'预测字符':<8} {'Top3候选':<30} {'置信度':<10}")
            print("-" * 70)

            all_chars = []
            for pos, logits in enumerate(outputs):
                probs = F.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
                top3_idx = probs.topk(3).indices.tolist()
                top3_prob = probs.topk(3).values.tolist()
                best_idx = top3_idx[0]
                best_char = idx_to_char[pos].get(best_idx, '?')

                candidates = []
                for i in range(3):
                    ch = idx_to_char[pos].get(top3_idx[i], '?')
                    candidates.append(f"{ch}({top3_prob[i]:.2f})")

                all_chars.append(best_char)
                print(f"P{pos:<3} {best_idx:<10} {best_char:<8} {' '.join(candidates):<30} {top3_prob[0]:.4f}")

            plate_str = ''.join(all_chars)
            print(f"\n  ✅ 最终预测: {plate_str}")
            print(f"  🎯 正确答案: 皖AZX345")
            print(f"\n  差异分析:")
            correct = "皖AZX345"
            for i, (p, c) in enumerate(zip(plate_str, correct)):
                match = "✓" if p == c else f"✗ (应为'{c}')"
                print(f"    P{i}: 预测='{p}' 正确='{c}' {match}")

