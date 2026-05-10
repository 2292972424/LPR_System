#!/usr/bin/env python3
"""批量测试10张图片 + 输出汇总报告"""

import os
import sys
import csv
import time
import argparse
from pathlib import Path
import cv2
from pipeline import LicensePlateRecognizer


def main():
    parser = argparse.ArgumentParser(description="批量车牌识别测试")
    parser.add_argument("--input-dir", default="test_images",
                        help="测试图片目录")
    parser.add_argument("--output-dir", default="batch_results",
                        help="结果输出目录")
    parser.add_argument("--yolo-weight", default="models/weights/best.pt")
    parser.add_argument("--lprnet-weight", default="models/weights/lprnet_best.pth")
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化识别器
    recognizer = LicensePlateRecognizer(
        yolo_weight_path=args.yolo_weight,
        lprnet_weight_path=args.lprnet_weight,
        device=args.device,
        conf_threshold=args.conf,
        bbox_margin=args.margin,
    )

    # 收集测试图片
    img_dir = Path(args.input_dir)
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if not img_paths:
        print(f"❌ {args.input_dir}/ 下没有找到 .jpg/.png 文件")
        return

    print(f"\n{'='*70}")
    print(f"批量测试: {len(img_paths)} 张图片")
    print(f"{'='*70}\n")

    # CSV 报告
    csv_path = os.path.join(args.output_dir, "report.csv")
    csv_file = open(csv_path, 'w', encoding='utf-8-sig', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["文件名", "图片类型", "真实车牌", "预测车牌", "检测置信度",
                      "识别置信度", "是否正确", "耗时(ms)"])

    # 统计
    total = 0
    correct = 0
    detected = 0
    blue_correct = blue_total = 0
    green_correct = green_total = 0
    total_time = 0

    for i, img_path in enumerate(img_paths, 1):
        fname = img_path.name

        # 从文件名提取真实车牌
        # 格式: blue_01_皖AZX345.jpg 或 green_01_京AD12345.jpg
        stem = img_path.stem
        parts = stem.split('_')
        gt_plate = parts[-1] if len(parts) >= 2 else "未知"

        # 判断类型
        img_type = "蓝牌" if fname.startswith("blue") else \
                   "绿牌" if fname.startswith("green") else "未知"

        # 推理
        t0 = time.time()
        results, annotated = recognizer.process_image(str(img_path), draw_result=True)
        elapsed = (time.time() - t0) * 1000

        total_time += elapsed

        if len(results) > 0:
            detected += 1
            pred_plate = results[0]["text"]
            det_conf = results[0]["detection_confidence"]
            recog_conf = results[0]["recognition_confidence"]
            is_correct = (pred_plate == gt_plate[:7])  # 绿牌只比较前7位
        else:
            pred_plate = "未检测到"
            det_conf = 0
            recog_conf = 0
            is_correct = False

        if is_correct:
            correct += 1
        if img_type == "蓝牌":
            blue_total += 1
            if is_correct:
                blue_correct += 1
        elif img_type == "绿牌":
            green_total += 1
            if is_correct:
                green_correct += 1
        total += 1

        # 打印
        status = "✅" if is_correct else "❌"
        print(f"[{i:2d}/{len(img_paths)}] {status} {fname:<30s} "
              f"真实={gt_plate:<10s} 预测={pred_plate:<10s} "
              f"检测={det_conf:.2f} 识别={recog_conf:.2f} {elapsed:.1f}ms")

        # 保存标注结果图
        out_img_path = os.path.join(args.output_dir, f"result_{fname}")
        cv2.imwrite(out_img_path, annotated)

        # 写 CSV
        writer.writerow([fname, img_type, gt_plate, pred_plate,
                         f"{det_conf:.4f}", f"{recog_conf:.4f}",
                         "是" if is_correct else "否", f"{elapsed:.1f}"])

    csv_file.close()

    # ── 汇总报告 ──
    print(f"\n{'='*70}")
    print(f"                     📊 汇总报告")
    print(f"{'='*70}")
    print(f"  总图片数:        {total}")
    print(f"  成功检测:        {detected}/{total} ({detected/total*100:.1f}%)")
    print(f"  识别正确:        {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"")
    print(f"  蓝牌 ({blue_total}张):  正确 {blue_correct}/{blue_total} ({blue_correct/max(blue_total,1)*100:.1f}%)")
    print(f"  绿牌 ({green_total}张):  正确 {green_correct}/{green_total} ({green_correct/max(green_total,1)*100:.1f}%)")
    print(f"")
    print(f"  平均耗时:        {total_time/max(total,1):.1f}ms/张")
    print(f"  总耗时:          {total_time:.0f}ms")
    print(f"")
    print(f"  CSV报告:         {csv_path}")
    print(f"  结果图片目录:     {os.path.abspath(args.output_dir)}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
