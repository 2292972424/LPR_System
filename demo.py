#!/usr/bin/env python3
"""车牌识别系统演示脚本"""
import os
import argparse
import cv2
from pipeline import LicensePlateRecognizer


def can_display():
    try:
        return os.environ.get('DISPLAY') is not None
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="车牌识别系统 (YOLO + LPRNet)")
    parser.add_argument("mode", choices=["image", "video", "webcam"])
    parser.add_argument("--input", "-i", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--yolo-weight", type=str, default="models/weights/best.pt")
    parser.add_argument("--lprnet-weight", type=str, default="models/weights/lprnet_best.pth")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.05,
                        help="裁剪扩边比例 (默认 0.05=5%%)")
    parser.add_argument("--draw-shrink", type=float, default=0.02,
                        help="绘制框缩边比例 (默认 0.02=2%%)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-display", action="store_true", default=not can_display())
    args = parser.parse_args()

    recognizer = LicensePlateRecognizer(
        yolo_weight_path=args.yolo_weight,
        lprnet_weight_path=args.lprnet_weight,
        device=args.device,
        conf_threshold=args.conf,
        bbox_margin=args.margin,
        draw_shrink=args.draw_shrink,
    )

    if args.mode == "image":
        if not args.input:
            raise ValueError("需要 --input 参数")
        results, annotated_img = recognizer.process_image(
            args.input, args.output, draw_result=True
        )

        print("\n" + "=" * 50)
        print("检测与识别结果:")
        print("=" * 50)
        if len(results) == 0:
            print("  ⚠️  未检测到车牌！试试 --conf 0.15 --margin 0.10")
        else:
            for i, res in enumerate(results, 1):
                print(f"  车牌 {i}: {res['text']}")
                print(f"    检测置信度: {res['detection_confidence']:.3f}")
                print(f"    识别置信度: {res['recognition_confidence']:.3f}")
        print("=" * 50)

        if not args.no_display and can_display():
            cv2.imshow("Result", annotated_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif args.output:
            print(f"[INFO] 结果已保存: {args.output}")

    elif args.mode == "video":
        if not args.input:
            raise ValueError("需要 --input 参数")
        recognizer.process_video(args.input, args.output,
                                 display=(not args.no_display and can_display()))

    elif args.mode == "webcam":
        if not can_display():
            print("[ERROR] 无桌面环境，不支持摄像头模式")
            return
        recognizer.process_webcam(camera_id=0)


if __name__ == "__main__":
    main()
