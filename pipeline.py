"""
车牌识别完整管线: YOLO检测 → LPRNet识别
"""
import cv2, os, subprocess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from models.lprnet import create_lprnet
from utils.plate_correction import preprocess_for_lprnet


# ── 中文字体 ──
def _find_chinese_font():
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    try:
        r = subprocess.run(['fc-list', ':lang=zh', '-f', '%{file}\n'],
                          capture_output=True, text=True, timeout=5)
        for line in r.stdout.strip().split('\n'):
            if line and os.path.isfile(line):
                candidates.insert(0, line)
    except: pass
    for p in candidates:
        if os.path.isfile(p): return p
    return None

_CN_FONT = None
_p = _find_chinese_font()
if _p:
    try:
        _CN_FONT = ImageFont.truetype(_p, 20)
        print(f"[INFO] 中文字体: {_p}")
    except: pass
if _CN_FONT is None:
    print("[WARN] 无中文字体")
    _CN_FONT = ImageFont.load_default()


def _draw_text_cv2(img, text, x, y, color=(255, 0, 255)):
    has_cn = any('\u4e00' <= c <= '\u9fff' for c in text)
    if not has_cn:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    b = d.textbbox((0, 0), text, font=_CN_FONT)
    d.text((x, max(0, y - (b[3]-b[1]))), text, font=_CN_FONT, fill=color)
    img[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


class LicensePlateRecognizer:
    def __init__(self, yolo_weight_path, lprnet_weight_path,
                 device='cuda', conf_threshold=0.25, iou_threshold=0.45,
                 bbox_margin=0.05, draw_shrink=0.02):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.bbox_margin = bbox_margin
        self.draw_shrink = draw_shrink

        print(f"[INFO] YOLO: {yolo_weight_path}")
        self.detector = YOLO(yolo_weight_path)
        self.detector.to(self.device)
        if hasattr(self.detector, 'names'):
            print(f"[INFO] 类别: {self.detector.names}")

        print(f"[INFO] LPRNet: {lprnet_weight_path}")
        self.recognizer = create_lprnet(lprnet_weight_path, self.device)
        self.recognizer.eval()
        print(f"[INFO] 设备: {self.device} | 裁剪扩边: {bbox_margin*100:.0f}% | 绘制缩边: {draw_shrink*100:.0f}%")

    def _expand_bbox(self, x1, y1, x2, y2, iw, ih):
        w, h = x2 - x1, y2 - y1
        mw, mh = int(w * self.bbox_margin), int(h * self.bbox_margin)
        return max(0, x1-mw), max(0, y1-mh), min(iw, x2+mw), min(ih, y2+mh)

    def _shrink_bbox(self, x1, y1, x2, y2):
        w, h = x2 - x1, y2 - y1
        sw, sh = int(w * self.draw_shrink), int(h * self.draw_shrink)
        return x1+sw, y1+sh, x2-sw, y2-sh

    def detect(self, image):
        results = self.detector(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        names = getattr(self.detector, 'names', {0:'license_plate'})
        h, w = image.shape[:2]
        plates = []
        for r in results:
            if r.boxes is not None:
                for b in r.boxes:
                    x1,y1,x2,y2 = b.xyxy[0].tolist()
                    cls = int(b.cls[0].item())
                    cn = names.get(cls, str(cls)).lower()
                    if 'plate' in cn or 'license' in cn or cls in (0,1):
                        cx1,cy1,cx2,cy2 = self._expand_bbox(int(x1),int(y1),int(x2),int(y2), w, h)
                        if cx2>cx1 and cy2>cy1:
                            plates.append((cx1,cy1,cx2,cy2, float(b.conf[0].item()),
                                           int(x1),int(y1),int(x2),int(y2)))
        return plates

    def recognize(self, plate_img):
        try:
            norm = preprocess_for_lprnet(plate_img, (94, 24))
            t = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)
            res = self.recognizer.predict(t)
            if isinstance(res, tuple) and len(res) >= 3:
                text, conf, is_green = res[0], res[1], res[2]
            elif isinstance(res, tuple) and len(res) == 2:
                text, conf, is_green = res[0], res[1], False
            else:
                text, conf, is_green = str(res), 0.5, False
            return text, conf, is_green
        except Exception as e:
            print(f"[WARN] 识别失败: {e}")
            return "失败", 0.0, False

    def process_image(self, image_path, output_path=None, draw_result=True):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        plates = self.detect(image)
        results = []

        for item in plates:
            x1,y1,x2,y2,det_conf,ox1,oy1,ox2,oy2 = item
            crop = image[y1:y2, x1:x2]
            if crop.size == 0: continue

            plate_text, recog_conf, is_green = self.recognize(crop)
            results.append({
                "bbox": (x1,y1,x2,y2), "text": plate_text,
                "detection_confidence": det_conf,
                "recognition_confidence": recog_conf,
                "is_green": is_green,
            })

            if draw_result:
                dx1,dy1,dx2,dy2 = self._shrink_bbox(ox1,oy1,ox2,oy2)
                cv2.rectangle(image, (dx1,dy1), (dx2,dy2), (255,0,255), 2)
                label = f"{plate_text} {'绿' if is_green else ''} ({recog_conf:.2f})"
                _draw_text_cv2(image, label, dx1, dy1-12, (255,0,255))

        if output_path and draw_result:
            cv2.imwrite(output_path, image)
            print(f"[INFO] 结果已保存: {output_path}")

        return (results, image) if draw_result else results

    def process_video(self, video_path, output_path=None, display=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开: {video_path}")
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if output_path:
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
        fc = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            fc += 1
            for item in self.detect(frame):
                x1,y1,x2,y2,_,ox1,oy1,ox2,oy2 = item
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                text, conf, is_green = self.recognize(crop)
                dx1,dy1,dx2,dy2 = self._shrink_bbox(ox1,oy1,ox2,oy2)
                cv2.rectangle(frame, (dx1,dy1), (dx2,dy2), (255,0,255), 2)
                _draw_text_cv2(frame, f"{text} {'绿' if is_green else ''}", dx1, dy1-12, (255,0,255))
            if writer: writer.write(frame)
            if display:
                try:
                    cv2.imshow('LPR', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                except: display = False
            if fc % 100 == 0: print(f"[INFO] 已处理 {fc} 帧")
        cap.release()
        if writer: writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] 完成，共 {fc} 帧")

    def process_webcam(self, camera_id=0):
        self.process_video(camera_id, output_path=None, display=True)
