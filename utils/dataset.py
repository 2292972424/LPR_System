"""
LPRNet 数据集加载器（支持蓝牌7位 + 绿牌8位）
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.plate_correction import preprocess_for_lprnet
from utils.charset import build_char_maps, CHARSET_CONFIG

TOTAL_LENGTH = CHARSET_CONFIG["max_plate_length"]      # 8
IGNORE_INDEX = CHARSET_CONFIG["ignore_index"]           # -100
NUM_CLASSES = CHARSET_CONFIG["num_classes_per_pos"]     # [31,24,34,34,34,34,34,34]


class CCPDDataset(Dataset):
    """CCPD 数据集：文件名含车牌号"""

    def __init__(self, data_dir, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.char_to_idx, self.idx_to_char = build_char_maps()

        self.image_paths = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.image_paths.append(os.path.join(root, f))

        print(f"[INFO] CCPD模式: 找到 {len(self.image_paths)} 张图片")

    def _parse_label(self, filepath):
        basename = os.path.splitext(os.path.basename(filepath))[0]
        if len(basename) >= 7:
            return basename[:8]  # 取前8位（绿牌8位，蓝牌7位）
        for sep in ['-', '_', '.']:
            parts = basename.split(sep)
            for p in parts:
                if len(p) >= 7 and '\u4e00' <= p[0] <= '\u9fff':
                    return p[:8]
        return None

    def _encode_label(self, plate_str):
        """编码车牌号，不足8位用 ignore_index 填充"""
        labels = []
        for pos in range(TOTAL_LENGTH):
            if pos < len(plate_str):
                ch = plate_str[pos]
                labels.append(self.char_to_idx[pos].get(ch, 0))
            else:
                labels.append(IGNORE_INDEX)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        plate_str = self._parse_label(img_path)

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        normalized = preprocess_for_lprnet(image, target_size=(94, 24))
        tensor = torch.from_numpy(normalized).unsqueeze(0).float()

        if plate_str is None or len(plate_str) < 7:
            plate_str = "皖A00000"

        labels = self._encode_label(plate_str[:TOTAL_LENGTH])

        return {
            "image": tensor,
            "label": labels,
            "plate_str": plate_str[:TOTAL_LENGTH],
            "path": img_path,
        }


class CustomPlateDataset(Dataset):
    """自定义标注数据集"""

    def __init__(self, data_dir, annotation_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.char_to_idx, self.idx_to_char = build_char_maps()

        self.samples = []
        with open(annotation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    plate_str = parts[1]
                    full_path = os.path.join(data_dir, img_name)
                    if os.path.isfile(full_path):
                        self.samples.append((full_path, plate_str))
                    else:
                        # 尝试当前目录
                        alt_path = os.path.join(os.path.dirname(data_dir), img_name) if os.path.sep not in img_name else img_name
                        if os.path.isfile(alt_path):
                            self.samples.append((alt_path, plate_str))
                        else:
                            print(f"[WARN] 图片不存在，跳过: {full_path}")

        print(f"[INFO] 自定义模式: {len(self.samples)} 个有效样本")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, plate_str = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        normalized = preprocess_for_lprnet(image, target_size=(94, 24))
        tensor = torch.from_numpy(normalized).unsqueeze(0).float()

        labels = []
        for pos in range(TOTAL_LENGTH):
            if pos < len(plate_str):
                ch = plate_str[pos]
                labels.append(self.char_to_idx[pos].get(ch, 0))
            else:
                labels.append(IGNORE_INDEX)

        return {
            "image": tensor,
            "label": torch.tensor(labels, dtype=torch.long),
            "plate_str": plate_str[:TOTAL_LENGTH],
            "path": img_path,
        }
