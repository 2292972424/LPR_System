"""
车牌图像预处理模块 — 使用直接 resize，与训练一致
废除有问题的角点检测+透视变换方案
"""

import cv2
import numpy as np


def preprocess_for_lprnet(plate_img, target_size=(94, 24)):
    """
    LPRNet 标准预处理:
    灰度化 → 直接 resize 到 94×24 → 归一化
    
    与训练完全一致：训练时 CCPD 裁剪的车牌区域已经是方正对齐的，
    cv2.resize 就是训练时实际使用的预处理。
    """
    h, w = plate_img.shape[:2]
    if w == 0 or h == 0:
        return np.zeros((target_size[1], target_size[0]), dtype=np.float32)

    # 转灰度
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()

    # 直接 resize（训练时就是这个操作）
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_CUBIC)

    # 归一化
    normalized = resized.astype(np.float32) / 255.0

    return normalized


# 保留旧函数名兼容
def correct_plate(plate_img, target_width=94, target_height=24):
    """兼容旧接口，内部使用直接 resize"""
    return preprocess_for_lprnet(plate_img, (target_width, target_height))
