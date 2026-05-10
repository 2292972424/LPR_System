"""
字符集配置 — 支持蓝牌(7位) + 绿牌(8位)
"""

PROVINCES = "京沪粤苏浙鲁晋冀豫川渝辽吉黑皖闽赣鄂湘贵云陕甘青蒙桂宁新藏琼"  # 31
CITY_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"  # 24 (不含I,O)
SEQ_CHARS = "ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"  # 34

CHARSET_CONFIG = {
    "num_classes_per_pos": [31, 24, 34, 34, 34, 34, 34, 34],
    "max_plate_length": 8,
    "ignore_index": -100,
}


def build_char_maps():
    """构建字符↔索引映射"""
    num_classes = CHARSET_CONFIG["num_classes_per_pos"]
    char_to_idx = []
    idx_to_char = []

    p0 = {ch: i for i, ch in enumerate(PROVINCES)}
    char_to_idx.append(p0)
    idx_to_char.append({i: ch for ch, i in p0.items()})

    p1 = {ch: i for i, ch in enumerate(CITY_LETTERS)}
    char_to_idx.append(p1)
    idx_to_char.append({i: ch for ch, i in p1.items()})

    seq = {ch: i for i, ch in enumerate(SEQ_CHARS)}
    for _ in range(len(num_classes) - 2):
        char_to_idx.append(seq)
        idx_to_char.append({i: ch for ch, i in seq.items()})

    return char_to_idx, idx_to_char
