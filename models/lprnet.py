"""
LPRNet - 支持蓝牌(7位)+绿牌(8位)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.charset import CHARSET_CONFIG, build_char_maps


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LPRNet(nn.Module):
    def __init__(self, num_classes_per_pos=None, dropout_rate=0.3, input_channels=1):
        super().__init__()
        if num_classes_per_pos is None:
            num_classes_per_pos = CHARSET_CONFIG["num_classes_per_pos"]

        self.num_classes_per_pos = num_classes_per_pos
        self.total_length = len(num_classes_per_pos)       # 8
        self.ignore_index = CHARSET_CONFIG["ignore_index"]

        # Backbone
        self.stage1 = nn.Sequential(
            ConvBlock(input_channels, 64, 3, 1, 1),
            ConvBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d((2, 2), (2, 2)),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(256, 256, 3, 1, 1),
            ConvBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d((3, 1), (3, 1)),
        )
        self.stage5 = nn.Sequential(
            nn.Conv2d(256, 512, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(512, 512, (1, 3), 1, (0, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
        )

        self.width_adapt = nn.AdaptiveAvgPool2d((1, self.total_length))  # 动态 8 列

        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout_rate),
                nn.Linear(256, n),
            )
            for n in num_classes_per_pos
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)                    # (B, 512, 1, 47)
        x = self.width_adapt(x)               # (B, 512, 1, 8)
        x = x.squeeze(2)                      # (B, 512, 8)

        outputs = []
        for pos in range(self.total_length):
            col = x[:, :, pos]
            outputs.append(self.classifiers[pos](col))
        return outputs

    def predict(self, x, green_threshold=0.3):
        self.eval()
        was_single = x.dim() == 3
        if was_single:
            x = x.unsqueeze(0)

        with torch.no_grad():
            outputs = self.forward(x)

        _, idx_to_char = build_char_maps()
        B = x.size(0)
        results = []
        for b in range(B):
            chars, confs = [], []
            for pos in range(self.total_length):
                probs = F.softmax(outputs[pos][b], dim=0)
                conf, idx = probs.max(dim=0)
                chars.append(idx_to_char[pos].get(idx.item(), '?'))
                confs.append(conf.item())

            is_green = confs[7] > green_threshold
            plate_str = ''.join(chars) if is_green else ''.join(chars[:7])
            avg_conf = sum(confs[:8] if is_green else confs[:7]) / (8 if is_green else 7)
            results.append((plate_str, avg_conf, is_green))

        return results[0] if was_single else results


def create_lprnet(pretrained_path=None, device='cuda'):
    model = LPRNet(
        num_classes_per_pos=CHARSET_CONFIG["num_classes_per_pos"],
        dropout_rate=0.3,
        input_channels=1
    )
    if pretrained_path:
        print(f"[INFO] 加载预训练权重: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model
