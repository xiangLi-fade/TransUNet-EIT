import copy

import torch
import torch.nn as nn

from .vit_seg_configs import get_b16_config
from .vit_seg_modeling import DecoderCup, Embeddings, Encoder, SegmentationHead


def get_eit_config():
    config = get_b16_config()
    config.patches.size = (16, 16)
    config.n_classes = 1
    config.n_skip = 0
    config.skip_channels = [0, 0, 0, 0]
    return config


class VoltageEncoder(nn.Module):
    def __init__(self, hidden_size, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.encoder = nn.Sequential(
            nn.Linear(208, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_size * num_tokens),
        )
        self.position = nn.Parameter(torch.zeros(1, num_tokens, hidden_size))

    def forward(self, voltage):
        tokens = self.encoder(voltage)
        return tokens.view(voltage.size(0), self.num_tokens, -1) + self.position


class EITTransUNet(nn.Module):
    """初始 EIT 图像与 208 维边界电压的双分支重建网络。"""

    def __init__(self, img_size=128, voltage_tokens=4, config=None):
        super().__init__()
        self.config = copy.deepcopy(config or get_eit_config())
        self.image_embeddings = Embeddings(self.config, img_size=img_size)
        self.voltage_encoder = VoltageEncoder(self.config.hidden_size, voltage_tokens)
        self.encoder = Encoder(self.config, vis=False)
        self.decoder = DecoderCup(self.config)
        self.output_head = SegmentationHead(
            self.config.decoder_channels[-1], 1, kernel_size=3
        )

    def forward(self, image, voltage):
        if image.size(1) == 1:
            image = image.repeat(1, 3, 1, 1)
        image_tokens, features = self.image_embeddings(image)
        voltage_tokens = self.voltage_encoder(voltage)
        fused, _ = self.encoder(torch.cat([image_tokens, voltage_tokens], dim=1))
        image_tokens = fused[:, :image_tokens.size(1)]
        return self.output_head(self.decoder(image_tokens, features))
