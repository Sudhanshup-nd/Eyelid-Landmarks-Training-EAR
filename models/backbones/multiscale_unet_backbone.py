import torch
import torch.nn as nn
from .unet_wrapper import UnetEnc

class MultiScaleUNetBackbone(nn.Module):
    """
    Frozen UNet encoder-only backbone exposing multi-scale features:
    - enc64, enc32, enc16, enc8 (pre-pooling outputs)
    - final: 3x3 conv on enc8 output
    """
    def __init__(self):
        super().__init__()
        self.enc64 = UnetEnc(1, 16)
        self.enc32 = UnetEnc(16, 16)
        self.enc16 = UnetEnc(16, 24)
        self.enc8 = UnetEnc(24, 32)
        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.out_channels = 32

    def forward(self, x):
        x64, x = self.enc64(x)
        x32, x = self.enc32(x)
        x16, x = self.enc16(x)
        x8,  x = self.enc8(x)
        x_final = self.conv(x8)
        return {"enc64": x64, "enc32": x32, "enc16": x16, "enc8": x8, "final": x_final}

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False