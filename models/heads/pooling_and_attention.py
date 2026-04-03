import torch
import torch.nn as nn
import torch.nn.functional as F

class GeMPool(nn.Module):
    """
    Generalized Mean Pooling:
    y = (mean(x^p))^(1/p)
    Operates over spatial dims (H, W).
    """
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        if learnable:
            self.p = nn.Parameter(torch.tensor(float(p)))
        else:
            self.register_buffer("p", torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1.0 / self.p)
        return x.view(x.size(0), -1)  # [B, C]

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, h, w = x.shape
        s = F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)  # squeeze
        wgt = self.fc(s).view(b, c, 1, 1)                # excite
        return x * wgt