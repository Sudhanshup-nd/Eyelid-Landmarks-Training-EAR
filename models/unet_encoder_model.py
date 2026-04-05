import sys
import torch
import torch.nn as nn
from typing import Dict
sys.path.insert(0, "/inwdata2a/sudhanshu/Unet_training_script")
from models.backbones.unet_wrapper import UnetEnc, UnetDec, UNetBackbone



class EyeLandmarkUNetModel(nn.Module):
    def __init__(self,
                 hidden_landmarks: int = 128,
                 dropout: float = 0.0,
                 num_landmarks: int = 6,
                 use_aux_head: bool = False,
                 backbone: nn.Module =  UNetBackbone()):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.use_aux_head = use_aux_head

        self.backbone = backbone if backbone is not None else UNetBackbone()
        in_features = self.backbone.out_channels

        self.landmark_head = nn.Sequential(
            nn.Linear(in_features, hidden_landmarks),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_landmarks, num_landmarks * 2)
        )

        self.aux_head = None
        if self.use_aux_head:
            self.aux_head = nn.Sequential(
                nn.Linear(in_features, hidden_landmarks // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_landmarks // 2, num_landmarks)
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        lmks = self.landmark_head(feats).view(-1, self.num_landmarks, 2)
        out = {"landmarks": lmks}
        if self.aux_head is not None:
            out["aux"] = self.aux_head(feats)
        return out




import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict



class EyeLandmarkUNetHeatmapModel(nn.Module):
    """
    Heatmap regression model that reuses the UnetUpSample_modified encoder/decoder.

    Architecture:
      enc64 -> enc32 -> enc16 -> enc8 -> conv
      dec8 -> dec16 -> dec32 -> dec64
      hm_head -> num_landmarks heatmaps

    Outputs:
      - heatmaps: [B, num_landmarks, H, W]
      - coords:   [B, num_landmarks, 2]  (x,y in [0,1] via soft-argmax)
    """
    def __init__(self,
                 num_landmarks: int = 6,
                 base_channels: int = 16):
        super().__init__()
        self.num_landmarks = num_landmarks

        # Encoder: same channels as UnetUpSample / UnetUpSample_modified
        self.enc64 = UnetEnc(1, 16)    # => 16
        self.enc32 = UnetEnc(16, 16)   # => 16
        self.enc16 = UnetEnc(16, 24)   # => 24
        self.enc8  = UnetEnc(24, 32)   # => 32

        self.conv = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        # Decoder: same pattern as UnetUpSample/Unet
        self.dec8  = UnetDec(32, 32, 32)
        self.dec16 = UnetDec(32, 24, 24)
        self.dec32 = UnetDec(24, 16, 16)
        self.dec64 = UnetDec(16, 16, 16)

        # Heatmap head: from 16 channels -> num_landmarks
        self.hm_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, num_landmarks, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B,1,H,W] grayscale eye crops
        Returns:
          - heatmaps: [B,L,H,W]
          - coords:   [B,L,2] normalized to [0,1] relative to W,H
        """
        B, C, H, W = x.shape

        # Encoder (same as UnetUpSample_modified)
        x64, x = self.enc64(x)  # x64: [B,16,H,W],   x: [B,16,H/2,W/2]
        x32, x = self.enc32(x)  # x32: [B,16,H/2],   x: [B,16,H/4,W/4]
        x16, x = self.enc16(x)  # x16: [B,24,H/4],   x: [B,24,H/8,W/8]
        x8,  x = self.enc8(x)   # x8:  [B,32,H/8],   x: [B,32,H/16,W/16]
        x = self.conv(x)        # [B,32,H/16,W/16]

        # Decoder (same as UnetUpSample_modified, but no seg head)
        x = self.dec8(x, x8)    # [B,32,H/8,W/8]
        x = self.dec16(x, x16)  # [B,24,H/4,W/4]
        x = self.dec32(x, x32)  # [B,16,H/2,W/2]
        x = self.dec64(x, x64)  # [B,16,H,W]

        # Heatmaps
        heatmaps = self.hm_head(x)  # [B,L,H,W]

        # Differentiable coordinates via soft-argmax
        coords = self._soft_argmax_2d(heatmaps)  # [B,L,2] in [0,1]

        return {
            "heatmaps": heatmaps,
            "coords": coords,
        }

    def _soft_argmax_2d(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """
        Differentiable coordinates from heatmaps via soft-argmax.
        Returns normalized coords in [0,1] relative to W,H.
        heatmaps: [B,L,H,W]
        """
        B, L, H, W = heatmaps.shape
        hm_flat = heatmaps.view(B, L, -1)           # [B,L,H*W]
        prob = F.softmax(hm_flat, dim=-1)           # [B,L,H*W]

        ys = torch.linspace(0, 1, steps=H, device=heatmaps.device)
        xs = torch.linspace(0, 1, steps=W, device=heatmaps.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')   # [H,W]
        grid = torch.stack([xx, yy], dim=-1).view(-1, 2) # [H*W,2]

        coords = torch.matmul(prob, grid)  # [B,L,2]
        return coords


def generate_gaussian_heatmaps(lm: torch.Tensor,
                               mask: torch.Tensor,
                               H: int,
                               W: int,
                               sigma: float = 1.5) -> torch.Tensor:
    """
    lm:   [B,L,2] normalized coords in [0,1]
    mask: [B,L]  1 if landmark valid, 0 otherwise
    Returns:
      gt_hm: [B,L,H,W] unnormalized Gaussian heatmaps (peak ~1.0)
    """
    device = lm.device
    B, L, _ = lm.shape

    ys = torch.arange(0, H, device=device).float()
    xs = torch.arange(0, W, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]

    gt_hm = torch.zeros(B, L, H, W, device=device)

    for b in range(B):
        for i in range(L):
            if mask[b, i] <= 0:
                continue
            x_norm, y_norm = lm[b, i]   # [0,1]
            cx = x_norm * (W - 1)
            cy = y_norm * (H - 1)
            g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            gt_hm[b, i] = g

    # Don't normalize - keep Gaussian peaks at ~1.0 for better gradient signal
    return gt_hm





class EyeLandmarkUNetHeatmapModelWrapper(nn.Module):
    """
    Wraps EyeLandmarkUNetHeatmapModel that outputs extra landmarks,
    but exposes only the first `num_train_landmarks` to the training code.
    """
    def __init__(self,
                 num_train_landmarks: int = 6,
                 num_total_landmarks: int = 11,
                 base_channels: int = 16):
        super().__init__()
        self.num_train_landmarks = num_train_landmarks
        self.model = EyeLandmarkUNetHeatmapModel(
            num_landmarks=num_total_landmarks,
            base_channels=base_channels
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.model(x)
        heatmaps = out["heatmaps"]           # [B, 11, H, W]
        coords   = out["coords"]             # [B, 11, 2]

        # Keep only first 6 for training
        heatmaps_train = heatmaps[:, :self.num_train_landmarks, :, :]
        coords_train   = coords[:, :self.num_train_landmarks, :]

        # Return both: training views and full views (for latency / analysis)
        return {
            "heatmaps": heatmaps_train,      # used by current training code
            "coords": coords_train,          # used by current training code
            "heatmaps_full": heatmaps,       # [B,11,H,W] (optional, for analysis)
            "coords_full": coords,           # [B,11,2]   (optional, for analysis)
        }





# ============================================================================
# NEW: Frozen Segmentation Backbone + Lightweight Heatmap Head
# ============================================================================

class LandmarkHeatmapHead(nn.Module):
    """
    Lightweight heatmap regression head (Option 1).
    Takes decoder features [B, 16, H, W] → produces heatmaps [B, num_landmarks, H, W]
    
    Architecture:
        Conv2d(16→32) + BN + ReLU
        Conv2d(32→16) + BN + ReLU
        Conv2d(16→num_landmarks)
        Soft-argmax → coords
    
    Trainable params: ~5K
    """
    def __init__(self, in_channels=16, num_landmarks=6):
        super().__init__()
        self.num_landmarks = num_landmarks
        
        # Simpler head without batch norm (avoids gradient dampening)
        self.head = nn.Sequential(
            # Expand and refine
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Final heatmaps with proper initialization
            nn.Conv2d(32, num_landmarks, kernel_size=1),
        )
        
        # Use default Kaiming initialization for better gradients
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with Kaiming for ReLU activations"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 16, H, W] decoder features
        Returns: [B, num_landmarks, H, W] heatmaps (raw logits for soft-argmax)
        """
        return self.head(x)  # Return raw logits - softmax in soft-argmax handles normalization

class MouthLandmarkModel(nn.Module):
    """
    End-to-end trainable mouth landmark detector.

    Unlike EyeLandmarkWithFrozenSegmentationBackbone, the backbone here
    is fully trainable — gradients flow through the entire network.
    This is necessary because the eye-pretrained backbone has never seen
    mouth data and needs to adapt its features.

    Architecture:
        UnetUpSample_modified (backbone — fully trainable)
            ↓ dec64 features [B, 16, H, W]
        LandmarkHeatmapHead (heatmap prediction)
            ↓
        soft-argmax → coords [B, L, 2]
    """
    def __init__(self, backbone: nn.Module, num_landmarks: int = 4):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.backbone      = backbone   # fully trainable — no freezing

        self.heatmap_head = LandmarkHeatmapHead(
            in_channels  = 16,           # dec64 output channels
            num_landmarks = num_landmarks
        )

        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[INFO] MouthLandmarkModel created")
        print(f"[INFO] Total params: {total}  |  Trainable: {trainable}")

    def _extract_decoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through backbone up to dec64.
        Gradients flow freely — backbone updates during training.
        """
        # Encoder
        x64, x = self.backbone.enc64(x)
        x32, x = self.backbone.enc32(x)
        x16, x = self.backbone.enc16(x)
        x8,  x = self.backbone.enc8(x)
        x      = self.backbone.conv(x)

        # Decoder
        x = self.backbone.dec8(x,  x8)
        x = self.backbone.dec16(x, x16)
        x = self.backbone.dec32(x, x32)
        x = self.backbone.dec64(x, x64)   # [B, 16, H, W]

        return x   # no detach — gradients flow back through here

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, 1, H, W] grayscale mouth crop

        Returns:
            heatmaps: [B, L, H, W]
            coords:   [B, L, 2] normalized [0,1]
        """
        feats    = self._extract_decoder_features(x)          # [B, 16, H, W]
        heatmaps = self.heatmap_head(feats)                    # [B, L, H, W]
        coords   = self._soft_argmax_2d(heatmaps * 100.0)     # [B, L, 2]

        return {"heatmaps": heatmaps, "coords": coords}

    def _soft_argmax_2d(self, heatmaps: torch.Tensor) -> torch.Tensor:
        B, L, H, W = heatmaps.shape
        prob = torch.softmax(heatmaps.view(B, L, -1), dim=-1)

        ys = torch.linspace(0, 1, H, device=heatmaps.device)
        xs = torch.linspace(0, 1, W, device=heatmaps.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        grid   = torch.stack([xx, yy], dim=-1).view(-1, 2)   # [H*W, 2]

        return torch.matmul(prob, grid)   # [B, L, 2]