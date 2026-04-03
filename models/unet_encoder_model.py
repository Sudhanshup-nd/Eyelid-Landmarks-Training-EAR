import torch
import torch.nn as nn
from typing import Dict

from Unet_training_script.models.backbones.unet_wrapper import UnetEnc, UnetDec, UNetBackbone



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


class EyeLandmarkWithFrozenSegmentationBackbone(nn.Module):
    """
    Multi-task model with frozen UnetUpSample_modified backbone.
    
    Architecture:
        1. Frozen UnetUpSample_modified (encoder + decoder + seg heads)
        2. Trainable LandmarkHeatmapHead operating on decoder features
    
    Outputs during training:
        - heatmaps: [B, num_landmarks, H, W]
        - coords: [B, num_landmarks, 2] via soft-argmax
    
    Outputs during inference (optional):
        - heatmaps: [B, num_landmarks, H, W]
        - coords: [B, num_landmarks, 2]
        - output1: segmentation mask (from frozen model)
        - output2: blink suppression (from frozen model)
    """
    def __init__(self, 
                 segmentation_model: nn.Module,
                 num_landmarks: int = 6,
                 return_segmentation_outputs: bool = True):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.return_segmentation_outputs = return_segmentation_outputs
        
        # Frozen segmentation backbone
        self.segmentation_model = segmentation_model
        self._freeze_segmentation_model()
        
        # Trainable heatmap head
        self.heatmap_head = LandmarkHeatmapHead(
            in_channels=16,  # dec64 output channels
            num_landmarks=num_landmarks
        )
        
        print(f"[INFO] Created EyeLandmarkWithFrozenSegmentationBackbone:")
        print(f"  - Frozen segmentation model params: {sum(p.numel() for p in self.segmentation_model.parameters())}")
        print(f"  - Trainable heatmap head params: {sum(p.numel() for p in self.heatmap_head.parameters())}")
        
    def _freeze_segmentation_model(self):
        """Freeze all parameters in the segmentation model."""
        for param in self.segmentation_model.parameters():
            param.requires_grad = False
        self.segmentation_model.eval()
        print("[INFO] Frozen segmentation model (all params set to requires_grad=False)")
    
    def _extract_decoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from dec64 layer (before seg1/seg2 heads).
        Returns: [B, 16, H, W]
        """
        # Forward through frozen segmentation encoder+decoder
        # We need to replicate the forward pass up to dec64 output
        with torch.no_grad():
            # Encoder
            x64, x = self.segmentation_model.enc64(x)
            x32, x = self.segmentation_model.enc32(x)
            x16, x = self.segmentation_model.enc16(x)
            x8, x = self.segmentation_model.enc8(x)
            x = self.segmentation_model.conv(x)
            
            # Decoder
            x = self.segmentation_model.dec8(x, x8)
            x = self.segmentation_model.dec16(x, x16)
            x = self.segmentation_model.dec32(x, x32)
            x = self.segmentation_model.dec64(x, x64)  # [B, 16, H, W]
        
        # CRITICAL: Detach to ensure no gradients flow to frozen model
        # But this should still allow gradients to flow through heatmap_head
        return x.detach()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, 1, H, W] grayscale input
        
        Returns dict with:
            - heatmaps: [B, num_landmarks, H, W]
            - coords: [B, num_landmarks, 2] normalized to [0,1]
            - output1: (optional) segmentation mask
            - output2: (optional) blink suppression
        """
        # Extract decoder features (frozen)
        decoder_feats = self._extract_decoder_features(x)  # [B, 16, H, W]
        
        # Generate heatmaps (trainable)
        heatmaps = self.heatmap_head(decoder_feats)  # [B, num_landmarks, H, W]
        
        # CRITICAL FIX: Scale heatmaps by 100x before soft-argmax to make softmax sharper
        # This compensates for the low-magnitude outputs from the randomly initialized head
        coords = self._soft_argmax_2d(heatmaps * 100.0, temperature=1.0)  # [B, num_landmarks, 2]
        
        output = {
            "heatmaps": heatmaps,
            "coords": coords,
        }
        
        # Optionally include segmentation outputs
        if self.return_segmentation_outputs:
            with torch.no_grad():
                output1, output2 = self.segmentation_model(x)
                output["output1"] = output1
                output["output2"] = output2
        
        return output
    
    def _soft_argmax_2d(self, heatmaps: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Differentiable coordinates from heatmaps via soft-argmax.
        Returns normalized coords in [0,1] relative to W,H.
        heatmaps: [B,L,H,W] raw logits
        temperature: softmax temperature (higher = softer distribution)
        """
        B, L, H, W = heatmaps.shape
        hm_flat = heatmaps.view(B, L, -1)           # [B,L,H*W]
        
        # Apply temperature scaling before softmax
        prob = torch.softmax(hm_flat / temperature, dim=-1)       # [B,L,H*W]

        ys = torch.linspace(0, 1, steps=H, device=heatmaps.device)
        xs = torch.linspace(0, 1, steps=W, device=heatmaps.device)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')   # [H,W]
        grid = torch.stack([xx, yy], dim=-1).view(-1, 2) # [H*W,2]

        coords = torch.matmul(prob, grid)  # [B,L,2]
        return coords
    
    def train(self, mode: bool = True):
        """Override train() to keep segmentation model in eval mode."""
        super().train(mode)
        self.segmentation_model.eval()  # Always keep frozen model in eval
        return self        