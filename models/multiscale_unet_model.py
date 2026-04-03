# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Dict
# from landmarks_only_training.models.backbones.multiscale_unet_backbone import MultiScaleUNetBackbone

# class MultiScaleEyeLandmarkUNetModel(nn.Module):
#     def __init__(
#         self,
#         num_landmarks: int = 6,
#         hidden_landmarks: int = 256,
#         dropout: float = 0.3,
#         backbone: nn.Module = None,
#         use_final_conv: bool = True,
#     ):
#         super().__init__()
#         self.num_landmarks = num_landmarks
#         self.use_final_conv = use_final_conv
#         self.backbone = backbone if backbone is not None else MultiScaleUNetBackbone()

#         # Freeze encoder
#         if hasattr(self.backbone, "freeze_backbone"):
#             self.backbone.freeze_backbone()
#         else:
#             for p in self.backbone.parameters():
#                 p.requires_grad = False

#         # Static feature dims from UnetEnc channels
#         base_dim = 16 + 16 + 24 + 32  # = 88 from enc64/32/16/8
#         self.in_features = base_dim + (32 if self.use_final_conv else 0)  # +32 for final

#         self.landmark_head = nn.Sequential(
#             nn.Linear(self.in_features, hidden_landmarks),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_landmarks, hidden_landmarks),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_landmarks, num_landmarks * 2),
#         )

#     def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
#         feats = self.backbone(x)
#         pooled = [
#             F.adaptive_avg_pool2d(feats["enc64"], (1, 1)).view(x.size(0), -1),
#             F.adaptive_avg_pool2d(feats["enc32"], (1, 1)).view(x.size(0), -1),
#             F.adaptive_avg_pool2d(feats["enc16"], (1, 1)).view(x.size(0), -1),
#             F.adaptive_avg_pool2d(feats["enc8"],  (1, 1)).view(x.size(0), -1),
#         ]
#         if self.use_final_conv and "final" in feats:
#             pooled.append(F.adaptive_avg_pool2d(feats["final"], (1, 1)).view(x.size(0), -1))
#         x_cat = torch.cat(pooled, dim=1)
#         lmks = self.landmark_head(x_cat).view(-1, self.num_landmarks, 2)
#         return {"landmarks": lmks}





import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from landmarks_only_training.models.backbones.multiscale_unet_backbone import MultiScaleUNetBackbone
from landmarks_only_training.models.heads.pooling_and_attention import GeMPool, SEBlock

class MultiScaleEyeLandmarkUNetModel(nn.Module):
    def __init__(
        self,
        num_landmarks: int = 6,
        hidden_landmarks: int = 256,
        dropout: float = 0.3,
        backbone: nn.Module = None,
        use_final_conv: bool = True,
        use_se: bool = True,          # enable SE per-scale
        use_gem: bool = True,         # use GeM pooling instead of GAP
        use_layernorm: bool = True,   # LayerNorm on concatenated vector
        use_scale_gates: bool = True, # learnable scalar gate per scale
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.use_final_conv = use_final_conv
        self.backbone = backbone if backbone is not None else MultiScaleUNetBackbone()

        # Freeze encoder
        if hasattr(self.backbone, "freeze_backbone"):
            self.backbone.freeze_backbone()
        else:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # Per-scale channel counts from your backbone
        self.scale_channels = {
            "enc64": 16,
            "enc32": 16,
            "enc16": 24,
            "enc8": 32,
            "final": 32,
        }

        # Build per-scale SE and pooling
        self.use_se = use_se
        self.use_gem = use_gem

        def make_pool():
            return GeMPool(p=3.0, learnable=True) if self.use_gem else nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
            )

        self.se_blocks = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        for name in ["enc64", "enc32", "enc16", "enc8"]:
            if self.use_se:
                self.se_blocks[name] = SEBlock(self.scale_channels[name], reduction=16)
            self.pools[name] = make_pool()

        if self.use_final_conv:
            if self.use_se:
                self.se_blocks["final"] = SEBlock(self.scale_channels["final"], reduction=16)
            self.pools["final"] = make_pool()

        # Learnable scale gates (one scalar per used scale)
        self.use_scale_gates = use_scale_gates
        scale_list = ["enc64", "enc32", "enc16", "enc8"] + (["final"] if self.use_final_conv else [])
        self.scales: List[str] = scale_list
        if self.use_scale_gates:
            self.scale_gates = nn.Parameter(torch.ones(len(self.scales)))

        # Feature dim after pooling and concat
        base_dim = self.scale_channels["enc64"] + self.scale_channels["enc32"] + \
                   self.scale_channels["enc16"] + self.scale_channels["enc8"]
        self.in_features = base_dim + (self.scale_channels["final"] if self.use_final_conv else 0)

        # Optional LayerNorm to stabilize feature scale
        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.pre_head_norm = nn.LayerNorm(self.in_features)
        else:
            self.pre_head_norm = nn.Identity()

        # Landmark regressor head
        self.landmark_head = nn.Sequential(
            nn.Linear(self.in_features, hidden_landmarks),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_landmarks, hidden_landmarks),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_landmarks, num_landmarks * 2),
        )

    def _apply_se(self, name: str, feat: torch.Tensor) -> torch.Tensor:
        if self.use_se and name in self.se_blocks:
            feat = self.se_blocks[name](feat)
        return feat

    def _pool(self, name: str, feat: torch.Tensor, batch_size: int) -> torch.Tensor:
        pooled = self.pools[name](feat)  # [B, C]
        if self.use_scale_gates:
            # gate is scalar per scale; expand to [B, 1] then broadcast
            idx = self.scales.index(name)
            gate = torch.sigmoid(self.scale_gates[idx])  # (0,1)
            pooled = pooled * gate
        return pooled

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)  # dict of feature maps

        pooled_vecs = []
        for name in ["enc64", "enc32", "enc16", "enc8"]:
            f = self._apply_se(name, feats[name])
            pooled_vecs.append(self._pool(name, f, x.size(0)))

        if self.use_final_conv and "final" in feats:
            f = self._apply_se("final", feats["final"])
            pooled_vecs.append(self._pool("final", f, x.size(0)))

        x_cat = torch.cat(pooled_vecs, dim=1)  # [B, in_features]
        x_cat = self.pre_head_norm(x_cat)
        lmks = self.landmark_head(x_cat).view(-1, self.num_landmarks, 2)
        return {"landmarks": lmks}