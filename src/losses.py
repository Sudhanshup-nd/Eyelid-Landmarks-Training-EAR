import torch
import torch.nn as nn
import math


class BaseLandmarkLoss(nn.Module):
    """
    Parent class just for interface consistency.
    All coordinate-based losses assume:
      pred: [B,L,2] normalized local coords
      gt:   [B,L,2]
      mask: [B,L] (1 if landmark valid)
      visibility: [B,1] (0/1)
    """
    def __init__(self, visibility_gating: bool = True, eps: float = 1e-6):
        super().__init__()
        self.visibility_gating = visibility_gating
        self.eps = eps

    def forward(self, pred, gt, mask, visibility):
        raise NotImplementedError

    def _apply_mask(self, pred, gt, mask, visibility):
        visibility = visibility.view(-1)  # [B]
        eff_mask = mask * visibility.unsqueeze(1) if self.visibility_gating else mask
        eff_mask_exp = eff_mask.unsqueeze(-1)  # [B,L,1]
        return eff_mask, eff_mask_exp


class L1LandmarkLoss(BaseLandmarkLoss):
    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = torch.abs(pred - gt) * eff_mask_exp
        valid_coords = eff_mask.sum() * 2.0
        if valid_coords.item() == 0:
            return pred.sum()*0.0, {"lmk_l1":0.0,"valid_coords":0.0}
        loss = diff.sum() / (valid_coords + self.eps)
        return loss, {"lmk_l1":loss.item(),"valid_coords":float(valid_coords.item())}


class MSELandmarkLoss(BaseLandmarkLoss):
    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = (pred - gt)**2 * eff_mask_exp
        valid_coords = eff_mask.sum() * 2.0
        if valid_coords.item() == 0:
            return pred.sum()*0.0, {"mse":0.0,"valid_coords":0.0}
        loss = diff.sum() / (valid_coords + self.eps)
        return loss, {"mse":loss.item(),"valid_coords":float(valid_coords.item())}


class SmoothL1LandmarkLoss(BaseLandmarkLoss):
    def __init__(self, visibility_gating=True, eps=1e-6, beta=1.0):
        super().__init__(visibility_gating, eps)
        self.beta = beta

    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = pred - gt
        abs_diff = torch.abs(diff)
        cond = abs_diff < self.beta
        smooth = torch.where(cond, 0.5 * (diff**2) / self.beta, abs_diff - 0.5*self.beta)
        smooth = smooth * eff_mask_exp
        valid_coords = eff_mask.sum()*2.0
        if valid_coords.item() == 0:
            return pred.sum()*0.0, {"smooth_l1":0.0,"valid_coords":0.0}
        loss = smooth.sum() / (valid_coords + self.eps)
        return loss, {"smooth_l1":loss.item(),"valid_coords":float(valid_coords.item())}


class HuberLandmarkLoss(BaseLandmarkLoss):
    def __init__(self, visibility_gating=True, eps=1e-6, delta=1.0):
        super().__init__(visibility_gating, eps)
        self.delta = delta

    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = pred - gt
        abs_diff = torch.abs(diff)
        quadratic = torch.minimum(abs_diff, torch.tensor(self.delta, device=abs_diff.device))
        linear = abs_diff - quadratic
        huber = 0.5 * quadratic**2 + self.delta * linear
        huber = huber * eff_mask_exp
        valid_coords = eff_mask.sum()*2.0
        if valid_coords.item()==0:
            return pred.sum()*0.0, {"huber":0.0,"valid_coords":0.0}
        loss = huber.sum()/(valid_coords + self.eps)
        return loss, {"huber":loss.item(),"valid_coords":float(valid_coords.item())}


class WingLandmarkLoss(BaseLandmarkLoss):
    """
    Wing Loss (used for facial landmark regression):
    loss(x) = w * log(1 + |x|/epsilon) if |x| < w
              |x| - C                     otherwise
    where C = w - w*log(1 + w/epsilon)
    """
    def __init__(self, visibility_gating=True, eps=1e-6, w=10.0, epsilon=2.0):
        super().__init__(visibility_gating, eps)
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * math.log(1 + w/epsilon)

    def forward(self, pred, gt, mask, visibility):
        eff_mask, eff_mask_exp = self._apply_mask(pred, gt, mask, visibility)
        diff = pred - gt
        abs_diff = torch.abs(diff)
        part_small = self.w * torch.log(1 + abs_diff / self.epsilon)
        part_large = abs_diff - self.C
        wing = torch.where(abs_diff < self.w, part_small, part_large)
        wing = wing * eff_mask_exp
        valid_coords = eff_mask.sum()*2.0
        if valid_coords.item()==0:
            return pred.sum()*0.0, {"wing":0.0,"valid_coords":0.0}
        loss = wing.sum()/(valid_coords + self.eps)
        return loss, {"wing":loss.item(),"valid_coords":float(valid_coords.item())}


# ----- HEATMAP LOSSES -----

def heatmap_mse_loss(pred_hm, gt_hm, mask, visibility, landmark_weights=None, eps: float = 1e-6):
    """
    pred_hm, gt_hm: [B,L,H,W]
    mask: [B,L], visibility: [B,1]
    landmark_weights: optional [L] tensor of per-landmark weights.
    """
    B, L, H, W = pred_hm.shape
    vis = visibility.view(-1, 1)    # [B,1]
    eff_mask = mask * vis           # [B,L]

    if landmark_weights is not None:
        w = landmark_weights.to(pred_hm.device).view(1, L, 1, 1)  # [1,L,1,1]
        eff_mask = eff_mask.unsqueeze(-1).unsqueeze(-1) * w       # [B,L,1,1]
    else:
        eff_mask = eff_mask.unsqueeze(-1).unsqueeze(-1)           # [B,L,1,1]

    # Simple MSE - no normalization to avoid gradient issues
    diff = (pred_hm - gt_hm) ** 2 * eff_mask
    num_valid = eff_mask.sum().clamp(min=1.0)
    loss = diff.sum() / num_valid
    return loss


def heatmap_ce_loss(pred_hm, gt_hm, mask, visibility, eps: float = 1e-6):
    """
    pred_hm, gt_hm: [B,L,H,W]
    mask: [B,L], visibility: [B,1]
    """
    B, L, H, W = pred_hm.shape
    vis = visibility.view(-1, 1)        # [B,1]
    eff_mask = mask * vis               # [B,L]
    eff_mask = eff_mask.unsqueeze(-1).unsqueeze(-1)  # [B,L,1,1]

    pred_flat = pred_hm.view(B, L, -1)  # [B,L,H*W]
    gt_flat   = gt_hm.view(B, L, -1)    # [B,L,H*W]

    # Pred distribution
    pred_prob = torch.softmax(pred_flat, dim=-1) + eps

    # GT distribution (normalize per heatmap)
    gt_prob = gt_flat / gt_flat.sum(dim=-1, keepdim=True).clamp(min=eps)

    ce = -(gt_prob * pred_prob.log()).sum(dim=-1)  # [B,L]
    ce = ce * eff_mask.squeeze(-1).squeeze(-1)
    num_valid = eff_mask.sum().clamp(min=1.0)
    loss = ce.sum() / num_valid
    return loss    


class HeatmapLossWrapper(nn.Module):
    """
    Wrapper to use weighted heatmap MSE with same interface as coordinate losses.
    Expects pred_hm, gt_hm, mask, visibility.
    Optionally supports auxiliary coord L1 (coords_pred/coords_gt).
    """
    def __init__(self,
                 lambda_coord: float = 0.0,
                 landmark_weights=None,
                 coord_loss_type: str = "l1",
                 wing_w: float = 10.0,
                 wing_epsilon: float = 2.0):
        """Heatmap loss with optional auxiliary coordinate loss.

        Args:
            lambda_coord: weight for the auxiliary coordinate loss.
            landmark_weights: optional per-landmark weights for heatmap MSE.
            coord_loss_type: "l1" or "wing". If "wing", uses WingLandmarkLoss.
            wing_w: Wing loss "w" parameter (only used when coord_loss_type == "wing").
            wing_epsilon: Wing loss epsilon parameter (only used when coord_loss_type == "wing").
        """
        super().__init__()
        self.lambda_coord = float(lambda_coord)

        if landmark_weights is not None:
            self.landmark_weights = torch.tensor(landmark_weights, dtype=torch.float32)
        else:
            self.landmark_weights = None

        coord_loss_type = coord_loss_type.lower().strip()
        if coord_loss_type not in ("l1", "wing"):
            coord_loss_type = "l1"
        self.coord_loss_type = coord_loss_type

        # When using Wing as auxiliary coord loss, reuse WingLandmarkLoss implementation
        # so masking and visibility gating stay consistent with other coord losses.
        self.wing_coord_loss = None
        if self.coord_loss_type == "wing" and self.lambda_coord > 0.0:
            self.wing_coord_loss = WingLandmarkLoss(
                visibility_gating=True,
                w=wing_w,
                epsilon=wing_epsilon
            )

    def forward(self, pred_hm, gt_hm, mask, visibility, coords_pred=None, coords_gt=None):
        """
        pred_hm: [B,L,H,W]
        gt_hm:   [B,L,H,W]
        mask:    [B,L]
        visibility: [B,1]
        coords_pred: [B,L,2] optional (from soft-argmax)
        coords_gt:   [B,L,2] optional (normalized GT)
        """
        # Use MSE loss for sigmoid-activated heatmaps (more stable than CE)
        hm_loss = heatmap_mse_loss(
            pred_hm, gt_hm, mask, visibility, self.landmark_weights
        )

        total_loss = hm_loss
        stats = {"hm_loss": float(hm_loss.item())}

        if self.lambda_coord > 0.0 and coords_pred is not None and coords_gt is not None:
            if self.coord_loss_type == "wing" and self.wing_coord_loss is not None:
                # Use WingLandmarkLoss on coordinates with mask + visibility
                coord_loss, _ = self.wing_coord_loss(coords_pred, coords_gt, mask, visibility)
                stats["coord_wing_aux"] = float(coord_loss.item())
            else:
                # Default auxiliary coord loss: masked L1
                vis = visibility.view(-1, 1)   # [B,1]
                eff_mask = mask * vis          # [B,L]
                eff_mask_exp = eff_mask.unsqueeze(-1)  # [B,L,1]
                coord_diff = torch.abs(coords_pred - coords_gt) * eff_mask_exp
                valid_coords = eff_mask.sum() * 2.0
                if valid_coords.item() == 0:
                    coord_loss = coords_pred.sum() * 0.0
                else:
                    coord_loss = coord_diff.sum() / (valid_coords + 1e-6)
                stats["coord_l1_aux"] = float(coord_loss.item())

            total_loss = total_loss + self.lambda_coord * coord_loss

        stats["total"] = float(total_loss.item())
        return total_loss, stats


def build_landmark_loss(cfg):
    t = cfg['training'].get('landmark_loss_type', 'l1').lower()
    params = cfg['training'].get('landmark_loss', {})

    if t == 'l1':
        return L1LandmarkLoss(visibility_gating=True)
    if t == 'mse':
        return MSELandmarkLoss(visibility_gating=True)
    if t == 'smooth_l1':
        beta = params.get('beta', 1.0)
        return SmoothL1LandmarkLoss(visibility_gating=True, beta=beta)
    if t == 'huber':
        delta = params.get('huber_delta', 1.0)
        return HuberLandmarkLoss(visibility_gating=True, delta=delta)
    if t == 'wing':
        w = params.get('wing_w', 10.0)
        eps = params.get('wing_epsilon', 2.0)
        return WingLandmarkLoss(visibility_gating=True, w=w, epsilon=eps)
    if t == 'heatmap':
        # Read weights + aux coord factor from config if present
        lmk_w = params.get("heatmap_landmark_weights", None)
        print("Using landmark weights:", lmk_w)

        lambda_coord = float(params.get("heatmap_lambda_coord", 0.0))

        # Coordinate auxiliary loss configuration for heatmap training
        coord_loss_type = params.get("coord_loss_type", "l1")
        wing_w = float(params.get("coord_wing_w", 10.0))
        wing_epsilon = float(params.get("coord_wing_epsilon", 2.0))

        return HeatmapLossWrapper(
            lambda_coord=lambda_coord,
            landmark_weights=lmk_w,
            coord_loss_type=coord_loss_type,
            wing_w=wing_w,
            wing_epsilon=wing_epsilon,
        )

    raise ValueError(f"Unknown landmark_loss_type: {t}")