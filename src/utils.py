import os
import yaml
import random
import torch
import numpy as np
import pickle
import datetime
import platform
from typing import Any, Dict, Optional

def timestamp():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def debug(msg: str, enabled: bool = True):
    if enabled:
        print(f"[DEBUG {timestamp()}] {msg}")

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if "debug" not in cfg:
        cfg["debug"] = {}
    return cfg

def seed_everything(seed: Any, dbg: bool = False):
    seed_int = int(seed)
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    torch.cuda.manual_seed_all(seed_int)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    debug(f"Set seeds to {seed_int}", dbg)

def ensure_dir(path: str, dbg: bool = False):
    try:
        os.makedirs(path, exist_ok=True)
        debug(f"Ensured directory: {path}", dbg)
    except PermissionError as e:
        raise PermissionError(f"Cannot create directory '{path}'. {e}")

def save_checkpoint(state: dict, path: str, dbg: bool = False):
    torch.save(state, path)
    debug(f"Saved checkpoint: {path}", dbg)

def save_weights_only(model_state: dict, path: str, dbg: bool = False):
    torch.save(model_state, path)
    debug(f"Saved weights-only file: {path}", dbg)

def load_checkpoint(path: str, map_location: Optional[str] = None,
                    allow_unsafe_fallback: bool = True, dbg: bool = False):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        ckpt = torch.load(path, map_location=map_location)
        debug(f"Loaded checkpoint (safe): {path}", dbg)
        return ckpt
    except pickle.UnpicklingError:
        if allow_unsafe_fallback:
            debug(f"Safe load failed, retry without weights_only: {path}", dbg)
            ckpt = torch.load(path, map_location=map_location, weights_only=False)
            debug(f"Loaded checkpoint (unsafe fallback): {path}", dbg)
            return ckpt
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{path}': {e}")

def reconcile_checkpoint(model: torch.nn.Module, ckpt_state: dict, cfg_debug: Dict[str, Any]):
    skip = bool(cfg_debug.get("skip_mismatch_layers", True))
    loaded, skipped = [], []
    model_state = model.state_dict()
    for k, v in ckpt_state.items():
        if k not in model_state:
            skipped.append(k); continue
        if model_state[k].shape != v.shape:
            if skip:
                skipped.append(k); continue
            else:
                raise RuntimeError(f"Shape mismatch for {k}: ckpt={tuple(v.shape)} model={tuple(model_state[k].shape)}")
        model_state[k].copy_(v)
        loaded.append(k)
    model.load_state_dict(model_state)
    return loaded, skipped

def print_environment(dbg: bool = True):
    if not dbg: return
    debug(f"Python: {platform.python_version()}", True)
    debug(f"Torch:  {torch.__version__}", True)
    debug(f"CUDA available: {torch.cuda.is_available()}", True)
    if torch.cuda.is_available():
        debug(f"CUDA devices: {torch.cuda.device_count()}", True)
        debug(f"Device name: {torch.cuda.get_device_name(0)}", True)
    debug(f"Working dir: {os.getcwd()}", True)


import os
import sys
import torch

def _ensure_src_on_path():
    # utils.py is inside landmarks_only_training/src
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)


def load_unet_encoder_backbone_from_ckpt(ckpt_path, cfg=None, device="cpu"):
    """
    Load a checkpoint saved via MLflow/wrapper that contains yacs CfgNode in ckpt['cfg'] and a model saved
    from eye_internal_segmentor.*. We:
      - allowlist safe globals (yacs.config.CfgNode) to keep weights_only=True
      - if weights_only=True still fails, we make src importable and fall back to weights_only=False (trusted)
      - extract encoder+bottleneck weights and freeze the backbone
    """
    # Allowlist yacs.config.CfgNode to enable safe weights-only loading
    try:
        import yacs.config
        torch.serialization.add_safe_globals([yacs.config.CfgNode])
    except Exception:
        pass

    ckpt = None
    try:
        # Preferred: tensors-only load; avoids importing modules like eye_internal_segmentor.*
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except Exception as e_safe:
        print(f"[WARN] weights_only=True load failed: {e_safe}")
        print("[INFO] Attempting trusted load with weights_only=False after making packages importable...")
        _ensure_src_on_path()
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Normalize to a state_dict
    model_state = None
    # Common MLflow structure: {'cfg': CfgNode, 'model': Module or dict, ...}
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            model_state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            model_state = ckpt["state_dict"]
        elif "model" in ckpt:
            model_obj = ckpt["model"]
            if hasattr(model_obj, "state_dict"):
                model_state = model_obj.state_dict()
            elif isinstance(model_obj, dict):
                model_state = model_obj
    # Fallbacks
    if model_state is None:
        if hasattr(ckpt, "state_dict"):
            model_state = ckpt.state_dict()
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            model_state = ckpt

    if model_state is None:
        raise RuntimeError("Could not resolve state_dict from checkpoint; please inspect checkpoint format.")

    # Extract only encoder + bottleneck keys from UnetUpSample: enc64/32/16/8 + conv
    encoder_parts = ["enc64", "enc32", "enc16", "enc8", "conv"]
    backbone_sd = {k: v for k, v in model_state.items() if any(k.startswith(part) for part in encoder_parts)}

    # Build backbone and load subset
    from landmarks_only_training.models.backbones.unet_wrapper import UNetBackbone

    backbone = UNetBackbone()
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing:
        print(f"[INFO] Missing keys in backbone load (expected for non-encoder layers): {missing}")
    if unexpected:
        print(f"[INFO] Unexpected keys ignored in backbone load: {unexpected}")

    print("[INFO] Freezing UNet encoder backbone weights.")
    backbone.freeze_backbone()
    return backbone





# def load_frozen_unet_segmentation_model(ckpt_path, cfg=None, device="cpu"):
#     """
#     Load a complete UnetUpSample_modified model from checkpoint and freeze it.
    
#     Similar to load_unet_encoder_backbone_from_ckpt but loads the FULL model
#     (encoder + decoder + seg heads) instead of just encoder.
    
#     Args:
#         ckpt_path: Path to checkpoint file
#         cfg: Config dict (unused, kept for API consistency)
#         device: Device to load model on
    
#     Returns:
#         Frozen UnetUpSample_modified model
#     """
#     # Allowlist yacs.config.CfgNode AND UnetUpSample_modified for safe weights_only=True loading
#     try:
#         import yacs.config
#         torch.serialization.add_safe_globals([yacs.config.CfgNode])
#     except Exception:
#         pass
    
#     # Add UnetUpSample_modified to safe globals
#     try:
#         from FINAL_TRAINING_SCRIPT.models.backbones.unet_wrapper import UnetUpSample_modified
#         torch.serialization.add_safe_globals([UnetUpSample_modified])
#     except ImportError:
#         try:
#             from models.backbones.unet_wrapper import UnetUpSample_modified
#             torch.serialization.add_safe_globals([UnetUpSample_modified])
#         except ImportError:
#             try:
#                 from uncertanity_trainig_script.models.backbones.unet_wrapper import UnetUpSample_modified
#                 torch.serialization.add_safe_globals([UnetUpSample_modified])
#             except ImportError:
#                 pass
    
#     # Also add eye_internal_segmentor version if it exists
#     try:
#         from eye_internal_segmentor.model.unet_wrapper import UnetUpSample_modified as UnetUpSample_eye
#         torch.serialization.add_safe_globals([UnetUpSample_eye])
#     except ImportError:
#         pass

#     ckpt = None
#     try:
#         ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
#     except Exception as e_safe:
#         print(f"[WARN] weights_only=True load failed: {e_safe}")
#         print("[INFO] Attempting trusted load with weights_only=False...")
#         _ensure_src_on_path()
#         ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

#     # Normalize to a state_dict
#     model_state = None
#     if isinstance(ckpt, dict):
#         if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
#             model_state = ckpt["model_state_dict"]
#         elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
#             model_state = ckpt["state_dict"]
#         elif "model" in ckpt:
#             model_obj = ckpt["model"]
#             if hasattr(model_obj, "state_dict"):
#                 model_state = model_obj.state_dict()
#             elif isinstance(model_obj, dict):
#                 model_state = model_obj
    
#     # Fallbacks
#     if model_state is None:
#         if hasattr(ckpt, "state_dict"):
#             model_state = ckpt.state_dict()
#         elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
#             model_state = ckpt

#     if model_state is None:
#         raise RuntimeError("Could not resolve state_dict from checkpoint; please inspect checkpoint format.")

#     # Build full UnetUpSample_modified model
#     try:
#         from FINAL_TRAINING_SCRIPT.models.backbones.unet_wrapper import UnetUpSample_modified
#     except ImportError:
#         try:
#             from models.backbones.unet_wrapper import UnetUpSample_modified
#         except ImportError:
#             from uncertanity_trainig_script.models.backbones.unet_wrapper import UnetUpSample_modified
    
#     model = UnetUpSample_modified().to(device)
    
#     # Load all weights
#     missing, unexpected = model.load_state_dict(model_state, strict=False)
#     if missing:
#         print(f"[WARN] Missing keys when loading full segmentation model: {missing}")
#     if unexpected:
#         print(f"[WARN] Unexpected keys when loading full segmentation model: {unexpected}")
    
#     # Freeze all parameters
#     for param in model.parameters():
#         param.requires_grad = False
#     model.eval()
    
#     print(f"[INFO] Loaded and froze complete UnetUpSample_modified model from: {ckpt_path}")
#     print(f"[INFO] Total frozen params: {sum(p.numel() for p in model.parameters())}")
    
#     return model



def load_frozen_unet_segmentation_model(ckpt_path, cfg=None, device="cpu"):
    """
    Load a complete UnetUpSample_modified model from checkpoint and freeze it.

    Args:
        ckpt_path: Path to checkpoint file
        cfg:       Unused, kept for API consistency
        device:    Device to load model on ("cpu" or "cuda")

    Returns:
        Frozen UnetUpSample_modified model in eval mode
    """
    import sys
    import types

    # ── 1. FAKE MODULE TRICK ──────────────────────────────────────────────────
    # The checkpoint was saved as torch.save({"cfg": cfg, "model": model})
    # where model was a UnetUpSample instance from the old codebase at path
    # "eye_internal_segmentor.model.unet_wrapper" — which no longer exists.
    # Fix: inject fake modules at those exact paths into sys.modules so pickle
    # finds them and redirects to our real local classes without touching disk.
    from Unet_training_script.models.backbones.unet_wrapper import (
        UnetDec,
        UnetEnc,
        Unet,
        UNetBackbone,
        UnetUpSampleDec,
        UnetUpSample,
        UnetUpSample_modified
    )

    fake_top   = types.ModuleType("eye_internal_segmentor")
    fake_model = types.ModuleType("eye_internal_segmentor.model")
    fake_unet  = types.ModuleType("eye_internal_segmentor.model.unet_wrapper")

    fake_unet.UnetDec               = UnetDec
    fake_unet.UnetEnc               = UnetEnc
    fake_unet.Unet                  = Unet
    fake_unet.UNetBackbone          = UNetBackbone
    fake_unet.UnetUpSampleDec       = UnetUpSampleDec
    fake_unet.UnetUpSample          = UnetUpSample
    fake_unet.UnetUpSample_modified = UnetUpSample_modified

    sys.modules["eye_internal_segmentor"]                    = fake_top
    sys.modules["eye_internal_segmentor.model"]              = fake_model
    sys.modules["eye_internal_segmentor.model.unet_wrapper"] = fake_unet

    # ── 2. LOAD CHECKPOINT ───────────────────────────────────────────────────
    # weights_only=False is required because the checkpoint contains a full
    # model object + yacs CfgNode (not just tensors), so pickle needs full access
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── 3. EXTRACT STATE DICT ────────────────────────────────────────────────
    # Checkpoint format is: {"cfg": <yacs CfgNode>, "model": <UnetUpSample instance>}
    # Confirmed by inspection of dyotana-model.pt
    model_state = ckpt["model"].state_dict()

    # ── 4. BUILD MODEL AND LOAD WEIGHTS ──────────────────────────────────────
    # Both UnetUpSample and UnetUpSample_modified have identical __init__ and
    # identical weight shapes — only forward() differs.
    # We use UnetUpSample_modified to get sigmoid on the blink output channel
    # (channel 5) instead of raw logits that UnetUpSample returns.
    model = UnetUpSample_modified().to(device)
    model.load_state_dict(model_state, strict=True)

    # ── 5. FREEZE ────────────────────────────────────────────────────────────
    # requires_grad=False → excludes all params from gradient computation
    # model.eval()        → disables Dropout + fixes BatchNorm running stats
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    print(f"[INFO] Loaded and froze UnetUpSample_modified from: {ckpt_path}")
    print(f"[INFO] Total frozen params: {sum(p.numel() for p in model.parameters())}")

    return model





def compute_fsr_torch(preds: torch.Tensor,
                      targets: torch.Tensor,
                      visibility: torch.Tensor = None,
                      threshold: float = 0.05) -> float:
    """
    Frame Success Rate (FSR) — fraction of frames where ALL landmarks
    are within threshold * eye_width of their GT positions.

    Args:
        preds:      [N, L, 2] predicted landmark coords in [0,1]
        targets:    [N, L, 2] GT landmark coords in [0,1]
        visibility: [N] optional — if provided, only visible frames are counted
        threshold:  success threshold as fraction of eye width (default 0.05 = 5%)

    Returns:
        FSR as float in [0, 1]
    """
    # Per-landmark L2 distance
    dist = torch.norm(preds - targets, p=2, dim=2)   # [N, L]

    # Normalize by GT eye width (x-span of landmarks) per frame
    target_xs  = targets[:, :, 0]                                        # [N, L]
    eye_width  = target_xs.max(dim=1)[0] - target_xs.min(dim=1)[0]       # [N]
    eye_width  = torch.clamp(eye_width, min=1e-6)

    norm_dist  = dist / eye_width.unsqueeze(1)                            # [N, L]

    # Frame succeeds only if ALL landmarks are within threshold
    max_error_per_frame = norm_dist.max(dim=1)[0]                         # [N]
    success_mask        = (max_error_per_frame <= threshold).float()      # [N]

    if visibility is not None:
        vis_mask    = (visibility > 0.5).float()
        total_valid = vis_mask.sum()
        if total_valid == 0:
            return 0.0
        return (success_mask * vis_mask).sum().item() / total_valid.item()

    return success_mask.mean().item()



def save_val_heatmaps(heatmaps: dict, epoch: int, save_dir: str, num_landmarks: int = 3):
    """
    Save predicted vs GT heatmap visualizations for the first val sample.
    Called once per epoch to visually track if the model is learning.

    Args:
        heatmaps:      dict with keys "pred" and "gt", each [L, H, W] cpu tensors
        epoch:         current epoch number (used for folder name)
        save_dir:      root directory to save under
        num_landmarks: how many landmarks to visualize (default first 3)
    """
    import matplotlib.pyplot as plt

    if heatmaps is None:
        return

    epoch_dir = os.path.join(save_dir, f"val_epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    pred_hm = heatmaps["pred"]   # [L, H, W]
    gt_hm   = heatmaps["gt"]     # [L, H, W]

    for j in range(min(num_landmarks, pred_hm.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(pred_hm[j])
        axes[0].set_title(f"Pred Heatmap — Epoch {epoch} Landmark {j}")

        axes[1].imshow(gt_hm[j])
        axes[1].set_title(f"GT Heatmap — Epoch {epoch} Landmark {j}")

        plt.tight_layout()
        plt.savefig(os.path.join(epoch_dir, f"landmark_{j}.png"))
        plt.close()