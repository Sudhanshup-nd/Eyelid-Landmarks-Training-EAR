from typing import Dict, Any
from PIL import Image
import torch
from torchvision import transforms
from .utils import debug  

class ToTensorContiguous:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def __call__(self, img: Image.Image) -> torch.Tensor:     # if a class defines a __call__(self, ...) method, instances of that class can be used like functions.
        if self.verbose:
            debug(f"ToTensor input size={img.size}, mode={img.mode}", True)
        
        # Always convert to grayscale if not already
        if img.mode != 'L':
            img = img.convert('L')
        
        t = transforms.ToTensor()(img).contiguous()
        
        if self.verbose:
            debug(f"ToTensor output shape={tuple(t.shape)}, min={t.min():.3f}, max={t.max():.3f}", True)
        return t

def build_train_transforms(cfg: Dict[str, Any]):
    """
    Build training transforms for single-channel (grayscale) model.
    Only photometric aug (if any), grayscale conversion, and normalization are applied.
    """
    data_cfg = cfg.get("data", {})
    dbg_cfg = cfg.get("debug", {})
    verbose = bool(dbg_cfg.get("verbose_transforms", False))

    tf_list = []
    
  
    tf_list.append(ToTensorContiguous(verbose=verbose))

    # Normalization for grayscale (1 channel)
    if "mean" in data_cfg and "std" in data_cfg:
        tf_list.append(transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]))

    if verbose:
        print(f"[TRANSFORM] Train: normalize={'mean' in data_cfg and 'std' in data_cfg}")
    
    return transforms.Compose(tf_list)

def build_val_transforms(cfg: Dict[str, Any]):
    """
    Validation transforms: grayscale tensor + normalize.
    """
    data_cfg = cfg.get("data", {})
    dbg_cfg = cfg.get("debug", {})
    verbose = bool(dbg_cfg.get("verbose_transforms", False))

    tf_list = [ToTensorContiguous(verbose=verbose)]

    if "mean" in data_cfg and "std" in data_cfg:
        tf_list.append(transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]))

    return transforms.Compose(tf_list)







