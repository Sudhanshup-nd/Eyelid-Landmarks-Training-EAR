import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

from .augmentations import apply_augmentations


# ── DEBUG HELPER ──────────────────────────────────────────────────────────────
def _dbg(enabled: bool, msg: str):
    if enabled:
        print(f"[DATASET-DEBUG] {msg}")


class EyeDataset(Dataset):
    """
    Dataset for eye landmark detection.

    Each CSV row represents one eye from one face frame, containing:
      - path to full face image
      - eye bounding box (derived from landmark extents during data preparation)
      - 6 eyelid landmark coordinates in face pixel space
      - eye visibility flag

    Only rows with valid landmarks are used — invisible eye rows have no bbox
    and no landmarks, so there is genuinely nothing to train on from them.
    Invisible eye handling is done at inference time via a confidence filter.
    """

    def __init__(self, csv_path: str, cfg: Dict[str, Any], transform=None, is_train: bool = True):
        self.cfg        = cfg
        self.transform  = transform
        self.is_train   = is_train
        self.df         = pd.read_csv(csv_path)

        # Validate required columns exist before anything runs
        required = [
            "video_id", "frame_key", "eye_side", "eye_visibility",
            "path_to_dataset", "eye_bbox_face", "landmarks_coordinates_inside_eye_bbox"
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns: {missing}")

        dcfg               = cfg.get("data", {})
        self.num_landmarks = int(dcfg.get("num_landmarks", 6))
        self.image_size    = int(dcfg.get("image_size", 64))
        self.debug_enabled = bool(cfg.get("debug", {}).get("enabled", False))

        # Only keep rows that have valid landmark annotations.
        # Invisible eye rows have neither landmarks nor bbox — nothing to train on.
        self.valid_indices = [
            i for i in range(len(self.df))
            if self._has_landmarks(str(self.df.iloc[i].landmarks_coordinates_inside_eye_bbox))
        ]

        _dbg(self.debug_enabled,
             f"Loaded {len(self.df)} total rows → {len(self.valid_indices)} usable (have landmarks)")

    def __len__(self):
        return len(self.valid_indices)

    # ── PARSING HELPERS ───────────────────────────────────────────────────────

    def _has_landmarks(self, s: str) -> bool:
        """Check if a landmark string is non-empty and parseable."""
        s = s.strip().lower()
        if s in ("", "[]", "nan", "none", "null"):
            return False
        return "," in s

    def _parse_visibility(self, raw: Any) -> int:
        """
        Normalize visibility from any format → 0 or 1.
        CSV can contain: True/False (bool), 1/0 (int), "visible"/"invisible" (str).
        """
        if raw is None:
            return 0
        if isinstance(raw, (int, float)):
            return 0 if int(raw) == 0 else 1
        token = re.search(r"[a-z0-9]+", str(raw).strip().lower())
        token = token.group(0) if token else ""
        if token in ("true", "1", "yes", "y", "visible"):
            return 1
        if token in ("false", "0", "no", "n", "invisible"):
            return 0
        try:
            return 1 if int(token) != 0 else 0
        except Exception:
            return 0

    def _parse_bbox(self, raw: Any) -> Tuple[int, int, int, int]:
        """
        Parse bbox string 'x1,y1,x2,y2' → (x1, y1, x2, y2) as ints.
        Returns (0,0,0,0) if unparseable.
        """
        if not isinstance(raw, str) or raw.strip().lower() in ("", "nan", "none", "null"):
            return (0, 0, 0, 0)
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 4:
            return (0, 0, 0, 0)
        try:
            x1, y1, x2, y2 = map(int, map(float, parts))
            return (x1, y1, x2, y2)
        except Exception:
            return (0, 0, 0, 0)

    def _parse_landmarks(self, raw: Any) -> List[Tuple[float, float]]:
        """
        Parse landmark string 'x1,y1;x2,y2;...' → list of (x, y) in face pixel space.
        Returns empty list if unparseable.
        """
        if not isinstance(raw, str) or raw.strip().lower() in ("", "[]", "nan", "none", "null"):
            return []
        pts = []
        for token in raw.strip().split(";"):
            parts = token.strip().split(",")
            if len(parts) != 2:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        return pts

    # ── CROP HELPER ───────────────────────────────────────────────────────────

    def _crop_eye(self, img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop eye region from face image using bbox.
        Applies CLAHE equalization on the full face image before cropping
        so the eye region has better local contrast.
        """
        import cv2
        import numpy as np

        # Convert PIL → numpy grayscale for CLAHE
        img_gray = np.array(img.convert('L'))

        # Apply CLAHE — enhances local contrast without blowing out highlights
        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq    = clahe.apply(img_gray)

        # Convert back to PIL for the rest of the pipeline
        img = Image.fromarray(img_eq)

        # Now crop as usual
        w_face, h_face = img.size
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([max(0, min(x1, w_face - 1)), max(0, min(x2, w_face - 1))])
        y1, y2 = sorted([max(0, min(y1, h_face - 1)), max(0, min(y2, h_face - 1))])
        crop = img.crop((x1, y1, x2 + 1, y2 + 1))

        if crop.size != (self.image_size, self.image_size):

            crop = crop.resize((self.image_size, self.image_size), Image.BILINEAR)

        return crop



    # ── MAIN PIPELINE ─────────────────────────────────────────────────────────

    def __getitem__(self, index: int):
        row = self.df.iloc[self.valid_indices[index]]

        # ── 1. Parse visibility ────────────────────────────────────────────────
        visibility = self._parse_visibility(row.eye_visibility)

        # ── 2. Parse bbox and landmarks ───────────────────────────────────────
        bbox = self._parse_bbox(row.eye_bbox_face)
        pts  = self._parse_landmarks(row.landmarks_coordinates_inside_eye_bbox)

        # ── 3. Load face image and crop eye region ────────────────────────────
        if not os.path.exists(row.path_to_dataset):
            raise FileNotFoundError(f"Image missing: {row.path_to_dataset}")

        face_img = Image.open(row.path_to_dataset)
        eye_img  = self._crop_eye(face_img, bbox)

        # ── 4. Normalize landmarks to [0,1] relative to crop ──────────────────
        # Landmarks from CSV are in face pixel space.
        # Convert to crop-local coordinates then normalize to [0,1]:
        #   lx = (x - bbox_x1) / bbox_width
        #   ly = (y - bbox_y1) / bbox_height
        # Discard any landmark that falls outside the crop boundary.
        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1 + 1))
        bh = max(1.0, float(y2 - y1 + 1))

        landmarks_norm = []
        for (x, y) in pts:
            if x < x1 or x > x2 or y < y1 or y > y2:
                continue    # landmark outside crop — discard
            landmarks_norm.append(((x - x1) / bw, (y - y1) / bh))

        # ── 5. Apply augmentations (image + landmarks together) ───────────────
        # Augmentations operate on normalized [0,1] landmarks so flips/rotations
        # stay consistent between the image and its landmark coordinates.
        eye_img, landmarks_norm = apply_augmentations(
            eye_img, landmarks_norm, self.cfg, self.is_train
        )

        # ── 6. Build landmark tensor and mask ─────────────────────────────────
        # mask[i] = 1 → landmark i is valid (loss uses it)
        # mask[i] = 0 → landmark i missing or outside crop (loss ignores it)
        lm   = torch.zeros((self.num_landmarks, 2), dtype=torch.float32)
        mask = torch.zeros((self.num_landmarks,),   dtype=torch.float32)
        for i, (lx, ly) in enumerate(landmarks_norm[:self.num_landmarks]):
            lm[i]   = torch.tensor([lx, ly])
            mask[i] = 1.0

        # ── 7. Apply transforms (ToTensor + Normalize) ────────────────────────
        eye_img = self.transform(eye_img)
    

        _dbg(
            self.debug_enabled and index < 5,
            f"idx={index} frame={row.frame_key} eye={row.eye_side} "
            f"vis={visibility} landmarks_used={int(mask.sum().item())}/{self.num_landmarks}"
        )

        return {
            "image":      eye_img,                                          # [1, H, W]
            "landmarks":  lm,                                               # [num_landmarks, 2] normalized [0,1]
            "mask":       mask,                                             # [num_landmarks] 1=valid 0=ignore
            "visibility": torch.tensor([visibility], dtype=torch.float32), # [1] 0 or 1
            "bbox":       torch.tensor(bbox,         dtype=torch.float32), # [4] original face-space bbox
            "img_path":   row.path_to_dataset,
            "video_id":   row.video_id,
            "frame_key":  row.frame_key,
            "eye_side":   row.eye_side,
        }




def generate_gaussian_heatmaps(lm: torch.Tensor,
                               mask: torch.Tensor,
                               H: int,
                               W: int,
                               sigma: float = 1.5) -> torch.Tensor:
    """
    lm:   [B,L,2] normalized coords in [0,1]
    mask: [B,L]  1 if landmark valid, 0 otherwise
    Returns:
      gt_hm: [B,L,H,W]
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
            # 2D Gaussian
            g = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            gt_hm[b, i] = g

    # Optional: normalize each heatmap to sum to 1 for stability
  #  gt_hm = gt_hm / (gt_hm.view(B, L, -1).sum(dim=-1, keepdim=True).clamp(min=1e-6).view(B, L, 1, 1))
    return gt_hm  