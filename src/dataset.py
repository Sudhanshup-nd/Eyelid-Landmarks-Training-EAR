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


class MouthDataset(Dataset):
    """
    Dataset for mouth landmark detection.

    Each CSV row represents one mouth from one DMS frame, containing:
      - path to full DMS frame
      - face bounding box in DMS frame pixel space
      - mouth bounding box in face crop pixel space
      - 4 mouth landmark coordinates in face crop pixel space
      - mouth visibility flag (always 1 for now)

    Pipeline per sample:
      DMS frame
        → face crop  (via face_bbox_dms)
        → CLAHE equalization (grayscale)
        → mouth crop (via mouth_bbox_face)
        → landmark normalization to [0,1] inside mouth crop
        → augmentations
        → transform (ToTensor + Normalize)
    """

    def __init__(self, csv_path: str, cfg: Dict[str, Any], transform=None, is_train: bool = True):
        self.cfg        = cfg
        self.transform  = transform
        self.is_train   = is_train
        self.df         = pd.read_csv(csv_path)

        # Validate required columns
        required = [
            "video_id", "frame_key", "mouth_visibility",
            "path_to_dataset", "face_bbox_dms",
            "mouth_bbox_face", "landmarks_coordinates_inside_mouth_bbox"
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV {csv_path} missing columns: {missing}")

        dcfg               = cfg.get("data", {})
        self.num_landmarks = int(dcfg.get("num_landmarks", 4))
        self.image_size    = int(dcfg.get("image_size", 64))
        self.debug_enabled = bool(cfg.get("debug", {}).get("enabled", False))

        # Only keep rows that have valid landmark annotations
        self.valid_indices = [
            i for i in range(len(self.df))
            if self._has_landmarks(
                str(self.df.iloc[i].landmarks_coordinates_inside_mouth_bbox)
            )
        ]

        _dbg(self.debug_enabled,
             f"Loaded {len(self.df)} total rows → "
             f"{len(self.valid_indices)} usable (have landmarks)")

    def __len__(self):
        return len(self.valid_indices)

    # ── PARSING HELPERS ───────────────────────────────────────────────────────

    def _has_landmarks(self, s: str) -> bool:
        s = s.strip().lower()
        if s in ("", "[]", "nan", "none", "null"):
            return False
        return "," in s

    def _parse_visibility(self, raw: Any) -> int:
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

    # ── CROP HELPERS ──────────────────────────────────────────────────────────

    def _extract_face_crop(self, dms_img: Image.Image,
                           face_bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Crop face region from full DMS frame using face_bbox_dms.
        """
        w_dms, h_dms = dms_img.size
        x1, y1, x2, y2 = face_bbox
        x1, x2 = sorted([max(0, min(x1, w_dms-1)), max(0, min(x2, w_dms-1))])
        y1, y2 = sorted([max(0, min(y1, h_dms-1)), max(0, min(y2, h_dms-1))])
        return dms_img.crop((x1, y1, x2+1, y2+1))

    def _crop_mouth(self, face_img: Image.Image,
                    mouth_bbox: Tuple[int, int, int, int]) -> Image.Image:
        """
        Apply CLAHE on face crop (grayscale), then crop mouth region
        and resize to image_size × image_size.
        """
        import cv2
        import numpy as np

        # PIL → numpy grayscale → CLAHE
        img_gray  = np.array(face_img.convert('L'))
        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_eq    = clahe.apply(img_gray)

        # Back to PIL
        face_eq   = Image.fromarray(img_eq)

        # Crop mouth region
        w_face, h_face = face_eq.size
        x1, y1, x2, y2 = mouth_bbox
        x1, x2 = sorted([max(0, min(x1, w_face-1)), max(0, min(x2, w_face-1))])
        y1, y2 = sorted([max(0, min(y1, h_face-1)), max(0, min(y2, h_face-1))])
        crop = face_eq.crop((x1, y1, x2+1, y2+1))

        if crop.size != (self.image_size, self.image_size):
            crop = crop.resize((self.image_size, self.image_size), Image.BILINEAR)

        return crop

    # ── MAIN PIPELINE ───────────────────────────────────────────���─────────────

    def __getitem__(self, index: int):
        row = self.df.iloc[self.valid_indices[index]]

        # ── 1. Parse visibility ────────────────────────────────────────────────
        visibility = self._parse_visibility(row.mouth_visibility)

        # ── 2. Parse bboxes and landmarks ─────────────────────────────────────
        face_bbox  = self._parse_bbox(row.face_bbox_dms)
        mouth_bbox = self._parse_bbox(row.mouth_bbox_face)
        pts        = self._parse_landmarks(
                         row.landmarks_coordinates_inside_mouth_bbox
                     )
        
        # ── 2a. Clamp mouth bbox to face crop bounds ────────���────────────────
        # face_bbox is (fx1, fy1, fx2, fy2): describes face crop region, so mouth_bbox is relative within that crop
        fw = face_bbox[2] - face_bbox[0]
        fh = face_bbox[3] - face_bbox[1]
        mx1, my1, mx2, my2 = mouth_bbox
        mx1 = max(0, min(mx1, fw-1))
        mx2 = max(0, min(mx2, fw-1))
        my1 = max(0, min(my1, fh-1))
        my2 = max(0, min(my2, fh-1))
        mx1, mx2 = sorted([mx1, mx2])
        my1, my2 = sorted([my1, my2])
        mouth_bbox = (mx1, my1, mx2, my2)

        # ── 3. Load DMS frame → face crop → mouth crop ────────────────────────
        if not os.path.exists(row.path_to_dataset):
            raise FileNotFoundError(f"Image missing: {row.path_to_dataset}")

        dms_img   = Image.open(row.path_to_dataset)
        face_img  = self._extract_face_crop(dms_img, face_bbox)
        mouth_img = self._crop_mouth(face_img, mouth_bbox)

        # ── 4. Normalize landmarks to [0,1] relative to mouth crop ────────────
        # Landmarks in CSV are in face crop pixel space.
        # Normalize inside mouth_bbox:
        #   lx = (x - mouth_x1) / mouth_bbox_width
        #   ly = (y - mouth_y1) / mouth_bbox_height
        x1, y1, x2, y2 = mouth_bbox
        bw = max(1.0, float(x2 - x1 + 1))
        bh = max(1.0, float(y2 - y1 + 1))

        landmarks_norm = []
        for (x, y) in pts:
            if x < x1 or x > x2 or y < y1 or y > y2:
                continue    # landmark outside mouth crop — discard
            landmarks_norm.append(((x - x1) / bw, (y - y1) / bh))

        # ── 5. Apply augmentations (image + landmarks together) ───────────────
        # mouth_img, landmarks_norm = apply_augmentations(
        #     mouth_img, landmarks_norm, self.cfg, self.is_train
        # )

        # ── 6. Build landmark tensor and mask ─────────────────────────────────
        lm   = torch.zeros((self.num_landmarks, 2), dtype=torch.float32)
        mask = torch.zeros((self.num_landmarks,),   dtype=torch.float32)
        for i, (lx, ly) in enumerate(landmarks_norm[:self.num_landmarks]):
            lm[i]   = torch.tensor([lx, ly])
            mask[i] = 1.0

        # ── 7. Apply transforms (ToTensor + Normalize) ────────────────────────
        mouth_img = self.transform(mouth_img)

        _dbg(
            self.debug_enabled and index < 5,
            f"idx={index} frame={row.frame_key} "
            f"vis={visibility} landmarks_used={int(mask.sum().item())}/{self.num_landmarks}"
        )

        return {
            "image":      mouth_img,
            "landmarks":  lm,
            "mask":       mask,
            "visibility": torch.tensor([visibility], dtype=torch.float32),
            "bbox":       torch.tensor(mouth_bbox,  dtype=torch.float32),
            "face_bbox":  torch.tensor(face_bbox,   dtype=torch.float32),  # ← add this
            "img_path":   row.path_to_dataset,
            "video_id":   row.video_id,
            "frame_key":  row.frame_key,
        }


# ── GAUSSIAN HEATMAP GENERATOR ────────────────────────────────────────────────
def generate_gaussian_heatmaps(lm: torch.Tensor,
                               mask: torch.Tensor,
                               H: int,
                               W: int,
                               sigma: float = 1.5) -> torch.Tensor:
    """
    lm:   [B, L, 2] normalized coords in [0,1]
    mask: [B, L]    1 if landmark valid, 0 otherwise
    Returns:
      gt_hm: [B, L, H, W]
    """
    device  = lm.device
    B, L, _ = lm.shape

    ys = torch.arange(0, H, device=device).float()
    xs = torch.arange(0, W, device=device).float()
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')   # [H, W]

    gt_hm = torch.zeros(B, L, H, W, device=device)

    for b in range(B):
        for i in range(L):
            if mask[b, i] <= 0:
                continue
            x_norm, y_norm = lm[b, i]
            cx = x_norm * (W - 1)
            cy = y_norm * (H - 1)
            g  = torch.exp(-((xx - cx)**2 + (yy - cy)**2) / (2 * sigma**2))
            gt_hm[b, i] = g

    return gt_hm