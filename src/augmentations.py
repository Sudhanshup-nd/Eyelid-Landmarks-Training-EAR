"""
Landmark-aware augmentations for eye crops.

Augmentations operate on the cropped eye image (PIL) before normalization.
Landmarks are expected in normalized [0,1] crop-local coordinates.

Supported ops (controlled by cfg['data']['augment']):
- random_horizontal_flip: probability in [0,1]
- random_vertical_flip:   probability in [0,1]
- random_rotation_deg:    max absolute rotation degrees (uniform in [-deg, +deg])
- occlusion_p:            probability to add a random rectangular black occluder
- blur_p:                 probability to apply slight Gaussian blur

Notes:
- All geometric transforms update landmarks to stay consistent with the image.
- Rotation is counter-clockwise around image center, matching PIL convention.
- Occlusion and blur are photometric only — landmarks are not changed.
"""

from typing import List, Tuple, Dict, Any
import random
import math
from PIL import Image, ImageFilter, ImageDraw


def _rand_bool(p: float) -> bool:
    return p > 0 and random.random() < p


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _rotate_point_norm(x: float, y: float, angle_rad: float) -> Tuple[float, float]:
    """
    Rotate normalized point (x,y) around center (0.5,0.5) by angle_rad.
    Uses image coordinate convention (y increases downward) to match PIL.rotate()
    which rotates counter-clockwise for positive angles.
    """
    cx, cy = 0.5, 0.5
    dx = x - cx
    dy = y - cy
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    # CCW rotation in y-down image coordinates
    rx = dx * cos_a + dy * sin_a
    ry = -dx * sin_a + dy * cos_a
    return (_clip01(cx + rx), _clip01(cy + ry))


def _horizontal_flip_norm(x: float, y: float) -> Tuple[float, float]:
    return (_clip01(1.0 - x), y)


def _vertical_flip_norm(x: float, y: float) -> Tuple[float, float]:
    return (x, _clip01(1.0 - y))


def _apply_random_occlusion(img: Image.Image) -> Image.Image:
    """Add a random black rectangle to the image — simulates partial occlusion."""
    w, h = img.size
    if w < 4 or h < 4:
        return img
    occ_w = random.randint(max(2, w // 8), max(3, w // 3))
    occ_h = random.randint(max(2, h // 8), max(3, h // 3))
    x1    = random.randint(0, max(0, w - occ_w))
    y1    = random.randint(0, max(0, h - occ_h))

    # Use scalar fill for grayscale images, RGB tuple for colour images
    fill_val = 0 if img.mode == 'L' else (0, 0, 0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([x1, y1, x1 + occ_w, y1 + occ_h], fill=fill_val)
    return img


def _apply_random_blur(img: Image.Image) -> Image.Image:
    """Apply slight Gaussian blur — simulates out-of-focus or motion blur."""
    return img.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))


def apply_augmentations(
    img: Image.Image,
    landmarks_norm: List[Tuple[float, float]],
    cfg: Dict[str, Any],
    is_train: bool
) -> Tuple[Image.Image, List[Tuple[float, float]]]:
    """
    Apply landmark-aware augmentations to a cropped eye image and its landmarks.

    Args:
        img:            PIL eye crop (any mode)
        landmarks_norm: list of (x,y) in [0,1] relative to the crop
        cfg:            training config dict
        is_train:       if False, returns inputs unchanged

    Returns:
        Augmented image and updated landmarks in [0,1]
    """
    if not is_train:
        return img, landmarks_norm

    aug = cfg.get("data", {}).get("augment", {}) or {}
    lmk = [(float(x), float(y)) for (x, y) in landmarks_norm]

    # ── Geometric transforms (image + landmarks updated together) ─────────────

    if _rand_bool(float(aug.get("random_horizontal_flip", 0.0))):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        lmk = [_horizontal_flip_norm(x, y) for (x, y) in lmk]

    if _rand_bool(float(aug.get("random_vertical_flip", 0.0))):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        lmk = [_vertical_flip_norm(x, y) for (x, y) in lmk]

    max_deg = float(aug.get("random_rotation_deg", 0.0))
    if max_deg > 0.0:
        angle     = random.uniform(-max_deg, max_deg)
        angle_rad = math.radians(angle)
        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        lmk = [_rotate_point_norm(x, y, angle_rad) for (x, y) in lmk]

    # ── Photometric transforms (landmarks unchanged) ───────────────────────────

    if _rand_bool(float(aug.get("occlusion_p", 0.0))):
        img = _apply_random_occlusion(img)

    if _rand_bool(float(aug.get("blur_p", 0.0))):
        img = _apply_random_blur(img)

    return img, lmk