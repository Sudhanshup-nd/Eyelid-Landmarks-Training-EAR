#!/usr/bin/env python3
"""
Read eye landmark CSV and create:
  1. Eye crop images (cropped using eye_bbox_face)
  2. (Optional) full face image with eye bbox + landmarks

This UPDATED version removes:
  - Landmark index numbers (no text drawn)
  - Connecting lines between landmarks (only circles)

CSV columns expected:
video_id,frame_key,eye_side,eye_visibility,path_to_dataset,eye_bbox_face,landmarks_coordinates_inside_eye_bbox
"""

import os
import csv
import sys
import math
import traceback
from typing import List, Tuple, Optional

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV (cv2) not installed. Install with: pip install opencv-python")
    sys.exit(1)

# =============================================================================
# CONFIGURATION
# =============================================================================
CSV_PATH = "/inwdata2a/sudhanshu/Unet_training_script/data-and-labelles/annotation-phase-2-random-padded.csv"
OUTPUT_DIR = "/inwdata2a/sudhanshu/Unet_training_script/data-and-labelles"

SAVE_PER_VIDEO_SUBDIR = True            # Organize outputs per video_id
ROW_LIMIT: Optional[int] = 100          # None to process all rows
SAVE_FACE_WITH_OVERLAY = True           # Also save full face image with bbox + landmarks

# Drawing settings (no numbers, no lines)
LEFT_EYE_COLOR = (0, 255, 0)            # BGR green
RIGHT_EYE_COLOR = (0, 128, 255)         # BGR orange
BBOX_COLOR = (255, 0, 0)                # BGR blue
LANDMARK_RADIUS = 3
LANDMARK_THICKNESS = -1                 # Filled circle
BBOX_THICKNESS = 2

# Behavior controls
DEBUG = True
PRINT_EVERY = 50
SKIP_INCOMPLETE_ROWS = True             # Skip if bbox or landmarks missing
# =============================================================================


def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def parse_bbox(bbox_str: str) -> Optional[Tuple[int, int, int, int]]:
    if not bbox_str:
        return None
    parts = bbox_str.split(",")
    if len(parts) != 4:
        debug(f"Invalid bbox string (wrong parts): {bbox_str}")
        return None
    try:
        xmin, ymin, xmax, ymax = map(int, parts)
    except ValueError as e:
        debug(f"Invalid bbox string (non-int): {bbox_str} err={e}")
        return None
    if xmin > xmax or ymin > ymax:
        debug(f"Invalid bbox ordering: {bbox_str}")
        return None
    return xmin, ymin, xmax, ymax


def parse_landmarks(landmark_str: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not landmark_str:
        return pts
    for seg in landmark_str.split(";"):
        seg = seg.strip()
        if not seg:
            continue
        try:
            x_str, y_str = seg.split(",")
            pts.append((float(x_str), float(y_str)))
        except Exception as e:
            debug(f"Failed to parse landmark '{seg}' in '{landmark_str}': {e}")
    return pts


def draw_landmarks_on_crop(crop_img, landmarks_crop_coords: List[Tuple[float, float]], eye_side: str):
    """
    Draw only landmark circles on the crop (no indices, no connecting lines).
    """
    color = LEFT_EYE_COLOR if eye_side == "left" else RIGHT_EYE_COLOR
    for (x, y) in landmarks_crop_coords:
        cv2.circle(crop_img, (int(round(x)), int(round(y))), LANDMARK_RADIUS, color, LANDMARK_THICKNESS)
    return crop_img


def draw_on_face(face_img, bbox: Tuple[int, int, int, int],
                 landmarks_abs: List[Tuple[float, float]],
                 eye_side: str):
    xmin, ymin, xmax, ymax = bbox
    # Draw bbox
    cv2.rectangle(face_img, (xmin, ymin), (xmax, ymax), BBOX_COLOR, BBOX_THICKNESS)
    color = LEFT_EYE_COLOR if eye_side == "left" else RIGHT_EYE_COLOR
    # Draw only circles (no indices, no lines)
    for (x, y) in landmarks_abs:
        cv2.circle(face_img, (int(round(x)), int(round(y))), LANDMARK_RADIUS, color, LANDMARK_THICKNESS)
    return face_img


def process_row(row: dict) -> Optional[Tuple[str, Optional[str]]]:
    """
    Process a CSV row: load image, parse bbox & landmarks, create crop with landmarks.
    Returns (crop_output_path, face_output_path or None)
    """
    video_id = row.get("video_id", "").strip()
    frame_key = row.get("frame_key", "").strip()
    eye_side = row.get("eye_side", "").strip()
    visibility = row.get("eye_visibility", "").strip()
    img_path = row.get("path_to_dataset", "").strip()
    bbox_str = row.get("eye_bbox_face", "").strip()
    lm_str = row.get("landmarks_coordinates_inside_eye_bbox", "").strip()

    if not os.path.isfile(img_path):
        debug(f"Image file not found, skipping: {img_path}")
        return None

    bbox = parse_bbox(bbox_str)
    landmarks_abs = parse_landmarks(lm_str)

    if (bbox is None or len(landmarks_abs) == -1) and SKIP_INCOMPLETE_ROWS:
        debug(f"Skipping incomplete row (bbox or landmarks missing): frame_key={frame_key}, eye_side={eye_side}")
        return None

    face_img = cv2.imread(img_path)
    if face_img is None:
        debug(f"Failed to read image: {img_path}")
        return None

    # Approximate bbox if missing and landmarks present
    if bbox is None and landmarks_abs:
        xs = [p[0] for p in landmarks_abs]
        ys = [p[1] for p in landmarks_abs]
        xmin = int(math.floor(min(xs)))
        xmax = int(math.ceil(max(xs)))
        ymin = int(math.floor(min(ys)))
        ymax = int(math.ceil(max(ys)))
        bbox = (xmin, ymin, xmax, ymax)
        debug(f"Approximated bbox from landmarks for frame_key={frame_key}, eye_side={eye_side}: {bbox}")

    if bbox is None:
        debug("No bbox available; skipping row.")
        return None

    xmin, ymin, xmax, ymax = bbox
    h, w = face_img.shape[:2]
    xmin = max(0, min(xmin, w - 1))
    xmax = max(0, min(xmax, w - 1))
    ymin = max(0, min(ymin, h - 1))
    ymax = max(0, min(ymax, h - 1))

    crop_img = face_img[ymin:ymax + 1, xmin:xmax + 1].copy()
    if crop_img.size == 0:
        debug(f"Empty crop for frame_key={frame_key}, bbox={bbox}; skipping.")
        return None

    landmarks_crop_coords = [(x - xmin, y - ymin) for (x, y) in landmarks_abs]
    crop_img = draw_landmarks_on_crop(crop_img, landmarks_crop_coords, eye_side)

    # Build output dirs
    if SAVE_PER_VIDEO_SUBDIR:
        crop_dir = os.path.join(OUTPUT_DIR, video_id, "eye_crops")
        face_dir = os.path.join(OUTPUT_DIR, video_id, "face_overlay") if SAVE_FACE_WITH_OVERLAY else None
    else:
        crop_dir = os.path.join(OUTPUT_DIR, "eye_crops")
        face_dir = os.path.join(OUTPUT_DIR, "face_overlay") if SAVE_FACE_WITH_OVERLAY else None

    os.makedirs(crop_dir, exist_ok=True)
    if SAVE_FACE_WITH_OVERLAY and face_dir:
        os.makedirs(face_dir, exist_ok=True)

    crop_filename = f"{frame_key}_{eye_side}_vis-{visibility}.png"
    crop_output_path = os.path.join(crop_dir, crop_filename)
    cv2.imwrite(crop_output_path, crop_img)

    face_output_path = None
    if SAVE_FACE_WITH_OVERLAY:
        face_img_overlay = face_img.copy()
        face_img_overlay = draw_on_face(face_img_overlay, (xmin, ymin, xmax, ymax), landmarks_abs, eye_side)
        face_filename = f"{frame_key}_{eye_side}_face_overlay_vis-{visibility}.png"
        face_output_path = os.path.join(face_dir, face_filename)
        cv2.imwrite(face_output_path, face_img_overlay)

    return crop_output_path, face_output_path


def main():
    print("=== Eye Crop Generation (No numbers, no lines) ===")
    print(f"CSV_PATH: {CSV_PATH}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"ROW_LIMIT: {ROW_LIMIT}")
    print(f"SAVE_FACE_WITH_OVERLAY: {SAVE_FACE_WITH_OVERLAY}")

    if not os.path.isfile(CSV_PATH):
        print(f"ERROR: CSV not found: {CSV_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_rows = 0
    processed = 0
    saved_crops = 0
    saved_faces = 0

    try:
        with open(CSV_PATH, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                if ROW_LIMIT is not None and processed >= ROW_LIMIT:
                    break
                res = process_row(row)
                if res is not None:
                    crop_path, face_path = res
                    saved_crops += 1
                    if face_path:
                        saved_faces += 1
                processed += 1
                if processed % PRINT_EVERY == 0:
                    debug(f"Progress: processed={processed}, crops={saved_crops}, faces={saved_faces}")
    except Exception as e:
        print("FATAL: error while processing CSV.")
        traceback.print_exc()
        sys.exit(2)

    print("\n=== SUMMARY ===")
    print(f"CSV rows read: {total_rows}")
    print(f"Rows processed: {processed}")
    print(f"Eye crops saved: {saved_crops}")
    if SAVE_FACE_WITH_OVERLAY:
        print(f"Face overlays saved: {saved_faces}")
    print(f"Output root: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()