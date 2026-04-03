# #!/usr/bin/env python3
# """
# Same as previous CSV generator, but makes EYE BBOXES SQUARE.

# Only change from your working script:
# - The bbox computation now expands to a square whose side = max(width, height),
#   then applies the padding ratio uniformly to that side, and clamps to image bounds.

# All other behavior (columns, logging, etc.) is unchanged.

# If you want to revert to rectangular, just replace the square_bbox_with_padding()
# call with your original bbox_with_padding().

# Configuration section remains the same.
# """

# import csv
# import os
# import sys
# import traceback
# import xml.etree.ElementTree as ET
# from typing import List, Tuple, Dict

# # =============================================================================
# # CONFIGURATION (EDIT THESE PATHS / FLAGS)
# # =============================================================================
# XML_PATH = "/inwdata2a/sudhanshu/data-and-labels/annotations_phase_2.xml"   # Path to CVAT export XML
# ROOT_DIR = "/inwdata2a/sudhanshu/video-path-creation/phase-2-face-crops_for_labelling"      # Root dir containing video_id subfolders
# OUT_CSV = "/inwdata2a/sudhanshu/data-and-labels/annotation-phase-2.csv"  # Destination CSV
# PADDING_RATIO = 0.5                 # Fractional padding added to the SQUARE side length
# SKIP_MISSING_EYE = False            # If True, skip row entirely when an eye lacks landmarks
# EXPECTED_POINTS_PER_EYE = 6         # For validation; just warns if mismatch
# DEBUG = True                        # Toggle verbose debug
# SHOW_SAMPLE_ROWS = 5                # Number of sample rows to print after processing
# # =============================================================================

# LEFT_LANDMARK_LABEL = "left-eye-landmarks"
# RIGHT_LANDMARK_LABEL = "right-eye-landmarks"
# LEFT_VIS_LABEL = "left-eye-visibility"
# RIGHT_VIS_LABEL = "right-eye-visibility"


# def debug(msg: str):
#     if DEBUG:
#         print(f"[DEBUG] {msg}")


# def parse_points(points_str: str) -> List[Tuple[float, float]]:
#     pts: List[Tuple[float, float]] = []
#     if not points_str:
#         return pts
#     for raw_pair in points_str.strip().split(";"):
#         pair = raw_pair.strip()
#         if not pair:
#             continue
#         try:
#             x_str, y_str = pair.split(",")
#             pts.append((float(x_str), float(y_str)))
#         except Exception as e:
#             debug(f"Failed parsing point '{pair}' in '{points_str}': {e}")
#     return pts


# def square_bbox_with_padding(
#     pts: List[Tuple[float, float]],
#     img_w: int,
#     img_h: int,
#     padding_ratio: float
# ) -> Tuple[int, int, int, int]:
#     """
#     Create a square bbox around the landmarks with uniform padding.

#     Steps:
#       1. Get min/max x,y.
#       2. side = max(width, height).
#       3. padded_side = side * (1 + 2*padding_ratio)  (padding on both sides)
#       4. Center square around the original bbox center.
#       5. Clamp to image boundaries.
#     """
#     xs = [p[0] for p in pts]
#     ys = [p[1] for p in pts]
#     xmin = min(xs)
#     xmax = max(xs)
#     ymin = min(ys)
#     ymax = max(ys)

#     width = xmax - xmin
#     height = ymax - ymin
#     side = max(width, height)
#     # Ensure non-zero
#     if side == 0:
#         side = 1.0

#     # Apply padding (padding_ratio is fraction of original side per side)
#     padded_side = side * (1 + 2 * padding_ratio)

#     # Center coordinates
#     cx = (xmin + xmax) / 2.0
#     cy = (ymin + ymax) / 2.0

#     half = padded_side / 2.0
#     sq_xmin = int(round(cx - half))
#     sq_xmax = int(round(cx + half))
#     sq_ymin = int(round(cy - half))
#     sq_ymax = int(round(cy + half))

#     # Clamp to image boundaries
#     # Horizontal shift if out of bounds
#     if sq_xmin < 0:
#         shift = -sq_xmin
#         sq_xmin += shift
#         sq_xmax += shift
#     if sq_xmax >= img_w:
#         shift = sq_xmax - (img_w - 1)
#         sq_xmin -= shift
#         sq_xmax -= shift
#     # Vertical shift
#     if sq_ymin < 0:
#         shift = -sq_ymin
#         sq_ymin += shift
#         sq_ymax += shift
#     if sq_ymax >= img_h:
#         shift = sq_ymax - (img_h - 1)
#         sq_ymin -= shift
#         sq_ymax -= shift

#     # Final clamp
#     sq_xmin = max(0, sq_xmin)
#     sq_ymin = max(0, sq_ymin)
#     sq_xmax = min(img_w - 1, sq_xmax)
#     sq_ymax = min(img_h - 1, sq_ymax)

#     # Guarantee ordering
#     if sq_xmin > sq_xmax:
#         sq_xmin, sq_xmax = sq_xmax, sq_xmin
#     if sq_ymin > sq_ymax:
#         sq_ymin, sq_ymax = sq_ymax, sq_ymin

#     return sq_xmin, sq_ymin, sq_xmax, sq_ymax


# def extract_video_id_and_frame_key(image_name: str) -> Tuple[str, str]:
#     parts = image_name.strip().split("/")
#     if len(parts) < 3:
#         debug(f"Unexpected image name format (using fallback): {image_name}")
#         return ("UNKNOWN_VIDEO", os.path.splitext(os.path.basename(image_name))[0])
#     video_id = parts[-2]
#     frame_key = os.path.splitext(parts[-1])[0]
#     return video_id, frame_key


# def build_image_path(root_dir: str, video_id: str, frame_file: str) -> str:
#     return os.path.join(root_dir, video_id, frame_file)


# def collect_visibility_tags(image_elem: ET.Element) -> Dict[str, str]:
#     vis: Dict[str, str] = {}
#     for tag in image_elem.findall("tag"):
#         label = tag.get("label", "")
#         if label in (LEFT_VIS_LABEL, RIGHT_VIS_LABEL):
#             attr = tag.find("attribute[@name='is-visible']")
#             vis[label] = (attr.text.strip() if (attr is not None and attr.text) else "")
#     return vis


# def validate_point_count(side: str, pts: List[Tuple[float, float]]):
#     if EXPECTED_POINTS_PER_EYE is not None and len(pts) != EXPECTED_POINTS_PER_EYE:
#         debug(f"WARNING: {side} eye landmark count {len(pts)} != expected {EXPECTED_POINTS_PER_EYE}")


# def generate_rows(xml_path: str) -> List[Dict[str, str]]:
#     debug(f"Starting XML parse: {xml_path}")
#     if not os.path.isfile(xml_path):
#         raise FileNotFoundError(f"XML file not found: {xml_path}")

#     try:
#         tree = ET.parse(xml_path)
#     except Exception:
#         debug("XML parsing failed; FULL TRACEBACK:")
#         traceback.print_exc()
#         raise

#     root = tree.getroot()
#     images = root.findall("image")
#     debug(f"Discovered {len(images)} <image> elements.")

#     rows: List[Dict[str, str]] = []
#     total_images = 0
#     missing_left = 0
#     missing_right = 0

#     for image in images:
#         total_images += 1
#         image_name = image.get("name", "")
#         if not image_name:
#             debug("Skipping <image> without 'name' attribute.")
#             continue

#         try:
#             img_w = int(image.get("width", "0"))
#             img_h = int(image.get("height", "0"))
#         except ValueError:
#             debug(f"Invalid width/height for image '{image_name}' - skipping.")
#             continue

#         video_id, frame_key = extract_video_id_and_frame_key(image_name)
#         frame_file = os.path.basename(image_name)
#         path_to_dataset = build_image_path(ROOT_DIR, video_id, frame_file)

#         if not os.path.isfile(path_to_dataset):
#             debug(f"NOTE: Image path does not exist: {path_to_dataset}")

#         landmarks_map: Dict[str, List[Tuple[float, float]]] = {}
#         raw_points_str_map: Dict[str, str] = {}

#         for pe in image.findall("points"):
#             label = pe.get("label", "")
#             pts_str = pe.get("points", "")
#             if not label or not pts_str:
#                 continue
#             pts = parse_points(pts_str)
#             if not pts:
#                 debug(f"No valid points for label '{label}' in '{image_name}'")
#                 continue
#             landmarks_map[label] = pts
#             raw_points_str_map[label] = pts_str

#         visibility_map = collect_visibility_tags(image)

#         for label, side in [(LEFT_LANDMARK_LABEL, "left"), (RIGHT_LANDMARK_LABEL, "right")]:
#             pts = landmarks_map.get(label)
#             if pts is None:
#                 if side == "left":
#                     missing_left += 1
#                 else:
#                     missing_right += 1

#                 if SKIP_MISSING_EYE:
#                     debug(f"Skipping {side} eye for frame '{frame_key}' (no landmarks).")
#                     continue

#                 row = {
#                     "video_id": video_id,
#                     "frame_key": frame_key,
#                     "eye_side": side,
#                     "eye_visibility": visibility_map.get(
#                         LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
#                     ),
#                     "path_to_dataset": path_to_dataset,
#                     "eye_bbox_face": "",
#                     "landmarks_coordinates_inside_eye_bbox": ""
#                 }
#                 rows.append(row)
#                 debug(f"Appended placeholder row for missing {side} eye: frame_key={frame_key}")
#                 continue

#             validate_point_count(side, pts)

#             try:
#                 xmin, ymin, xmax, ymax = square_bbox_with_padding(pts, img_w, img_h, PADDING_RATIO)
#                 bbox_str = f"{xmin},{ymin},{xmax},{ymax}"
#             except Exception as e:
#                 debug(f"ERROR building square bbox for {side} eye in '{image_name}': {e}")
#                 bbox_str = ""

#             visibility = visibility_map.get(
#                 LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
#             )

#             row = {
#                 "video_id": video_id,
#                 "frame_key": frame_key,
#                 "eye_side": side,
#                 "eye_visibility": visibility,
#                 "path_to_dataset": path_to_dataset,
#                 "eye_bbox_face": bbox_str,
#                 "landmarks_coordinates_inside_eye_bbox": raw_points_str_map.get(label, "")
#             }
#             rows.append(row)
#             debug(f"Appended {side} eye row (SQUARE): frame_key={frame_key}, bbox={bbox_str}, visibility={visibility}")

#     debug("----- SUMMARY -----")
#     debug(f"Total <image> elements processed: {total_images}")
#     debug(f"Rows generated: {len(rows)} (aim: 2 per image minus skipped/missing eyes)")
#     debug(f"Missing left-eye landmarks:  {missing_left}")
#     debug(f"Missing right-eye landmarks: {missing_right}")
#     debug("-------------------")

#     return rows


# def write_csv(rows: List[Dict[str, str]], out_csv: str):
#     fieldnames = [
#         "video_id",
#         "frame_key",
#         "eye_side",
#         "eye_visibility",
#         "path_to_dataset",
#         "eye_bbox_face",
#         "landmarks_coordinates_inside_eye_bbox"
#     ]

#     out_dir = os.path.dirname(os.path.abspath(out_csv))
#     debug(f"Ensuring output directory exists: {out_dir}")

#     try:
#         os.makedirs(out_dir, exist_ok=True)
#     except PermissionError:
#         debug(f"Permission denied creating directory '{out_dir}'.")
#         raise
#     except Exception as e:
#         debug(f"Unexpected directory creation error for '{out_dir}': {e}")
#         raise

#     debug(f"Writing CSV to: {out_csv}")
#     try:
#         with open(out_csv, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=fieldnames)
#             writer.writeheader()
#             for r in rows:
#                 writer.writerow(r)
#     except PermissionError:
#         debug(f"Permission denied writing file '{out_csv}'.")
#         raise
#     except Exception as e:
#         debug(f"Unexpected error writing CSV '{out_csv}': {e}")
#         raise

#     debug("CSV write complete.")


# def main():
#     print("=== Eye Landmark CSV Generation (SQUARE BBOX) ===")
#     print(f"XML_PATH:          {XML_PATH}")
#     print(f"ROOT_DIR:          {ROOT_DIR}")
#     print(f"OUT_CSV:           {OUT_CSV}")
#     print(f"PADDING_RATIO:     {PADDING_RATIO}")
#     print(f"SKIP_MISSING_EYE:  {SKIP_MISSING_EYE}")
#     print(f"EXPECTED_POINTS:   {EXPECTED_POINTS_PER_EYE}")
#     print(f"DEBUG:             {DEBUG}")

#     try:
#         rows = generate_rows(XML_PATH)
#     except Exception as e:
#         print("FATAL: Error during row generation.")
#         traceback.print_exc()
#         sys.exit(1)

#     if rows:
#         print(f"\nSample {min(SHOW_SAMPLE_ROWS, len(rows))} rows:")
#         for sample in rows[:SHOW_SAMPLE_ROWS]:
#             print(sample)
#     else:
#         print("No rows generated (check XML, labels, paths).")

#     try:
#         write_csv(rows, OUT_CSV)
#     except Exception as e:
#         print("FATAL: Error writing CSV.")
#         traceback.print_exc()
#         sys.exit(2)

#     print(f"\nSUCCESS: Wrote {len(rows)} rows to {OUT_CSV}")
#     print("Each eye bbox is now square (with padding).")


# if __name__ == "__main__":
#     main()








#!/usr/bin/env python3
"""
CSV generator for eye crops with SQUARE bboxes, now with RANDOMIZED padding and randomized
bbox center (while still guaranteeing all landmarks remain inside the crop and the crop
stays within image bounds).

Key changes:
- Compute square side from landmarks as before: side = max(width, height).
- Apply a randomized padding ratio per sample: padding_ratio is sampled uniformly from
  [PADDING_RATIO_MIN, PADDING_RATIO_MAX].
- Randomize the square bbox center within the allowable range that keeps:
    a) all landmarks inside the bbox
    b) the bbox within image bounds
  This ensures the eyes are not always at the center of the crop.

If you want deterministic padding (no randomization), set PADDING_RATIO_MIN == PADDING_RATIO_MAX
and set RANDOMIZE_CENTER = False.

All other behavior (columns, logging, etc.) remains unchanged.
"""

import csv
import os
import sys
import random
import traceback
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

# =============================================================================
# CONFIGURATION (EDIT THESE PATHS / FLAGS)
# =============================================================================
XML_PATH = "/inwdata2a/sudhanshu/Unet_training_script/data-and-labelles/annotations_phase_2.xml"   # Path to CVAT export XML
ROOT_DIR = "/inwdata2a/sudhanshu/Unet_training_script/data-and-labelles/phase-2-face-crops-for-labelling"      # Root dir containing video_id subfolders
OUT_CSV = "/inwdata2a/sudhanshu/Unet_training_script/data-and-labelles/annotation-phase-2-random-padded.csv"  # Destination CSV

# Base padding ratio range (uniformly sampled per eye crop).
# Example: to match your prior 0.5 exactly sometimes and vary around it, set min=0.3, max=0.7, etc.
PADDING_RATIO_MIN = 0.4
PADDING_RATIO_MAX = 0.4

# Randomize the center position of the square bbox while keeping all landmarks inside
# and the bbox within image bounds.
RANDOMIZE_CENTER = False

# Global seed for reproducibility (set to None for fully random every run)
RANDOM_SEED = 42

SKIP_MISSING_EYE = False            # If True, skip row entirely when an eye lacks landmarks
EXPECTED_POINTS_PER_EYE = 6         # For validation; just warns if mismatch
DEBUG = True                        # Toggle verbose debug
SHOW_SAMPLE_ROWS = 5                # Number of sample rows to print after processing
# =============================================================================

LEFT_LANDMARK_LABEL = "left-eye-landmarks"
RIGHT_LANDMARK_LABEL = "right-eye-landmarks"
LEFT_VIS_LABEL = "left-eye-visibility"
RIGHT_VIS_LABEL = "right-eye-visibility"


def debug(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def parse_points(points_str: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    if not points_str:
        return pts
    for raw_pair in points_str.strip().split(";"):
        pair = raw_pair.strip()
        if not pair:
            continue
        try:
            x_str, y_str = pair.split(",")
            pts.append((float(x_str), float(y_str)))
        except Exception as e:
            debug(f"Failed parsing point '{pair}' in '{points_str}': {e}")
    return pts


def sample_padding_ratio() -> float:
    # Uniform sample in [PADDING_RATIO_MIN, PADDING_RATIO_MAX]
    if PADDING_RATIO_MIN == PADDING_RATIO_MAX:
        return PADDING_RATIO_MIN
    return random.uniform(PADDING_RATIO_MIN, PADDING_RATIO_MAX)


def square_bbox_with_random_padding_and_center(
    pts: List[Tuple[float, float]],
    img_w: int,
    img_h: int
) -> Tuple[int, int, int, int, float]:
    """
    Create a square bbox around the landmarks with random padding and random center.

    Steps:
      1. Compute tight bbox of landmarks: xmin, xmax, ymin, ymax.
      2. side = max(width, height); if side=0, set to 1.0.
      3. Sample padding_ratio ~ U[PADDING_RATIO_MIN, PADDING_RATIO_MAX].
      4. padded_side = side * (1 + 2*padding_ratio).
      5. Compute half = padded_side / 2.
      6. Compute valid range for the square center that keeps landmarks inside:
         - cx in [xmax - half, xmin + half]
         - cy in [ymax - half, ymin + half]
      7. Intersect with image bounds range for center:
         - cx in [half, (img_w - 1) - half]
         - cy in [half, (img_h - 1) - half]
      8. If RANDOMIZE_CENTER is True, sample cx, cy uniformly in the valid interval.
         Else use the original landmarks bbox center clamped to valid interval.
      9. Compute integer bbox coordinates from sampled center and half.
     10. Final clamp to [0..img_w-1], [0..img_h-1] and ensure ordering.

    Returns:
      (sq_xmin, sq_ymin, sq_xmax, sq_ymax, used_padding_ratio)
    """
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    width = xmax - xmin
    height = ymax - ymin
    side = max(width, height)
    if side <= 0:
        side = 1.0

    padding_ratio = sample_padding_ratio()
    padded_side = side * (1 + 2 * padding_ratio)
    half = padded_side / 2.0

    # Valid center ranges to include all points
    cx_min_include = xmax - half
    cx_max_include = xmin + half
    cy_min_include = ymax - half
    cy_max_include = ymin + half

    # Valid center ranges to keep bbox within image bounds
    cx_min_bounds = half
    cx_max_bounds = (img_w - 1) - half
    cy_min_bounds = half
    cy_max_bounds = (img_h - 1) - half

    # Intersect ranges
    cx_min = max(cx_min_include, cx_min_bounds)
    cx_max = min(cx_max_include, cx_max_bounds)
    cy_min = max(cy_min_include, cy_min_bounds)
    cy_max = min(cy_max_include, cy_max_bounds)

    # If intersection is empty due to extreme cases (e.g., padded_side > image),
    # fall back to clamping the original center and allowing partial inclusion
    # (but still try to keep as much as possible inside image).
    orig_cx = (xmin + xmax) / 2.0
    orig_cy = (ymin + ymax) / 2.0

    def clamp(v, lo, hi):
        if lo > hi:
            # degenerate; pick midpoint of lo and hi swapped (effectively orig clamped to bounds)
            return max(min(v, max(lo, hi)), min(lo, hi))
        return max(min(v, hi), lo)

    if RANDOMIZE_CENTER and cx_min <= cx_max and cy_min <= cy_max:
        cx = random.uniform(cx_min, cx_max)
        cy = random.uniform(cy_min, cy_max)
    else:
        # Deterministic center: original bbox center, then clamp to feasible/bounds
        # First clamp to include range; if invalid, clamp to bounds
        if cx_min <= cx_max:
            cx = clamp(orig_cx, cx_min, cx_max)
        else:
            cx = clamp(orig_cx, cx_min_bounds, cx_max_bounds)
        if cy_min <= cy_max:
            cy = clamp(orig_cy, cy_min, cy_max)
        else:
            cy = clamp(orig_cy, cy_min_bounds, cy_max_bounds)

    sq_xmin = int(round(cx - half))
    sq_xmax = int(round(cx + half))
    sq_ymin = int(round(cy - half))
    sq_ymax = int(round(cy + half))

    # Final clamp
    sq_xmin = max(0, sq_xmin)
    sq_ymin = max(0, sq_ymin)
    sq_xmax = min(img_w - 1, sq_xmax)
    sq_ymax = min(img_h - 1, sq_ymax)

    # Guarantee ordering
    if sq_xmin > sq_xmax:
        sq_xmin, sq_xmax = sq_xmax, sq_xmin
    if sq_ymin > sq_ymax:
        sq_ymin, sq_ymax = sq_ymax, sq_ymin

    return sq_xmin, sq_ymin, sq_xmax, sq_ymax, padding_ratio


def extract_video_id_and_frame_key(image_name: str) -> Tuple[str, str]:
    parts = image_name.strip().split("/")
    if len(parts) < 3:
        debug(f"Unexpected image name format (using fallback): {image_name}")
        return ("UNKNOWN_VIDEO", os.path.splitext(os.path.basename(image_name))[0])
    video_id = parts[-2]
    frame_key = os.path.splitext(parts[-1])[0]
    return video_id, frame_key


def build_image_path(root_dir: str, video_id: str, frame_file: str) -> str:
    return os.path.join(root_dir, video_id, frame_file)


def collect_visibility_tags(image_elem: ET.Element) -> Dict[str, str]:
    vis: Dict[str, str] = {}
    for tag in image_elem.findall("tag"):
        label = tag.get("label", "")
        if label in (LEFT_VIS_LABEL, RIGHT_VIS_LABEL):
            attr = tag.find("attribute[@name='is-visible']")
            vis[label] = (attr.text.strip() if (attr is not None and attr.text) else "")
    return vis


def validate_point_count(side: str, pts: List[Tuple[float, float]]):
    if EXPECTED_POINTS_PER_EYE is not None and len(pts) != EXPECTED_POINTS_PER_EYE:
        debug(f"WARNING: {side} eye landmark count {len(pts)} != expected {EXPECTED_POINTS_PER_EYE}")


def generate_rows(xml_path: str) -> List[Dict[str, str]]:
    debug(f"Starting XML parse: {xml_path}")
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except Exception:
        debug("XML parsing failed; FULL TRACEBACK:")
        traceback.print_exc()
        raise

    root = tree.getroot()
    images = root.findall("image")
    debug(f"Discovered {len(images)} <image> elements.")

    rows: List[Dict[str, str]] = []
    total_images = 0
    missing_left = 0
    missing_right = 0

    for image in images:
        total_images += 1
        image_name = image.get("name", "")
        if not image_name:
            debug("Skipping <image> without 'name' attribute.")
            continue

        try:
            img_w = int(image.get("width", "0"))
            img_h = int(image.get("height", "0"))
        except ValueError:
            debug(f"Invalid width/height for image '{image_name}' - skipping.")
            continue

        video_id, frame_key = extract_video_id_and_frame_key(image_name)
        frame_file = os.path.basename(image_name)
        path_to_dataset = build_image_path(ROOT_DIR, video_id, frame_file)

        if not os.path.isfile(path_to_dataset):
            debug(f"NOTE: Image path does not exist: {path_to_dataset}")

        landmarks_map: Dict[str, List[Tuple[float, float]]] = {}
        raw_points_str_map: Dict[str, str] = {}

        for pe in image.findall("points"):
            label = pe.get("label", "")
            pts_str = pe.get("points", "")
            if not label or not pts_str:
                continue
            pts = parse_points(pts_str)
            if not pts:
                debug(f"No valid points for label '{label}' in '{image_name}'")
                continue
            landmarks_map[label] = pts
            raw_points_str_map[label] = pts_str

        visibility_map = collect_visibility_tags(image)

        for label, side in [(LEFT_LANDMARK_LABEL, "left"), (RIGHT_LANDMARK_LABEL, "right")]:
            pts = landmarks_map.get(label)
            if pts is None:
                if side == "left":
                    missing_left += 1
                else:
                    missing_right += 1

                if SKIP_MISSING_EYE:
                    debug(f"Skipping {side} eye for frame '{frame_key}' (no landmarks).")
                    continue

                row = {
                    "video_id": video_id,
                    "frame_key": frame_key,
                    "eye_side": side,
                    "eye_visibility": visibility_map.get(
                        LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
                    ),
                    "path_to_dataset": path_to_dataset,
                    "eye_bbox_face": "",
                    "landmarks_coordinates_inside_eye_bbox": ""
                }
                rows.append(row)
                debug(f"Appended placeholder row for missing {side} eye: frame_key={frame_key}")
                continue

            validate_point_count(side, pts)

            try:
                xmin, ymin, xmax, ymax, used_pad = square_bbox_with_random_padding_and_center(pts, img_w, img_h)
                bbox_str = f"{xmin},{ymin},{xmax},{ymax}"
            except Exception as e:
                debug(f"ERROR building randomized square bbox for {side} eye in '{image_name}': {e}")
                bbox_str = ""

            visibility = visibility_map.get(
                LEFT_VIS_LABEL if side == "left" else RIGHT_VIS_LABEL, ""
            )

            row = {
                "video_id": video_id,
                "frame_key": frame_key,
                "eye_side": side,
                "eye_visibility": visibility,
                "path_to_dataset": path_to_dataset,
                "eye_bbox_face": bbox_str,
                "landmarks_coordinates_inside_eye_bbox": raw_points_str_map.get(label, "")
            }
            rows.append(row)
            debug(f"Appended {side} eye row (SQUARE + RANDOM): frame_key={frame_key}, bbox={bbox_str}, visibility={visibility}")

    debug("----- SUMMARY -----")
    debug(f"Total <image> elements processed: {total_images}")
    debug(f"Rows generated: {len(rows)} (aim: 2 per image minus skipped/missing eyes)")
    debug(f"Missing left-eye landmarks:  {missing_left}")
    debug(f"Missing right-eye landmarks: {missing_right}")
    debug("-------------------")

    return rows


def write_csv(rows: List[Dict[str, str]], out_csv: str):
    fieldnames = [
        "video_id",
        "frame_key",
        "eye_side",
        "eye_visibility",
        "path_to_dataset",
        "eye_bbox_face",
        "landmarks_coordinates_inside_eye_bbox"
    ]

    out_dir = os.path.dirname(os.path.abspath(out_csv))
    debug(f"Ensuring output directory exists: {out_dir}")

    try:
        os.makedirs(out_dir, exist_ok=True)
    except PermissionError:
        debug(f"Permission denied creating directory '{out_dir}'.")
        raise
    except Exception as e:
        debug(f"Unexpected directory creation error for '{out_dir}': {e}")
        raise

    debug(f"Writing CSV to: {out_csv}")
    try:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    except PermissionError:
        debug(f"Permission denied writing file '{out_csv}'.")
        raise
    except Exception as e:
        debug(f"Unexpected error writing CSV '{out_csv}': {e}")
        raise

    debug("CSV write complete.")


def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    print("=== Eye Landmark CSV Generation (SQUARE BBOX with RANDOM PADDING & CENTER) ===")
    print(f"XML_PATH:             {XML_PATH}")
    print(f"ROOT_DIR:             {ROOT_DIR}")
    print(f"OUT_CSV:              {OUT_CSV}")
    print(f"PADDING_RATIO_MIN:    {PADDING_RATIO_MIN}")
    print(f"PADDING_RATIO_MAX:    {PADDING_RATIO_MAX}")
    print(f"RANDOMIZE_CENTER:     {RANDOMIZE_CENTER}")
    print(f"SKIP_MISSING_EYE:     {SKIP_MISSING_EYE}")
    print(f"EXPECTED_POINTS:      {EXPECTED_POINTS_PER_EYE}")
    print(f"DEBUG:                {DEBUG}")
    print(f"RANDOM_SEED:          {RANDOM_SEED}")

    try:
        rows = generate_rows(XML_PATH)
    except Exception as e:
        print("FATAL: Error during row generation.")
        traceback.print_exc()
        sys.exit(1)

    if rows:
        print(f"\nSample {min(SHOW_SAMPLE_ROWS, len(rows))} rows:")
        for sample in rows[:SHOW_SAMPLE_ROWS]:
            print(sample)
    else:
        print("No rows generated (check XML, labels, paths).")

    try:
        write_csv(rows, OUT_CSV)
    except Exception as e:
        print("FATAL: Error writing CSV.")
        traceback.print_exc()
        sys.exit(2)

    print(f"\nSUCCESS: Wrote {len(rows)} rows to {OUT_CSV}")
    print("Each eye bbox is now square, with randomized padding and randomized center (when possible).")


if __name__ == "__main__":
    main()