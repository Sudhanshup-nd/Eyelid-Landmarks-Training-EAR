import os
import csv
import cv2
import numpy as np

# ── PATHS ─────────────────────────────────────────────────────────────────────
IMG_DIR    = "/inwdata2a/sudhanshu/mouth_keypoints/mouth_frames_dms"
LBL_DIR    = "/inwdata2a/sudhanshu/mouth_keypoints/frame_labels/all_frames"
OUTPUT_CSV = "/inwdata2a/sudhanshu/mouth_keypoints/mouth_labels_all.csv"   # ← new name

# ── LABEL STRUCTURE ───────────────────────────────────────────────────────────
# [0]      class
# [1:5]    face box            (xc, yc, w, h) normalized
# [5:9]    left_eye box        (xc, yc, w, h) normalized
# [9:13]   person box          (xc, yc, w, h) normalized
# [13:17]  right_eye box       (xc, yc, w, h) normalized
# [17:21]  mouth box           (xc, yc, w, h) normalized
# [21:23]  left_shoulder       (x, y) normalized  — body kp, unused
# [23:25]  mouth_left_corner   (x, y) normalized
# [25:27]  mouth_upper_centre  (x, y) normalized
# [27:29]  mouth_right_corner  (x, y) normalized
# [29:31]  mouth_lower_centre  (x, y) normalized

# ── HELPERS ───────────────────────────────────────────────────────────────────
def xywh_to_xyxy(vals):
    xc, yc, w, h = vals
    return np.array([xc - w/2, yc - h/2, xc + w/2, yc + h/2])

def unnormalize_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return (int(x1*img_w), int(y1*img_h), int(x2*img_w), int(y2*img_h))

def unnormalize_kp(kp, img_w, img_h):
    x, y = kp
    return float(x * img_w), float(y * img_h)

def clamp_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return (max(0, min(x1, img_w-1)), max(0, min(y1, img_h-1)),
            max(0, min(x2, img_w-1)), max(0, min(y2, img_h-1)))

def box_to_str(box):
    return ",".join(str(int(v)) for v in box)

def kps_to_str(kps):
    return ";".join(f"{x:.2f},{y:.2f}" for x, y in kps)

# ── MAIN ──────────────────────────────────────────────────────────────────────
all_files   = sorted(os.listdir(LBL_DIR))    # ← all files, sorted for consistency
total       = len(all_files)
rows        = []
skipped     = 0

print(f"Total label files: {total}")
print("Processing...")

for i, lbl_file in enumerate(all_files):

    # ── Progress ───────────────────────────────────────────────────────────────
    if (i + 1) % 1000 == 0 or (i + 1) == total:
        print(f"  [{i+1:6d}/{total}]  rows={len(rows)}  skipped={skipped}")

    # ── Parse filename ─────────────────────────────────────────────────────────
    name      = lbl_file.replace(".txt", "")
    parts     = name.rsplit("_", 1)
    video_id  = parts[0]
    frame_idx = parts[1]

    img_path = os.path.join(IMG_DIR, name + ".png")
    lbl_path = os.path.join(LBL_DIR, lbl_file)

    # ── Load image ─────────────────────────────────────────────────────────────
    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue
    img_h, img_w = img.shape[:2]

    # ── Load label ─────────────────────────────────────────────────────────────
    with open(lbl_path, 'r') as f:
        label = list(map(float, f.readline().strip().split()))

    if len(label) < 31:
        skipped += 1
        continue

    # ── face_bbox_dms ──────────────────────────────────────────────────────────
    face_box_dms = clamp_box(
        unnormalize_box(xywh_to_xyxy(label[1:5]), img_w, img_h),
        img_w, img_h
    )
    fx1, fy1, fx2, fy2 = face_box_dms

    # ── mouth_bbox_face ────────────────────────────────────────────────────────
    mouth_box_dms = clamp_box(
        unnormalize_box(xywh_to_xyxy(label[17:21]), img_w, img_h),
        img_w, img_h
    )
    mx1_dms, my1_dms, mx2_dms, my2_dms = mouth_box_dms
    mouth_bbox_face = (mx1_dms-fx1, my1_dms-fy1, mx2_dms-fx1, my2_dms-fy1)

    # ── 4 mouth keypoints → face crop space ───────────────────────────────────
    kps_face = []
    for kp_norm in [label[23:25], label[25:27], label[27:29], label[29:31]]:
        x_dms, y_dms = unnormalize_kp(kp_norm, img_w, img_h)
        kps_face.append((x_dms - fx1, y_dms - fy1))

    # ── Build row ──────────────────────────────────────────────────────────────
    rows.append({
        "video_id"                               : video_id,
        "frame_key"                              : frame_idx,
        "mouth_visibility"                       : 1,
        "path_to_dataset"                        : img_path,
        "face_bbox_dms"                          : box_to_str(face_box_dms),
        "mouth_bbox_face"                        : box_to_str(mouth_bbox_face),
        "landmarks_coordinates_inside_mouth_bbox": kps_to_str(kps_face),
    })

# ── Write CSV ──────────────────────────────────────────────────────────────────
fieldnames = [
    "video_id",
    "frame_key",
    "mouth_visibility",
    "path_to_dataset",
    "face_bbox_dms",
    "mouth_bbox_face",
    "landmarks_coordinates_inside_mouth_bbox",
]

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'='*50}")
print(f"Saved  : {OUTPUT_CSV}")
print(f"Rows   : {len(rows)}")
print(f"Skipped: {skipped}")