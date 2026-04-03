#!/usr/bin/env python3
"""
Split a single CSV into train.csv, val.csv, and test.csv *by video_id*—
NO video_id will be in more than one split.

Input CSV columns:
video_id,frame_key,eye_side,eye_visibility,path_to_dataset,eye_bbox_face,landmarks_coordinates_inside_eye_bbox

Result:
- Disjoint train/val/test splits by video_id
- Print number of unique video_id and rows (frames) in each split
"""

import pandas as pd
from pathlib import Path
import random
import math

# ------------------ PATHS ------------------
INPUT_CSV   = Path("/inwdata2a/sudhanshu/Unet_training_script/data-and-labelles/annotation-phase-1-and-2-combined.csv")
OUTPUT_DIR  = Path("/inwdata2a/sudhanshu/Unet_training_script/data-and-labelles/final_split")
TRAIN_OUT   = OUTPUT_DIR / "train.csv"
VAL_OUT     = OUTPUT_DIR / "val.csv"
TEST_OUT    = OUTPUT_DIR / "test.csv"

# Split ratios
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

RANDOM_SEED = 42

MIN_TRAIN_VIDEOS = 1
MIN_VAL_VIDEOS   = 1
MIN_TEST_VIDEOS  = 1
# --------------------------------------------

def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float):
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0. Got {total:.6f}.")

def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")
    validate_ratios(TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    if "video_id" not in df.columns:
        raise ValueError("Expected column 'video_id' not found in CSV.")

    # Unique video IDs
    video_ids = sorted(df["video_id"].unique())
    if len(video_ids) == 0:
        raise ValueError("No video_id values found in CSV.")

    # Shuffle video IDs for randomization
    random.seed(RANDOM_SEED)
    random.shuffle(video_ids)

    # Split counts
    n_total = len(video_ids)
    n_train_target = int(round(n_total * TRAIN_RATIO))
    n_val_target   = int(round(n_total * VAL_RATIO))
    n_test_target  = n_total - n_train_target - n_val_target

    # Minimums
    n_train = max(n_train_target, MIN_TRAIN_VIDEOS)
    n_val   = max(n_val_target, MIN_VAL_VIDEOS)
    n_test  = max(n_test_target, MIN_TEST_VIDEOS)

    # Fix over-allocation if sum > n_total (after minimums)
    total_assigned = n_train + n_val + n_test
    if total_assigned > n_total:
        over = total_assigned - n_total
        for var in ['n_train', 'n_val', 'n_test']:
            take = min(over, locals()[var])
            locals()[var] -= take
            over -= take
            if over == 0: break

    # Assign splits
    idx_train = n_train
    idx_val   = idx_train + n_val
    train_video_ids = set(video_ids[:idx_train])
    val_video_ids   = set(video_ids[idx_train:idx_val])
    test_video_ids  = set(video_ids[idx_val:])

    # Disjointness
    assert len(train_video_ids & val_video_ids) == 0
    assert len(train_video_ids & test_video_ids) == 0
    assert len(val_video_ids & test_video_ids) == 0

    # Dataframes per split
    train_df = df[df["video_id"].isin(train_video_ids)].copy()
    val_df   = df[df["video_id"].isin(val_video_ids)].copy()
    test_df  = df[df["video_id"].isin(test_video_ids)].copy()

    # Save splits
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    test_df.to_csv(TEST_OUT, index=False)

    # Print stats
    def stats(df, video_ids, name):
        n_vids = len(video_ids)
        n_frames = len(df)
        print(f"[{name.upper()}] videos: {n_vids} | frames/rows: {n_frames}")

    print(f"Total unique videos: {n_total}")
    stats(train_df, train_video_ids, "train")
    stats(val_df, val_video_ids, "val")
    stats(test_df, test_video_ids, "test")
    print(f"Train CSV: {TRAIN_OUT}")
    print(f"Val CSV:   {VAL_OUT}")
    print(f"Test CSV:  {TEST_OUT}")
    print("Done.")

if __name__ == "__main__":
    main()